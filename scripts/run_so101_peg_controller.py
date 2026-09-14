#!/usr/bin/env python3
"""Execute the SO-101 peg state machine through real Newton joint commands.

The script commissions named task-space waypoints with Newton FK, resets the
scene, and then executes only rate-limited joint-position commands through the
normal Isaac Lab actuator path.  It records enough state to distinguish joint
motion, a grasp/lift, transport and insertion instead of inferring success from
the animation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import traceback

if os.environ.get("ACCEPT_EULA") != "Y":
    raise RuntimeError("Review the NVIDIA Isaac Sim license and export ACCEPT_EULA=Y before running.")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", required=True)
parser.add_argument("--output", default="/workspace/output/so101_peg_controller")
parser.add_argument("--mode", choices=("mvp1", "mvp2", "mvp3"), default="mvp3")
parser.add_argument("--max-seconds", type=float, default=38.0)
parser.add_argument("--render", action="store_true")
parser.add_argument("--arm-stiffness", type=float)
parser.add_argument("--arm-damping", type=float)
parser.add_argument("--arm-effort-limit", type=float)
parser.add_argument(
    "--tcp-servo-gain",
    type=float,
    default=5.0,
    help=(
        "Free-space TCP proportional gain in 1/s. The 6 mm/s descent and "
        "1.5 mm tracking gate require a value greater than 4 1/s to leave "
        "finite steady-state margin."
    ),
)
parser.add_argument(
    "--lift-speed",
    type=float,
    default=0.020,
    help=(
        "Audited Cartesian lift speed in m/s. Commissioning may slow the lift "
        "to diagnose contact retention, but may not exceed the 0.020 m/s default."
    ),
)
parser.add_argument("--gripper-stiffness", type=float)
parser.add_argument("--gripper-damping", type=float)
parser.add_argument("--gripper-effort-limit", type=float)
parser.add_argument(
    "--gripper-close-rate",
    type=float,
    default=0.20,
    help=(
        "Maximum closing command rate in rad/s. The default keeps first contact "
        "inside the SO-101 pad-commissioning envelope; opening remains 1.25 rad/s."
    ),
)
parser.add_argument("--grasp-height", type=float)
parser.add_argument("--transport-height", type=float)
parser.add_argument("--align-height", type=float)
parser.add_argument(
    "--tcp-offset-x",
    type=float,
    default=0.00155,
    help=(
        "Task-TCP X offset in the gripper frame. This is a commissioned grasp "
        "geometry value, not a Newton physics parameter."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch

from newton_calibration.isaaclab.tasks.so101_peg_insertion import (
    BilateralContactGripperLatch,
    ControllerObservation,
    ControllerPhase,
    PegInsertionController,
    PegInsertionMode,
    PegInsertionSceneSpec,
    SO101JointCommandAdapter,
    advance_tcp_setpoint,
    interpolate_phase_servo_nominal,
    resolve_tcp_setpoint_speeds,
)
from newton_calibration.isaaclab.tasks.so101_peg_insertion.run_evaluation import (
    BilateralGraspContactTracker,
    GuardedInsertionTracker,
    PreInsertionReadinessThresholds,
    PreInsertionReadinessTracker,
    PrecontactTCPTrackingTracker,
    PrecloseDisturbanceTracker,
    TaskEvidenceSample,
    TaskEvidenceTracker,
)
from newton_calibration.isaaclab.tasks.so101_peg_insertion.scene import SO101PegInsertionScene


MODE = {
    "mvp1": PegInsertionMode.ROBOT_ONLY,
    "mvp2": PegInsertionMode.GRASP_TRANSPORT,
    "mvp3": PegInsertionMode.INSERTION,
}


TCP_SERVO_PHASES = {
    ControllerPhase.APPROACH_PEG,
    ControllerPhase.STABILIZE_ABOVE_PEG,
    ControllerPhase.DESCEND_TO_GRASP,
    ControllerPhase.CLOSE,
    ControllerPhase.VERIFY_GRASP,
    ControllerPhase.LIFT,
    ControllerPhase.MOVE_ABOVE_SOCKET,
    ControllerPhase.ALIGN,
    ControllerPhase.GUARDED_INSERT,
    ControllerPhase.VERIFY_DEPTH,
    ControllerPhase.COMPLETE,
    ControllerPhase.ABORT,
}


def _tensor_numpy(value) -> np.ndarray:
    tensor = value.torch if hasattr(value, "torch") else value
    return tensor[0].detach().cpu().numpy().astype(np.float64, copy=True)


def _peg_pose(scene: SO101PegInsertionScene) -> np.ndarray:
    return _tensor_numpy(scene.peg.data.root_pose_w)


def _peg_velocity(scene: SO101PegInsertionScene) -> np.ndarray:
    return _tensor_numpy(scene.peg.data.root_vel_w)


def _peg_position(scene: SO101PegInsertionScene) -> np.ndarray:
    return _peg_pose(scene)[:3]


def _phase_solution_name(
    phase: ControllerPhase,
    mode: PegInsertionMode,
) -> str | None:
    if phase is ControllerPhase.COMPLETE:
        return {
            PegInsertionMode.ROBOT_ONLY: "approach_peg",
            PegInsertionMode.GRASP_TRANSPORT: "align_socket",
            PegInsertionMode.INSERTION: "insert_socket",
        }[mode]
    return {
        ControllerPhase.APPROACH_PEG: "approach_peg",
        ControllerPhase.STABILIZE_ABOVE_PEG: "approach_peg",
        ControllerPhase.DESCEND_TO_GRASP: "grasp_peg",
        ControllerPhase.CLOSE: "grasp_peg",
        ControllerPhase.VERIFY_GRASP: "grasp_peg",
        ControllerPhase.LIFT: "lift_peg",
        ControllerPhase.MOVE_ABOVE_SOCKET: "above_socket",
        ControllerPhase.ALIGN: "align_socket",
        ControllerPhase.GUARDED_INSERT: "insert_socket",
        ControllerPhase.VERIFY_DEPTH: "insert_socket",
    }.get(phase)


def _commission(
    scene: SO101PegInsertionScene,
    adapter: SO101JointCommandAdapter,
) -> dict[str, object]:
    spec = scene.spec
    px, py = spec.peg_start_xy_m
    sx, sy = spec.socket_center_xy_m
    targets = [
        ("approach_peg", (px, py, spec.transport_tcp_height_m)),
        ("grasp_peg", (px, py, spec.grasp_tcp_height_m)),
        ("lift_peg", (px, py, spec.transport_tcp_height_m)),
        ("above_socket", (sx, sy, spec.transport_tcp_height_m)),
        ("align_socket", (sx, sy, spec.align_tcp_height_m)),
        ("insert_socket", (sx, sy, spec.insertion_tcp_height_m)),
    ]
    targets = targets[
        : {
            PegInsertionMode.ROBOT_ONLY: 1,
            PegInsertionMode.GRASP_TRANSPORT: 5,
            PegInsertionMode.INSERTION: 6,
        }[scene.mode]
    ]
    downward_pitch_sum = 0.5 * math.pi
    seed = np.asarray([0.30, -0.40, 1.20, 0.77, 0.0, adapter.open_gripper_rad])
    solutions = {}
    for name, target in targets:
        solution = adapter.solve_tcp_position(
            name,
            target,
            seed_joint_position_rad=seed,
            wrist_pitch_sum_rad=downward_pitch_sum,
            position_tolerance_m=0.0008 if name in {"align_socket", "insert_socket"} else 0.0015,
        )
        solutions[name] = solution
        if not solution.converged:
            raise RuntimeError(f"Waypoint {name!r} did not converge: {solution.to_dict()}")
        seed = np.asarray(solution.joint_position_rad, dtype=np.float64)
    return solutions


def _rate_limit(
    command: np.ndarray,
    desired: np.ndarray,
    *,
    dt_s: float,
    arm_rate_rad_s: float = 0.75,
    gripper_rate_rad_s: float = 1.25,
) -> np.ndarray:
    limit = np.asarray([arm_rate_rad_s] * 5 + [gripper_rate_rad_s], dtype=np.float64) * dt_s
    return command + np.clip(desired - command, -limit, limit)


def main() -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(args.tcp_servo_gain) or args.tcp_servo_gain <= 4.0:
        raise ValueError(
            "--tcp-servo-gain must be finite and greater than 4 1/s so the "
            "6 mm/s descent has margin inside the 1.5 mm tracking gate"
        )
    tcp_speed_by_phase = resolve_tcp_setpoint_speeds(
        lift_speed_m_s=args.lift_speed,
    )
    if (
        not math.isfinite(args.gripper_close_rate)
        or args.gripper_close_rate <= 0.0
        or args.gripper_close_rate > 1.25
    ):
        raise ValueError(
            "--gripper-close-rate must be finite, positive, and no greater than "
            "1.25 rad/s"
        )
    if not math.isfinite(args.tcp_offset_x) or abs(args.tcp_offset_x) > 0.02:
        raise ValueError("--tcp-offset-x must be finite and within +/-20 mm")
    spec = PegInsertionSceneSpec()
    overrides = {
        name: value
        for name, value in {
            "arm_control_stiffness_nm_per_rad": args.arm_stiffness,
            "arm_control_damping_nm_s_per_rad": args.arm_damping,
            "arm_control_effort_limit_nm": args.arm_effort_limit,
            "gripper_control_stiffness_nm_per_rad": args.gripper_stiffness,
            "gripper_control_damping_nm_s_per_rad": args.gripper_damping,
            "gripper_control_effort_limit_nm": args.gripper_effort_limit,
            "grasp_tcp_height_m": args.grasp_height,
            "transport_tcp_height_m": args.transport_height,
            "align_tcp_height_m": args.align_height,
        }.items()
        if value is not None
    }
    if overrides:
        spec = replace(spec, **overrides)
    if args.grasp_height is not None:
        # The TCP-to-peg-center relation is part of the grasp geometry.  Keep
        # insertion/retention logic coupled to a user-selected grasp station
        # rather than silently retaining the original 55 mm station's offset.
        nominal_grasp_offset_m = args.grasp_height - spec.peg_rest_center_z_m
        if nominal_grasp_offset_m <= 0.0:
            raise ValueError("--grasp-height must remain above the nominal peg center")
        spec = replace(
            spec,
            grasp_tcp_above_peg_center_m=nominal_grasp_offset_m,
        )
        spec.validate()
    scene = None
    try:
        scene = SO101PegInsertionScene(
            usd_path=args.asset,
            device=args.device,
            mode=MODE[args.mode],
            spec=spec,
        )
        tcp_offset_gripper_m = (
            args.tcp_offset_x,
            0.0,
            -0.085,
        )
        adapter = SO101JointCommandAdapter(
            scene,
            tcp_offset_gripper_m=tcp_offset_gripper_m,
        )
        solutions = _commission(scene, adapter)
        scene.reset()
        # Contact modes start at the commissioned pre-grasp pose.  The earlier
        # level-yaw sweep was kinematically collision-free at the TCP but the
        # open finger geometry could clip the peg before the grasp phase.  MVP1
        # retains that yaw move because no contact outcome is under test there.
        home_q = np.asarray(solutions["approach_peg"].joint_position_rad, dtype=np.float64)
        if MODE[args.mode] is PegInsertionMode.ROBOT_ONLY:
            home_q[0] = 0.0
        home_q[5] = adapter.open_gripper_rad
        scene.write_kinematic_joint_state(home_q)
        adapter = SO101JointCommandAdapter(
            scene,
            tcp_offset_gripper_m=tcp_offset_gripper_m,
        )
        controller = PegInsertionController(
            scene.spec,
            mode=MODE[args.mode],
            settle_s=1.0,
            close_s=(
                adapter.open_gripper_rad - adapter.closed_gripper_rad
            )
            / args.gripper_close_rate
            + 0.25,
            max_phase_s=8.0,
            max_lift_phase_s=max(
                8.0,
                (
                    scene.spec.transport_tcp_height_m
                    - scene.spec.grasp_tcp_height_m
                )
                / tcp_speed_by_phase[ControllerPhase.LIFT]
                + 2.0,
            ),
        )

        initial_q = scene.joint_position_rad()
        command_q = initial_q.copy()
        applied_joint_torque_available = (
            getattr(scene.robot.data, "applied_torque", None) is not None
        )
        if (
            MODE[args.mode] is PegInsertionMode.INSERTION
            and not applied_joint_torque_available
        ):
            raise RuntimeError(
                "MVP3 requires Newton applied joint torque for the guarded-insertion "
                "effort limit; refusing to substitute an all-zero signal"
            )
        initial_tcp = adapter.current_tcp_position()
        initial_peg_pose = _peg_pose(scene)
        initial_peg = initial_peg_pose[:3]
        maximum_peg_height = float(initial_peg[2])
        minimum_relative_distance: float | None = None
        phase_first_seen: dict[str, float] = {}
        records: list[dict[str, object]] = []
        evidence_tracker = TaskEvidenceTracker()
        preclose_tracker = PrecloseDisturbanceTracker()
        precontact_tracking_tracker = PrecontactTCPTrackingTracker()
        bilateral_grasp_tracker = BilateralGraspContactTracker()
        gripper_contact_latch = BilateralContactGripperLatch(
            minimum_partner_force_n=(
                bilateral_grasp_tracker.thresholds.minimum_partner_force_n
            )
        )
        guarded_insertion_tracker = GuardedInsertionTracker()
        effective_radial_clearance_m = (
            scene.spec.radial_clearance_m - scene.spec.contact_gap_m
        )
        pre_insertion_readiness_tracker = PreInsertionReadinessTracker(
            PreInsertionReadinessThresholds(
                effective_radial_clearance_m=effective_radial_clearance_m,
                insertion_depth_basis_m=scene.spec.target_insertion_depth_m,
                maximum_lateral_offset_m=0.5 * effective_radial_clearance_m,
                maximum_tilt_deg=1.0,
                minimum_peg_bottom_clearance_m=0.00050,
                maximum_pre_guard_socket_force_n=0.05,
                readiness_window_s=0.10,
            )
        )
        preclose_abort_reason: str | None = None
        precontact_tracking_abort_reason: str | None = None
        last_phase: ControllerPhase | None = None
        finite_stable = True
        # A kinematically valid reset pose is not necessarily dynamically
        # supportable by the selected actuator profile.  Qualify the profile
        # before allowing the state machine to approach the workpiece.
        effort_caps_nm = np.asarray(
            [scene.spec.arm_control_effort_limit_nm] * 5
            + [scene.spec.gripper_control_effort_limit_nm],
            dtype=np.float64,
        )
        settle_sample_count = 0
        settle_saturation_count = np.zeros(len(scene.joint_names), dtype=np.int64)
        maximum_settle_tcp_drift_m = 0.0
        hold_authority_evaluated = False
        hold_authority_passed = False
        task_dt = scene.spec.simulation_dt_s * scene.spec.controller_decimation
        max_steps = int(args.max_seconds / scene.spec.simulation_dt_s)
        command = None
        last_tcp_servo = None
        tcp_setpoint_m: np.ndarray | None = None
        last_tcp_setpoint = None
        servo_phase: ControllerPhase | None = None
        servo_phase_entry_q: np.ndarray | None = None
        servo_phase_entry_tcp_setpoint_m: np.ndarray | None = None

        for physics_step in range(max_steps):
            now_s = physics_step * scene.spec.simulation_dt_s
            if physics_step % scene.spec.controller_decimation == 0:
                tcp = adapter.current_tcp_position()
                peg_pose = _peg_pose(scene)
                peg = peg_pose[:3]
                peg_velocity = _peg_velocity(scene)
                measured_q = scene.joint_position_rad()
                measured_dq = scene.joint_velocity_rad_s()
                torque = scene.applied_joint_torque_nm()
                contact_force_evidence = scene.contact_force_evidence_n()
                if not all(
                    np.isfinite(values).all()
                    for values in (
                        tcp,
                        peg_pose,
                        peg_velocity,
                        measured_q,
                        measured_dq,
                        torque,
                    )
                ):
                    finite_stable = False
                    controller.phase = ControllerPhase.ABORT
                    print(f"[controller] t={now_s:6.2f}s abort=non_finite_state", flush=True)
                    break
                relative = tcp - peg
                relative_distance = float(np.linalg.norm(relative))
                minimum_relative_distance = (
                    relative_distance
                    if minimum_relative_distance is None
                    else min(minimum_relative_distance, relative_distance)
                )
                maximum_peg_height = max(maximum_peg_height, float(peg[2]))

                near_grasp_tcp = (
                    float(np.linalg.norm(relative[:2])) <= 0.004
                    and abs(float(relative[2]) - scene.spec.grasp_tcp_above_peg_center_m) <= 0.004
                )
                gripper_gap_m = adapter.estimated_gripper_aperture_m(measured_q[5])
                jaw_near_closed = bool(
                    gripper_gap_m <= scene.spec.peg_diameter_m + 0.003
                )
                bilateral_contact_qualified = bilateral_grasp_tracker.update(
                    time_s=now_s,
                    phase=controller.phase.value,
                    fixed_finger_force_n=float(
                        contact_force_evidence["fixed_finger_force_n"]
                    ),
                    moving_finger_force_n=float(
                        contact_force_evidence["moving_finger_force_n"]
                    ),
                )
                gripper_contact_latch.update(
                    time_s=now_s,
                    phase=controller.phase,
                    current_command_rad=float(command_q[5]),
                    fixed_finger_force_n=float(
                        contact_force_evidence["fixed_finger_force_n"]
                    ),
                    moving_finger_force_n=float(
                        contact_force_evidence["moving_finger_force_n"]
                    ),
                )
                grasp_candidate = bool(
                    near_grasp_tcp
                    and jaw_near_closed
                    and bilateral_contact_qualified
                )
                current_peg_lift_m = float(peg[2] - initial_peg[2])
                lifted = bool(
                    evidence_tracker.thresholds.minimum_lift_m
                    <= current_peg_lift_m
                    <= evidence_tracker.thresholds.maximum_lift_m
                )
                metrics = scene.task_metrics(settled=False)
                peg_tilt_rad = math.radians(metrics.tilt_deg)
                peg_vertical_half_extent_m = (
                    0.5 * scene.spec.peg_height_m * abs(math.cos(peg_tilt_rad))
                    + scene.spec.peg_radius_m * abs(math.sin(peg_tilt_rad))
                )
                peg_bottom_clearance_m = float(
                    peg[2]
                    - peg_vertical_half_extent_m
                    - scene.spec.socket_top_z_m
                )
                if MODE[args.mode] is PegInsertionMode.INSERTION:
                    pre_insertion_ready = pre_insertion_readiness_tracker.update(
                        time_s=now_s,
                        phase=controller.phase.value,
                        lateral_offset_m=metrics.lateral_offset_m,
                        tilt_deg=metrics.tilt_deg,
                        peg_bottom_clearance_m=peg_bottom_clearance_m,
                        socket_normal_force_n=float(
                            contact_force_evidence["socket_normal_force_n"]
                        ),
                    )
                    pre_insertion_gate_failed = (
                        pre_insertion_readiness_tracker.violated
                    )
                else:
                    pre_insertion_ready = False
                    pre_insertion_gate_failed = False
                pre_insertion_readiness_summary = (
                    pre_insertion_readiness_tracker.summary()
                )
                arm_effort_fraction = float(
                    np.max(np.abs(torque[:5]) / effort_caps_nm[:5])
                )
                insertion_guard_clear = guarded_insertion_tracker.update(
                    time_s=now_s,
                    phase=controller.phase.value,
                    insertion_depth_m=metrics.insertion_depth_m,
                    target_insertion_depth_m=scene.spec.target_insertion_depth_m,
                    maximum_effort_fraction=arm_effort_fraction,
                    socket_normal_force_n=float(
                        contact_force_evidence["socket_normal_force_n"]
                    ),
                )
                insertion_guard_summary = guarded_insertion_tracker.summary()
                insertion_stalled = bool(
                    not insertion_guard_clear
                    and insertion_guard_summary["stop_reason"]
                    == "insertion_depth_progress_stalled"
                )
                precontact_tracking_clear = True
                if tcp_setpoint_m is not None:
                    precontact_tracking_clear = precontact_tracking_tracker.update(
                        time_s=now_s,
                        phase=controller.phase.value,
                        measured_tcp_position_m=tuple(float(value) for value in tcp),
                        commanded_tcp_setpoint_m=tuple(
                            float(value) for value in tcp_setpoint_m
                        ),
                    )
                if (
                    not precontact_tracking_clear
                    and precontact_tracking_abort_reason is None
                ):
                    tracking = precontact_tracking_tracker.summary()
                    first = tracking["first_violation"]
                    precontact_tracking_abort_reason = (
                        "precontact_tcp_tracking_error: "
                        f"lateral={first['lateral_error_m']:.6f}m, "
                        f"vertical={first['vertical_error_m']:.6f}m"
                    )
                    print(
                        f"[controller] t={now_s:6.2f}s "
                        f"abort={precontact_tracking_abort_reason}",
                        flush=True,
                    )
                if controller.phase is ControllerPhase.SETTLE:
                    settle_sample_count += 1
                    maximum_settle_tcp_drift_m = max(
                        maximum_settle_tcp_drift_m,
                        float(np.linalg.norm(tcp - initial_tcp)),
                    )
                    settle_saturation_count += (
                        np.abs(torque) >= 0.98 * effort_caps_nm
                    ).astype(np.int64)
                elif not hold_authority_evaluated:
                    hold_authority_evaluated = True
                    saturation_fraction = settle_saturation_count / max(settle_sample_count, 1)
                    hold_authority_passed = (
                        maximum_settle_tcp_drift_m <= 0.005
                        and bool(np.all(saturation_fraction < 0.25))
                    )
                    if not hold_authority_passed:
                        controller.phase = ControllerPhase.ABORT
                        controller.phase_time_s = 0.0
                        print(
                            f"[controller] t={now_s:6.2f}s abort=hold_authority "
                            f"tcp_drift_m={maximum_settle_tcp_drift_m:.6f} "
                            f"max_saturation_fraction={float(np.max(saturation_fraction)):.3f}",
                            flush=True,
                        )
                inferred_events = evidence_tracker.update(
                    TaskEvidenceSample(
                        time_s=now_s,
                        phase=controller.phase.value,
                        tcp_position_m=tuple(float(value) for value in tcp),
                        peg_position_m=tuple(float(value) for value in peg),
                        peg_linear_velocity_m_s=tuple(
                            float(value) for value in peg_velocity[:3]
                        ),
                        peg_angular_velocity_rad_s=tuple(
                            float(value) for value in peg_velocity[3:]
                        ),
                        grasp_candidate=grasp_candidate,
                        peg_lift_m=current_peg_lift_m,
                        insertion_depth_m=metrics.insertion_depth_m,
                        lateral_offset_m=metrics.lateral_offset_m,
                        tilt_deg=metrics.tilt_deg,
                        target_insertion_depth_m=scene.spec.target_insertion_depth_m,
                        lateral_tolerance_m=scene.spec.success_lateral_tolerance_m,
                        tilt_tolerance_deg=scene.spec.success_tilt_tolerance_deg,
                        inferred_jam_candidate=insertion_stalled,
                    )
                )
                lost_after_lift = bool(
                    inferred_events.drop_candidate
                    or inferred_events.full_transfer_violated
                )
                observation = ControllerObservation(
                    ee_position_m=tuple(float(value) for value in tcp),
                    peg_position_m=tuple(float(value) for value in peg),
                    gripper_gap_m=gripper_gap_m,
                    insertion_depth_m=metrics.insertion_depth_m,
                    peg_grasped=grasp_candidate or inferred_events.retention_candidate,
                    peg_dropped=lost_after_lift,
                    jammed=insertion_stalled,
                    force_limit_exceeded=False,
                    guard_stop_requested=(
                        not insertion_guard_clear
                        or not precontact_tracking_clear
                    ),
                    pre_insertion_ready=pre_insertion_ready,
                    pre_insertion_gate_failed=pre_insertion_gate_failed,
                )
                command = controller.step(observation, task_dt)
                preclose_clear = preclose_tracker.update(
                    phase=command.phase.value,
                    peg_position_m=tuple(float(value) for value in peg),
                    fixed_finger_force_n=float(
                        contact_force_evidence["fixed_finger_force_n"]
                    ),
                    moving_finger_force_n=float(
                        contact_force_evidence["moving_finger_force_n"]
                    ),
                )
                if not preclose_clear and command.phase in {
                    ControllerPhase.APPROACH_PEG,
                    ControllerPhase.STABILIZE_ABOVE_PEG,
                    ControllerPhase.DESCEND_TO_GRASP,
                }:
                    preclose = preclose_tracker.summary()
                    preclose_abort_reason = (
                        "peg_moved_before_close: "
                        f"xy={preclose['maximum_xy_drift_m']:.6f}m, "
                        f"z={preclose['maximum_z_drift_m']:.6f}m, "
                        "gripper_contact="
                        f"{preclose['maximum_gripper_contact_force_n']:.3f}N"
                    )
                    command = controller.request_abort(observation)
                    print(
                        f"[controller] t={now_s:6.2f}s abort={preclose_abort_reason}",
                        flush=True,
                    )
                phase_first_seen.setdefault(command.phase.value, now_s)
                if command.phase is not last_phase:
                    print(
                        f"[controller] t={now_s:6.2f}s phase={command.phase.value} "
                        f"tcp={tcp.round(4).tolist()} peg={peg.round(4).tolist()}",
                        flush=True,
                    )
                    last_phase = command.phase

                desired_q = command_q.copy()
                solution_name = _phase_solution_name(command.phase, MODE[args.mode])
                if solution_name is None:
                    if command.phase is ControllerPhase.SETTLE:
                        desired_q = initial_q.copy()
                        desired_q[5] = adapter.open_gripper_rad
                else:
                    desired_q = adapter.command_from_solution(
                        solutions[solution_name],
                        gripper_closed=command.gripper_closed,
                    )
                    if command.gripper_closed:
                        desired_q[5] = gripper_contact_latch.target(desired_q[5])

                records.append(
                    {
                        "time_s": now_s,
                        "phase": command.phase.value,
                        "command_q_rad": [float(value) for value in command_q],
                        "desired_q_rad": [float(value) for value in desired_q],
                        "measured_q_rad": [float(value) for value in measured_q],
                        "measured_dq_rad_s": [float(value) for value in measured_dq],
                        "applied_torque_nm": [float(value) for value in torque],
                        "gripper_gap_m": gripper_gap_m,
                        "gripper_closed_commanded": command.gripper_closed,
                        "gripper_contact_latch": gripper_contact_latch.summary(),
                        "tcp_position_m": [float(value) for value in tcp],
                        "peg_position_m": [float(value) for value in peg],
                        "peg_pose_world_m_xyzw": [
                            float(value) for value in peg_pose
                        ],
                        "peg_velocity_world_m_s_rad_s": [
                            float(value) for value in peg_velocity
                        ],
                        "peg_tcp_relative_m": [float(value) for value in relative],
                        "peg_position_relative_to_tcp_m": [
                            float(value) for value in -relative
                        ],
                        "grasp_candidate": grasp_candidate,
                        "peg_lifted": lifted,
                        "inferred_events": inferred_events.to_dict(),
                        "raw_contact_force_evidence_n": contact_force_evidence,
                        "bilateral_grasp_contact": bilateral_grasp_tracker.summary(),
                        "guarded_insertion": insertion_guard_summary,
                        "insertion_depth_m": metrics.insertion_depth_m,
                        "lateral_offset_m": metrics.lateral_offset_m,
                        "tilt_deg": metrics.tilt_deg,
                        "peg_bottom_clearance_above_socket_rim_m": (
                            peg_bottom_clearance_m
                        ),
                        "pre_insertion_readiness": (
                            pre_insertion_readiness_summary
                        ),
                        "precontact_tcp_tracking": (
                            precontact_tracking_tracker.summary()
                        ),
                        "tcp_servo": last_tcp_servo.to_dict() if last_tcp_servo is not None else None,
                        "tcp_setpoint": (
                            last_tcp_setpoint.to_dict()
                            if last_tcp_setpoint is not None
                            else None
                        ),
                    }
                )
            if command is None:
                continue

            solution_name = _phase_solution_name(command.phase, MODE[args.mode])
            desired_q = command_q.copy()
            if solution_name is None:
                if command.phase is ControllerPhase.SETTLE:
                    desired_q = initial_q.copy()
                    desired_q[5] = adapter.open_gripper_rad
            else:
                nominal_q = adapter.command_from_solution(
                    solutions[solution_name],
                    gripper_closed=command.gripper_closed,
                )
                if command.gripper_closed:
                    nominal_q[5] = gripper_contact_latch.target(nominal_q[5])
                desired_q = nominal_q
            if command.phase in TCP_SERVO_PHASES:
                # IK supplies the feed-forward posture.  Newton's measured link
                # Jacobian follows a progressive Cartesian setpoint rather than
                # seeing the distant phase endpoint in one discontinuous step.
                # Cartesian speed and joint speed are independent safety bounds:
                # the former protects the workpiece, the latter the mechanism.
                final_tcp_target = command.target_position_m
                if solution_name is None:
                    # ABORT has no commissioned endpoint: retain the current
                    # arm posture as the null-space nominal while retreating,
                    # and open the gripper under its independent rate limit.
                    nominal_q = command_q.copy()
                    nominal_q[5] = adapter.open_gripper_rad
                if tcp_setpoint_m is None:
                    tcp_setpoint_m = adapter.current_tcp_position()
                if servo_phase is not command.phase:
                    servo_phase = command.phase
                    servo_phase_entry_q = command_q.copy()
                    servo_phase_entry_tcp_setpoint_m = tcp_setpoint_m.copy()
                last_tcp_setpoint = advance_tcp_setpoint(
                    current_setpoint_m=tcp_setpoint_m,
                    final_target_m=final_tcp_target,
                    max_speed_m_s=tcp_speed_by_phase[command.phase],
                    dt_s=scene.spec.simulation_dt_s,
                )
                tcp_setpoint_m = np.asarray(
                    last_tcp_setpoint.position_m,
                    dtype=np.float64,
                )
                # Blend the null-space reference from the phase-entry command
                # to the endpoint IK using the same Cartesian setpoint
                # progress.  This prevents DESCEND from jumping toward the
                # final grasp IK faster than the audited Cartesian setpoint.
                assert servo_phase_entry_q is not None
                assert servo_phase_entry_tcp_setpoint_m is not None
                servo_nominal_q = interpolate_phase_servo_nominal(
                    servo_phase_entry_q,
                    nominal_q,
                    servo_phase_entry_tcp_setpoint_m,
                    tcp_setpoint_m,
                    final_tcp_target,
                    arm_joint_count=adapter.arm_joint_count,
                )
                last_tcp_servo = adapter.correct_tcp_position_command(
                    target_tcp_position_m=tcp_setpoint_m,
                    nominal_joint_position_rad=servo_nominal_q,
                    previous_joint_command_rad=command_q,
                    dt_s=scene.spec.simulation_dt_s,
                    max_arm_rate_rad_s=(
                        0.12
                        if command.guarded_contact
                        else 0.20 if command.phase is ControllerPhase.ABORT else 0.35
                    ),
                    proportional_gain_per_s=(
                        2.5
                        if command.guarded_contact
                        else 3.0
                        if command.phase is ControllerPhase.ABORT
                        else args.tcp_servo_gain
                    ),
                    max_position_error_m=(
                        0.004
                        if command.guarded_contact
                        else 0.010 if command.phase is ControllerPhase.ABORT else 0.015
                    ),
                    target_wrist_pitch_sum_rad=0.5 * math.pi,
                    max_gripper_rate_rad_s=(
                        args.gripper_close_rate
                        if command.gripper_closed
                        else 1.25
                    ),
                )
                command_q = np.asarray(last_tcp_servo.joint_position_rad, dtype=np.float64)
            else:
                last_tcp_servo = None
                last_tcp_setpoint = None
                tcp_setpoint_m = None
                servo_phase = None
                servo_phase_entry_q = None
                servo_phase_entry_tcp_setpoint_m = None
                command_q = _rate_limit(
                    command_q,
                    desired_q,
                    dt_s=scene.spec.simulation_dt_s,
                    arm_rate_rad_s=0.28 if command.guarded_contact else 0.75,
                )
            scene.step(
                torch.as_tensor(command_q, dtype=torch.float32, device=scene.device),
                render=args.render,
            )

            if controller.phase in {ControllerPhase.COMPLETE, ControllerPhase.ABORT}:
                # Hold long enough for the final classification to be physical,
                # not just a transient pose crossing.
                if controller.phase_time_s >= 0.5:
                    break

        final_peg_pose = _peg_pose(scene)
        final_peg = final_peg_pose[:3]
        final_peg_velocity = _peg_velocity(scene)
        final_is_finite = bool(
            np.isfinite(final_peg_pose).all()
            and np.isfinite(final_peg_velocity).all()
        )
        finite_stable = finite_stable and final_is_finite
        final_metrics = scene.task_metrics(settled=True) if final_is_finite else None
        evidence_summary = evidence_tracker.summary()
        preclose_summary = preclose_tracker.summary()
        precontact_tracking_summary = precontact_tracking_tracker.summary()
        bilateral_grasp_summary = bilateral_grasp_tracker.summary()
        guarded_insertion_summary = guarded_insertion_tracker.summary()
        pre_insertion_readiness_summary = (
            pre_insertion_readiness_tracker.summary()
        )
        saturation_fraction = settle_saturation_count / max(settle_sample_count, 1)
        if not hold_authority_evaluated:
            # A run that never completed the settle window did not qualify the
            # control profile, even if the terminal state happened to be finite.
            hold_authority_passed = False
        peg_lift_m = maximum_peg_height - float(initial_peg[2])
        approach_target = np.asarray(
            solutions["approach_peg"].target_tcp_position_m,
            dtype=np.float64,
        )
        approach_reached = any(
            row["phase"] == ControllerPhase.APPROACH_PEG.value
            and np.linalg.norm(np.asarray(row["tcp_position_m"], dtype=np.float64) - approach_target) <= 0.008
            for row in records
        )
        final_events = evidence_summary["final_events"]
        transport_aligned = bool(
            records
            and final_events
            and final_events["retention_candidate"]
            and records[-1]["lateral_offset_m"]
            <= evidence_tracker.thresholds.transport_lateral_tolerance_m
        )
        basic_grasp_and_lift = bool(
            finite_stable
            and evidence_tracker.thresholds.minimum_lift_m
            <= peg_lift_m
            <= evidence_tracker.thresholds.maximum_lift_m
        )
        gates = {
            "finite_stable": finite_stable,
            "applied_joint_torque_available": bool(
                MODE[args.mode] is not PegInsertionMode.INSERTION
                or applied_joint_torque_available
            ),
            "hold_authority": hold_authority_passed,
            "waypoints_converged": all(solution.converged for solution in solutions.values()),
            "arm_executed": approach_reached,
            "preclose_workpiece_stable": bool(preclose_summary["passed"]),
            "precontact_tcp_tracking": bool(
                MODE[args.mode] is PegInsertionMode.ROBOT_ONLY
                or precontact_tracking_summary["passed"]
            ),
            "exact_finger_contact_attribution": bool(
                MODE[args.mode] is PegInsertionMode.ROBOT_ONLY
                or scene.grasp_contact_shape_profile is not None
            ),
            "bilateral_gripper_contact": bool(
                bilateral_grasp_summary["qualified"]
            ),
            "grasp_and_lift": basic_grasp_and_lift,
            "sustained_grasp_retention": bool(
                evidence_summary["sustained_grasp_retention_seen"]
                and not evidence_summary["inferred_drop_candidate_seen"]
            ),
            "transport_aligned": transport_aligned,
            "stable_transport": evidence_summary["stable_transport_final"],
            "full_transfer_qualification": evidence_summary[
                "full_transfer_qualified"
            ],
            "task_complete": controller.phase is ControllerPhase.COMPLETE,
            "seated": bool(final_metrics and final_metrics.seated),
            "stable_seating": evidence_summary["stable_seating_final"],
            "insertion_guard_not_triggered": not bool(
                guarded_insertion_summary["triggered"]
            ),
            "pre_insertion_readiness": bool(
                MODE[args.mode] is not PegInsertionMode.INSERTION
                or pre_insertion_readiness_summary["passed"]
            ),
        }
        required = [
            "finite_stable",
            "hold_authority",
            "waypoints_converged",
            "arm_executed",
            "task_complete",
        ]
        if MODE[args.mode] is not PegInsertionMode.ROBOT_ONLY:
            required.extend(
                (
                    "preclose_workpiece_stable",
                    "precontact_tcp_tracking",
                    "exact_finger_contact_attribution",
                    "bilateral_gripper_contact",
                    "grasp_and_lift",
                    "sustained_grasp_retention",
                    "transport_aligned",
                    "full_transfer_qualification",
                    "stable_transport",
                )
            )
        if MODE[args.mode] is PegInsertionMode.INSERTION:
            required.extend(
                (
                    "pre_insertion_readiness",
                    "applied_joint_torque_available",
                    "insertion_guard_not_triggered",
                    "seated",
                    "stable_seating",
                )
            )
        passed = all(gates[name] for name in required)
        result = {
            "scope": (
                "actual SO-101 joint-command execution in Isaac Lab/Newton; "
                "the task-controller profile commissions the environment and is not a calibration overlay"
            ),
            "mode": MODE[args.mode].value,
            "passed": passed,
            "required_gates": required,
            "gates": gates,
            "controller_terminal_phase": controller.phase.value,
            "applied_joint_torque_available": applied_joint_torque_available,
            "phase_first_seen_s": phase_first_seen,
            "environment_commissioning_control_profile": {
                "arm_stiffness_nm_per_rad": scene.spec.arm_control_stiffness_nm_per_rad,
                "arm_damping_nm_s_per_rad": scene.spec.arm_control_damping_nm_s_per_rad,
                "arm_effort_cap_nm": scene.spec.arm_control_effort_limit_nm,
                "gripper_stiffness_nm_per_rad": scene.spec.gripper_control_stiffness_nm_per_rad,
                "gripper_damping_nm_s_per_rad": scene.spec.gripper_control_damping_nm_s_per_rad,
                "gripper_effort_cap_nm": scene.spec.gripper_control_effort_limit_nm,
                "gripper_open_rad": adapter.open_gripper_rad,
                "gripper_close_rad": adapter.closed_gripper_rad,
                "gripper_close_rate_rad_s": args.gripper_close_rate,
                "nominal_grasp_tcp_above_peg_center_m": (
                    scene.spec.grasp_tcp_above_peg_center_m
                ),
                "gripper_contact_latch": gripper_contact_latch.summary(),
                "gripper_aperture_basis": (
                    "USD mesh/FK estimate at peg-centre station; Newton contact "
                    "and sustained retention remain authoritative"
                ),
            },
            "commissioned_waypoints": {
                name: solution.to_dict() for name, solution in solutions.items()
            },
            "task_reset_joint_position_rad": home_q.tolist(),
            "hold_authority_preflight": {
                "settle_duration_s": controller.settle_s,
                "sample_count": settle_sample_count,
                "maximum_tcp_drift_m": maximum_settle_tcp_drift_m,
                "maximum_tcp_drift_allowed_m": 0.005,
                "torque_saturation_definition": "abs(applied_torque) >= 98% of configured effort cap",
                "maximum_saturation_fraction_allowed_exclusive": 0.25,
                "per_joint_saturation_fraction": {
                    name: float(value)
                    for name, value in zip(scene.joint_names, saturation_fraction, strict=True)
                },
                "passed": hold_authority_passed,
            },
            "tcp_servo": {
                "enabled_phases": sorted(phase.value for phase in TCP_SERVO_PHASES),
                "jacobian_source": "Newton body_link_jacobian_w",
                "orientation_feedback": (
                    "SO-101 wrist-pitch sum controlled in the TCP-position null space"
                ),
                "tcp_offset_gripper_m": adapter.tcp_offset_gripper_m.tolist(),
                "approach_position_tolerance_m": 0.002,
                "grasp_position_tolerance_m": 0.0015,
                "transport_position_tolerance_m": 0.002,
                "align_position_tolerance_m": 0.00075,
                "insertion_position_tolerance_m": 0.0005,
                "align_max_arm_rate_rad_s": 0.35,
                "guarded_insert_max_arm_rate_rad_s": 0.12,
                "free_space_proportional_gain_per_s": args.tcp_servo_gain,
                "guarded_insert_proportional_gain_per_s": 2.5,
                "max_joint_offset_from_commissioned_ik_rad": 0.18,
                "phase_nominal": (
                    "joint interpolation from phase-entry command to endpoint IK "
                    "using projected Cartesian setpoint progress"
                ),
                "target_wrist_pitch_sum_rad": 0.5 * math.pi,
                "progressive_cartesian_setpoint": {
                    "source": "previous commanded Cartesian setpoint",
                    "phase_speed_limits_m_s": {
                        phase.value: speed
                        for phase, speed in tcp_speed_by_phase.items()
                    },
                },
            },
            "evidence_inference": evidence_summary,
            "full_transfer_gate": {
                "scope": (
                    "sticky retention/stability qualification from lift through "
                    "the commissioned 150 mm move and arrival at alignment"
                ),
                "required_phases": evidence_summary[
                    "full_transfer_required_phases"
                ],
                "phases_seen": evidence_summary["full_transfer_phases_seen"],
                "minimum_tcp_displacement_m": evidence_tracker.thresholds.minimum_transport_displacement_m,
                "maximum_tcp_displacement_observed_m": evidence_summary[
                    "maximum_full_transfer_displacement_m"
                ],
                "retention_window_s": evidence_tracker.thresholds.transport_window_s,
                "violated": evidence_summary["full_transfer_violated"],
                "violation_reason": evidence_summary[
                    "full_transfer_violation_reason"
                ],
                "passed": evidence_summary["full_transfer_qualified"],
            },
            "raw_grasp_contact_evidence": {
                **bilateral_grasp_summary,
                "attribution": scene.grasp_contact_attribution,
                "collision_profile": (
                    scene.grasp_contact_shape_profile.profile_id
                    if scene.grasp_contact_shape_profile is not None
                    else None
                ),
                "fixed_filter_objects": contact_force_evidence[
                    "fixed_finger_filter_objects"
                ],
                "moving_filter_objects": contact_force_evidence[
                    "moving_finger_filter_objects"
                ],
            },
            "preclose_disturbance_gate": {
                **preclose_summary,
                "abort_reason": preclose_abort_reason,
                "scope": "approach and open-gripper descent after scene settling",
            },
            "precontact_tcp_tracking_gate": {
                **precontact_tracking_summary,
                "abort_reason": precontact_tracking_abort_reason,
                "signal_basis": (
                    "measured Newton TCP pose versus the audited progressive "
                    "Cartesian setpoint"
                ),
            },
            "guarded_insertion_gate": guarded_insertion_summary,
            "pre_insertion_readiness_gate": pre_insertion_readiness_summary,
            "evidence_field_semantics": {
                "gripper_contact": (
                    "raw Newton/MJWarp per-partner normal-force matrix for the peg "
                    "against separately filtered finger channels; MVP2/3 require "
                    "exactly one versioned fixed-follower and moving-jaw proxy scope"
                ),
                "grasp_contact_attribution": scene.grasp_contact_attribution,
                "retention_slip_drop_socket_events": (
                    "kinematic inference; raw socket normal force is consumed by the "
                    "pre-insertion readiness and guarded-insertion safety gates"
                ),
                "peg_pose_world_m_xyzw": (
                    "position meters followed by Isaac Lab/Newton xyzw quaternion"
                ),
                "peg_velocity_world_m_s_rad_s": (
                    "world linear velocity followed by world angular velocity"
                ),
            },
            "initial_peg_position_m": initial_peg.tolist(),
            "initial_peg_pose_world_m_xyzw": initial_peg_pose.tolist(),
            "final_peg_position_m": final_peg.tolist() if final_is_finite else None,
            "final_peg_pose_world_m_xyzw": (
                final_peg_pose.tolist() if final_is_finite else None
            ),
            "final_peg_velocity_world_m_s_rad_s": (
                final_peg_velocity.tolist() if final_is_finite else None
            ),
            "maximum_peg_lift_m": peg_lift_m,
            "minimum_tcp_to_peg_distance_m": minimum_relative_distance,
            "final_metrics": (
                {
                    "insertion_depth_m": final_metrics.insertion_depth_m,
                    "lateral_offset_m": final_metrics.lateral_offset_m,
                    "tilt_deg": final_metrics.tilt_deg,
                    "seated": final_metrics.seated,
                    "jammed": final_metrics.jammed,
                }
                if final_metrics is not None
                else None
            ),
            "records": records,
        }
        result_path = output / "controller_result.json"
        result_path.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "records"},
                indent=2,
                allow_nan=False,
            )
        )
        print(f"RESULT={result_path}", flush=True)
        if not passed:
            raise RuntimeError(f"SO-101 controller did not pass required gates: {gates}")
    except Exception as exc:
        # A failed physical run is still valuable evidence.  Persist the exact
        # failure before simulator teardown so agent-managed jobs never end as
        # an unexplained empty output directory.
        failure = {
            "scope": "SO-101 joint-command execution in Isaac Lab/Newton",
            "mode": MODE[args.mode].value,
            "asset": str(args.asset),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_path = output / "controller_failure.json"
        failure_path.write_text(
            json.dumps(failure, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2, allow_nan=False), flush=True)
        print(f"FAILURE={failure_path}", flush=True)
        raise
    finally:
        if scene is not None:
            scene.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
