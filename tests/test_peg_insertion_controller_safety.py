from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from newton_calibration.isaaclab.tasks.so101_peg_insertion import (
    BilateralContactGripperLatch,
    ControllerObservation,
    ControllerPhase,
    PegInsertionController,
    PegInsertionMode,
    PegInsertionSceneSpec,
    PegInsertionTaskMetrics,
)


def test_gripper_latches_rate_limited_command_at_first_bilateral_contact() -> None:
    latch = BilateralContactGripperLatch(minimum_partner_force_n=0.05)

    assert latch.update(
        time_s=1.0,
        phase=ControllerPhase.CLOSE,
        current_command_rad=0.03,
        fixed_finger_force_n=0.0,
        moving_finger_force_n=0.2,
    ) is None
    assert latch.update(
        time_s=1.01,
        phase=ControllerPhase.CLOSE,
        current_command_rad=0.02,
        fixed_finger_force_n=0.1,
        moving_finger_force_n=0.2,
    ) == pytest.approx(0.02)
    assert latch.update(
        time_s=1.02,
        phase=ControllerPhase.CLOSE,
        current_command_rad=-0.04,
        fixed_finger_force_n=1.0,
        moving_finger_force_n=1.0,
    ) == pytest.approx(0.02)
    assert latch.target(-0.04) == pytest.approx(0.02)
    assert latch.summary()["latched"]


def test_gripper_latch_ignores_contact_outside_close_phase() -> None:
    latch = BilateralContactGripperLatch()
    assert latch.update(
        time_s=1.0,
        phase=ControllerPhase.DESCEND_TO_GRASP,
        current_command_rad=0.3,
        fixed_finger_force_n=0.2,
        moving_finger_force_n=0.2,
    ) is None


def _observation(
    spec: PegInsertionSceneSpec,
    *,
    ee_position_m: tuple[float, float, float] | None = None,
    peg_grasped: bool = False,
    insertion_depth_m: float = 0.0,
    peg_dropped: bool = False,
    jammed: bool = False,
    force_limit_exceeded: bool = False,
    guard_stop_requested: bool = False,
    pre_insertion_ready: bool = False,
    pre_insertion_gate_failed: bool = False,
) -> ControllerObservation:
    return ControllerObservation(
        ee_position_m=ee_position_m
        or (*spec.peg_start_xy_m, spec.transport_tcp_height_m),
        peg_position_m=(*spec.peg_start_xy_m, spec.peg_rest_center_z_m),
        gripper_gap_m=0.04,
        insertion_depth_m=insertion_depth_m,
        peg_grasped=peg_grasped,
        peg_dropped=peg_dropped,
        jammed=jammed,
        force_limit_exceeded=force_limit_exceeded,
        guard_stop_requested=guard_stop_requested,
        pre_insertion_ready=pre_insertion_ready,
        pre_insertion_gate_failed=pre_insertion_gate_failed,
    )


def _advance_to_grasp_verification(
    controller: PegInsertionController,
    spec: PegInsertionSceneSpec,
) -> None:
    controller.step(_observation(spec), 0.01)
    assert controller.phase is ControllerPhase.APPROACH_PEG
    controller.step(_observation(spec), 0.01)
    assert controller.phase is ControllerPhase.STABILIZE_ABOVE_PEG
    controller.step(_observation(spec), controller.approach_stabilize_s)
    assert controller.phase is ControllerPhase.DESCEND_TO_GRASP
    at_grasp = _observation(
        spec,
        ee_position_m=(*spec.peg_start_xy_m, spec.grasp_tcp_height_m),
    )
    controller.step(at_grasp, 0.01)
    assert controller.phase is ControllerPhase.CLOSE
    controller.step(at_grasp, 0.01)
    assert controller.phase is ControllerPhase.VERIFY_GRASP


def _runner_dict_keys(variable_name: str) -> set[str]:
    """Read a runner output dictionary without importing Isaac Lab."""

    script = Path(__file__).parents[1] / "scripts" / "run_so101_peg_controller.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        return {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise AssertionError(f"Could not find dictionary assignment for {variable_name!r}")


def _runner_record_keys() -> set[str]:
    script = Path(__file__).parents[1] / "scripts" / "run_so101_peg_controller.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "append" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "records" or len(node.args) != 1:
            continue
        payload = node.args[0]
        if isinstance(payload, ast.Dict):
            return {
                key.value
                for key in payload.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError("Could not find records.append({...}) in controller runner")


def test_reference_fixture_placements_fit_on_the_table_and_do_not_overlap() -> None:
    spec = PegInsertionSceneSpec()
    table_center_xy_m = (0.20, 0.0)
    half_x = 0.5 * spec.table_size_m[0]
    half_y = 0.5 * spec.table_size_m[1]
    table_x = (table_center_xy_m[0] - half_x, table_center_xy_m[0] + half_x)
    table_y = (table_center_xy_m[1] - half_y, table_center_xy_m[1] + half_y)

    px, py = spec.peg_start_xy_m
    sx, sy = spec.socket_center_xy_m
    half_socket = 0.5 * spec.socket_outer_m
    assert table_x[0] + spec.peg_radius_m < px < table_x[1] - spec.peg_radius_m
    assert table_y[0] + spec.peg_radius_m < py < table_y[1] - spec.peg_radius_m
    assert table_x[0] < sx - half_socket < sx + half_socket < table_x[1]
    assert table_y[0] < sy - half_socket < sy + half_socket < table_y[1]
    assert abs(py - sy) > spec.peg_radius_m + half_socket


def test_transport_and_insertion_heights_encode_the_declared_geometry() -> None:
    spec = PegInsertionSceneSpec()
    transported_peg_bottom = (
        spec.transport_tcp_height_m
        - spec.grasp_tcp_above_peg_center_m
        - 0.5 * spec.peg_height_m
    )
    inserted_peg_bottom = (
        spec.insertion_tcp_height_m
        - spec.grasp_tcp_above_peg_center_m
        - 0.5 * spec.peg_height_m
    )
    assert transported_peg_bottom > spec.socket_top_z_m + spec.contact_gap_m
    assert abs(
        (spec.socket_top_z_m - inserted_peg_bottom) - spec.target_insertion_depth_m
    ) < 1.0e-12


def test_robot_only_stops_after_reaching_the_approach_waypoint() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(
        spec,
        mode=PegInsertionMode.ROBOT_ONLY,
        settle_s=0.0,
    )
    controller.step(_observation(spec), 0.01)
    approach = controller.step(_observation(spec), 0.01)
    assert approach.phase is ControllerPhase.APPROACH_PEG
    assert not approach.gripper_closed
    assert controller.phase is ControllerPhase.COMPLETE
    assert controller.step(_observation(spec), 0.01).done


def test_contact_phases_use_millimetre_scale_tcp_tolerances() -> None:
    controller = PegInsertionController(PegInsertionSceneSpec())
    expected = {
        ControllerPhase.APPROACH_PEG: 0.00075,
        ControllerPhase.STABILIZE_ABOVE_PEG: 0.00075,
        ControllerPhase.DESCEND_TO_GRASP: 0.0015,
        ControllerPhase.LIFT: 0.002,
        ControllerPhase.MOVE_ABOVE_SOCKET: 0.002,
        ControllerPhase.ALIGN: 0.00075,
        ControllerPhase.GUARDED_INSERT: 0.0005,
    }
    for phase, tolerance_m in expected.items():
        controller.phase = phase
        assert controller._position_tolerance_for_phase() == tolerance_m

    robot_only = PegInsertionController(
        PegInsertionSceneSpec(), mode=PegInsertionMode.ROBOT_ONLY
    )
    robot_only.phase = ControllerPhase.APPROACH_PEG
    assert robot_only._position_tolerance_for_phase() == 0.002


def test_contact_mode_requires_continuous_stable_dwell_above_peg() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(
        spec,
        mode=PegInsertionMode.GRASP_TRANSPORT,
        settle_s=0.0,
        approach_stabilize_s=0.25,
    )
    on_target = _observation(spec)
    controller.step(on_target, 0.01)
    controller.step(on_target, 0.01)
    assert controller.phase is ControllerPhase.STABILIZE_ABOVE_PEG

    controller.step(on_target, 0.15)
    assert controller.phase is ControllerPhase.STABILIZE_ABOVE_PEG
    off_target = _observation(
        spec,
        ee_position_m=(
            spec.peg_start_xy_m[0] + 0.001,
            spec.peg_start_xy_m[1],
            spec.transport_tcp_height_m,
        ),
    )
    controller.step(off_target, 0.01)
    assert controller.phase is ControllerPhase.STABILIZE_ABOVE_PEG
    assert controller.phase_time_s == 0.0
    controller.step(on_target, 0.24)
    assert controller.phase is ControllerPhase.STABILIZE_ABOVE_PEG
    controller.step(on_target, 0.02)
    assert controller.phase is ControllerPhase.DESCEND_TO_GRASP


def test_approach_does_not_advance_with_old_eight_millimetre_error() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(spec, settle_s=0.0)
    controller.step(_observation(spec), 0.01)
    assert controller.phase is ControllerPhase.APPROACH_PEG
    controller.step(
        _observation(
            spec,
            ee_position_m=(
                spec.peg_start_xy_m[0] + 0.003,
                spec.peg_start_xy_m[1],
                spec.transport_tcp_height_m,
            ),
        ),
        0.01,
    )
    assert controller.phase is ControllerPhase.APPROACH_PEG


def test_grasp_transport_runs_through_alignment_but_not_insertion() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(
        spec,
        mode=PegInsertionMode.GRASP_TRANSPORT,
        settle_s=0.0,
        close_s=0.0,
        verify_grasp_s=0.01,
    )
    _advance_to_grasp_verification(controller, spec)
    controller.step(_observation(spec, peg_grasped=True), 0.01)
    assert controller.phase is ControllerPhase.LIFT
    controller.step(
        _observation(
            spec,
            ee_position_m=(*spec.peg_start_xy_m, spec.transport_tcp_height_m),
            peg_grasped=True,
        ),
        0.01,
    )
    assert controller.phase is ControllerPhase.MOVE_ABOVE_SOCKET
    controller.step(
        _observation(
            spec,
            ee_position_m=(*spec.socket_center_xy_m, spec.transport_tcp_height_m),
            peg_grasped=True,
        ),
        0.01,
    )
    assert controller.phase is ControllerPhase.ALIGN
    controller.step(
        _observation(
            spec,
            ee_position_m=(*spec.socket_center_xy_m, spec.align_tcp_height_m),
            peg_grasped=True,
            pre_insertion_ready=True,
        ),
        0.01,
    )
    assert controller.phase is ControllerPhase.COMPLETE


def test_mvp3_alignment_waits_for_pre_insertion_readiness() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(
        spec,
        mode=PegInsertionMode.INSERTION,
        settle_s=0.0,
    )
    controller.phase = ControllerPhase.ALIGN
    aligned = _observation(
        spec,
        ee_position_m=(*spec.socket_center_xy_m, spec.align_tcp_height_m),
        peg_grasped=True,
    )
    controller.step(aligned, 0.01)
    assert controller.phase is ControllerPhase.ALIGN
    controller.step(replace(aligned, pre_insertion_ready=True), 0.01)
    assert controller.phase is ControllerPhase.GUARDED_INSERT


def test_mvp3_pre_insertion_failure_aborts_but_mvp2_is_unchanged() -> None:
    spec = PegInsertionSceneSpec()
    failed = _observation(
        spec,
        ee_position_m=(*spec.socket_center_xy_m, spec.align_tcp_height_m),
        peg_grasped=True,
        pre_insertion_gate_failed=True,
    )
    insertion = PegInsertionController(spec, mode=PegInsertionMode.INSERTION)
    insertion.phase = ControllerPhase.MOVE_ABOVE_SOCKET
    assert insertion.step(failed, 0.01).aborted

    transport = PegInsertionController(spec, mode=PegInsertionMode.GRASP_TRANSPORT)
    transport.phase = ControllerPhase.ALIGN
    transport.step(failed, 0.01)
    assert transport.phase is ControllerPhase.COMPLETE


def test_insertion_requires_all_prior_phases_even_if_depth_is_reported_early() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(
        spec,
        mode=PegInsertionMode.INSERTION,
        settle_s=0.0,
        close_s=0.0,
        verify_grasp_s=0.01,
    )
    deep = _observation(spec, insertion_depth_m=spec.target_insertion_depth_m)
    controller.step(deep, 0.01)
    assert controller.phase is ControllerPhase.APPROACH_PEG
    controller.step(deep, 0.01)
    assert controller.phase is ControllerPhase.STABILIZE_ABOVE_PEG
    controller.step(deep, controller.approach_stabilize_s)
    assert controller.phase is ControllerPhase.DESCEND_TO_GRASP


def test_insertion_reaches_complete_only_after_guarded_depth_verification() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(
        spec,
        mode=PegInsertionMode.INSERTION,
        settle_s=0.0,
        close_s=0.0,
        verify_grasp_s=0.01,
    )
    _advance_to_grasp_verification(controller, spec)
    controller.step(_observation(spec, peg_grasped=True), 0.01)
    controller.step(
        _observation(
            spec,
            ee_position_m=(*spec.peg_start_xy_m, spec.transport_tcp_height_m),
            peg_grasped=True,
        ),
        0.01,
    )
    controller.step(
        _observation(
            spec,
            ee_position_m=(*spec.socket_center_xy_m, spec.transport_tcp_height_m),
            peg_grasped=True,
        ),
        0.01,
    )
    controller.step(
        _observation(
            spec,
            ee_position_m=(*spec.socket_center_xy_m, spec.align_tcp_height_m),
            peg_grasped=True,
            pre_insertion_ready=True,
        ),
        0.01,
    )
    assert controller.phase is ControllerPhase.GUARDED_INSERT
    controller.step(
        _observation(
            spec,
            ee_position_m=(*spec.socket_center_xy_m, spec.insertion_tcp_height_m),
            peg_grasped=True,
            insertion_depth_m=spec.target_insertion_depth_m,
        ),
        0.01,
    )
    assert controller.phase is ControllerPhase.VERIFY_DEPTH
    controller.step(
        _observation(
            spec,
            ee_position_m=(*spec.socket_center_xy_m, spec.insertion_tcp_height_m),
            peg_grasped=True,
            insertion_depth_m=spec.target_insertion_depth_m,
        ),
        0.01,
    )
    assert controller.phase is ControllerPhase.COMPLETE


def test_failed_grasp_aborts_before_lift() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(
        spec,
        mode=PegInsertionMode.GRASP_TRANSPORT,
        settle_s=0.0,
        close_s=0.0,
        verify_grasp_s=0.01,
    )
    _advance_to_grasp_verification(controller, spec)
    controller.step(_observation(spec, peg_grasped=False), 0.01)
    assert controller.phase is ControllerPhase.ABORT
    assert controller.step(_observation(spec), 0.01).aborted


def test_drop_jam_and_force_limit_are_immediate_abort_conditions() -> None:
    spec = PegInsertionSceneSpec()
    for failure in ("peg_dropped", "jammed", "force_limit_exceeded"):
        controller = PegInsertionController(spec, settle_s=10.0)
        command = controller.step(_observation(spec, **{failure: True}), 0.01)
        assert controller.phase is ControllerPhase.ABORT
        assert command.aborted


def test_abort_opens_gripper_and_latches_one_upward_retreat() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(spec, settle_s=10.0, abort_retract_m=0.020)
    observation = _observation(
        spec,
        ee_position_m=(0.235, 0.070, 0.061),
        guard_stop_requested=True,
    )
    first = controller.step(observation, 0.01)
    second = controller.step(
        replace(observation, ee_position_m=(0.235, 0.070, 0.065)),
        0.01,
    )
    assert first.aborted and second.aborted
    assert not first.gripper_closed and not second.gripper_closed
    assert first.target_position_m == pytest.approx((0.235, 0.070, 0.081))
    assert second.target_position_m == first.target_position_m
    assert controller.phase_time_s == pytest.approx(0.02)


def test_metrics_reject_deep_but_laterally_misaligned_or_tilted_poses() -> None:
    spec = PegInsertionSceneSpec()
    seated_center_z = (
        spec.socket_top_z_m
        + 0.5 * spec.peg_height_m
        - spec.target_insertion_depth_m
    )
    misaligned = PegInsertionTaskMetrics.from_pose(
        spec,
        (
            spec.socket_center_xy_m[0] + 1.01 * spec.success_lateral_tolerance_m,
            spec.socket_center_xy_m[1],
            seated_center_z,
        ),
        (0.0, 0.0, 1.0),
        settled=True,
    )
    tilted = PegInsertionTaskMetrics.from_pose(
        spec,
        (*spec.socket_center_xy_m, seated_center_z),
        (0.2, 0.0, 0.9797958971),
        settled=True,
    )
    assert not misaligned.seated
    assert not tilted.seated


def test_controller_result_keeps_traceability_and_gate_fields() -> None:
    result_keys = _runner_dict_keys("result")
    gate_keys = _runner_dict_keys("gates")
    assert {
        "scope",
        "mode",
        "passed",
        "required_gates",
        "gates",
        "controller_terminal_phase",
        "phase_first_seen_s",
        "environment_commissioning_control_profile",
        "commissioned_waypoints",
        "task_reset_joint_position_rad",
        "evidence_inference",
        "full_transfer_gate",
        "raw_grasp_contact_evidence",
        "evidence_field_semantics",
        "guarded_insertion_gate",
        "pre_insertion_readiness_gate",
        "precontact_tcp_tracking_gate",
        "applied_joint_torque_available",
        "initial_peg_position_m",
        "initial_peg_pose_world_m_xyzw",
        "final_peg_position_m",
        "final_peg_pose_world_m_xyzw",
        "final_peg_velocity_world_m_s_rad_s",
        "maximum_peg_lift_m",
        "minimum_tcp_to_peg_distance_m",
        "final_metrics",
        "records",
    } <= result_keys
    assert {
        "finite_stable",
        "hold_authority",
        "waypoints_converged",
        "arm_executed",
        "preclose_workpiece_stable",
        "precontact_tcp_tracking",
        "exact_finger_contact_attribution",
        "bilateral_gripper_contact",
        "grasp_and_lift",
        "sustained_grasp_retention",
        "transport_aligned",
        "stable_transport",
        "full_transfer_qualification",
        "pre_insertion_readiness",
        "applied_joint_torque_available",
        "task_complete",
        "seated",
        "stable_seating",
        "insertion_guard_not_triggered",
    } <= gate_keys


def test_controller_record_keeps_commands_measurements_and_task_metrics() -> None:
    record_keys = _runner_record_keys()
    assert {
        "time_s",
        "phase",
        "command_q_rad",
        "desired_q_rad",
        "measured_q_rad",
        "measured_dq_rad_s",
        "applied_torque_nm",
        "gripper_gap_m",
        "gripper_closed_commanded",
        "tcp_position_m",
        "peg_position_m",
        "peg_pose_world_m_xyzw",
        "peg_velocity_world_m_s_rad_s",
        "peg_tcp_relative_m",
        "peg_position_relative_to_tcp_m",
        "grasp_candidate",
        "peg_lifted",
        "inferred_events",
        "raw_contact_force_evidence_n",
        "bilateral_grasp_contact",
        "guarded_insertion",
        "insertion_depth_m",
        "lateral_offset_m",
        "tilt_deg",
        "peg_bottom_clearance_above_socket_rim_m",
        "pre_insertion_readiness",
        "precontact_tcp_tracking",
    } <= record_keys


def test_mvp2_and_mvp3_contact_evidence_uses_separate_versioned_shape_filters() -> None:
    scene_path = (
        Path(__file__).parents[1]
        / "src/newton_calibration/isaaclab/tasks/so101_peg_insertion/scene.py"
    )
    source = scene_path.read_text(encoding="utf-8")
    assert (
        '"/World/Env_0/Robot/gripper_link/"\n'
        '    "newton_collision_v.*_fixed_follower/part_.*"'
    ) in source
    assert (
        '"/World/Env_0/Robot/moving_jaw_so101_v1_link/"\n'
        '    "newton_collision_v.*_moving_jaw/part_.*"'
    ) in source
    assert (
        '"/World/Env_0/Robot/gripper_link/"\n'
        '    "newton_collision_v4_fixed_pad/part_000"'
    ) in source
    assert (
        '"/World/Env_0/Robot/moving_jaw_so101_v1_link/"\n'
        '    "newton_collision_v4_moving_pad/part_000"'
    ) in source
    assert '"so101_gripper_prebaked_v3"' in source
    assert '"so101_task_planar_pads_v4"' in source
    assert "self.fixed_finger_contact_sensor" in source
    assert "self.moving_finger_contact_sensor" in source
    assert "self.grasp_contact_attribution = profile.attribution" in source
    assert "asset-declared Newton grasp-contact" in source
    assert "profile.expected_part_names" in source

    runner_path = Path(__file__).parents[1] / "scripts/run_so101_peg_controller.py"
    runner_source = runner_path.read_text(encoding="utf-8")
    assert '"exact_finger_contact_attribution"' in runner_source
    assert '"fixed_finger_force_n"' in runner_source
    assert '"moving_finger_force_n"' in runner_source


def test_whole_body_contact_fallback_is_explicitly_mvp1_only() -> None:
    scene_path = (
        Path(__file__).parents[1]
        / "src/newton_calibration/isaaclab/tasks/so101_peg_insertion/scene.py"
    )
    source = scene_path.read_text(encoding="utf-8")
    assert "if self.mode is PegInsertionMode.ROBOT_ONLY:" in source
    assert 'self.grasp_contact_attribution = "body_fallback_mvp1_safety_only"' in source
    assert "whole-body readings" in source


def test_controller_result_rejects_non_standard_nan_and_infinity_json() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_so101_peg_controller.py"
    source = script.read_text(encoding="utf-8")
    assert "minimum_relative_distance = float(\"inf\")" not in source
    assert source.count("allow_nan=False") >= 2
