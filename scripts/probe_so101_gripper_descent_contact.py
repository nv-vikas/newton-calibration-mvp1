#!/usr/bin/env python3
"""Attribute the first peg disturbance during an open SO-101 descent.

The probe executes the same dynamically controlled pre-grasp descent as the
task runner, but stops before commanding gripper closure.  At every physics
step it measures the signed distance from the peg cylinder to the fixed and
moving jaw collision meshes using Newton's live link poses.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if os.environ.get("ACCEPT_EULA") != "Y":
    raise RuntimeError("Review the NVIDIA Isaac Sim license and export ACCEPT_EULA=Y before running.")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", required=True)
parser.add_argument("--geometry", required=True, help="NPZ from extract_so101_gripper_mesh.py")
parser.add_argument("--output", required=True)
parser.add_argument("--tcp-x", type=float, default=0.002)
parser.add_argument("--open-q", type=float, default=0.30)
parser.add_argument("--approach-height", type=float, default=0.090)
parser.add_argument("--grasp-height", type=float, default=0.055)
parser.add_argument("--max-arm-rate", type=float, default=0.35)
parser.add_argument(
    "--post-rate-limit",
    action="store_true",
    help="Diagnostic workaround: enforce the configured rate after the servo returns",
)
parser.add_argument("--settle-seconds", type=float, default=1.0)
parser.add_argument("--descent-seconds", type=float, default=1.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
from scipy.spatial import ConvexHull

from newton_calibration.isaaclab.tasks.so101_peg_insertion import PegInsertionMode
from newton_calibration.isaaclab.tasks.so101_peg_insertion.robot_adapter import (
    SO101JointCommandAdapter,
)
from newton_calibration.isaaclab.tasks.so101_peg_insertion.scene import (
    SO101PegInsertionScene,
)


def _rotate_xyzw(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    xyz = quaternion[:3]
    w = float(quaternion[3])
    repeated = np.broadcast_to(xyz, vectors.shape)
    twice_cross = 2.0 * np.cross(repeated, vectors)
    return vectors + w * twice_cross + np.cross(repeated, twice_cross)


def _inverse_xyzw(quaternion: np.ndarray) -> np.ndarray:
    return np.asarray(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float64,
    )


def _body_points_world(
    scene: SO101PegInsertionScene,
    body_name: str,
    points_body: np.ndarray,
) -> np.ndarray:
    position, quaternion = scene.body_pose_world(body_name)
    return position + _rotate_xyzw(quaternion, points_body)


def _points_world_to_body(
    scene: SO101PegInsertionScene,
    body_name: str,
    points_world: np.ndarray,
) -> np.ndarray:
    position, quaternion = scene.body_pose_world(body_name)
    return _rotate_xyzw(
        _inverse_xyzw(quaternion),
        points_world - position,
    )


def _peg_pose(scene: SO101PegInsertionScene) -> tuple[np.ndarray, np.ndarray]:
    pose = scene.peg.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
    # The pinned Isaac Lab/Newton rigid-object interface publishes xyzw.
    return pose[:3], np.asarray(pose[3:7], dtype=np.float64)


def _peg_velocity(scene: SO101PegInsertionScene) -> np.ndarray:
    return (
        scene.peg.data.root_vel_w.torch[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _signed_distance_to_cylinder(
    points_world: np.ndarray,
    *,
    center_world: np.ndarray,
    quaternion_xyzw: np.ndarray,
    radius_m: float,
    half_height_m: float,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    points_local = _rotate_xyzw(
        _inverse_xyzw(quaternion_xyzw),
        points_world - center_world,
    )
    radial = np.linalg.norm(points_local[:, :2], axis=1)
    radial_delta = radial - radius_m
    axial_delta = np.abs(points_local[:, 2]) - half_height_m
    outside = np.linalg.norm(
        np.maximum(np.stack((radial_delta, axial_delta), axis=1), 0.0),
        axis=1,
    )
    inside = np.minimum(np.maximum(radial_delta, axial_delta), 0.0)
    signed = outside + inside
    index = int(np.argmin(signed))
    region = "side" if radial_delta[index] >= axial_delta[index] else "cap"
    return (
        float(signed[index]),
        points_world[index].copy(),
        points_local[index].copy(),
        region,
    )


def _sample_cylinder_volume(
    *, radius_m: float, half_height_m: float
) -> np.ndarray:
    """Return deterministic interior/surface samples in the cylinder frame."""

    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    radii = radius_m * np.asarray((0.0, 0.25, 0.50, 0.75, 1.0))
    heights = np.linspace(-half_height_m, half_height_m, 13)
    rings = []
    for height in heights:
        for radius in radii:
            if radius == 0.0:
                rings.append(np.asarray(((0.0, 0.0, height),)))
            else:
                rings.append(
                    np.stack(
                        (
                            radius * np.cos(angles),
                            radius * np.sin(angles),
                            np.full_like(angles, height),
                        ),
                        axis=1,
                    )
                )
    return np.concatenate(rings, axis=0).astype(np.float64)


def _normalized_hull_equations(points_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = ConvexHull(points_body)
    equations = np.asarray(hull.equations, dtype=np.float64).copy()
    normal_norm = np.linalg.norm(equations[:, :3], axis=1)
    equations /= normal_norm[:, None]
    return equations, np.asarray(hull.vertices, dtype=np.int64)


def _sampled_cylinder_hull_margin(
    scene: SO101PegInsertionScene,
    *,
    body_name: str,
    hull_equations: np.ndarray,
    peg_samples_local: np.ndarray,
) -> dict[str, object]:
    """Approximate the cylinder-to-convex-hull signed clearance.

    Qhull equations use ``n dot x + b <= 0`` for points inside the hull.
    The maximum plane violation is therefore positive outside and non-positive
    inside.  Taking the minimum over dense cylinder-volume samples provides a
    contact attribution signal, including the case where a hull protrusion lies
    completely inside the peg volume.
    """

    peg_position, peg_quaternion = _peg_pose(scene)
    samples_world = peg_position + _rotate_xyzw(peg_quaternion, peg_samples_local)
    samples_body = _points_world_to_body(scene, body_name, samples_world)
    normals = hull_equations[:, :3]
    offsets = hull_equations[:, 3]
    best_margin = np.inf
    best_sample_index = -1
    best_plane_index = -1
    inside_count = 0
    for begin in range(0, samples_body.shape[0], 256):
        end = min(begin + 256, samples_body.shape[0])
        plane_values = samples_body[begin:end] @ normals.T + offsets
        outside_margin = np.max(plane_values, axis=1)
        inside_count += int(np.count_nonzero(outside_margin <= 0.0))
        local_index = int(np.argmin(outside_margin))
        margin = float(outside_margin[local_index])
        if margin < best_margin:
            best_margin = margin
            best_sample_index = begin + local_index
            best_plane_index = int(np.argmax(plane_values[local_index]))
    sample_body = samples_body[best_sample_index]
    sample_world = samples_world[best_sample_index]
    return {
        "sampled_cylinder_hull_margin_m": best_margin,
        "overlap_detected": best_margin <= 0.0,
        "penetrating_sample_count": inside_count,
        "closest_peg_sample_body_m": sample_body.tolist(),
        "closest_peg_sample_world_m": sample_world.tolist(),
        "active_hull_plane_index": best_plane_index,
        "active_hull_plane_normal_body": normals[best_plane_index].tolist(),
        "active_hull_plane_offset_m": float(offsets[best_plane_index]),
    }
def _contact_measurement(
    scene: SO101PegInsertionScene,
    *,
    collision_geometries: dict[str, dict[str, object]],
    peg_samples_local: np.ndarray,
) -> dict[str, object]:
    peg_position, peg_quaternion = _peg_pose(scene)
    gripper_position, gripper_quaternion = scene.body_pose_world("gripper_link")
    peg_in_gripper = _rotate_xyzw(
        _inverse_xyzw(gripper_quaternion),
        (peg_position - gripper_position)[None, :],
    )[0]
    result: dict[str, object] = {
        "peg_center_in_gripper_m": peg_in_gripper.tolist(),
    }
    for label, geometry in collision_geometries.items():
        body_name = str(geometry["body_name"])
        hull_vertex_points_body = np.asarray(
            geometry["hull_vertex_points_body"], dtype=np.float64
        )
        points = _body_points_world(scene, body_name, hull_vertex_points_body)
        distance, closest_world, closest_peg, region = _signed_distance_to_cylinder(
            points,
            center_world=peg_position,
            quaternion_xyzw=peg_quaternion,
            radius_m=scene.spec.peg_radius_m,
            half_height_m=0.5 * scene.spec.peg_height_m,
        )
        closest_gripper = _rotate_xyzw(
            _inverse_xyzw(gripper_quaternion),
            (closest_world - gripper_position)[None, :],
        )[0]
        hull_measurement = _sampled_cylinder_hull_margin(
            scene,
            body_name=body_name,
            hull_equations=np.asarray(geometry["hull_equations"], dtype=np.float64),
            peg_samples_local=peg_samples_local,
        )
        result[label] = {
            "body_name": body_name,
            "signed_vertex_clearance_m": distance,
            "closest_point_world_m": closest_world.tolist(),
            "closest_point_peg_m": closest_peg.tolist(),
            "closest_point_gripper_m": closest_gripper.tolist(),
            "closest_region": region,
            **hull_measurement,
        }
    return result


def _sample(
    scene: SO101PegInsertionScene,
    adapter: SO101JointCommandAdapter,
    *,
    time_s: float,
    phase: str,
    initial_peg_position: np.ndarray,
    collision_geometries: dict[str, dict[str, object]],
    peg_samples_local: np.ndarray,
    command_q: np.ndarray,
) -> dict[str, object]:
    peg_position, peg_quaternion = _peg_pose(scene)
    peg_velocity = _peg_velocity(scene)
    gripper_position, gripper_quaternion = scene.body_pose_world("gripper_link")
    jaw_position, jaw_quaternion = scene.body_pose_world("moving_jaw_so101_v1_link")
    displacement = peg_position - initial_peg_position
    return {
        "time_s": time_s,
        "phase": phase,
        "tcp_position_world_m": adapter.current_tcp_position().tolist(),
        "gripper_link_pose_world": {
            "position_m": gripper_position.tolist(),
            "quaternion_xyzw": gripper_quaternion.tolist(),
        },
        "moving_jaw_pose_world": {
            "position_m": jaw_position.tolist(),
            "quaternion_xyzw": jaw_quaternion.tolist(),
        },
        "command_q_rad": command_q.tolist(),
        "measured_q_rad": scene.joint_position_rad().tolist(),
        "peg_position_world_m": peg_position.tolist(),
        "peg_quaternion_xyzw": peg_quaternion.tolist(),
        "peg_velocity_world_m_s_rad_s": peg_velocity.tolist(),
        "peg_displacement_m": displacement.tolist(),
        "peg_displacement_norm_m": float(np.linalg.norm(displacement)),
        "jaw_clearance": _contact_measurement(
            scene,
            collision_geometries=collision_geometries,
            peg_samples_local=peg_samples_local,
        ),
    }


def _solve(
    adapter: SO101JointCommandAdapter,
    scene: SO101PegInsertionScene,
    name: str,
    height_m: float,
    seed: np.ndarray,
):
    return adapter.solve_tcp_position(
        name,
        (*scene.spec.peg_start_xy_m, height_m),
        seed_joint_position_rad=seed,
        wrist_pitch_sum_rad=0.5 * np.pi,
        position_tolerance_m=0.0015,
    )


def main() -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.geometry) as geometry:
        fixed_follower_points = np.asarray(
            geometry[
                "fixed_follower_collision_body_points"
                if "fixed_follower_collision_body_points" in geometry
                else "fixed_collision_body_points"
            ],
            dtype=np.float64,
        )
        fixed_servo_points = (
            np.asarray(geometry["fixed_servo_collision_body_points"], dtype=np.float64)
            if "fixed_servo_collision_body_points" in geometry
            else np.empty((0, 3), dtype=np.float64)
        )
        moving_points = np.asarray(
            geometry[
                "moving_jaw_collision_body_points"
                if "moving_jaw_collision_body_points" in geometry
                else "moving_collision_body_points"
            ],
            dtype=np.float64,
        )

    collision_geometries: dict[str, dict[str, object]] = {}
    for label, body_name, points in (
        ("fixed_follower", "gripper_link", fixed_follower_points),
        ("fixed_servo", "gripper_link", fixed_servo_points),
        ("moving_jaw", "moving_jaw_so101_v1_link", moving_points),
    ):
        if points.shape[0] < 4:
            continue
        equations, hull_vertex_indices = _normalized_hull_equations(points)
        collision_geometries[label] = {
            "body_name": body_name,
            "hull_equations": equations,
            "hull_vertex_points_body": points[hull_vertex_indices],
            "source_vertex_count": int(points.shape[0]),
            "hull_vertex_count": int(hull_vertex_indices.shape[0]),
            "hull_plane_count": int(equations.shape[0]),
        }

    scene = SO101PegInsertionScene(
        usd_path=args.asset,
        device=args.device,
        mode=PegInsertionMode.GRASP_TRANSPORT,
    )
    try:
        adapter = SO101JointCommandAdapter(
            scene,
            tcp_offset_gripper_m=(args.tcp_x, 0.0, -0.085),
            open_gripper_rad=args.open_q,
            closed_gripper_rad=-0.04,
        )
        seed = np.asarray((0.30, -0.40, 1.20, 0.77, 0.0, args.open_q))
        approach = _solve(adapter, scene, "approach", args.approach_height, seed)
        if not approach.converged:
            raise RuntimeError(f"Approach IK did not converge: {approach.to_dict()}")
        grasp = _solve(
            adapter,
            scene,
            "grasp",
            args.grasp_height,
            np.asarray(approach.joint_position_rad),
        )
        if not grasp.converged:
            raise RuntimeError(f"Grasp IK did not converge: {grasp.to_dict()}")

        scene.reset()
        home_q = np.asarray(approach.joint_position_rad, dtype=np.float64)
        home_q[5] = args.open_q
        scene.write_kinematic_joint_state(home_q)
        adapter = SO101JointCommandAdapter(
            scene,
            tcp_offset_gripper_m=(args.tcp_x, 0.0, -0.085),
            open_gripper_rad=args.open_q,
            closed_gripper_rad=-0.04,
        )
        command_q = home_q.copy()
        initial_peg_position, _ = _peg_pose(scene)
        dt_s = scene.spec.simulation_dt_s
        records: list[dict[str, object]] = []
        peg_samples_local = _sample_cylinder_volume(
            radius_m=scene.spec.peg_radius_m,
            half_height_m=0.5 * scene.spec.peg_height_m,
        )

        settle_steps = int(round(args.settle_seconds / dt_s))
        for step in range(settle_steps):
            scene.step(torch.as_tensor(command_q, dtype=torch.float32, device=scene.device))
            if step % scene.spec.controller_decimation == 0:
                records.append(
                    _sample(
                        scene,
                        adapter,
                        time_s=(step + 1) * dt_s,
                        phase="settle",
                        initial_peg_position=initial_peg_position,
                        collision_geometries=collision_geometries,
                        peg_samples_local=peg_samples_local,
                        command_q=command_q,
                    )
                )

        nominal_q = np.asarray(grasp.joint_position_rad, dtype=np.float64)
        nominal_q[5] = args.open_q
        settled_peg_position, _ = _peg_pose(scene)
        descent_steps = int(round(args.descent_seconds / dt_s))
        first_disturbance_index: int | None = None
        for descent_step in range(descent_steps):
            servo = adapter.correct_tcp_position_command(
                target_tcp_position_m=grasp.target_tcp_position_m,
                nominal_joint_position_rad=nominal_q,
                previous_joint_command_rad=command_q,
                dt_s=dt_s,
                max_arm_rate_rad_s=args.max_arm_rate,
                proportional_gain_per_s=4.0,
                max_position_error_m=0.015,
            )
            raw_servo_q = np.asarray(servo.joint_position_rad, dtype=np.float64)
            if args.post_rate_limit:
                per_step_limit = np.asarray(
                    [args.max_arm_rate] * 5 + [1.25],
                    dtype=np.float64,
                ) * dt_s
                command_q = command_q + np.clip(
                    raw_servo_q - command_q,
                    -per_step_limit,
                    per_step_limit,
                )
            else:
                command_q = raw_servo_q
            scene.step(torch.as_tensor(command_q, dtype=torch.float32, device=scene.device))
            sample = _sample(
                scene,
                adapter,
                time_s=args.settle_seconds + (descent_step + 1) * dt_s,
                phase="descent",
                initial_peg_position=settled_peg_position,
                collision_geometries=collision_geometries,
                peg_samples_local=peg_samples_local,
                command_q=command_q,
            )
            sample["tcp_servo"] = servo.to_dict()
            sample["raw_tcp_servo_command_q_rad"] = raw_servo_q.tolist()
            records.append(sample)
            if (
                first_disturbance_index is None
                and np.linalg.norm(np.asarray(sample["peg_displacement_m"][:2]))
                >= 0.0005
            ):
                first_disturbance_index = len(records) - 1
                clearances = sample["jaw_clearance"]
                nearest_label = min(
                    collision_geometries,
                    key=lambda name: clearances[name][
                        "sampled_cylinder_hull_margin_m"
                    ],
                )
                print(
                    "[DESCENT-PROBE] first disturbance "
                    f"t={sample['time_s']:.4f}s "
                    f"peg={np.asarray(sample['peg_position_world_m']).round(6).tolist()} "
                    f"nearest_hull={nearest_label} "
                    f"hull_margin="
                    f"{clearances[nearest_label]['sampled_cylinder_hull_margin_m']:.6f}",
                    flush=True,
                )
            if np.linalg.norm(np.asarray(sample["peg_displacement_m"][:2])) >= 0.060:
                break

        if first_disturbance_index is None:
            attribution = "no_peg_disturbance"
            evidence_window: list[dict[str, object]] = records[-8:]
        else:
            window_start = max(0, first_disturbance_index - 6)
            window_end = min(len(records), first_disturbance_index + 7)
            evidence_window = records[window_start:window_end]
            onset = records[first_disturbance_index]
            attribution = min(
                collision_geometries,
                key=lambda name: onset["jaw_clearance"][name][
                    "sampled_cylinder_hull_margin_m"
                ],
            )

        result = {
            "scope": "open-gripper dynamic descent only; no closure or grasp claim",
            "settings": {
                "tcp_offset_gripper_m": [args.tcp_x, 0.0, -0.085],
                "open_gripper_rad": args.open_q,
                "approach_height_m": args.approach_height,
                "grasp_height_m": args.grasp_height,
                "max_arm_rate_rad_s": args.max_arm_rate,
                "post_rate_limit": args.post_rate_limit,
                "simulation_dt_s": dt_s,
            },
            "approach_solution": approach.to_dict(),
            "grasp_solution": grasp.to_dict(),
            "initial_peg_position_world_m": initial_peg_position.tolist(),
            "settled_peg_position_world_m": settled_peg_position.tolist(),
            "first_disturbance_index": first_disturbance_index,
            "contact_attribution": attribution,
            "collision_geometry": {
                label: {
                    "body_name": geometry["body_name"],
                    "source_vertex_count": geometry["source_vertex_count"],
                    "hull_vertex_count": geometry["hull_vertex_count"],
                    "hull_plane_count": geometry["hull_plane_count"],
                    "usd_approximation": "convexHull",
                }
                for label, geometry in collision_geometries.items()
            },
            "peg_volume_sample_count": int(peg_samples_local.shape[0]),
            "evidence_window": evidence_window,
            "records": records,
        }
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"ATTRIBUTION={attribution}", flush=True)
        print(f"RESULT={output}", flush=True)
    finally:
        scene.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
