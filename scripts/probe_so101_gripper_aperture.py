#!/usr/bin/env python3
"""Measure the SO-101 fingertip geometry over the authored gripper sweep.

This is a kinematic commissioning probe.  It combines mesh vertices from the
source USD with Newton's authoritative link poses; it does not execute contact
or claim that any command produces a successful grasp.
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
parser.add_argument("--asset", required=True, help="SO-101 USD path")
parser.add_argument("--geometry", required=True, help="NPZ from extract_so101_gripper_mesh.py")
parser.add_argument(
    "--output",
    default="/workspace/output/controller_commission/gripper_aperture.json",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
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
    twice_cross = 2.0 * np.cross(np.broadcast_to(xyz, vectors.shape), vectors)
    return vectors + w * twice_cross + np.cross(
        np.broadcast_to(xyz, vectors.shape), twice_cross
    )


def _points_in_gripper_frame(
    scene: SO101PegInsertionScene,
    *,
    body_name: str,
    points_body: np.ndarray,
) -> np.ndarray:
    gripper_position, gripper_quaternion = scene.body_pose_world("gripper_link")
    body_position, body_quaternion = scene.body_pose_world(body_name)
    points_world = body_position + _rotate_xyzw(body_quaternion, points_body)
    inverse_gripper = np.asarray(
        (-gripper_quaternion[0], -gripper_quaternion[1], -gripper_quaternion[2], gripper_quaternion[3]),
        dtype=np.float64,
    )
    return _rotate_xyzw(inverse_gripper, points_world - gripper_position)


def _bounds(points: np.ndarray) -> dict[str, list[float] | int]:
    return {
        "count": int(points.shape[0]),
        "min_m": [float(value) for value in points.min(axis=0)],
        "max_m": [float(value) for value in points.max(axis=0)],
    }


def _slab_bounds(points: np.ndarray, low_z: float, high_z: float) -> dict[str, object] | None:
    selected = points[(points[:, 2] >= low_z) & (points[:, 2] < high_z)]
    return _bounds(selected) if selected.size else None


def _interval_separation(a: tuple[float, float], b: tuple[float, float]) -> float:
    if a[1] < b[0]:
        return b[0] - a[1]
    if b[1] < a[0]:
        return a[0] - b[1]
    return 0.0


def _tip_summary(fixed: np.ndarray, moving: np.ndarray) -> dict[str, object]:
    # The controller's TCP is at gripper-local z=-85 mm, while the peg centre
    # at the grasp target is z=-104.5 mm.  These slabs expose the jaw envelope
    # around both stations rather than reporting one opaque aggregate bound.
    slabs = []
    for low_z, high_z in ((-0.110, -0.100), (-0.100, -0.090), (-0.090, -0.080), (-0.080, -0.070)):
        fixed_bounds = _slab_bounds(fixed, low_z, high_z)
        moving_bounds = _slab_bounds(moving, low_z, high_z)
        row: dict[str, object] = {
            "z_range_gripper_m": [low_z, high_z],
            "fixed": fixed_bounds,
            "moving": moving_bounds,
        }
        if fixed_bounds is not None and moving_bounds is not None:
            fixed_min = np.asarray(fixed_bounds["min_m"])
            fixed_max = np.asarray(fixed_bounds["max_m"])
            moving_min = np.asarray(moving_bounds["min_m"])
            moving_max = np.asarray(moving_bounds["max_m"])
            row["separation_x_m"] = _interval_separation(
                (float(fixed_min[0]), float(fixed_max[0])),
                (float(moving_min[0]), float(moving_max[0])),
            )
            row["separation_y_m"] = _interval_separation(
                (float(fixed_min[1]), float(fixed_max[1])),
                (float(moving_min[1]), float(moving_max[1])),
            )
            fixed_inner_x = float(fixed_max[0])
            moving_inner_x = float(moving_min[0])
            if fixed_inner_x < moving_inner_x:
                row["fixed_inner_x_m"] = fixed_inner_x
                row["moving_inner_x_m"] = moving_inner_x
                row["aperture_x_m"] = moving_inner_x - fixed_inner_x
                row["aperture_midpoint_x_m"] = 0.5 * (
                    fixed_inner_x + moving_inner_x
                )
        slabs.append(row)
    return {"slabs": slabs}


def main() -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry_path = Path(args.geometry)
    with np.load(geometry_path) as geometry:
        fixed_body_points = np.asarray(
            geometry["fixed_collision_body_points"], dtype=np.float64
        )
        moving_body_points = np.asarray(
            geometry["moving_collision_body_points"], dtype=np.float64
        )

    scene = SO101PegInsertionScene(
        usd_path=args.asset,
        device=args.device,
        mode=PegInsertionMode.GRASP_TRANSPORT,
    )
    try:
        adapter = SO101JointCommandAdapter(scene)
        # Use the actual top-down grasp posture.  Only the gripper coordinate is
        # varied, so all aperture measurements share one arm geometry.
        seed = np.asarray((0.30, -0.40, 1.20, 0.77, 0.0, 0.0), dtype=np.float64)
        solution = adapter.solve_tcp_position(
            "grasp_peg",
            (*scene.spec.peg_start_xy_m, scene.spec.grasp_tcp_height_m),
            seed_joint_position_rad=seed,
            wrist_pitch_sum_rad=np.pi / 2.0,
        )
        if not solution.converged:
            raise RuntimeError(f"Top-down grasp IK did not converge: {solution.to_dict()}")

        sweep = []
        for command in (
            -0.15,
            0.0,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.50,
            0.70,
            0.90,
            1.10,
            1.30,
            1.50,
            1.70,
        ):
            q = np.asarray(solution.joint_position_rad, dtype=np.float64)
            q[5] = command
            scene.write_kinematic_joint_state(q)
            fixed = _points_in_gripper_frame(
                scene,
                body_name="gripper_link",
                points_body=fixed_body_points,
            )
            moving = _points_in_gripper_frame(
                scene,
                body_name="moving_jaw_so101_v1_link",
                points_body=moving_body_points,
            )
            jaw_origin = _points_in_gripper_frame(
                scene,
                body_name="moving_jaw_so101_v1_link",
                points_body=np.zeros((1, 3), dtype=np.float64),
            )[0]
            sweep.append(
                {
                    "command_rad": command,
                    "jaw_origin_in_gripper_m": [float(value) for value in jaw_origin],
                    "fixed_mesh_in_gripper": _bounds(fixed),
                    "moving_mesh_in_gripper": _bounds(moving),
                    "tip": _tip_summary(fixed, moving),
                }
            )

        gripper_position, gripper_quaternion = scene.body_pose_world("gripper_link")
        gripper_axes_world = _rotate_xyzw(
            gripper_quaternion,
            np.eye(3, dtype=np.float64),
        )
        inverse_gripper = np.asarray(
            (
                -gripper_quaternion[0],
                -gripper_quaternion[1],
                -gripper_quaternion[2],
                gripper_quaternion[3],
            ),
            dtype=np.float64,
        )
        peg_center_world = np.asarray(
            (*scene.spec.peg_start_xy_m, scene.spec.peg_rest_center_z_m),
            dtype=np.float64,
        )
        peg_center_gripper = _rotate_xyzw(
            inverse_gripper,
            (peg_center_world - gripper_position)[None, :],
        )[0]

        result = {
            "scope": (
                "kinematic collision-mesh/FK commissioning only; "
                "no dynamic-contact or grasp claim"
            ),
            "asset": str(Path(args.asset).resolve()),
            "joint_limit_rad": [float(scene.joint_position_limits_rad()[5, 0]), float(scene.joint_position_limits_rad()[5, 1])],
            "peg_diameter_m": scene.spec.peg_diameter_m,
            "tcp_offset_gripper_m": adapter.tcp_offset_gripper_m.tolist(),
            "peg_center_at_grasp_in_gripper_m": peg_center_gripper.tolist(),
            "grasp_solution": solution.to_dict(),
            "gripper_pose_world_at_grasp": {
                "position_m": gripper_position.tolist(),
                "quaternion_xyzw": gripper_quaternion.tolist(),
                "local_axes_in_world": {
                    "x": gripper_axes_world[0].tolist(),
                    "y": gripper_axes_world[1].tolist(),
                    "z": gripper_axes_world[2].tolist(),
                },
            },
            "fixed_mesh_body": _bounds(fixed_body_points),
            "moving_mesh_body": _bounds(moving_body_points),
            "sweep": sweep,
        }
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"RESULT={output}", flush=True)
    finally:
        scene.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
