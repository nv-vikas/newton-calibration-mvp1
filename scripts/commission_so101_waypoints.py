#!/usr/bin/env python3
"""Solve SO-101 peg-task waypoints against Newton's actual forward kinematics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

if os.environ.get("ACCEPT_EULA") != "Y":
    raise RuntimeError("Review the NVIDIA Isaac Sim license and export ACCEPT_EULA=Y before running.")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", required=True)
parser.add_argument("--output", default="/workspace/output/controller_commission/waypoints.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np

from newton_calibration.isaaclab.tasks.so101_peg_insertion import (
    PegInsertionMode,
    SO101JointCommandAdapter,
)
from newton_calibration.isaaclab.tasks.so101_peg_insertion.scene import SO101PegInsertionScene


def main() -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = SO101PegInsertionScene(
        usd_path=args.asset,
        device=args.device,
        mode=PegInsertionMode.ROBOT_ONLY,
    )
    try:
        adapter = SO101JointCommandAdapter(scene)
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
        # The fingers approach along the gripper's -Z axis.  A +90 degree
        # shoulder/elbow/wrist pitch sum turns that tool axis from horizontal
        # at reset to a top-down grasp while keeping wrist roll free.
        downward_pitch_sum = 0.5 * math.pi
        seed = np.asarray([0.30, -0.40, 1.20, 0.77, 0.0, adapter.open_gripper_rad])
        solutions = []
        for name, target in targets:
            solution = adapter.solve_tcp_position(
                name,
                target,
                seed_joint_position_rad=seed,
                wrist_pitch_sum_rad=downward_pitch_sum,
                position_tolerance_m=0.0012 if name != "insert_socket" else 0.0008,
            )
            solutions.append(solution)
            if solution.converged:
                seed = np.asarray(solution.joint_position_rad, dtype=np.float64)

        result = {
            "scope": "offline Newton-FK waypoint commissioning; no dynamics or contact claim",
            "tcp_offset_gripper_m": adapter.tcp_offset_gripper_m.tolist(),
            "open_gripper_rad": adapter.open_gripper_rad,
            "closed_gripper_rad": adapter.closed_gripper_rad,
            "all_converged": all(solution.converged for solution in solutions),
            "solutions": [solution.to_dict() for solution in solutions],
        }
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"RESULT={output_path}", flush=True)
        if not result["all_converged"]:
            raise RuntimeError("One or more SO-101 task waypoints are not reachable")
    finally:
        scene.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
