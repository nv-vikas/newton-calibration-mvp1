#!/usr/bin/env python3
"""Conformance-check Newton's native SO-101 TCP Jacobian against FK deltas."""

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
parser.add_argument("--output", default="/workspace/output/so101_native_jacobian.json")
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
        q = np.asarray((0.30, -0.40, 1.20, 0.77, 0.0, adapter.open_gripper_rad))
        q = np.clip(q, adapter.lower, adapter.upper)
        scene.write_kinematic_joint_state(q)
        origin = adapter.current_tcp_position()
        native = adapter.native_tcp_position_jacobian()
        finite_difference = np.zeros_like(native)
        epsilon = 1.0e-4
        for joint_index in range(adapter.arm_joint_count):
            displaced_q = q.copy()
            displaced_q[joint_index] += epsilon
            scene.write_kinematic_joint_state(displaced_q)
            finite_difference[:, joint_index] = (
                adapter.current_tcp_position() - origin
            ) / epsilon
        scene.write_kinematic_joint_state(q)
        difference = native - finite_difference
        max_abs_error = float(np.max(np.abs(difference)))
        rms_error = float(np.sqrt(np.mean(difference**2)))
        # A 1e-4-rad forward difference on float32 state is noisy at roughly
        # 1e-4 m/rad.  This gate catches convention, body-row and TCP-shift
        # mistakes by orders of magnitude without pretending to be metrology.
        passed = max_abs_error <= 0.003
        payload = {
            "passed": passed,
            "asset": scene.asset_path,
            "body": "gripper_link",
            "fixed_base_body_id": scene.ee_body_id,
            "fixed_base_jacobian_row": scene.ee_body_id - 1,
            "tcp_offset_gripper_m": adapter.tcp_offset_gripper_m.tolist(),
            "joint_position_rad": q.tolist(),
            "native_tcp_jacobian_m_per_rad": native.tolist(),
            "finite_difference_tcp_jacobian_m_per_rad": finite_difference.tolist(),
            "max_abs_error_m_per_rad": max_abs_error,
            "rms_error_m_per_rad": rms_error,
            "acceptance_max_abs_error_m_per_rad": 0.003,
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        print(f"RESULT={output_path}", flush=True)
        if not passed:
            raise RuntimeError("Newton native TCP Jacobian failed finite-difference conformance")
    finally:
        scene.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
