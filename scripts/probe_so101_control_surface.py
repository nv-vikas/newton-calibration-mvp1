#!/usr/bin/env python3
"""Commission the SO-101 joint and kinematic control surface in Newton.

This is deliberately a read-only commissioning probe: it teleports the robot
through safe joint configurations, asks Newton for the resulting body poses,
and records the finite-difference position Jacobian.  It does not claim a
grasp or execute the peg-insertion task.
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
parser.add_argument("--output", default="/workspace/output/so101_control_probe.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch

from newton_calibration.isaaclab.tasks.so101_peg_insertion import PegInsertionMode
from newton_calibration.isaaclab.tasks.so101_peg_insertion.scene import SO101PegInsertionScene


def _torch(value):
    return value.torch if hasattr(value, "torch") else value


def _body_state(scene: SO101PegInsertionScene) -> dict[str, object]:
    positions = _torch(scene.robot.data.body_link_pos_w)[0].detach().cpu().numpy()
    quaternions = _torch(scene.robot.data.body_link_quat_w)[0].detach().cpu().numpy()
    return {
        name: {
            "position_w_m": [float(value) for value in positions[index]],
            "quaternion_raw": [float(value) for value in quaternions[index]],
        }
        for index, name in enumerate(scene.robot.body_names)
    }


def _set_state(scene: SO101PegInsertionScene, q: np.ndarray) -> dict[str, object]:
    position = torch.as_tensor(q[None, :], dtype=torch.float32, device=scene.device)
    velocity = torch.zeros_like(position)
    scene.robot.write_joint_state_to_sim_index(
        position=position,
        velocity=velocity,
        env_ids=scene.env_ids_tensor,
        joint_ids=scene.joint_ids_tensor,
    )
    scene.robot.set_joint_position_target_index(
        target=position,
        env_ids=scene.env_ids_tensor,
        joint_ids=scene.joint_ids_tensor,
    )
    scene.sim.forward()
    scene.robot.update(scene.spec.simulation_dt_s)
    measured = _torch(scene.robot.data.joint_pos)[0, scene.joint_ids_tensor].detach().cpu().numpy()
    return {
        "command_q_rad": [float(value) for value in q],
        "measured_q_rad": [float(value) for value in measured],
        "bodies": _body_state(scene),
    }


def _finite_difference_jacobian(
    scene: SO101PegInsertionScene,
    q: np.ndarray,
    *,
    body_name: str,
    epsilon: float = 1.0e-3,
) -> list[list[float]]:
    body_id = scene.robot.body_names.index(body_name)
    _set_state(scene, q)
    origin = _torch(scene.robot.data.body_link_pos_w)[0, body_id].detach().cpu().numpy().copy()
    jacobian = np.zeros((3, 5), dtype=np.float64)
    for joint_index in range(5):
        perturbed = q.copy()
        perturbed[joint_index] += epsilon
        _set_state(scene, perturbed)
        position = _torch(scene.robot.data.body_link_pos_w)[0, body_id].detach().cpu().numpy().copy()
        jacobian[:, joint_index] = (position - origin) / epsilon
    _set_state(scene, q)
    return [[float(value) for value in row] for row in jacobian]


def main() -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = SO101PegInsertionScene(
        usd_path=args.asset,
        device=args.device,
        mode=PegInsertionMode.ROBOT_ONLY,
    )
    try:
        default_q = _torch(scene.robot.data.default_joint_pos)[0, scene.joint_ids_tensor].detach().cpu().numpy()
        limit_value = getattr(scene.robot.data, "soft_joint_pos_limits", None)
        limits = None
        if limit_value is not None:
            limits_array = _torch(limit_value)[0, scene.joint_ids_tensor].detach().cpu().numpy()
            limits = [[float(value) for value in row] for row in limits_array]

        configurations = {
            "default": default_q.copy(),
            "safe_bent": np.asarray([0.0, -0.45, 0.70, -0.25, 0.0, default_q[5]], dtype=np.float64),
            "safe_left": np.asarray([0.35, -0.45, 0.70, -0.25, 0.0, default_q[5]], dtype=np.float64),
            "safe_right": np.asarray([-0.35, -0.45, 0.70, -0.25, 0.0, default_q[5]], dtype=np.float64),
        }
        sampled = {name: _set_state(scene, values) for name, values in configurations.items()}

        gripper_sweep = []
        for value in (-0.17, 0.0, 0.35, 0.70, 1.05, 1.40, 1.74):
            q = default_q.copy()
            q[5] = value
            state = _set_state(scene, q)
            fixed = np.asarray(state["bodies"]["gripper_link"]["position_w_m"])
            moving = np.asarray(state["bodies"]["moving_jaw_so101_v1_link"]["position_w_m"])
            state["jaw_origin_distance_m"] = float(np.linalg.norm(moving - fixed))
            gripper_sweep.append(state)

        result = {
            "scope": "joint/FK commissioning only; no grasp or insertion claim",
            "joint_names": scene.joint_names,
            "body_names": list(scene.robot.body_names),
            "default_q_rad": [float(value) for value in default_q],
            "soft_joint_limits_rad": limits,
            "configurations": sampled,
            "gripper_sweep": gripper_sweep,
            "finite_difference_position_jacobian_m_per_rad": {
                "body": "gripper_link",
                "at_configuration": "safe_bent",
                "values": _finite_difference_jacobian(
                    scene,
                    configurations["safe_bent"],
                    body_name="gripper_link",
                ),
            },
        }
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"RESULT={output_path}", flush=True)
    finally:
        scene.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
