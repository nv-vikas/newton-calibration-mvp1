"""Actual Newton finite-difference backend bound to an existing Isaac Lab scene.

No new scene, real evidence, hardware connection or optimized parameters are
created. Each prediction restores initial state and finally restores the scene's
original actuator settings. Private Newton APIs stay confined to this adapter.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from newton_calibration.core.io import sha256_file


class IsaacLabPredictionBackend:
    def __init__(self, *, sim, robot, environment, motion, update_objects=(), probe_window_s=None):
        import torch
        from isaaclab_newton.physics.newton_manager import NewtonManager

        if NewtonManager._model is None:
            raise ValueError("Newton must be initialized before constructing the dynamics probe")
        self.sim, self.robot, self.env, self.motion = sim, robot, environment, motion
        self.objects, self.window = tuple(update_objects), probe_window_s
        self.dt = sim.get_physics_dt()
        if abs(self.dt - environment.dt) > 1e-12:
            raise ValueError("Probe runtime dt differs from the analyzed environment")
        if sha256_file(robot.cfg.spawn.usd_path) != sha256_file(environment.asset_path):
            raise ValueError("Probe scene has a different source USD")
        if probe_window_s is not None and (not np.isfinite(probe_window_s) or probe_window_s < 2):
            raise ValueError("Probe diagnostic windows must be at least two seconds")
        self.torch = torch
        self.tensor = lambda x: x.torch if hasattr(x, "torch") else x
        self.ids = [robot.joint_names.index(j) for j in motion.joint_names]
        self.q0 = self.tensor(robot.data.joint_pos).clone()
        self.v0 = self.tensor(robot.data.joint_vel).clone()
        if self.q0.shape[0] != 1:
            raise ValueError("This prediction adapter supports one scene instance")
        self.root0 = self.tensor(robot.data.root_link_pose_w).clone()
        self.rootv0 = self.tensor(robot.data.root_com_vel_w).clone()
        self.object_states = [
            (self.tensor(o.data.root_link_pose_w).clone(), self.tensor(o.data.root_com_vel_w).clone())
            for o in self.objects
        ]
        self.device = self.q0.device
        self.indices = torch.tensor(self.ids, device=self.device, dtype=torch.long)
        self.env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        self.armature0 = self.tensor(robot.data.joint_armature)[:, self.ids].clone()
        self.friction0 = self.tensor(robot.data.joint_friction_coeff)[:, self.ids].clone()
        self.actuator_original = {
            name: {key: getattr(a, key).clone() for key in ("stiffness", "damping", "effort_limit", "armature")}
            for name, a in robot.actuators.items()
        }
        self.bindings = {}
        for group, members in environment.joint_groups.items():
            usd = [environment.joint_map[j] for j in members]
            for kind in ("stiffness_scale", "damping_scale", "armature", "friction_nm"):
                self.bindings[f"{group}_{kind}"] = (kind, usd)
        self.calls = 0
        # RTX-compatible Newton defers CUDA graph capture until the first step;
        # capture itself performs an eager warm-up integration. Consume it before
        # any measured baseline and restore state, rather than loosening the
        # repeatability threshold or accepting a startup-contaminated response.
        self.warmup_steps = 4
        self._warmup()

    def _warmup(self):
        try:
            target = self._reset(self.motion.center_rad)
            for _ in range(self.warmup_steps):
                self.robot.set_joint_position_target_index(target=target)
                self.robot.write_data_to_sim()
                self.sim.step(render=False)
                self.robot.update(self.dt)
                for obj in self.objects:
                    obj.update(self.dt)
        finally:
            self._reset(self.motion.center_rad)

    def describe(self):
        return {
            "physics": "Newton",
            "backend": "isaaclab-scene-finite-difference/v1",
            "scene_id": self.motion.scene_id,
            "asset_sha256": sha256_file(self.env.asset_path),
            "newton_version": importlib.metadata.version("newton"),
            "runtime_dt_s": self.dt,
            "controlled_joints": list(self.motion.joint_names),
            "probe_window_s": self.window,
            "window_policy": "full command file"
            if self.window is None
            else "center diagnostic window, reset at its first command",
            "parameter_readback": True,
            "restores_actuator_settings": True,
            "startup_warmup_steps_excluded": self.warmup_steps,
            "adapter_source_sha256": sha256_file(__file__),
            "real_data": False,
        }

    def _apply(self, parameters):
        for name, original in self.actuator_original.items():
            for key, value in original.items():
                getattr(self.robot.actuators[name], key)[:] = value
        armature, friction = self.armature0.clone(), self.friction0.clone()
        expected = []
        for name, value in parameters.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid Newton probe parameter: {name}")
            if name == "command_delay_s":
                continue
            if name not in self.bindings:
                raise ValueError(f"Unsupported or invalid Newton probe parameter: {name}")
            kind, joints = self.bindings[name]
            for joint in joints:
                index = self.motion.joint_names.index(joint)
                if kind == "armature":
                    armature[:, index] = value
                elif kind == "friction_nm":
                    friction[:, index] = value
                else:
                    attribute = "stiffness" if kind == "stiffness_scale" else "damping"
                    matches = [(key, a) for key, a in self.robot.actuators.items() if joint in a.joint_names]
                    if len(matches) != 1:
                        raise ValueError(f"Joint {joint} has ambiguous actuator ownership")
                    key, actuator = matches[0]
                    local = actuator.joint_names.index(joint)
                    target = self.actuator_original[key][attribute][:, local] * value
                    getattr(actuator, attribute)[:, local] = target
                    expected.append((getattr(actuator, attribute)[:, local], target))
        self.robot.write_joint_armature_to_sim_index(armature=armature, env_ids=self.env_ids, joint_ids=self.indices)
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction, env_ids=self.env_ids, joint_ids=self.indices
        )
        expected += [
            (self.tensor(self.robot.data.joint_armature)[:, self.ids], armature),
            (self.tensor(self.robot.data.joint_friction_coeff)[:, self.ids], friction),
        ]
        if any(not self.torch.allclose(actual, target, rtol=1e-5, atol=1e-7) for actual, target in expected):
            raise RuntimeError("Newton sensitivity parameter write/readback mismatch")

    def _reset(self, initial):
        self.robot.reset()
        self.robot.write_root_pose_to_sim_index(root_pose=self.root0)
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.rootv0)
        q = self.q0.clone()
        q[:, self.ids] = self.torch.tensor(initial, device=self.device, dtype=q.dtype)
        self.robot.write_joint_state_to_sim_index(position=q, velocity=self.torch.zeros_like(self.v0))
        self.robot.set_joint_effort_target_index(target=self.torch.zeros_like(self.v0))
        for obj, (pose, velocity) in zip(self.objects, self.object_states):
            obj.reset()
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=velocity)
        return q

    def rollout(self, command_path: Path, experiment, parameters):
        table = np.loadtxt(command_path, delimiter=",", skiprows=1)
        times, q = table[:, 0], table[:, 1 : len(self.ids) + 1]
        duration = float(times[-1]) if self.window is None else min(float(times[-1]), self.window)
        start = (float(times[-1]) - duration) / 2 if self.window is not None else 0.0
        t = np.arange(round(duration / self.dt)) * self.dt + start
        delay = round(parameters.get("command_delay_s", 0.0) / self.dt) * self.dt
        values = np.column_stack([np.interp(t - delay, times, q[:, j]) for j in range(len(self.ids))])
        commands = self.torch.tensor(values, device=self.device, dtype=self.q0.dtype)
        sample_stride = max(1, round(0.01 / self.dt))
        sampled = []
        try:
            target = self._reset(values[0])
            self._apply(parameters)
            for step, command in enumerate(commands):
                target[:, self.ids] = command
                self.robot.set_joint_position_target_index(target=target)
                self.robot.write_data_to_sim()
                self.sim.step(render=False)
                self.robot.update(self.dt)
                for obj in self.objects:
                    obj.update(self.dt)
                if step % sample_stride == 0:
                    position = self.tensor(self.robot.data.joint_pos)[0, self.ids].cpu().numpy().copy()
                    velocity = self.tensor(self.robot.data.joint_vel)[0, self.ids].cpu().numpy().copy()
                    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
                        raise ValueError("Unstable Newton sensitivity prediction")
                    if np.any(position < np.array(self.motion.lower_rad) + self.motion.margin_rad) or np.any(
                        position > np.array(self.motion.upper_rad) - self.motion.margin_rad
                    ):
                        raise ValueError("Perturbed Newton prediction crossed the declared joint-limit margin")
                    sampled.append(np.concatenate((position, velocity)))
            self.calls += 1
            return np.array(sampled)
        finally:
            self._apply({})
            self._reset(self.motion.center_rad)
