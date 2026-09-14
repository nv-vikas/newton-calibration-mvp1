from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from newton_calibration.actuators import load_residual
from newton_calibration.core.models import EnvironmentSpec
from newton_calibration.validation.metrics import compare_trajectories


class AnalyticPDReplayRuntime:
    """CPU reference backend for contract tests; it is not the product physics backend."""

    def __init__(self, environment: EnvironmentSpec):
        self.environment = environment
        self.residual = load_residual(environment.residual_model_path, list(environment.joint_map))

    def describe(self) -> EnvironmentSpec:
        return self.environment

    def evaluate(self, candidate, episodes: Sequence, objective_weights):
        aggregate: list[dict[str, float]] = []
        per_episode: dict[str, dict[str, float]] = {}
        stable = True
        for episode in episodes:
            simulated_q, simulated_dq = self._rollout(candidate, episode)
            score, metrics = compare_trajectories(
                simulated_q,
                simulated_dq,
                episode.actual_q,
                episode.actual_dq,
                episode.command_q,
                self.environment.dt,
                objective_weights,
            )
            metrics["score"] = score
            per_episode[episode.name] = metrics
            aggregate.append(metrics)
            stable = stable and np.isfinite(simulated_q).all() and float(np.max(np.abs(simulated_q))) < 100.0
        names = aggregate[0].keys()
        means = {name: float(np.mean([item[name] for item in aggregate])) for name in names}
        return means["score"], means, per_episode, stable

    def _rollout(self, candidate, episode):
        dt = self.environment.dt
        q = episode.actual_q[0].copy()
        dq = episode.actual_dq[0].copy()
        q_history = np.empty_like(episode.actual_q)
        dq_history = np.empty_like(episode.actual_dq)
        delay_steps = max(0, round(candidate.get("command_delay_s", 0.0) / dt))
        arm_kp = 35.0 * candidate.get("arm_stiffness_scale", 1.0)
        arm_kd = 2.0 * candidate.get("arm_damping_scale", 1.0)
        grip_kp = 28.0 * candidate.get("gripper_stiffness_scale", 1.0)
        grip_kd = 1.5 * candidate.get("gripper_damping_scale", 1.0)
        kp = np.array([arm_kp] * 5 + [grip_kp])
        kd = np.array([arm_kd] * 5 + [grip_kd])
        friction = np.array([candidate.get("arm_friction_nm", 0.0)] * 5 + [candidate.get("gripper_friction_nm", 0.0)])
        effort = np.array(
            [self.environment.base_effort_limit * candidate.get("arm_effort_scale", 1.0)] * 5
            + [self.environment.base_effort_limit * candidate.get("gripper_effort_scale", 1.0)]
        )
        inertia = np.array([1.8, 1.6, 1.2, 0.7, 0.5, 0.35])
        inertia += np.array(
            [candidate.get("arm_armature", 0.0)] * 5 + [candidate.get("gripper_armature", 0.0)]
        )
        if self.residual is not None:
            self.residual.reset(episode.command_q[0])
        for step in range(len(episode.time_s)):
            command = episode.command_q[max(0, step - delay_steps)]
            torque = kp * (command - q) - kd * dq - friction * np.tanh(dq / 0.01)
            if self.residual is not None:
                torque += self.residual.compute(command, q, dq)
            torque = np.clip(torque, -effort, effort)
            ddq = torque / inertia
            dq += dt * ddq
            q += dt * dq
            q_history[step] = q
            dq_history[step] = dq
        return q_history, dq_history

    def close(self) -> None:
        return None
