from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from newton_calibration.actuators import load_residual
from newton_calibration.adapters.runtime.analytic import (
    _joint_properties,
    _resolve_joint_layout,
    _validate_episode_width,
)
from newton_calibration.core.attestation import (
    episode_inputs,
    evaluation_result_fingerprint,
    numeric_surface_fingerprint,
    parameter_fingerprint,
)
from newton_calibration.core.io import sha256_file
from newton_calibration.core.models import EnvironmentSpec
from newton_calibration.validation.metrics import compare_trajectories


class IsaacLabNewtonRuntime:
    """Thin Isaac Lab adapter that replays real commands in the Newton runtime."""

    def __init__(self, environment: EnvironmentSpec):
        self.environment = environment
        self.joint_layout = _resolve_joint_layout(environment)
        self.actuator_name = "so101" if self.joint_layout.legacy_so101 else "calibration"
        # Validate per-joint defaults before importing or launching Isaac Lab.
        _joint_properties(environment, self.joint_layout, {}, analytic=False)
        if environment.calibration_manifest_path:
            from newton_calibration.adapters.surface.package_loader import (
                VerifiedArticulationPackage,
                VerifiedSO101Package,
            )

            package_type = VerifiedSO101Package if self.joint_layout.legacy_so101 else VerifiedArticulationPackage
            verified = package_type.open(
                Path(environment.calibration_manifest_path).parent,
                expected_manifest_sha256=environment.calibration_manifest_sha256,
            )
            verified.assert_matches_environment(environment)
        self._build()

    def _build(self) -> None:
        try:
            import isaaclab.sim as sim_utils
            import torch
            from isaaclab.actuators import IdealPDActuatorCfg
            from isaaclab.assets import ArticulationCfg
            from isaaclab.sim import SimulationCfg, build_simulation_context
            from isaaclab_newton.assets import Articulation
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
        except ImportError as exc:  # pragma: no cover - exercised in the Isaac Lab container
            raise RuntimeError(
                "The isaaclab_newton adapter must run inside an Isaac Lab 3.0 image with the Newton backend."
            ) from exc

        self.torch = torch
        physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                iterations=self.environment.solver_iterations,
                tolerance=self.environment.solver_tolerance,
                integrator="implicitfast",
                cone="pyramidal",
                impratio=1.0,
                ls_parallel=False,
            ),
            num_substeps=self.environment.num_substeps,
            use_cuda_graph=False,
        )
        sim_cfg = SimulationCfg(
            dt=self.environment.dt,
            device=self.environment.device,
            gravity=self.environment.gravity,
            physics=physics,
            # Newton 1.2's native ControllerPD currently ignores ``max_effort``
            # when imported from USD. Use Isaac Lab's explicit PD model so the
            # command is torque-clipped, then apply that torque to Newton.
            use_newton_actuators=False,
        )
        # ``build_simulation_context`` is a context manager. Entering it creates
        # and registers the USD stage in kit-less Isaac Lab; merely calling the
        # function leaves the global stage unset.
        self._sim_context_manager = build_simulation_context(
            sim_cfg=sim_cfg,
            device=self.environment.device,
        )
        self.sim = self._sim_context_manager.__enter__()
        self.sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Env_0", "Xform")
        desired = list(self.joint_layout.runtime_names)
        selected_joint_patterns = [f"^{re.escape(name)}$" for name in desired]
        cfg = ArticulationCfg(
            prim_path="/World/Env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=self.environment.asset_path),
            actuators={
                self.actuator_name: IdealPDActuatorCfg(
                    joint_names_expr=selected_joint_patterns,
                    stiffness=self.environment.base_stiffness,
                    damping=self.environment.base_damping,
                    armature=self.environment.base_armature,
                    effort_limit=self.environment.base_effort_limit,
                    # The explicit actuator owns clipping. The physics-engine
                    # limit remains high to avoid applying the same cap twice.
                    effort_limit_sim=1.0e9,
                )
            },
        )
        self.robot = Articulation(cfg)
        self.sim.reset()
        if not self.robot.is_initialized:
            raise RuntimeError(f"Articulation {self.environment.robot_id!r} did not initialize in Newton")
        self.joint_ids, matched = self.robot.find_joints(selected_joint_patterns, preserve_order=True)
        if matched != desired:
            raise RuntimeError(
                f"Articulation {self.environment.robot_id!r} joint mapping mismatch. "
                f"Expected {desired}, found {matched}"
            )
        actuator = self.robot.actuators[self.actuator_name]
        actuator_names = list(getattr(actuator, "joint_names", ()))
        if actuator_names != desired:
            raise RuntimeError(
                f"Calibration actuator joint mismatch. Expected {desired}, found {actuator_names}"
            )
        actuator_global_ids = _actuator_global_indices(
            getattr(actuator, "joint_indices", None),
            total_joints=self.robot.num_joints,
        )
        if actuator_global_ids != list(self.joint_ids):
            raise RuntimeError(
                "Calibration actuator indices do not match the selected articulation joints: "
                f"actuator={actuator_global_ids}, articulation={list(self.joint_ids)}"
            )
        self.joint_ids_tensor = torch.tensor(self.joint_ids, device=self.environment.device, dtype=torch.long)
        self.env_ids_tensor = torch.tensor([0], device=self.environment.device, dtype=torch.long)
        residual_path = self.environment.residual_model_path
        residual_before = sha256_file(residual_path) if residual_path else None
        self.residual = load_residual(residual_path, list(self.joint_layout.logical_names))
        residual_after = sha256_file(residual_path) if residual_path else None
        if residual_before != residual_after:
            raise RuntimeError("Actuator residual changed while the Newton runtime was loading it")
        if self.environment.residual_model_sha256 not in (None, residual_after):
            raise RuntimeError("Actuator residual does not match the locked environment fingerprint")
        self._loaded_residual_sha256 = residual_after
        # Capture the complete post-import state once. Every episode restores
        # root pose/velocity and every DOF before selected evidence coordinates
        # are overwritten, so passive/unselected joints cannot leak state from
        # one candidate or episode into the next.
        self._canonical_root_pose = self.robot.data.root_link_pose_w.torch.clone()
        self._canonical_root_velocity = self.robot.data.root_com_vel_w.torch.clone()
        self._canonical_joint_position = self.robot.data.joint_pos.torch.clone()
        self._canonical_joint_velocity = self.robot.data.joint_vel.torch.clone()
        self._apply_candidate(dict(self.environment.calibration_parameters))
        self._verify_actuator_readback(dict(self.environment.calibration_parameters))

    def describe(self) -> EnvironmentSpec:
        return self.environment

    def attestation(self) -> dict[str, object]:
        if not hasattr(self, "_last_evaluation"):
            raise RuntimeError("Newton runtime cannot attest before a complete evidence evaluation succeeds")
        return {
            "schema": "newton.calibration.runtime-attestation/v2",
            "backend": "isaaclab_newton",
            "authoritative": True,
            "robot_id": self.environment.robot_id,
            "asset_sha256": sha256_file(self.environment.asset_path),
            "logical_joints": list(self.joint_layout.logical_names),
            "runtime_joints": list(self.joint_layout.runtime_names),
            "runtime_dt_s": self.environment.dt,
            "gravity": list(self.environment.gravity),
            "num_substeps": self.environment.num_substeps,
            "solver_iterations": self.environment.solver_iterations,
            "solver_tolerance": self.environment.solver_tolerance,
            "selected_joint_scoped": True,
            "full_state_reset_per_episode": True,
            "readback_parameters": ["stiffness", "damping", "effort_limit", "armature", "friction_nm"],
            "candidate_sha256": self._last_verified_candidate_sha256,
            "readback_values_sha256": self._last_readback_values_sha256,
            "residual_sha256": self._loaded_residual_sha256,
            **self._last_evaluation,
        }

    def evaluate(
        self,
        candidate,
        episodes: Sequence,
        objective_weights,
        *,
        phase: str = "unscoped",
        run_id: str = "",
        plan_sha256: str = "",
        evidence_fingerprint: str = "",
        mapping_fingerprint: str = "",
    ):
        candidate = self._resolve_candidate(candidate)
        aggregate: list[dict[str, float]] = []
        per_episode: dict[str, dict[str, float]] = {}
        stable = True
        for episode in episodes:
            simulated_q, simulated_dq, episode_stable = self._rollout(candidate, episode)
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
            stable = stable and episode_stable
        names = aggregate[0].keys()
        means = {name: float(np.mean([item[name] for item in aggregate])) for name in names}
        if not stable:
            means["score"] += 1_000.0
        self._last_evaluation = {
            "evaluation_phase": phase,
            "run_id": run_id,
            "plan_sha256": plan_sha256,
            "evidence_fingerprint": evidence_fingerprint,
            "mapping_fingerprint": mapping_fingerprint,
            "evidence_episodes": episode_inputs(episodes),
            "result_sha256": evaluation_result_fingerprint(
                score=means["score"], metrics=means, episodes=per_episode, stable=stable
            ),
        }
        return means["score"], means, per_episode, stable

    def _resolve_candidate(self, candidate: dict[str, float]) -> dict[str, float]:
        resolved = dict(self.environment.calibration_parameters)
        resolved.update(candidate)
        return resolved

    def _apply_candidate(self, candidate: dict[str, float]) -> None:
        torch = self.torch
        properties = self._parameter_vectors(candidate)

        def tensor(name: str):
            return torch.as_tensor(
                properties[name][None, :],
                dtype=torch.float32,
                device=self.environment.device,
            )

        stiffness = tensor("stiffness")
        damping = tensor("damping")
        friction = tensor("friction")
        effort = tensor("effort")
        armature = tensor("armature")
        actuator = self.robot.actuators[self.actuator_name]
        # Actuator tensors are actuator-local, even when its joints are a
        # non-contiguous subset of the articulation. Global articulation IDs
        # belong only on robot state/target/write APIs below.
        actuator.stiffness[:, :] = stiffness
        actuator.damping[:, :] = damping
        actuator.effort_limit[:, :] = effort
        actuator.armature[:, :] = armature
        self.robot.write_joint_armature_to_sim_index(
            armature=armature, env_ids=self.env_ids_tensor, joint_ids=self.joint_ids_tensor
        )
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction, env_ids=self.env_ids_tensor, joint_ids=self.joint_ids_tensor
        )

    def _parameter_vectors(self, candidate: dict[str, float]) -> dict[str, np.ndarray]:
        """Return ordered per-joint values without importing Isaac Lab or torch."""

        return _joint_properties(
            self.environment,
            self.joint_layout,
            candidate,
            analytic=False,
        )

    def _verify_actuator_readback(self, candidate: dict[str, float]) -> None:
        expected = self._parameter_vectors(candidate)
        actuator = self.robot.actuators[self.actuator_name]
        actual = {
            "stiffness": actuator.stiffness[:, :],
            "damping": actuator.damping[:, :],
            "effort_limit": actuator.effort_limit[:, :],
            # These two arrays are bound to Newton model attributes, rather
            # than to the explicit actuator buffers that issued the writes.
            "armature": self.robot.data.joint_armature.torch[:, self.joint_ids_tensor],
            "friction_nm": self.robot.data.joint_friction_coeff.torch[:, self.joint_ids_tensor],
        }
        expected_by_surface = {
            "stiffness": expected["stiffness"],
            "damping": expected["damping"],
            "effort_limit": expected["effort"],
            "armature": expected["armature"],
            "friction_nm": expected["friction"],
        }
        observed_by_surface: dict[str, list[float]] = {}
        for name, values in actual.items():
            observed = values.detach().cpu().numpy()[0]
            if not np.allclose(observed, expected_by_surface[name], rtol=1e-5, atol=1e-7):
                raise RuntimeError(f"Newton actuator {name} write/readback mismatch")
            observed_by_surface[name] = observed.tolist()
        self._last_verified_candidate_sha256 = parameter_fingerprint(candidate)
        self._last_readback_values_sha256 = numeric_surface_fingerprint(observed_by_surface)

    def _rollout(self, candidate, episode):
        torch = self.torch
        _validate_episode_width(episode, len(self.joint_layout.logical_names))
        self.robot.reset()
        self.robot.write_root_pose_to_sim_index(
            root_pose=self._canonical_root_pose,
            env_ids=self.env_ids_tensor,
        )
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self._canonical_root_velocity,
            env_ids=self.env_ids_tensor,
        )
        self.robot.write_joint_state_to_sim_index(
            position=self._canonical_joint_position,
            velocity=self._canonical_joint_velocity,
            env_ids=self.env_ids_tensor,
        )
        self._apply_candidate(candidate)
        self._verify_actuator_readback(candidate)
        initial_q = torch.as_tensor(episode.actual_q[0:1], dtype=torch.float32, device=self.environment.device)
        initial_dq = torch.as_tensor(episode.actual_dq[0:1], dtype=torch.float32, device=self.environment.device)
        self.robot.write_joint_state_to_sim_index(
            position=initial_q,
            velocity=initial_dq,
            env_ids=self.env_ids_tensor,
            joint_ids=self.joint_ids_tensor,
        )
        self.robot.set_joint_effort_target_index(
            target=torch.zeros_like(initial_q),
            env_ids=self.env_ids_tensor,
            joint_ids=self.joint_ids_tensor,
        )
        delay_steps = max(0, round(candidate.get("command_delay_s", 0.0) / self.environment.dt))
        q_history = np.empty_like(episode.actual_q)
        dq_history = np.empty_like(episode.actual_dq)
        stable = True
        if self.residual is not None:
            self.residual.reset(episode.command_q[0])
        for step in range(len(episode.time_s)):
            command = torch.as_tensor(
                episode.command_q[max(0, step - delay_steps) : max(0, step - delay_steps) + 1],
                dtype=torch.float32,
                device=self.environment.device,
            )
            self.robot.set_joint_position_target_index(
                target=command,
                env_ids=self.env_ids_tensor,
                joint_ids=self.joint_ids_tensor,
            )
            if self.residual is not None:
                current_q = self.robot.data.joint_pos.torch[0, self.joint_ids_tensor].detach().cpu().numpy()
                current_dq = self.robot.data.joint_vel.torch[0, self.joint_ids_tensor].detach().cpu().numpy()
                residual_effort = self.residual.compute(command.detach().cpu().numpy()[0], current_q, current_dq)
                self.robot.set_joint_effort_target_index(
                    target=torch.as_tensor(
                        residual_effort[None, :], dtype=torch.float32, device=self.environment.device
                    ),
                    env_ids=self.env_ids_tensor,
                    joint_ids=self.joint_ids_tensor,
                )
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.environment.dt)
            q = self.robot.data.joint_pos.torch[0, self.joint_ids_tensor].detach().cpu().numpy()
            dq = self.robot.data.joint_vel.torch[0, self.joint_ids_tensor].detach().cpu().numpy()
            q_history[step], dq_history[step] = q, dq
            if not np.isfinite(q).all() or float(np.max(np.abs(q))) > 100.0:
                q_history[step:] = q
                dq_history[step:] = dq
                stable = False
                break
        return q_history, dq_history, stable

    def close(self) -> None:
        if getattr(self, "sim", None) is not None:
            self._sim_context_manager.__exit__(None, None, None)
            self.sim = None


def _actuator_global_indices(value, *, total_joints: int) -> list[int]:
    """Normalize Isaac Lab actuator joint indices for an exact scope check."""

    if value is None:
        raise RuntimeError("Calibration actuator did not expose joint_indices")
    if isinstance(value, slice):
        return list(range(total_joints))[value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [int(index) for index in value]
