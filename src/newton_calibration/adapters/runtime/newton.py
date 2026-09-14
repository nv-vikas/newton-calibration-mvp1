from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from newton_calibration.actuators import load_residual
from newton_calibration.core.models import EnvironmentSpec
from newton_calibration.validation.metrics import compare_trajectories


class IsaacLabNewtonRuntime:
    """Thin Isaac Lab adapter that replays real commands in the Newton runtime."""

    def __init__(self, environment: EnvironmentSpec):
        self.environment = environment
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
        cfg = ArticulationCfg(
            prim_path="/World/Env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=self.environment.asset_path),
            actuators={
                "so101": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
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
            raise RuntimeError("SO-101 articulation did not initialize in Newton")
        desired = [self.environment.joint_map[name] for name in self.environment.joint_map]
        self.joint_ids, matched = self.robot.find_joints(desired, preserve_order=True)
        if matched != desired:
            raise RuntimeError(f"SO-101 joint mapping mismatch. Expected {desired}, found {matched}")
        self.joint_ids_tensor = torch.tensor(self.joint_ids, device=self.environment.device, dtype=torch.long)
        self.env_ids_tensor = torch.tensor([0], device=self.environment.device, dtype=torch.long)
        self.residual = load_residual(
            self.environment.residual_model_path,
            list(self.environment.joint_map),
        )

    def describe(self) -> EnvironmentSpec:
        return self.environment

    def evaluate(self, candidate, episodes: Sequence, objective_weights):
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
        return means["score"], means, per_episode, stable

    def _apply_candidate(self, candidate: dict[str, float]) -> None:
        torch = self.torch
        stiffness = torch.tensor(
            [[self.environment.base_stiffness * candidate["arm_stiffness_scale"]] * 5
             + [self.environment.base_stiffness * candidate["gripper_stiffness_scale"]]],
            dtype=torch.float32,
            device=self.environment.device,
        )
        damping = torch.tensor(
            [[self.environment.base_damping * candidate["arm_damping_scale"]] * 5
             + [self.environment.base_damping * candidate["gripper_damping_scale"]]],
            dtype=torch.float32,
            device=self.environment.device,
        )
        friction = torch.tensor(
            [[candidate["arm_friction_nm"]] * 5 + [candidate["gripper_friction_nm"]]],
            dtype=torch.float32,
            device=self.environment.device,
        )
        effort = torch.tensor(
            [[self.environment.base_effort_limit * candidate["arm_effort_scale"]] * 5
             + [self.environment.base_effort_limit * candidate["gripper_effort_scale"]]],
            dtype=torch.float32,
            device=self.environment.device,
        )
        armature = torch.tensor(
            [[candidate["arm_armature"]] * 5 + [candidate["gripper_armature"]]],
            dtype=torch.float32,
            device=self.environment.device,
        )
        actuator = self.robot.actuators["so101"]
        actuator.stiffness[:, self.joint_ids_tensor] = stiffness
        actuator.damping[:, self.joint_ids_tensor] = damping
        actuator.effort_limit[:, self.joint_ids_tensor] = effort
        actuator.armature[:, self.joint_ids_tensor] = armature
        self.robot.write_joint_armature_to_sim_index(
            armature=armature, env_ids=self.env_ids_tensor, joint_ids=self.joint_ids_tensor
        )
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction, env_ids=self.env_ids_tensor, joint_ids=self.joint_ids_tensor
        )

    def _rollout(self, candidate, episode):
        torch = self.torch
        self.robot.reset()
        self._apply_candidate(candidate)
        initial_q = torch.as_tensor(episode.actual_q[0:1], dtype=torch.float32, device=self.environment.device)
        initial_dq = torch.as_tensor(episode.actual_dq[0:1], dtype=torch.float32, device=self.environment.device)
        self.robot.write_joint_state_to_sim_index(
            position=initial_q,
            velocity=initial_dq,
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
