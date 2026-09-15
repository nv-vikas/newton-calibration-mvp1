"""Thin, robot-agnostic adapter for a *running* Isaac Lab/Newton scene.

Imports the GPU runtime only when invoked. The application owns scene creation,
camera placement and unloaded/payload setup; this adapter replays USD-coordinate
commands and records real renderer frames. It never connects to hardware.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from newton_calibration.collection.planning import MotionSpec, verify_commands
from newton_calibration.core.io import sha256_file, write_json


class IsaacLabScenePreview:
    def __init__(
        self,
        *,
        sim,
        robot,
        camera,
        scene_id: str,
        fixture_prefixes: tuple[str, ...],
        backend_record: dict,
        update_objects=(),
        fixed_robot_prefixes: tuple[str, ...] = (),
    ):
        self.sim, self.robot, self.camera = sim, robot, camera
        self.scene_id, self.fixture_prefixes = scene_id, fixture_prefixes
        self.backend_record, self.update_objects = backend_record, update_objects
        self.fixed_robot_prefixes = fixed_robot_prefixes

    def __call__(self, plan_path: Path, output: Path) -> dict:
        import torch
        from isaaclab_newton.physics.newton_manager import NewtonManager

        from .video import MotionRecorder

        plan = verify_commands(plan_path)
        spec = MotionSpec(**plan["motion_spec"])
        if spec.scene_id != self.scene_id or self.backend_record.get("physics") != "Newton":
            raise ValueError("Wrong scene/backend bound to collection preview")
        if sha256_file(self.robot.cfg.spawn.usd_path) != plan["asset_sha256"]:
            raise ValueError("Active articulation's source USD differs from analyzed asset")
        dt = self.sim.get_physics_dt()
        model = NewtonManager._model  # pinned Isaac Lab 3 beta2 adapter, not a stable public API
        if model is None:
            raise RuntimeError("Newton model is not initialized")
        tensor = lambda value: value.torch if hasattr(value, "torch") else value
        base = tensor(self.robot.data.default_joint_pos).clone()
        if base.shape[0] != 1:
            raise ValueError("Collection preview currently supports one scene instance")
        ids = [self.robot.joint_names.index(j) for j in spec.joint_names]
        base[:, ids] = torch.tensor(spec.center_rad, device=base.device, dtype=base.dtype)
        output.mkdir(parents=True, exist_ok=True)
        backend = dict(
            self.backend_record,
            scene_id=self.scene_id,
            asset_sha256=plan["asset_sha256"],
            physics="Newton",
            newton_version=importlib.metadata.version("newton"),
            physics_dt_s=dt,
            controlled_joint_names=list(spec.joint_names),
            actual_joint_names=list(self.robot.joint_names),
            real_data=False,
        )
        backend_path = write_json(output / "backend.json", backend)
        result = {
            "schema": "newton.collection-screen/v1",
            "scene_id": self.scene_id,
            "physics": "Newton",
            "real_data": False,
            "real_execution_approved": False,
            "command_plan_sha256": sha256_file(plan_path),
            "tests": [],
            "passed": False,
            "self_collision_certified": False,
            "limitations": [
                "A simulation screen is not hardware safety approval",
                "Reported contact-buffer pairs are candidates, not proof of active force",
                "Self-contact candidates need geometry/penetration review; not silently filtered",
                "Real command smoothing, cables, payload and workspace may differ",
                "Simulated trajectories do not establish real parameter identifiability",
            ],
        }
        recorder = MotionRecorder(self.sim, self.camera, output, plan_path)
        labels = list(model.shape_label)
        robot_prefix = str(self.robot.cfg.prim_path).rstrip("/") + "/"

        def advance(target):
            self.robot.set_joint_position_target_index(target=target)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(dt)
            for obj in self.update_objects:
                obj.update(dt)

        try:
            for index, episode in enumerate(plan["episodes"]):
                table = np.genfromtxt(plan_path.parent / episode["command_file"], delimiter=",", names=True)
                values = np.column_stack([table[f"q{i + 1}_rad"] for i in range(len(ids))])
                self.robot.write_joint_position_to_sim_index(position=base)
                self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(base))
                self.robot.reset()
                for _ in range(round(1 / dt)):
                    advance(base)
                recorder.begin_episode(episode, index)
                error, speed, margin = 0.0, 0.0, float("inf")
                fixtures, self_pairs = set(), set()
                steps = round(episode["duration_s"] / dt)
                trace = []
                for k in range(steps):
                    t = k * dt
                    command = np.array([np.interp(t, table["time_s"], values[:, j]) for j in range(len(ids))])
                    target = base.clone()
                    target[:, ids] = torch.tensor(command, device=base.device, dtype=base.dtype)
                    advance(target)
                    actual = tensor(self.robot.data.joint_pos)[0, ids].cpu().numpy()
                    dq = tensor(self.robot.data.joint_vel)[0, ids].cpu().numpy()
                    if not np.isfinite(actual).all() or not np.isfinite(dq).all():
                        raise RuntimeError(f"Nonfinite Newton state in {episode['name']}")
                    error = max(error, float(abs(command - actual).max()))
                    speed = max(speed, float(abs(dq).max()))
                    margin = min(
                        margin,
                        float(np.min(actual - np.asarray(spec.lower_rad))),
                        float(np.min(np.asarray(spec.upper_rad) - actual)),
                    )
                    contacts = NewtonManager.get_contacts()
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    aa = contacts.rigid_contact_shape0.numpy()[:count]
                    bb = contacts.rigid_contact_shape1.numpy()[:count]
                    for a, b in zip(aa, bb):
                        if a < 0 or b < 0:
                            continue
                        la, lb = labels[int(a)], labels[int(b)]
                        # Ignore only fixed-base vs mount; record all other candidate pairs.
                        moving = lambda s: s.startswith(robot_prefix) and not s.startswith(self.fixed_robot_prefixes)
                        fixture = lambda s: s.startswith(self.fixture_prefixes)
                        if (moving(la) and fixture(lb)) or (moving(lb) and fixture(la)):
                            fixtures.add(tuple(sorted((la, lb))))
                        if la.startswith(robot_prefix) and lb.startswith(robot_prefix):
                            self_pairs.add(tuple(sorted((la, lb))))
                    if k % recorder.steps_per_frame == 0:
                        recorder.frame(t, command, actual, error, speed, margin, len(fixtures), len(self_pairs))
                    if k % max(1, round(0.05 / dt)) == 0:
                        trace.append(
                            {
                                "time_s": t,
                                "state_time_s": t + dt,
                                "command_q": command.tolist(),
                                "simulated_q": actual.tolist(),
                            }
                        )
                limits_ok = error < 0.1 and speed < max(spec.max_velocity_rad_s) * 1.5 and margin > spec.margin_rad
                item = {
                    "name": episode["name"],
                    "split": episode["split"],
                    "finite": True,
                    "max_position_error_rad": error,
                    "max_joint_velocity_rad_s": speed,
                    "min_joint_limit_margin_rad": margin,
                    "kinematic_screen_passed": bool(limits_ok),
                    "fixture_contact_candidates": sorted(fixtures),
                    "self_contact_candidates": sorted(self_pairs),
                    "contact_status": "review_required" if fixtures or self_pairs else "none_reported",
                    "passed": bool(
                        limits_ok
                        and not fixtures
                        and not self_pairs
                        and backend.get("self_collision_requested") is True
                    ),
                    "trace": trace,
                }
                result["tests"].append(item)
                result["passed"] = len(result["tests"]) == len(plan["episodes"]) and all(
                    x["passed"] for x in result["tests"]
                )
                write_json(output / "screen.json", result)
                recorder.end_episode(item)
                print(
                    "[COLLECTION] "
                    + episode["name"]
                    + " kinematics="
                    + str(limits_ok)
                    + " contact_candidates="
                    + str(len(fixtures) + len(self_pairs)),
                    flush=True,
                )
            recorder.finish(result)
        except BaseException:
            recorder.abort()
            raise
        return {
            "physics": "Newton",
            "scene_id": self.scene_id,
            "command_plan_sha256": sha256_file(plan_path),
            "video_path": str(recorder.path),
            "screen_path": str(output / "screen.json"),
            "backend_record_path": str(backend_path),
            "screen_passed": result["passed"],
            "hardware_approved": False,
        }
