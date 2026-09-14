"""Actual Isaac Lab + Newton scene for the SO-101 peg-insertion task.

Import this module only after ``isaaclab.app.AppLauncher`` has started the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab_newton.assets import Articulation, RigidObject
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors.contact_sensor import (
    ContactSensorCfg as NewtonContactSensorCfg,
)
from isaaclab_newton.sim.schemas import (
    NewtonCollisionPropertiesCfg,
    NewtonMaterialPropertiesCfg,
    NewtonRigidBodyPropertiesCfg,
)
from pxr import Usd

from .contract import (
    PegInsertionMode,
    PegInsertionSceneSpec,
    PegInsertionTaskMetrics,
    local_z_axis_from_quaternion_xyzw,
)


_V3_FIXED_FINGER_SHAPE_EXPR = (
    "/World/Env_0/Robot/gripper_link/"
    "newton_collision_v.*_fixed_follower/part_.*"
)
_V3_MOVING_FINGER_SHAPE_EXPR = (
    "/World/Env_0/Robot/moving_jaw_so101_v1_link/"
    "newton_collision_v.*_moving_jaw/part_.*"
)
_V4_FIXED_PAD_SHAPE_EXPR = (
    "/World/Env_0/Robot/gripper_link/"
    "newton_collision_v4_fixed_pad/part_000"
)
_V4_MOVING_PAD_SHAPE_EXPR = (
    "/World/Env_0/Robot/moving_jaw_so101_v1_link/"
    "newton_collision_v4_moving_pad/part_000"
)
_SOCKET_SHAPE_EXPR = "/World/Env_0/Socket.*"


@dataclass(frozen=True)
class _GraspContactShapeProfile:
    """Exact force-attribution contract declared by the robot asset."""

    profile_id: str
    attribution: str
    fixed_shape_expr: str
    moving_shape_expr: str
    expected_part_names: tuple[str, ...]


_GRASP_CONTACT_PROFILES = {
    "so101_gripper_prebaked_v3": _GraspContactShapeProfile(
        profile_id="so101_gripper_prebaked_v3",
        attribution="versioned_finger_proxy_shapes",
        fixed_shape_expr=_V3_FIXED_FINGER_SHAPE_EXPR,
        moving_shape_expr=_V3_MOVING_FINGER_SHAPE_EXPR,
        expected_part_names=tuple(f"part_{index:03d}" for index in range(32)),
    ),
    "so101_task_planar_pads_v4": _GraspContactShapeProfile(
        profile_id="so101_task_planar_pads_v4",
        attribution="task_planar_pad_shapes_v4",
        fixed_shape_expr=_V4_FIXED_PAD_SHAPE_EXPR,
        moving_shape_expr=_V4_MOVING_PAD_SHAPE_EXPR,
        expected_part_names=("part_000",),
    ),
}


class SO101PegInsertionScene:
    """One-robot reference scene shared by MVP 1, MVP 2 and MVP 3."""

    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    def __init__(
        self,
        *,
        usd_path: str,
        device: str = "cuda:0",
        mode: PegInsertionMode = PegInsertionMode.INSERTION,
        spec: PegInsertionSceneSpec | None = None,
    ) -> None:
        self.asset_path = str(Path(usd_path).expanduser().resolve())
        if not Path(self.asset_path).is_file():
            raise FileNotFoundError(f"SO-101 USD does not exist: {self.asset_path}")
        self.device = device
        self.mode = mode
        self.spec = spec or PegInsertionSceneSpec()
        self.spec.validate()
        self.grasp_contact_shape_profile = self._resolve_grasp_contact_shape_profile()
        self._build()

    def _resolve_grasp_contact_shape_profile(
        self,
    ) -> _GraspContactShapeProfile | None:
        """Resolve the exact contact contract from versioned USD metadata.

        MVP1 may use a canonical robot without task collision metadata. MVP2/3
        fail before model construction unless the asset declares a supported
        collision profile; guessing from a file name would make force evidence
        non-reproducible.
        """

        stage = Usd.Stage.Open(self.asset_path, Usd.Stage.LoadNone)
        if stage is None:
            raise RuntimeError(f"Could not inspect SO-101 USD metadata: {self.asset_path}")
        default_prim = stage.GetDefaultPrim()
        profile_id = (
            default_prim.GetCustomDataByKey(
                "newtonCalibration:collisionProfile"
            )
            if default_prim
            else None
        )
        if self.mode is PegInsertionMode.ROBOT_ONLY:
            return _GRASP_CONTACT_PROFILES.get(str(profile_id))
        profile = _GRASP_CONTACT_PROFILES.get(str(profile_id))
        if profile is None:
            raise RuntimeError(
                "MVP2/3 requires a supported asset-declared Newton grasp-contact "
                "profile; found "
                f"{profile_id!r} in {self.asset_path}. Supported profiles: "
                f"{sorted(_GRASP_CONTACT_PROFILES)}"
            )
        return profile

    def _build(self) -> None:
        print("[scene] stage=configure_simulation begin", flush=True)
        physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                iterations=self.spec.solver_iterations,
                tolerance=self.spec.solver_tolerance,
                integrator="implicitfast",
                cone="pyramidal",
                impratio=1.0,
                ls_parallel=False,
                nconmax=256,
                njmax=512,
                # Per-partner force matrices are qualified in the pinned
                # Isaac Lab/Newton tests only through MuJoCo contact mode.
                use_mujoco_contacts=True,
            ),
            num_substeps=self.spec.solver_substeps,
            use_cuda_graph=False,
        )
        sim_cfg = SimulationCfg(
            dt=self.spec.simulation_dt_s,
            device=self.device,
            gravity=(0.0, 0.0, -9.81),
            physics=physics,
            render_interval=self.spec.controller_decimation,
            use_newton_actuators=False,
        )
        self._sim_manager = build_simulation_context(sim_cfg=sim_cfg, device=self.device)
        self.sim = self._sim_manager.__enter__()
        print("[scene] stage=configure_simulation complete", flush=True)
        self.sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Env_0", "Xform")
        print("[scene] stage=spawn_static begin", flush=True)
        self._spawn_static_scene()
        print("[scene] stage=spawn_static complete", flush=True)
        print("[scene] stage=spawn_robot begin", flush=True)
        self.robot = self._spawn_robot()
        print("[scene] stage=spawn_robot complete", flush=True)
        print("[scene] stage=spawn_peg begin", flush=True)
        self.peg = self._spawn_peg()
        print("[scene] stage=spawn_peg complete", flush=True)
        print("[scene] stage=spawn_contact_sensors begin", flush=True)
        (
            self.fixed_finger_contact_sensor,
            self.moving_finger_contact_sensor,
            self.socket_contact_sensor,
        ) = self._spawn_contact_sensors()
        print("[scene] stage=spawn_contact_sensors complete", flush=True)
        print("[scene] stage=simulation_reset begin", flush=True)
        self.sim.reset()
        print("[scene] stage=simulation_reset complete", flush=True)
        if not all(
            component.is_initialized
            for component in (
                self.robot,
                self.peg,
                self.fixed_finger_contact_sensor,
                self.moving_finger_contact_sensor,
                self.socket_contact_sensor,
            )
        ):
            raise RuntimeError("SO-101, peg, or contact sensors failed to initialize in Newton")
        self._validate_contact_sensor_attribution()
        self.joint_ids, matched = self.robot.find_joints(self.joint_names, preserve_order=True)
        if matched != self.joint_names:
            raise RuntimeError(f"SO-101 joint mapping mismatch: expected {self.joint_names}, found {matched}")
        self.joint_ids_tensor = torch.tensor(self.joint_ids, device=self.device, dtype=torch.long)
        self.env_ids_tensor = torch.tensor([0], device=self.device, dtype=torch.long)
        self.ee_body_id = self.robot.body_names.index("gripper_link")
        self.jaw_body_id = self.robot.body_names.index("moving_jaw_so101_v1_link")
        self._configure_robot_hold_actuator()
        self.reset()

    def _collision(self) -> NewtonCollisionPropertiesCfg:
        return NewtonCollisionPropertiesCfg(
            collision_enabled=True,
            contact_margin=0.0,
            contact_gap=self.spec.contact_gap_m,
        )

    @staticmethod
    def _material(
        color: tuple[float, float, float],
        *,
        friction: float,
        metallic: float = 0.0,
    ) -> tuple[NewtonMaterialPropertiesCfg, sim_utils.PreviewSurfaceCfg]:
        physics = NewtonMaterialPropertiesCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
            torsional_friction=0.002,
            rolling_friction=0.0005,
        )
        visual = sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic, roughness=0.35)
        return physics, visual

    def _spawn_static_scene(self) -> None:
        spec = self.spec
        table_physics, table_visual = self._material((0.15, 0.17, 0.20), friction=0.65, metallic=0.15)
        table = sim_utils.CuboidCfg(
            size=spec.table_size_m,
            collision_props=self._collision(),
            physics_material=table_physics,
            visual_material=table_visual,
        )
        table.func(
            "/World/Env_0/Table",
            table,
            translation=(0.20, 0.0, spec.table_top_z_m - 0.5 * spec.table_size_m[2]),
        )

        socket_physics, socket_visual = self._material((0.12, 0.34, 0.62), friction=0.35, metallic=0.65)
        wall_thickness = 0.5 * (spec.socket_outer_m - spec.socket_aperture_m)
        cx, cy = spec.socket_center_xy_m
        z = spec.table_top_z_m + 0.5 * spec.socket_wall_height_m
        offset = 0.5 * (spec.socket_aperture_m + wall_thickness)
        walls = {
            "North": ((spec.socket_outer_m, wall_thickness, spec.socket_wall_height_m), (cx, cy + offset, z)),
            "South": ((spec.socket_outer_m, wall_thickness, spec.socket_wall_height_m), (cx, cy - offset, z)),
            "East": ((wall_thickness, spec.socket_aperture_m, spec.socket_wall_height_m), (cx + offset, cy, z)),
            "West": ((wall_thickness, spec.socket_aperture_m, spec.socket_wall_height_m), (cx - offset, cy, z)),
        }
        for name, (size, position) in walls.items():
            wall = sim_utils.CuboidCfg(
                size=size,
                collision_props=self._collision(),
                physics_material=socket_physics,
                visual_material=socket_visual,
            )
            wall.func(f"/World/Env_0/Socket{name}", wall, translation=position)

        ground = sim_utils.GroundPlaneCfg(
            physics_material=table_physics,
            color=(0.055, 0.060, 0.070),
        )
        ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.08))
        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.92, 0.95, 1.0))
        light.func("/World/Light", light)

    def _spawn_robot(self) -> Articulation:
        spec = self.spec
        cfg = ArticulationCfg(
            class_type=Articulation,
            prim_path="/World/Env_0/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.asset_path,
                activate_contact_sensors=True,
            ),
            actuators={
                "so101": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=spec.arm_control_stiffness_nm_per_rad,
                    damping=spec.arm_control_damping_nm_s_per_rad,
                    armature=0.0,
                    effort_limit=spec.arm_control_effort_limit_nm,
                    effort_limit_sim=1.0e9,
                )
            },
        )
        return Articulation(cfg)

    def _spawn_peg(self) -> RigidObject:
        spec = self.spec
        peg_physics, peg_visual = self._material((0.72, 0.74, 0.78), friction=0.40, metallic=0.9)
        cfg = RigidObjectCfg(
            class_type=RigidObject,
            prim_path="/World/Env_0/Peg",
            spawn=sim_utils.CylinderCfg(
                radius=spec.peg_radius_m,
                height=spec.peg_height_m,
                axis="Z",
                rigid_props=NewtonRigidBodyPropertiesCfg(rigid_body_enabled=True),
                collision_props=self._collision(),
                mass_props=sim_utils.MassPropertiesCfg(mass=spec.peg_mass_kg),
                physics_material=peg_physics,
                visual_material=peg_visual,
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(spec.peg_start_xy_m[0], spec.peg_start_xy_m[1], spec.peg_rest_center_z_m),
                # Isaac Lab 3.0/Newton uses xyzw quaternions.
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        return RigidObject(cfg)

    def _spawn_contact_sensors(self) -> tuple[ContactSensor, ContactSensor, ContactSensor]:
        """Create separately attributed fixed-finger, moving-finger and socket sensors.

        MVP 2/3 contact claims require exactly one split, versioned collision-
        proxy scope. MVP 1 may
        still run against a canonical robot asset, but its whole-body readings
        are explicitly labelled as a safety-only fallback and never qualify a
        grasp.
        """

        if self.mode is PegInsertionMode.ROBOT_ONLY:
            self.grasp_contact_attribution = "body_fallback_mvp1_safety_only"
            fixed_finger = ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/Env_0/Peg",
                    update_period=0.0,
                    history_length=1,
                    filter_prim_paths_expr=["/World/Env_0/Robot/gripper_link"],
                )
            )
            moving_finger = ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/Env_0/Peg",
                    update_period=0.0,
                    history_length=1,
                    filter_prim_paths_expr=[
                        "/World/Env_0/Robot/moving_jaw_so101_v1_link"
                    ],
                )
            )
        else:
            profile = self.grasp_contact_shape_profile
            if profile is None:
                raise RuntimeError("MVP2/3 grasp-contact profile was not resolved")
            self.grasp_contact_attribution = profile.attribution
            fixed_finger = ContactSensor(
                NewtonContactSensorCfg(
                    prim_path="/World/Env_0/Peg",
                    update_period=0.0,
                    history_length=1,
                    filter_shape_prim_expr=[profile.fixed_shape_expr],
                )
            )
            moving_finger = ContactSensor(
                NewtonContactSensorCfg(
                    prim_path="/World/Env_0/Peg",
                    update_period=0.0,
                    history_length=1,
                    filter_shape_prim_expr=[profile.moving_shape_expr],
                )
            )
        socket = ContactSensor(
            NewtonContactSensorCfg(
                prim_path="/World/Env_0/Peg",
                update_period=0.0,
                history_length=1,
                filter_shape_prim_expr=[_SOCKET_SHAPE_EXPR],
            )
        )
        return fixed_finger, moving_finger, socket

    def _validate_contact_sensor_attribution(self) -> None:
        """Fail closed when a contact channel does not resolve its declared shapes."""

        fixed_names = list(self.fixed_finger_contact_sensor.filter_object_names or [])
        moving_names = list(self.moving_finger_contact_sensor.filter_object_names or [])
        if self.mode is PegInsertionMode.ROBOT_ONLY:
            if fixed_names != ["gripper_link"] or moving_names != [
                "moving_jaw_so101_v1_link"
            ]:
                raise RuntimeError(
                    "MVP1 body-contact fallback did not resolve the declared SO-101 bodies: "
                    f"fixed={fixed_names}, moving={moving_names}"
                )
        else:
            profile = self.grasp_contact_shape_profile
            if profile is None:
                raise RuntimeError("MVP2/3 grasp-contact profile was not resolved")
            expected_parts = set(profile.expected_part_names)
            if set(fixed_names) != expected_parts or len(fixed_names) != len(expected_parts):
                raise RuntimeError(
                    "MVP2/3 fixed force channel did not match the asset-declared "
                    f"{profile.profile_id!r} profile; "
                    f"resolved {len(fixed_names)} shapes: {fixed_names}"
                )
            if set(moving_names) != expected_parts or len(moving_names) != len(expected_parts):
                raise RuntimeError(
                    "MVP2/3 moving force channel did not match the asset-declared "
                    f"{profile.profile_id!r} profile; "
                    f"resolved {len(moving_names)} shapes: {moving_names}"
                )

        socket_names = list(self.socket_contact_sensor.filter_object_names or [])
        # The pinned Newton backend preserves the four matching shape indices,
        # but ContactSensor exposes each generated cuboid's leaf basename as
        # ``mesh`` rather than its SocketNorth/South/East/West parent path.
        # Exact coverage is therefore established by the anchored path pattern
        # plus a strict four-shape count; names alone cannot distinguish walls.
        if (
            self.socket_contact_sensor.num_filter_objects != 4
            or len(socket_names) != 4
        ):
            raise RuntimeError(
                "Socket contact filter must resolve exactly the four wall shapes; "
                f"found {socket_names}"
            )

    def _configure_robot_hold_actuator(self) -> None:
        """Apply the explicit environment-commissioning controller profile.

        This controller is only used to prove that the task can be physically
        exercised.  It is not a calibration result and, in particular, its
        gripper effort is not an MVP2 force calibration.
        """

        actuator = self.robot.actuators["so101"]
        stiffness = torch.tensor(
            [[self.spec.arm_control_stiffness_nm_per_rad] * 5
             + [self.spec.gripper_control_stiffness_nm_per_rad]],
            dtype=torch.float32,
            device=self.device,
        )
        damping = torch.tensor(
            [[self.spec.arm_control_damping_nm_s_per_rad] * 5
             + [self.spec.gripper_control_damping_nm_s_per_rad]],
            dtype=torch.float32,
            device=self.device,
        )
        effort = torch.tensor(
            [[self.spec.arm_control_effort_limit_nm] * 5
             + [self.spec.gripper_control_effort_limit_nm]],
            dtype=torch.float32,
            device=self.device,
        )
        armature = torch.tensor(
            [[self.spec.arm_armature_kg_m2] * 5 + [self.spec.gripper_armature_kg_m2]],
            dtype=torch.float32,
            device=self.device,
        )
        friction = torch.tensor(
            [[self.spec.arm_joint_friction_nm] * 5 + [self.spec.gripper_joint_friction_nm]],
            dtype=torch.float32,
            device=self.device,
        )
        actuator.stiffness[:, self.joint_ids_tensor] = stiffness
        actuator.damping[:, self.joint_ids_tensor] = damping
        actuator.effort_limit[:, self.joint_ids_tensor] = effort
        actuator.armature[:, self.joint_ids_tensor] = armature
        self.robot.write_joint_armature_to_sim_index(
            armature=armature,
            env_ids=self.env_ids_tensor,
            joint_ids=self.joint_ids_tensor,
        )
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction,
            env_ids=self.env_ids_tensor,
            joint_ids=self.joint_ids_tensor,
        )

    def reset(self, *, probe: str | None = None) -> None:
        self.robot.reset()
        self._configure_robot_hold_actuator()
        default_q = self.robot.data.default_joint_pos.torch.clone()
        default_dq = torch.zeros_like(default_q)
        self._parked_joint_position = default_q
        self._parked_joint_velocity = default_dq
        self._write_parked_robot_state()
        if probe in {"centered", "offset"}:
            offset = 0.0 if probe == "centered" else 0.75 * self.spec.socket_aperture_m
            peg_position = (
                self.spec.socket_center_xy_m[0] + offset,
                self.spec.socket_center_xy_m[1],
                self.spec.socket_top_z_m + 0.5 * self.spec.peg_height_m + 0.012,
            )
        else:
            # Calibration mode controls how far the controller runs; it must not
            # silently change the task geometry.  In particular, MVP 1 still
            # approaches the recipe peg pose.  The former ROBOT_ONLY parking
            # pose (0.50, 0.30) was outside the tabletop's Y extent, so the peg
            # immediately fell below the table and the run measured the wrong
            # initial condition.
            peg_position = (
                self.spec.peg_start_xy_m[0],
                self.spec.peg_start_xy_m[1],
                self.spec.peg_rest_center_z_m,
            )
        pose = torch.tensor(
            [[*peg_position, 0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        self.peg.write_root_pose_to_sim_index(root_pose=pose)
        self.peg.write_root_velocity_to_sim_index(root_velocity=velocity)
        self.peg.reset()
        self.fixed_finger_contact_sensor.reset()
        self.moving_finger_contact_sensor.reset()
        self.socket_contact_sensor.reset()
        self.sim.forward()
        self.robot.update(self.spec.simulation_dt_s)
        self.peg.update(self.spec.simulation_dt_s)

    @staticmethod
    def _filtered_force_magnitudes_n(sensor: ContactSensor) -> tuple[list[str], np.ndarray]:
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError("Filtered contact sensor did not publish a force matrix")
        values = matrix.torch
        if values.ndim != 4 or values.shape[0] != 1 or values.shape[-1] != 3:
            raise RuntimeError(f"Unexpected contact force-matrix shape: {tuple(values.shape)}")
        # Sum force magnitudes rather than norming a vector sum: opposing
        # contacts must not cancel and hide a real constraint.
        magnitudes = torch.linalg.vector_norm(values[0], dim=-1).sum(dim=0)
        names = list(sensor.filter_object_names or [])
        if len(names) != int(magnitudes.shape[0]):
            raise RuntimeError(
                f"Contact-filter metadata/data mismatch: names={names}, shape={tuple(values.shape)}"
            )
        result = magnitudes.detach().cpu().numpy().astype(np.float64, copy=True)
        if not np.isfinite(result).all() or np.any(result < 0.0):
            raise RuntimeError(f"Contact sensor published invalid force magnitudes: {result}")
        return names, result

    def contact_force_evidence_n(self) -> dict[str, object]:
        """Return per-partner normal-force evidence from the pinned solver."""

        fixed_names, fixed_forces = self._filtered_force_magnitudes_n(
            self.fixed_finger_contact_sensor
        )
        moving_names, moving_forces = self._filtered_force_magnitudes_n(
            self.moving_finger_contact_sensor
        )
        socket_names, socket_forces = self._filtered_force_magnitudes_n(
            self.socket_contact_sensor
        )
        return {
            "fixed_finger_force_n": float(fixed_forces.sum()),
            "moving_finger_force_n": float(moving_forces.sum()),
            "socket_normal_force_n": float(socket_forces.sum()),
            "grasp_contact_attribution": self.grasp_contact_attribution,
            "grasp_contact_profile": (
                self.grasp_contact_shape_profile.profile_id
                if self.grasp_contact_shape_profile is not None
                else None
            ),
            "fixed_finger_filter_objects": fixed_names,
            "moving_finger_filter_objects": moving_names,
            "socket_filter_objects": socket_names,
            "source": "Newton/MJWarp ContactSensor.force_matrix_w",
        }

    def _write_parked_robot_state(self) -> None:
        """Restore the fixed reference pose used by isolated object-contact probes."""

        self.robot.write_joint_state_to_sim_index(
            position=self._parked_joint_position,
            velocity=self._parked_joint_velocity,
            env_ids=self.env_ids_tensor,
            joint_ids=self.joint_ids_tensor,
        )
        self.robot.set_joint_position_target_index(
            target=self._parked_joint_position,
            env_ids=self.env_ids_tensor,
            joint_ids=self.joint_ids_tensor,
        )

    @staticmethod
    def _torch(value):
        return value.torch if hasattr(value, "torch") else value

    def joint_position_rad(self) -> np.ndarray:
        """Return the six commanded SO-101 joints in canonical order."""

        return (
            self._torch(self.robot.data.joint_pos)[0, self.joint_ids_tensor]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
        )

    def joint_velocity_rad_s(self) -> np.ndarray:
        return (
            self._torch(self.robot.data.joint_vel)[0, self.joint_ids_tensor]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
        )

    def joint_position_limits_rad(self) -> np.ndarray:
        limits = self._torch(self.robot.data.soft_joint_pos_limits)[0, self.joint_ids_tensor]
        return limits.detach().cpu().numpy().astype(np.float64, copy=True)

    def applied_joint_torque_nm(self) -> np.ndarray:
        value = getattr(self.robot.data, "applied_torque", None)
        if value is None:
            return np.zeros(len(self.joint_ids), dtype=np.float64)
        torque = self._torch(value)[0, self.joint_ids_tensor]
        return torque.detach().cpu().numpy().astype(np.float64, copy=True)

    def body_pose_world(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return one link pose as position plus Isaac Lab xyzw quaternion."""

        body_id = self.robot.body_names.index(body_name)
        position = self._torch(self.robot.data.body_link_pos_w)[0, body_id]
        quaternion = self._torch(self.robot.data.body_link_quat_w)[0, body_id]
        return (
            position.detach().cpu().numpy().astype(np.float64, copy=True),
            quaternion.detach().cpu().numpy().astype(np.float64, copy=True),
        )

    def body_link_jacobian_world(self, body_name: str) -> np.ndarray:
        """Return Newton's world-frame 6xN link-origin spatial Jacobian.

        This SO-101 is fixed-base.  Newton therefore omits the root body from
        ``body_link_jacobian_w``: body id 5 (``gripper_link``) is Jacobian row
        4.  Rows 0:3 are linear velocity, rows 3:6 angular velocity.
        """

        body_id = self.robot.body_names.index(body_name)
        if body_id == 0:
            raise ValueError("The fixed root body has no articulated-link Jacobian row")
        jacobians = self._torch(self.robot.data.body_link_jacobian_w)
        if jacobians.ndim != 4 or jacobians.shape[2] != 6:
            raise RuntimeError(
                "Unexpected Newton link-Jacobian layout: "
                f"expected [env, link, 6, dof], got {tuple(jacobians.shape)}"
            )
        row = body_id - 1
        if row >= jacobians.shape[1]:
            raise RuntimeError(
                f"Newton Jacobian has no row {row} for fixed-base body {body_name!r}"
            )
        link_jacobian = jacobians[0, row]
        canonical = torch.index_select(link_jacobian, 1, self.joint_ids_tensor)
        result = canonical.detach().cpu().numpy().astype(np.float64, copy=True)
        if not np.isfinite(result).all():
            raise RuntimeError(f"Newton published a non-finite Jacobian for {body_name!r}")
        return result

    def write_kinematic_joint_state(self, joint_position_rad: np.ndarray) -> None:
        """Publish one joint state and refresh Newton FK without integrating.

        This method exists for commissioning and offline waypoint planning.  The
        task runner uses :meth:`step` so actuator and contact dynamics remain
        authoritative during execution.
        """

        values = np.asarray(joint_position_rad, dtype=np.float64)
        if values.shape != (len(self.joint_ids),):
            raise ValueError(f"Expected {len(self.joint_ids)} joint values, got {values.shape}")
        position = torch.as_tensor(values[None, :], dtype=torch.float32, device=self.device)
        velocity = torch.zeros_like(position)
        self.robot.write_joint_state_to_sim_index(
            position=position,
            velocity=velocity,
            env_ids=self.env_ids_tensor,
            joint_ids=self.joint_ids_tensor,
        )
        self.robot.set_joint_position_target_index(
            target=position,
            env_ids=self.env_ids_tensor,
            joint_ids=self.joint_ids_tensor,
        )
        self.sim.forward()
        self.robot.update(self.spec.simulation_dt_s)

    def step(
        self,
        joint_target: torch.Tensor | None = None,
        *,
        render: bool = False,
        hold_robot: bool = False,
    ) -> None:
        if hold_robot:
            self._write_parked_robot_state()
        if joint_target is not None:
            target = joint_target.reshape(1, -1).to(device=self.device, dtype=torch.float32)
            self.robot.set_joint_position_target_index(
                target=target,
                env_ids=self.env_ids_tensor,
                joint_ids=self.joint_ids_tensor,
            )
        self.robot.write_data_to_sim()
        self.peg.write_data_to_sim()
        self.sim.step(render=render)
        if hold_robot:
            # The conformance probe is intentionally about peg/socket contact,
            # not free-fall of an uncalibrated robot. Restore the parked state
            # after integration so rendering and metrics cannot inherit drift.
            self._write_parked_robot_state()
            self.sim.forward()
        self.robot.update(self.spec.simulation_dt_s)
        self.peg.update(self.spec.simulation_dt_s)
        self.fixed_finger_contact_sensor.update(
            self.spec.simulation_dt_s,
            force_recompute=True,
        )
        self.moving_finger_contact_sensor.update(
            self.spec.simulation_dt_s,
            force_recompute=True,
        )
        self.socket_contact_sensor.update(
            self.spec.simulation_dt_s,
            force_recompute=True,
        )

    def task_metrics(self, *, settled: bool = False) -> PegInsertionTaskMetrics:
        pose = self.peg.data.root_pose_w.torch[0].detach().cpu()
        position = tuple(float(value) for value in pose[:3])
        axis = local_z_axis_from_quaternion_xyzw(
            tuple(float(value) for value in pose[3:7])
        )
        return PegInsertionTaskMetrics.from_pose(self.spec, position, axis, settled=settled)

    def snapshot(self) -> dict[str, object]:
        metrics = self.task_metrics()
        peg_pose = self.peg.data.root_pose_w.torch[0].detach().cpu().tolist()
        joint_position = self.joint_position_rad().tolist()
        ee_position = self.body_pose_world("gripper_link")[0].tolist()
        return {
            "task_id": self.spec.task_id,
            "mode": self.mode.value,
            "asset_path": self.asset_path,
            "joint_names": list(self.robot.joint_names),
            "body_names": list(self.robot.body_names),
            "joint_position_rad": [float(value) for value in joint_position],
            "end_effector_position_w_m": [float(value) for value in ee_position],
            "peg_pose_w": [float(value) for value in peg_pose],
            "metrics": {
                "insertion_depth_m": metrics.insertion_depth_m,
                "lateral_offset_m": metrics.lateral_offset_m,
                "tilt_deg": metrics.tilt_deg,
                "seated": metrics.seated,
                "jammed": metrics.jammed,
            },
            "scene_spec": self.spec.to_dict(),
        }

    def close(self) -> None:
        if getattr(self, "sim", None) is not None:
            self._sim_manager.__exit__(None, None, None)
            self.sim = None
