from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite, sqrt


def local_z_axis_from_quaternion_xyzw(
    quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotate local +Z into world using Isaac Lab/Newton's xyzw convention.

    Keeping the conversion in the pure-Python contract layer makes the pose
    convention explicit and independently testable.  Newton quaternions can
    accumulate small normalization error, so normalize before deriving the
    axis used by the insertion-success gate.
    """

    if len(quaternion_xyzw) != 4 or any(
        not isfinite(value) for value in quaternion_xyzw
    ):
        raise ValueError("Quaternion must contain four finite xyzw values")
    norm = sqrt(sum(value * value for value in quaternion_xyzw))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm must be positive")
    x, y, z, w = (value / norm for value in quaternion_xyzw)
    return (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


class PegInsertionMode(str, Enum):
    """Progressive calibration boundaries exposed by one shared task."""

    ROBOT_ONLY = "mvp1_robot_only"
    GRASP_TRANSPORT = "mvp2_grasp_transport"
    INSERTION = "mvp3_insertion"


@dataclass(frozen=True)
class PegInsertionSceneSpec:
    """Serializable geometry and task contract for the reference scene.

    The socket is intentionally assembled from four convex walls.  It is a
    conformance fixture for bringing up Newton contact, not a substitute for
    measured customer CAD.
    """

    task_id: str = "Isaac-PegInsertion-SO101-Newton-v0"
    table_size_m: tuple[float, float, float] = (0.62, 0.46, 0.03)
    table_top_z_m: float = 0.0
    peg_radius_m: float = 0.009
    peg_height_m: float = 0.070
    peg_mass_kg: float = 0.048
    peg_start_xy_m: tuple[float, float] = (0.235, -0.080)
    socket_center_xy_m: tuple[float, float] = (0.235, 0.070)
    socket_aperture_m: float = 0.021
    socket_outer_m: float = 0.070
    socket_wall_height_m: float = 0.030
    contact_gap_m: float = 0.0002
    target_insertion_depth_m: float = 0.024
    success_lateral_tolerance_m: float = 0.0015
    success_tilt_tolerance_deg: float = 5.0
    grasp_tcp_height_m: float = 0.055
    transport_tcp_height_m: float = 0.090
    # Pre-insertion alignment must leave the peg bottom above the socket rim.
    # Actual peg/socket contact begins only in the guarded-insert phase.
    align_tcp_height_m: float = 0.086
    grasp_tcp_above_peg_center_m: float = 0.0195
    # Task-controller commissioning defaults.  These are deliberately separate
    # from a calibration overlay: a weak or bound-hitting fitted parameter must
    # not silently become the controller that qualifies the task environment.
    arm_control_stiffness_nm_per_rad: float = 45.0
    arm_control_damping_nm_s_per_rad: float = 0.30
    arm_control_effort_limit_nm: float = 3.0
    arm_armature_kg_m2: float = 0.001
    arm_joint_friction_nm: float = 0.0556
    gripper_control_stiffness_nm_per_rad: float = 2.0
    gripper_control_damping_nm_s_per_rad: float = 0.10
    gripper_control_effort_limit_nm: float = 0.35
    gripper_armature_kg_m2: float = 0.0001
    gripper_joint_friction_nm: float = 0.0243
    simulation_dt_s: float = 1.0 / 240.0
    controller_decimation: int = 4
    solver_substeps: int = 2
    solver_iterations: int = 100
    solver_tolerance: float = 1.0e-6

    @property
    def peg_diameter_m(self) -> float:
        return 2.0 * self.peg_radius_m

    @property
    def radial_clearance_m(self) -> float:
        return 0.5 * (self.socket_aperture_m - self.peg_diameter_m)

    @property
    def socket_top_z_m(self) -> float:
        return self.table_top_z_m + self.socket_wall_height_m

    @property
    def socket_neighborhood_radius_m(self) -> float:
        """Lateral radius in which vertical insertion depth is meaningful."""

        return 0.5 * self.socket_outer_m + self.peg_radius_m

    @property
    def peg_rest_center_z_m(self) -> float:
        return self.table_top_z_m + 0.5 * self.peg_height_m + 0.0005

    @property
    def insertion_tcp_height_m(self) -> float:
        desired_peg_center = (
            self.socket_top_z_m + 0.5 * self.peg_height_m - self.target_insertion_depth_m
        )
        return desired_peg_center + self.grasp_tcp_above_peg_center_m

    @property
    def aligned_peg_bottom_clearance_m(self) -> float:
        aligned_peg_bottom = (
            self.align_tcp_height_m
            - self.grasp_tcp_above_peg_center_m
            - 0.5 * self.peg_height_m
        )
        return aligned_peg_bottom - self.socket_top_z_m

    def validate(self) -> None:
        if self.socket_aperture_m <= self.peg_diameter_m + 2.0 * self.contact_gap_m:
            raise ValueError(
                "Socket aperture must exceed peg diameter plus both declared contact gaps; "
                f"got aperture={self.socket_aperture_m}, peg={self.peg_diameter_m}, "
                f"gap={self.contact_gap_m}."
            )
        if self.socket_outer_m <= self.socket_aperture_m:
            raise ValueError("Socket outer width must exceed its aperture.")
        if self.target_insertion_depth_m <= 0.0 or self.target_insertion_depth_m > self.socket_wall_height_m:
            raise ValueError("Target insertion depth must lie inside the socket wall height.")
        if not (
            self.grasp_tcp_height_m < self.align_tcp_height_m <= self.transport_tcp_height_m
        ):
            raise ValueError("Controller TCP heights must order grasp < align <= transport.")
        if self.aligned_peg_bottom_clearance_m <= self.contact_gap_m:
            raise ValueError(
                "Alignment must keep the peg above the socket rim; peg/socket contact "
                "belongs exclusively to guarded insertion."
            )
        transported_peg_bottom = (
            self.transport_tcp_height_m
            - self.grasp_tcp_above_peg_center_m
            - 0.5 * self.peg_height_m
        )
        if transported_peg_bottom <= self.socket_top_z_m + self.contact_gap_m:
            raise ValueError("Transport TCP height does not clear the socket rim.")
        control_values = (
            self.arm_control_stiffness_nm_per_rad,
            self.arm_control_effort_limit_nm,
            self.gripper_control_stiffness_nm_per_rad,
            self.gripper_control_effort_limit_nm,
        )
        nonnegative_values = (
            self.arm_control_damping_nm_s_per_rad,
            self.arm_armature_kg_m2,
            self.arm_joint_friction_nm,
            self.gripper_control_damping_nm_s_per_rad,
            self.gripper_armature_kg_m2,
            self.gripper_joint_friction_nm,
        )
        if any(value <= 0.0 for value in control_values) or any(
            value < 0.0 for value in nonnegative_values
        ):
            raise ValueError("Robot control gains, limits, armature and friction must be physically valid.")
        if self.simulation_dt_s <= 0.0 or self.solver_substeps < 1 or self.solver_iterations < 1:
            raise ValueError("Simulation and solver settings must be positive.")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            peg_diameter_m=self.peg_diameter_m,
            radial_clearance_m=self.radial_clearance_m,
            socket_top_z_m=self.socket_top_z_m,
            socket_neighborhood_radius_m=self.socket_neighborhood_radius_m,
            insertion_tcp_height_m=self.insertion_tcp_height_m,
            aligned_peg_bottom_clearance_m=self.aligned_peg_bottom_clearance_m,
        )
        return payload


@dataclass(frozen=True)
class PegInsertionTaskMetrics:
    insertion_depth_m: float
    lateral_offset_m: float
    tilt_deg: float
    seated: bool
    jammed: bool

    @classmethod
    def from_pose(
        cls,
        spec: PegInsertionSceneSpec,
        peg_position_m: tuple[float, float, float],
        peg_axis_world: tuple[float, float, float],
        *,
        settled: bool,
    ) -> "PegInsertionTaskMetrics":
        dx = peg_position_m[0] - spec.socket_center_xy_m[0]
        dy = peg_position_m[1] - spec.socket_center_xy_m[1]
        lateral_offset = sqrt(dx * dx + dy * dy)
        peg_bottom_z = peg_position_m[2] - 0.5 * spec.peg_height_m
        within_socket_neighborhood = lateral_offset <= spec.socket_neighborhood_radius_m
        insertion_depth = (
            min(
                spec.socket_wall_height_m,
                max(0.0, spec.socket_top_z_m - peg_bottom_z),
            )
            if within_socket_neighborhood
            else 0.0
        )
        axis_z = min(1.0, max(-1.0, abs(peg_axis_world[2])))
        # Stable near zero without importing numpy in the contract layer.
        tilt_rad = __import__("math").acos(axis_z)
        tilt_deg = tilt_rad * 180.0 / __import__("math").pi
        seated = (
            insertion_depth >= spec.target_insertion_depth_m
            and lateral_offset <= spec.success_lateral_tolerance_m
            and tilt_deg <= spec.success_tilt_tolerance_deg
        )
        jammed = (
            settled
            and within_socket_neighborhood
            and not seated
            and insertion_depth < 0.5 * spec.target_insertion_depth_m
        )
        return cls(
            insertion_depth_m=insertion_depth,
            lateral_offset_m=lateral_offset,
            tilt_deg=tilt_deg,
            seated=seated,
            jammed=jammed,
        )


def required_evidence_channels(mode: PegInsertionMode) -> tuple[str, ...]:
    """Canonical real/sim observation contract for each calibration stage."""

    robot = (
        "timestamp",
        "joint.command_position",
        "joint.measured_position",
        "joint.measured_velocity",
        "gripper.command",
        "gripper.measured_position",
    )
    if mode is PegInsertionMode.ROBOT_ONLY:
        return robot
    grasp = robot + (
        "peg.pose_world",
        "peg.velocity_world",
        "peg.pose_in_gripper",
        "event.grasp",
        "event.slip",
        "event.drop",
    )
    if mode is PegInsertionMode.GRASP_TRANSPORT:
        return grasp
    return grasp + (
        "hole.pose_world",
        "peg.pose_in_hole",
        "insertion.depth",
        "event.first_contact",
        "event.jam",
        "event.seated",
        "wrist.force_torque_optional",
    )
