from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported only inside Isaac Lab
    from .scene import SO101PegInsertionScene


# Kinematic measurements from the source USD mesh at the peg-centre station
# (gripper-local z approximately -104.5 mm).  This mapping is only used to
# report aperture and infer a contact candidate; Newton contact remains the
# authority for whether the peg is actually retained.
SO101_APERTURE_AT_ZERO_RAD_M = 0.015800002818840475
SO101_APERTURE_SLOPE_M_PER_RAD = 0.07727098939348247
# Measured from the two inner fingertip surfaces at q_gripper=0.30 rad.
SO101_OPEN_GRIPPER_CENTER_X_M = 0.01159065
# Only the moving SO-101 jaw closes.  At the 18 mm peg-contact configuration
# (q≈0.0285 rad), the opening midpoint is about +1.1 mm.  The task TCP is
# placed 0.45 mm toward the moving jaw.  The original 0.2 mm margin still
# produced 0.176 N of fixed-fingertip contact during the live v3 open descent;
# adding 0.25 mm cleared that measured tracking/settling error without changing
# the physics or weakening the pre-close gate.  Live bilateral-contact, drift,
# and retention gates remain authoritative; this nominal offset alone does not
# prove a grasp.
SO101_CONTACT_GRIPPER_CENTER_X_M = 0.00110
SO101_TASK_TCP_CENTER_X_M = 0.00155
SO101_TASK_TCP_OFFSET_GRIPPER_M = (
    SO101_TASK_TCP_CENTER_X_M,
    0.0,
    -0.085,
)


def estimate_so101_fingertip_aperture_m(gripper_position_rad: float) -> float:
    """Estimate the SO-101 inner fingertip gap near the task's grasp plane."""

    value = float(gripper_position_rad)
    if not isfinite(value):
        raise ValueError("gripper_position_rad must be finite")
    return max(
        0.0,
        SO101_APERTURE_AT_ZERO_RAD_M + SO101_APERTURE_SLOPE_M_PER_RAD * value,
    )


def interpolate_phase_servo_nominal(
    phase_entry_joint_command_rad: Iterable[float],
    phase_endpoint_joint_position_rad: Iterable[float],
    phase_entry_tcp_setpoint_m: Iterable[float],
    current_tcp_setpoint_m: Iterable[float],
    phase_endpoint_tcp_m: Iterable[float],
    *,
    arm_joint_count: int = 5,
) -> np.ndarray:
    """Interpolate a phase-continuous arm nominal along Cartesian progress.

    A Cartesian phase endpoint can be many joint degrees away from the prior
    phase.  Feeding that distant IK solution directly to the DLS null-space
    term (and its nominal-offset bound) creates a joint-space step that can
    outrun the deliberately slow Cartesian setpoint.  Start at the phase-entry
    arm command and blend toward the endpoint IK by projected Cartesian
    setpoint progress.  Retain the endpoint's gripper target independently.
    The measured TCP and fixed wrist-pitch feedback close the remaining error.
    """

    entry = np.asarray(tuple(phase_entry_joint_command_rad), dtype=np.float64)
    endpoint = np.asarray(tuple(phase_endpoint_joint_position_rad), dtype=np.float64)
    start_tcp = np.asarray(tuple(phase_entry_tcp_setpoint_m), dtype=np.float64)
    current_tcp = np.asarray(tuple(current_tcp_setpoint_m), dtype=np.float64)
    endpoint_tcp = np.asarray(tuple(phase_endpoint_tcp_m), dtype=np.float64)
    if entry.ndim != 1 or endpoint.shape != entry.shape:
        raise ValueError("Entry and endpoint joint commands must be equal 1-D vectors")
    if any(value.shape != (3,) for value in (start_tcp, current_tcp, endpoint_tcp)):
        raise ValueError("TCP setpoints must contain exactly three values")
    if arm_joint_count < 1 or entry.size <= arm_joint_count:
        raise ValueError("A gripper joint is required after the arm joints")
    if not all(
        np.isfinite(value).all()
        for value in (entry, endpoint, start_tcp, current_tcp, endpoint_tcp)
    ):
        raise ValueError("Servo nominal inputs must be finite")

    path = endpoint_tcp - start_tcp
    path_norm_squared = float(path @ path)
    progress = (
        0.0
        if path_norm_squared <= 1.0e-16
        else float(np.clip(((current_tcp - start_tcp) @ path) / path_norm_squared, 0.0, 1.0))
    )
    nominal = endpoint.copy()
    nominal[:arm_joint_count] = (
        entry[:arm_joint_count]
        + progress * (endpoint[:arm_joint_count] - entry[:arm_joint_count])
    )
    return nominal


@dataclass(frozen=True)
class IKSolution:
    name: str
    target_tcp_position_m: tuple[float, float, float]
    joint_position_rad: tuple[float, ...]
    achieved_tcp_position_m: tuple[float, float, float]
    position_error_m: float
    wrist_pitch_error_rad: float
    iterations: int
    converged: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "target_tcp_position_m": list(self.target_tcp_position_m),
            "joint_position_rad": list(self.joint_position_rad),
            "achieved_tcp_position_m": list(self.achieved_tcp_position_m),
            "position_error_m": self.position_error_m,
            "wrist_pitch_error_rad": self.wrist_pitch_error_rad,
            "iterations": self.iterations,
            "converged": self.converged,
        }


@dataclass(frozen=True)
class TCPServoCommand:
    """One bounded outer-loop TCP correction and its diagnostics."""

    joint_position_rad: tuple[float, ...]
    position_error_m: tuple[float, float, float]
    clipped_position_error_m: tuple[float, float, float]
    arm_velocity_rad_s: tuple[float, ...]
    wrist_pitch_error_rad: float | None
    clipped_wrist_pitch_error_rad: float | None
    wrist_pitch_control_available: bool
    jacobian_condition: float
    error_limited: bool
    rate_limited: bool
    offset_limited: bool
    joint_limit_limited: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_position_rad": list(self.joint_position_rad),
            "position_error_m": list(self.position_error_m),
            "clipped_position_error_m": list(self.clipped_position_error_m),
            "arm_velocity_rad_s": list(self.arm_velocity_rad_s),
            "wrist_pitch_error_rad": self.wrist_pitch_error_rad,
            "clipped_wrist_pitch_error_rad": self.clipped_wrist_pitch_error_rad,
            "wrist_pitch_control_available": self.wrist_pitch_control_available,
            "jacobian_condition": (
                self.jacobian_condition if isfinite(self.jacobian_condition) else None
            ),
            "error_limited": self.error_limited,
            "rate_limited": self.rate_limited,
            "offset_limited": self.offset_limited,
            "joint_limit_limited": self.joint_limit_limited,
        }


def skew_symmetric(vector: Iterable[float]) -> np.ndarray:
    """Return the matrix ``S(v)`` for which ``S(v) @ x == v cross x``."""

    x, y, z = np.asarray(tuple(vector), dtype=np.float64)
    return np.asarray(
        (
            (0.0, -z, y),
            (z, 0.0, -x),
            (-y, x, 0.0),
        ),
        dtype=np.float64,
    )


def shift_linear_jacobian_to_point(
    link_spatial_jacobian_w: np.ndarray,
    point_offset_w_m: Iterable[float],
) -> np.ndarray:
    """Shift a world-frame link-origin Jacobian to an offset point.

    Newton publishes spatial Jacobians with linear rows first and angular rows
    second.  For a point ``p = origin + r``, ``v_p = v_origin + omega x r``;
    therefore ``Jv_p = Jv_origin - skew(r) @ Jw``.
    """

    jacobian = np.asarray(link_spatial_jacobian_w, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != 6:
        raise ValueError(f"Expected a 6xN spatial Jacobian, got {jacobian.shape}")
    offset = np.asarray(tuple(point_offset_w_m), dtype=np.float64)
    if offset.shape != (3,):
        raise ValueError("point_offset_w_m must contain exactly three values")
    if not np.isfinite(jacobian).all() or not np.isfinite(offset).all():
        raise ValueError("Jacobian and point offset must be finite")
    return jacobian[:3] - skew_symmetric(offset) @ jacobian[3:]


def bounded_dls_tcp_step(
    *,
    tcp_jacobian_m_per_rad: np.ndarray,
    current_tcp_position_m: Iterable[float],
    target_tcp_position_m: Iterable[float],
    measured_joint_position_rad: Iterable[float],
    previous_joint_command_rad: Iterable[float],
    nominal_joint_position_rad: Iterable[float],
    joint_lower_rad: Iterable[float],
    joint_upper_rad: Iterable[float],
    dt_s: float,
    arm_joint_count: int = 5,
    damping_m_per_rad: float = 0.012,
    proportional_gain_per_s: float = 4.0,
    nullspace_gain_per_s: float = 0.35,
    max_position_error_m: float = 0.015,
    max_arm_rate_rad_s: float = 0.45,
    max_gripper_rate_rad_s: float = 1.25,
    max_offset_from_nominal_rad: float = 0.18,
    target_wrist_pitch_sum_rad: float | None = None,
    wrist_pitch_gain_per_s: float = 4.0,
    max_wrist_pitch_error_rad: float = 0.12,
) -> TCPServoCommand:
    """Take one safe resolved-rate DLS step toward a measured TCP target.

    This function is intentionally independent of Isaac Lab so the safety math
    can be unit tested.  The joint command integrates from the previous command
    (which supplies enough position error to hold against gravity), while a
    null-space term keeps the solution near the commissioned IK posture.  Four
    independent bounds apply: Cartesian error, joint velocity, distance from
    the nominal posture, and the robot's safe joint limits.
    """

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if arm_joint_count < 1:
        raise ValueError("arm_joint_count must be positive")
    if damping_m_per_rad <= 0.0:
        raise ValueError("damping_m_per_rad must be positive")
    if proportional_gain_per_s <= 0.0:
        raise ValueError("proportional_gain_per_s must be positive")
    if max_position_error_m <= 0.0 or max_arm_rate_rad_s <= 0.0:
        raise ValueError("servo error and arm-rate bounds must be positive")
    if max_gripper_rate_rad_s <= 0.0 or max_offset_from_nominal_rad <= 0.0:
        raise ValueError("gripper-rate and nominal-offset bounds must be positive")
    if wrist_pitch_gain_per_s <= 0.0 or max_wrist_pitch_error_rad <= 0.0:
        raise ValueError("wrist-pitch gain and error bound must be positive")

    jacobian = np.asarray(tcp_jacobian_m_per_rad, dtype=np.float64)
    current_tcp = np.asarray(tuple(current_tcp_position_m), dtype=np.float64)
    target_tcp = np.asarray(tuple(target_tcp_position_m), dtype=np.float64)
    measured = np.asarray(tuple(measured_joint_position_rad), dtype=np.float64)
    previous = np.asarray(tuple(previous_joint_command_rad), dtype=np.float64)
    nominal = np.asarray(tuple(nominal_joint_position_rad), dtype=np.float64)
    lower = np.asarray(tuple(joint_lower_rad), dtype=np.float64)
    upper = np.asarray(tuple(joint_upper_rad), dtype=np.float64)
    joint_count = measured.size
    if jacobian.shape != (3, arm_joint_count):
        raise ValueError(
            f"Expected a 3x{arm_joint_count} TCP Jacobian, got {jacobian.shape}"
        )
    if any(values.shape != (joint_count,) for values in (previous, nominal, lower, upper)):
        raise ValueError("All joint vectors must have the same length")
    if joint_count <= arm_joint_count:
        raise ValueError("A gripper joint is required after the arm joints")
    arrays = (jacobian, current_tcp, target_tcp, measured, previous, nominal, lower, upper)
    if current_tcp.shape != (3,) or target_tcp.shape != (3,):
        raise ValueError("TCP positions must contain exactly three values")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Servo inputs must be finite")
    if not np.all(lower < upper):
        raise ValueError("Joint lower limits must be below upper limits")

    error = target_tcp - current_tcp
    error_norm = float(np.linalg.norm(error))
    error_limited = error_norm > max_position_error_m
    clipped_error = error.copy()
    if error_limited:
        clipped_error *= max_position_error_m / error_norm

    normal = jacobian @ jacobian.T + (damping_m_per_rad**2) * np.eye(3)
    pseudo_inverse = jacobian.T @ np.linalg.solve(normal, np.eye(3))
    desired_tcp_velocity = proportional_gain_per_s * clipped_error
    arm_velocity = pseudo_inverse @ desired_tcp_velocity
    nullspace = np.eye(arm_joint_count) - pseudo_inverse @ jacobian
    arm_velocity += nullspace @ (
        nullspace_gain_per_s * (nominal[:arm_joint_count] - measured[:arm_joint_count])
    )

    # Position-only TCP feedback can hide a tilted gripper: the TCP reaches the
    # target while a fingertip sweeps laterally into the peg.  For SO-101 the
    # top-down tool pitch is q1+q2+q3.  Correct that scalar in the null space of
    # the position task so the jaws remain vertical during approach and close.
    wrist_pitch_error: float | None = None
    clipped_wrist_pitch_error: float | None = None
    wrist_pitch_control_available = False
    if target_wrist_pitch_sum_rad is not None:
        target_pitch = float(target_wrist_pitch_sum_rad)
        if not isfinite(target_pitch) or arm_joint_count < 4:
            raise ValueError("A finite wrist-pitch target requires at least four arm joints")
        current_pitch = float(measured[1] + measured[2] + measured[3])
        wrist_pitch_error = (target_pitch - current_pitch + pi) % (2.0 * pi) - pi
        clipped_wrist_pitch_error = float(
            np.clip(
                wrist_pitch_error,
                -max_wrist_pitch_error_rad,
                max_wrist_pitch_error_rad,
            )
        )
        pitch_axis = np.zeros(arm_joint_count, dtype=np.float64)
        pitch_axis[1:4] = 1.0
        pitch_direction = nullspace @ pitch_axis
        pitch_authority = float(pitch_axis @ pitch_direction)
        wrist_pitch_control_available = abs(pitch_authority) > 1.0e-8
        if wrist_pitch_control_available:
            arm_velocity += pitch_direction * (
                wrist_pitch_gain_per_s * clipped_wrist_pitch_error / pitch_authority
            )

    unclipped_velocity = arm_velocity.copy()
    arm_velocity = np.clip(arm_velocity, -max_arm_rate_rad_s, max_arm_rate_rad_s)
    rate_limited = not np.allclose(arm_velocity, unclipped_velocity, rtol=0.0, atol=1.0e-12)

    target = previous.copy()
    rate_safe_arm_target = previous[:arm_joint_count] + arm_velocity * dt_s
    lower_offset = nominal[:arm_joint_count] - max_offset_from_nominal_rad
    upper_offset = nominal[:arm_joint_count] + max_offset_from_nominal_rad
    offset_bounded = rate_safe_arm_target.copy()
    arm_step_limit = max_arm_rate_rad_s * dt_s
    for joint_index in range(arm_joint_count):
        prior = previous[joint_index]
        if prior < lower_offset[joint_index]:
            # A waypoint change can put the previous command outside the new
            # nominal band.  Re-enter at the declared rate instead of jumping
            # directly to the band edge after the rate limit was applied.
            offset_bounded[joint_index] = min(
                prior + arm_step_limit,
                lower_offset[joint_index],
            )
        elif prior > upper_offset[joint_index]:
            offset_bounded[joint_index] = max(
                prior - arm_step_limit,
                upper_offset[joint_index],
            )
        else:
            # Because the prior command is inside this interval, clipping the
            # already rate-safe candidate cannot increase its step size.
            offset_bounded[joint_index] = np.clip(
                rate_safe_arm_target[joint_index],
                lower_offset[joint_index],
                upper_offset[joint_index],
            )
    offset_limited = not np.allclose(
        offset_bounded, rate_safe_arm_target, rtol=0.0, atol=1.0e-12
    )
    target[:arm_joint_count] = offset_bounded

    # The Cartesian loop does not control the gripper.  It follows the nominal
    # command under its own velocity limit.
    gripper_delta_limit = max_gripper_rate_rad_s * dt_s
    gripper_delta = float(np.clip(
        nominal[arm_joint_count] - previous[arm_joint_count],
        -gripper_delta_limit,
        gripper_delta_limit,
    ))
    target[arm_joint_count] = previous[arm_joint_count] + gripper_delta
    rate_limited = rate_limited or not np.isclose(
        gripper_delta,
        nominal[arm_joint_count] - previous[arm_joint_count],
        rtol=0.0,
        atol=1.0e-12,
    )

    joint_bounded = np.clip(target, lower, upper)
    joint_limit_limited = not np.allclose(joint_bounded, target, rtol=0.0, atol=1.0e-12)
    final_step_limits = np.asarray(
        [max_arm_rate_rad_s] * arm_joint_count
        + [max_gripper_rate_rad_s] * (joint_count - arm_joint_count),
        dtype=np.float64,
    ) * dt_s
    if np.any(np.abs(joint_bounded - previous) > final_step_limits + 1.0e-12):
        raise RuntimeError("Final joint command violated the declared rate limit")
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    jacobian_condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 1.0e-12
        else float("inf")
    )
    return TCPServoCommand(
        joint_position_rad=tuple(float(value) for value in joint_bounded),
        position_error_m=tuple(float(value) for value in error),
        clipped_position_error_m=tuple(float(value) for value in clipped_error),
        arm_velocity_rad_s=tuple(float(value) for value in arm_velocity),
        wrist_pitch_error_rad=wrist_pitch_error,
        clipped_wrist_pitch_error_rad=clipped_wrist_pitch_error,
        wrist_pitch_control_available=wrist_pitch_control_available,
        jacobian_condition=jacobian_condition,
        error_limited=error_limited,
        rate_limited=rate_limited,
        offset_limited=offset_limited,
        joint_limit_limited=joint_limit_limited,
    )


class SO101JointCommandAdapter:
    """Map task-space TCP targets to valid SO-101 joint commands.

    The Newton backend exposes a backend-neutral link Jacobian, while the usual
    PhysX ``root_physx_view`` is intentionally unavailable.  This commissioning
    implementation obtains forward kinematics from Newton itself and builds a
    finite-difference TCP Jacobian; that also serves as a conformance check for
    the native Newton Jacobian before online guarded insertion uses it.
    The resulting joint targets are then executed through the normal dynamic
    actuator path; the robot is never teleported during task execution.

    Only the first four joints are used for position plus horizontal wrist-pitch
    control.  ``wrist_roll`` is retained from the seed because the SO-101 has
    five arm degrees of freedom and cannot satisfy an arbitrary 6-D pose.
    """

    arm_joint_count = 5
    planar_joint_indices = (0, 1, 2, 3)

    def __init__(
        self,
        scene: "SO101PegInsertionScene",
        *,
        # Local +X is the commissioned 18 mm contact centre plus one contact
        # gap toward the moving jaw; see the constants above.
        tcp_offset_gripper_m: tuple[float, float, float] = SO101_TASK_TCP_OFFSET_GRIPPER_M,
        # Positive gripper angle opens this USD.  0.30 rad gives ~39 mm at the
        # peg plane.  The theoretical 18 mm contact is ~0.0285 rad; commanding
        # -0.04 rad supplies squeeze and lets the effort limit stop the jaw.
        open_gripper_rad: float = 0.30,
        closed_gripper_rad: float = -0.04,
        joint_limit_margin_rad: float = 0.035,
    ) -> None:
        self.scene = scene
        self.tcp_offset_gripper_m = np.asarray(tcp_offset_gripper_m, dtype=np.float64)
        self.open_gripper_rad = float(open_gripper_rad)
        self.closed_gripper_rad = float(closed_gripper_rad)
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        limits = self.scene.joint_position_limits_rad()
        self.lower = limits[:, 0] + self.joint_limit_margin_rad
        self.upper = limits[:, 1] - self.joint_limit_margin_rad
        if not np.all(self.lower < self.upper):
            raise ValueError("SO-101 joint-limit margin leaves an empty command range")
        for name, value in (
            ("open_gripper_rad", self.open_gripper_rad),
            ("closed_gripper_rad", self.closed_gripper_rad),
        ):
            if value < self.lower[5] or value > self.upper[5]:
                raise ValueError(f"{name}={value} lies outside the safe gripper range")
        if self.open_gripper_rad <= self.closed_gripper_rad:
            raise ValueError("open_gripper_rad must exceed closed_gripper_rad for this USD")

    @staticmethod
    def estimated_gripper_aperture_m(gripper_position_rad: float) -> float:
        return estimate_so101_fingertip_aperture_m(gripper_position_rad)

    @staticmethod
    def _rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Rotate ``vector`` by an Isaac Lab xyzw quaternion."""

        xyz = quaternion[:3]
        w = float(quaternion[3])
        twice_cross = 2.0 * np.cross(xyz, vector)
        return vector + w * twice_cross + np.cross(xyz, twice_cross)

    def current_joint_position(self) -> np.ndarray:
        return self.scene.joint_position_rad()

    def current_tcp_position(self) -> np.ndarray:
        position, quaternion = self.scene.body_pose_world("gripper_link")
        return position + self._rotate_xyzw(quaternion, self.tcp_offset_gripper_m)

    def native_tcp_position_jacobian(self) -> np.ndarray:
        """Return Newton's world-frame 3x5 Jacobian at the configured TCP."""

        link_jacobian = self.scene.body_link_jacobian_world("gripper_link")[:, : self.arm_joint_count]
        _, quaternion = self.scene.body_pose_world("gripper_link")
        offset_w = self._rotate_xyzw(quaternion, self.tcp_offset_gripper_m)
        return shift_linear_jacobian_to_point(link_jacobian, offset_w)

    def correct_tcp_position_command(
        self,
        *,
        target_tcp_position_m: Iterable[float],
        nominal_joint_position_rad: Iterable[float],
        previous_joint_command_rad: Iterable[float],
        dt_s: float,
        max_arm_rate_rad_s: float = 0.45,
        damping_m_per_rad: float = 0.012,
        proportional_gain_per_s: float = 4.0,
        max_position_error_m: float = 0.015,
        max_offset_from_nominal_rad: float = 0.18,
        max_gripper_rate_rad_s: float = 1.25,
        target_wrist_pitch_sum_rad: float | None = None,
    ) -> TCPServoCommand:
        """Correct one dynamic command using measured state and Newton's Jacobian.

        The result is a joint-position target for the ordinary actuator path;
        this method never writes or teleports simulation state.
        """

        nominal = np.asarray(tuple(nominal_joint_position_rad), dtype=np.float64)
        if target_wrist_pitch_sum_rad is None:
            target_wrist_pitch_sum_rad = float(
                nominal[1] + nominal[2] + nominal[3]
            )
        return bounded_dls_tcp_step(
            tcp_jacobian_m_per_rad=self.native_tcp_position_jacobian(),
            current_tcp_position_m=self.current_tcp_position(),
            target_tcp_position_m=target_tcp_position_m,
            measured_joint_position_rad=self.current_joint_position(),
            previous_joint_command_rad=previous_joint_command_rad,
            nominal_joint_position_rad=nominal,
            joint_lower_rad=self.lower,
            joint_upper_rad=self.upper,
            dt_s=dt_s,
            arm_joint_count=self.arm_joint_count,
            damping_m_per_rad=damping_m_per_rad,
            proportional_gain_per_s=proportional_gain_per_s,
            max_position_error_m=max_position_error_m,
            max_arm_rate_rad_s=max_arm_rate_rad_s,
            max_gripper_rate_rad_s=max_gripper_rate_rad_s,
            max_offset_from_nominal_rad=max_offset_from_nominal_rad,
            target_wrist_pitch_sum_rad=target_wrist_pitch_sum_rad,
        )

    def _evaluate_kinematic_state(self, joint_position: np.ndarray) -> np.ndarray:
        self.scene.write_kinematic_joint_state(joint_position)
        return self.current_tcp_position()

    @staticmethod
    def _norm(values: np.ndarray) -> float:
        return sqrt(float(values @ values))

    def solve_tcp_position(
        self,
        name: str,
        target_position_m: Iterable[float],
        *,
        seed_joint_position_rad: Iterable[float] | None = None,
        wrist_pitch_sum_rad: float = 0.0,
        position_tolerance_m: float = 0.0015,
        wrist_pitch_tolerance_rad: float = 0.04,
        max_iterations: int = 90,
        finite_difference_rad: float = 1.0e-3,
        damping: float = 0.012,
        orientation_weight_m_per_rad: float = 0.07,
        max_joint_step_rad: float = 0.14,
    ) -> IKSolution:
        """Solve one position-only waypoint using Newton forward kinematics."""

        target = np.asarray(tuple(target_position_m), dtype=np.float64)
        if target.shape != (3,):
            raise ValueError("target_position_m must contain exactly three values")
        original = self.current_joint_position()
        q = np.asarray(
            tuple(seed_joint_position_rad) if seed_joint_position_rad is not None else tuple(original),
            dtype=np.float64,
        )
        if q.shape != (6,):
            raise ValueError("seed_joint_position_rad must contain all six SO-101 joints")
        q = np.clip(q, self.lower, self.upper)
        converged = False
        iteration = 0
        try:
            for iteration in range(1, max_iterations + 1):
                position = self._evaluate_kinematic_state(q)
                position_error = target - position
                pitch_error = wrist_pitch_sum_rad - float(q[1] + q[2] + q[3])
                if (
                    self._norm(position_error) <= position_tolerance_m
                    and abs(pitch_error) <= wrist_pitch_tolerance_rad
                ):
                    converged = True
                    break

                jacobian = np.zeros((4, 4), dtype=np.float64)
                for column, joint_index in enumerate(self.planar_joint_indices):
                    perturbed = q.copy()
                    perturbed[joint_index] = min(self.upper[joint_index], q[joint_index] + finite_difference_rad)
                    actual_delta = perturbed[joint_index] - q[joint_index]
                    if actual_delta <= 0.0:
                        perturbed[joint_index] = max(
                            self.lower[joint_index], q[joint_index] - finite_difference_rad
                        )
                        actual_delta = perturbed[joint_index] - q[joint_index]
                    displaced = self._evaluate_kinematic_state(perturbed)
                    jacobian[:3, column] = (displaced - position) / actual_delta
                    if joint_index in (1, 2, 3):
                        jacobian[3, column] = orientation_weight_m_per_rad

                residual = np.concatenate(
                    (position_error, np.asarray([orientation_weight_m_per_rad * pitch_error]))
                )
                normal = jacobian @ jacobian.T + (damping * damping) * np.eye(4)
                delta = jacobian.T @ np.linalg.solve(normal, residual)
                delta_norm = self._norm(delta)
                if delta_norm > max_joint_step_rad:
                    delta *= max_joint_step_rad / delta_norm
                q[list(self.planar_joint_indices)] += delta
                q = np.clip(q, self.lower, self.upper)

            achieved = self._evaluate_kinematic_state(q)
            final_position_error = self._norm(target - achieved)
            final_pitch_error = wrist_pitch_sum_rad - float(q[1] + q[2] + q[3])
            converged = converged or (
                final_position_error <= position_tolerance_m
                and abs(final_pitch_error) <= wrist_pitch_tolerance_rad
            )
            return IKSolution(
                name=name,
                target_tcp_position_m=tuple(float(value) for value in target),
                joint_position_rad=tuple(float(value) for value in q),
                achieved_tcp_position_m=tuple(float(value) for value in achieved),
                position_error_m=final_position_error,
                wrist_pitch_error_rad=abs(final_pitch_error),
                iterations=iteration,
                converged=converged,
            )
        finally:
            self.scene.write_kinematic_joint_state(original)

    def command_from_solution(self, solution: IKSolution, *, gripper_closed: bool) -> np.ndarray:
        if not solution.converged:
            raise ValueError(f"Cannot execute unconverged waypoint {solution.name!r}")
        target = np.asarray(solution.joint_position_rad, dtype=np.float64)
        target[5] = self.closed_gripper_rad if gripper_closed else self.open_gripper_rad
        return np.clip(target, self.lower, self.upper)
