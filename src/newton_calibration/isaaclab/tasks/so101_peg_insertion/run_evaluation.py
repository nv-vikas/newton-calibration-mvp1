"""Pure-Python evidence inference and sustained task gates.

This module deliberately does not import Isaac Lab, Newton, NumPy, or the task
controller.  The runtime runner converts simulator state into
``TaskEvidenceSample`` values, while these classes make the pass/fail semantics
unit-testable.  All contact-like events are explicitly labelled as inferred;
they are not substitutes for raw contact-sensor evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import dist, isfinite, radians, sqrt, tan


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class EvaluationThresholds:
    minimum_lift_m: float = 0.018
    maximum_lift_m: float = 0.150
    relative_retention_tolerance_m: float = 0.006
    drop_relative_drift_m: float = 0.020
    retention_window_s: float = 0.30
    transport_lateral_tolerance_m: float = 0.008
    transport_linear_speed_limit_m_s: float = 0.030
    transport_angular_speed_limit_rad_s: float = 0.50
    transport_window_s: float = 0.30
    # The commissioned task moves the TCP 150 mm from peg to socket.  Its
    # transition tolerance is 2 mm, so an observed 148 mm horizontal move is
    # the smallest motion that demonstrates the complete commanded transfer.
    minimum_transport_displacement_m: float = 0.148
    seating_linear_speed_limit_m_s: float = 0.010
    seating_angular_speed_limit_rad_s: float = 0.25
    seating_window_s: float = 0.30

    def validate(self) -> None:
        values = asdict(self)
        if any(not isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("Evaluation thresholds must be finite and positive")
        if self.maximum_lift_m <= self.minimum_lift_m:
            raise ValueError("maximum_lift_m must exceed minimum_lift_m")
        if self.drop_relative_drift_m <= self.relative_retention_tolerance_m:
            raise ValueError(
                "drop_relative_drift_m must exceed relative_retention_tolerance_m"
            )


@dataclass(frozen=True)
class PrecloseDisturbanceThresholds:
    """Maximum workpiece motion allowed before the gripper is commanded shut."""

    maximum_xy_drift_m: float = 0.00025
    maximum_z_drift_m: float = 0.00025
    maximum_gripper_contact_force_n: float = 0.05

    def validate(self) -> None:
        if any(
            not isfinite(value) or value <= 0.0
            for value in (
                self.maximum_xy_drift_m,
                self.maximum_z_drift_m,
                self.maximum_gripper_contact_force_n,
            )
        ):
            raise ValueError("Pre-close disturbance thresholds must be finite and positive")


class PrecloseDisturbanceTracker:
    """Detect contact or instability before grasp closure is commanded.

    The reference is latched at the first approach sample, after the scene's
    settle window.  This avoids treating normal initial table settling as a
    gripper collision while still making approach/descent contact a hard gate.
    """

    _TRACKED_PHASES = {
        "approach_peg",
        "stabilize_above_peg",
        "descend_to_grasp",
    }

    def __init__(
        self,
        thresholds: PrecloseDisturbanceThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or PrecloseDisturbanceThresholds()
        self.thresholds.validate()
        self._reference_m: Vector3 | None = None
        self._maximum_xy_drift_m = 0.0
        self._maximum_z_drift_m = 0.0
        self._maximum_gripper_contact_force_n = 0.0
        self._violated = False

    def update(
        self,
        *,
        phase: str,
        peg_position_m: Vector3,
        fixed_finger_force_n: float = 0.0,
        moving_finger_force_n: float = 0.0,
    ) -> bool:
        if len(peg_position_m) != 3 or any(not isfinite(value) for value in peg_position_m):
            raise ValueError("Pre-close peg position must contain three finite values")
        if any(
            not isfinite(value) or value < 0.0
            for value in (fixed_finger_force_n, moving_finger_force_n)
        ):
            raise ValueError("Pre-close contact forces must be finite and non-negative")
        if phase not in self._TRACKED_PHASES:
            return not self._violated
        if self._reference_m is None:
            self._reference_m = peg_position_m
        dx = peg_position_m[0] - self._reference_m[0]
        dy = peg_position_m[1] - self._reference_m[1]
        dz = abs(peg_position_m[2] - self._reference_m[2])
        xy = sqrt(dx * dx + dy * dy)
        self._maximum_xy_drift_m = max(self._maximum_xy_drift_m, xy)
        self._maximum_z_drift_m = max(self._maximum_z_drift_m, dz)
        self._maximum_gripper_contact_force_n = max(
            self._maximum_gripper_contact_force_n,
            fixed_finger_force_n,
            moving_finger_force_n,
        )
        self._violated |= bool(
            xy > self.thresholds.maximum_xy_drift_m
            or dz > self.thresholds.maximum_z_drift_m
            or self._maximum_gripper_contact_force_n
            > self.thresholds.maximum_gripper_contact_force_n
        )
        return not self._violated

    def summary(self) -> dict[str, object]:
        return {
            "reference_peg_position_m": (
                list(self._reference_m) if self._reference_m is not None else None
            ),
            "maximum_xy_drift_m": self._maximum_xy_drift_m,
            "maximum_z_drift_m": self._maximum_z_drift_m,
            "maximum_xy_drift_allowed_m": self.thresholds.maximum_xy_drift_m,
            "maximum_z_drift_allowed_m": self.thresholds.maximum_z_drift_m,
            "maximum_gripper_contact_force_n": self._maximum_gripper_contact_force_n,
            "maximum_gripper_contact_force_allowed_n": (
                self.thresholds.maximum_gripper_contact_force_n
            ),
            "violated": self._violated,
            "passed": self._reference_m is not None and not self._violated,
        }


@dataclass(frozen=True)
class PrecontactTCPTrackingThresholds:
    """Maximum measured TCP lag allowed before commanded grasp contact."""

    maximum_lateral_error_m: float = 0.00075
    maximum_vertical_error_m: float = 0.00150

    def validate(self) -> None:
        if any(
            not isfinite(value) or value <= 0.0
            for value in (
                self.maximum_lateral_error_m,
                self.maximum_vertical_error_m,
            )
        ):
            raise ValueError("Pre-contact TCP tracking thresholds must be finite and positive")


class PrecontactTCPTrackingTracker:
    """Fail closed when the physical TCP departs from its slow setpoint.

    The gate is active only after the arm has reached the above-peg waypoint
    and while it descends with the gripper open.  It prevents actuator sag or
    a controller transient from being mistaken for a collision-geometry or
    contact-parameter problem.
    """

    _TRACKED_PHASES = {"stabilize_above_peg", "descend_to_grasp"}

    def __init__(
        self,
        thresholds: PrecontactTCPTrackingThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or PrecontactTCPTrackingThresholds()
        self.thresholds.validate()
        self._sample_count = 0
        self._maximum_lateral_error_m = 0.0
        self._maximum_vertical_error_m = 0.0
        self._first_violation: dict[str, object] | None = None
        self._violated = False

    def update(
        self,
        *,
        time_s: float,
        phase: str,
        measured_tcp_position_m: Vector3,
        commanded_tcp_setpoint_m: Vector3,
    ) -> bool:
        values = (
            time_s,
            *measured_tcp_position_m,
            *commanded_tcp_setpoint_m,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("Pre-contact TCP tracking samples must be finite")
        if phase not in self._TRACKED_PHASES:
            return not self._violated
        self._sample_count += 1
        dx = measured_tcp_position_m[0] - commanded_tcp_setpoint_m[0]
        dy = measured_tcp_position_m[1] - commanded_tcp_setpoint_m[1]
        dz = measured_tcp_position_m[2] - commanded_tcp_setpoint_m[2]
        lateral = sqrt(dx * dx + dy * dy)
        vertical = abs(dz)
        self._maximum_lateral_error_m = max(
            self._maximum_lateral_error_m,
            lateral,
        )
        self._maximum_vertical_error_m = max(
            self._maximum_vertical_error_m,
            vertical,
        )
        violated_now = bool(
            lateral > self.thresholds.maximum_lateral_error_m
            or vertical > self.thresholds.maximum_vertical_error_m
        )
        if violated_now and self._first_violation is None:
            self._first_violation = {
                "time_s": time_s,
                "phase": phase,
                "measured_tcp_position_m": list(measured_tcp_position_m),
                "commanded_tcp_setpoint_m": list(commanded_tcp_setpoint_m),
                "lateral_error_m": lateral,
                "vertical_error_m": vertical,
            }
        self._violated |= violated_now
        return not self._violated

    @property
    def violated(self) -> bool:
        return self._violated

    def summary(self) -> dict[str, object]:
        return {
            "thresholds": asdict(self.thresholds),
            "sample_count": self._sample_count,
            "maximum_lateral_error_m": self._maximum_lateral_error_m,
            "maximum_vertical_error_m": self._maximum_vertical_error_m,
            "first_violation": self._first_violation,
            "violated": self._violated,
            "passed": self._sample_count > 0 and not self._violated,
            "scope": "stabilize-above-peg and open-gripper descent",
        }


@dataclass(frozen=True)
class BilateralGraspThresholds:
    minimum_partner_force_n: float = 0.05
    bilateral_window_s: float = 0.10

    def validate(self) -> None:
        if any(
            not isfinite(value) or value <= 0.0
            for value in (self.minimum_partner_force_n, self.bilateral_window_s)
        ):
            raise ValueError("Bilateral-grasp thresholds must be finite and positive")


class BilateralGraspContactTracker:
    """Require sustained peg contact with both independently attributed fingers."""

    _TRACKED_PHASES = {"close", "verify_grasp"}

    def __init__(self, thresholds: BilateralGraspThresholds | None = None) -> None:
        self.thresholds = thresholds or BilateralGraspThresholds()
        self.thresholds.validate()
        self._last_time_s: float | None = None
        self._bilateral_start_s: float | None = None
        self._qualified = False
        self._maximum_fixed_force_n = 0.0
        self._maximum_moving_force_n = 0.0
        self._latest_bilateral = False
        self._latest_duration_s = 0.0

    def update(
        self,
        *,
        time_s: float,
        phase: str,
        fixed_finger_force_n: float,
        moving_finger_force_n: float,
    ) -> bool:
        values = (time_s, fixed_finger_force_n, moving_finger_force_n)
        if any(not isfinite(value) for value in values):
            raise ValueError("Bilateral-grasp samples must be finite")
        if fixed_finger_force_n < 0.0 or moving_finger_force_n < 0.0:
            raise ValueError("Bilateral-grasp forces must be non-negative")
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError("Bilateral-grasp timestamps must be monotonic")
        self._last_time_s = time_s
        self._maximum_fixed_force_n = max(
            self._maximum_fixed_force_n,
            fixed_finger_force_n,
        )
        self._maximum_moving_force_n = max(
            self._maximum_moving_force_n,
            moving_finger_force_n,
        )
        self._latest_bilateral = bool(
            phase in self._TRACKED_PHASES
            and fixed_finger_force_n >= self.thresholds.minimum_partner_force_n
            and moving_finger_force_n >= self.thresholds.minimum_partner_force_n
        )
        if self._latest_bilateral:
            if self._bilateral_start_s is None:
                self._bilateral_start_s = time_s
            self._latest_duration_s = time_s - self._bilateral_start_s
            self._qualified |= bool(
                self._latest_duration_s >= self.thresholds.bilateral_window_s
            )
        else:
            self._bilateral_start_s = None
            self._latest_duration_s = 0.0
        return self._qualified

    def summary(self) -> dict[str, object]:
        return {
            "source": "Newton/MJWarp per-partner normal-force matrix",
            "thresholds": asdict(self.thresholds),
            "latest_bilateral_contact": self._latest_bilateral,
            "latest_bilateral_duration_s": self._latest_duration_s,
            "maximum_fixed_finger_force_n": self._maximum_fixed_force_n,
            "maximum_moving_finger_force_n": self._maximum_moving_force_n,
            "qualified": self._qualified,
        }


@dataclass(frozen=True)
class PreInsertionReadinessThresholds:
    """Conservative geometric and contact limits before guarded insertion.

    The reference fixture has 1.5 mm nominal radial clearance and a 0.2 mm
    declared contact gap, leaving 1.3 mm effective radial clearance.  The
    0.65 mm lateral limit allocates half of that budget to translation.  A
    one-degree tilt sweeps about 0.42 mm over the 24 mm target depth, leaving
    roughly 0.23 mm of residual geometric margin at the worst declared pose.
    """

    effective_radial_clearance_m: float = 0.00130
    insertion_depth_basis_m: float = 0.024
    maximum_lateral_offset_m: float = 0.00065
    maximum_tilt_deg: float = 1.0
    minimum_peg_bottom_clearance_m: float = 0.00050
    maximum_pre_guard_socket_force_n: float = 0.05
    readiness_window_s: float = 0.10

    def validate(self) -> None:
        values = asdict(self)
        if any(not isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError(
                "Pre-insertion readiness thresholds must be finite and positive"
            )
        worst_case_radial_sweep_m = (
            self.maximum_lateral_offset_m
            + self.insertion_depth_basis_m * tan(radians(self.maximum_tilt_deg))
        )
        if worst_case_radial_sweep_m >= self.effective_radial_clearance_m:
            raise ValueError(
                "Declared lateral and tilt limits must leave positive effective "
                "radial clearance"
            )


class PreInsertionReadinessTracker:
    """Require a stable, collision-free actual peg pose before insertion.

    Geometry is evaluated only during ``align`` because the peg is intentionally
    travelling toward the socket during ``move_above_socket``.  Socket contact
    is watched in both phases and is sticky: once contact occurs before the
    guarded phase, later motion cannot make the run eligible for insertion.
    """

    _CONTACT_WATCH_PHASES = {"move_above_socket", "align"}
    _READINESS_PHASE = "align"

    def __init__(
        self,
        thresholds: PreInsertionReadinessThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or PreInsertionReadinessThresholds()
        self.thresholds.validate()
        self._last_time_s: float | None = None
        self._ready_start_s: float | None = None
        self._ready_duration_s = 0.0
        self._ready = False
        self._violated = False
        self._stop_reason: str | None = None
        self._maximum_pre_guard_socket_force_n = 0.0
        self._latest_lateral_offset_m: float | None = None
        self._latest_tilt_deg: float | None = None
        self._latest_peg_bottom_clearance_m: float | None = None
        self._latest_checks: dict[str, bool] | None = None

    @property
    def violated(self) -> bool:
        return self._violated

    def update(
        self,
        *,
        time_s: float,
        phase: str,
        lateral_offset_m: float,
        tilt_deg: float,
        peg_bottom_clearance_m: float,
        socket_normal_force_n: float,
    ) -> bool:
        values = (
            time_s,
            lateral_offset_m,
            tilt_deg,
            peg_bottom_clearance_m,
            socket_normal_force_n,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("Pre-insertion readiness samples must be finite")
        if lateral_offset_m < 0.0 or tilt_deg < 0.0 or socket_normal_force_n < 0.0:
            raise ValueError(
                "Lateral offset, tilt and socket force must be non-negative"
            )
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError("Pre-insertion readiness timestamps must be monotonic")
        self._last_time_s = time_s

        if phase in self._CONTACT_WATCH_PHASES:
            self._maximum_pre_guard_socket_force_n = max(
                self._maximum_pre_guard_socket_force_n,
                socket_normal_force_n,
            )
            if (
                socket_normal_force_n
                >= self.thresholds.maximum_pre_guard_socket_force_n
            ):
                self._violated = True
                self._stop_reason = "socket_contact_before_guarded_insert"

        if phase != self._READINESS_PHASE:
            self._ready_start_s = None
            self._ready_duration_s = 0.0
            return False

        self._latest_lateral_offset_m = lateral_offset_m
        self._latest_tilt_deg = tilt_deg
        self._latest_peg_bottom_clearance_m = peg_bottom_clearance_m
        self._latest_checks = {
            "lateral_offset": (
                lateral_offset_m <= self.thresholds.maximum_lateral_offset_m
            ),
            "tilt": tilt_deg <= self.thresholds.maximum_tilt_deg,
            "peg_bottom_clearance": (
                peg_bottom_clearance_m
                >= self.thresholds.minimum_peg_bottom_clearance_m
            ),
            "no_early_socket_contact": not self._violated,
        }
        candidate = all(self._latest_checks.values())
        if candidate:
            if self._ready_start_s is None:
                self._ready_start_s = time_s
            self._ready_duration_s = time_s - self._ready_start_s
            self._ready |= bool(
                self._ready_duration_s >= self.thresholds.readiness_window_s
            )
        else:
            self._ready_start_s = None
            self._ready_duration_s = 0.0
        return self._ready and not self._violated

    def summary(self) -> dict[str, object]:
        return {
            "thresholds": asdict(self.thresholds),
            "worst_case_radial_sweep_m": (
                self.thresholds.maximum_lateral_offset_m
                + self.thresholds.insertion_depth_basis_m
                * tan(radians(self.thresholds.maximum_tilt_deg))
            ),
            "latest_lateral_offset_m": self._latest_lateral_offset_m,
            "latest_tilt_deg": self._latest_tilt_deg,
            "latest_peg_bottom_clearance_m": self._latest_peg_bottom_clearance_m,
            "latest_checks": self._latest_checks,
            "maximum_pre_guard_socket_force_n": (
                self._maximum_pre_guard_socket_force_n
            ),
            "ready_duration_s": self._ready_duration_s,
            "ready": self._ready and not self._violated,
            "violated": self._violated,
            "stop_reason": self._stop_reason,
            "passed": self._ready and not self._violated,
        }


@dataclass(frozen=True)
class GuardedInsertionThresholds:
    """Safety limits for slow insertion when no wrist force/torque is wired."""

    maximum_effort_fraction: float = 0.90
    effort_window_s: float = 0.15
    maximum_socket_normal_force_n: float = 8.0
    socket_force_window_s: float = 0.05
    progress_window_s: float = 0.50
    minimum_progress_m: float = 0.00025
    target_depth_tolerance_m: float = 0.00050

    def validate(self) -> None:
        values = asdict(self)
        if any(not isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("Guarded-insertion thresholds must be finite and positive")
        if self.maximum_effort_fraction > 1.0:
            raise ValueError("maximum_effort_fraction must not exceed one")


class GuardedInsertionTracker:
    """Stop a slow insertion on sustained effort or absent depth progress.

    The effort signal is a joint-effort saturation proxy, not a measured TCP
    force.  The output records that limitation explicitly so this cannot be
    mistaken for force-controlled insertion.  A future wrist F/T or Newton
    contact-wrench plug-in can feed the same stop decision.
    """

    _PHASE = "guarded_insert"

    def __init__(
        self,
        thresholds: GuardedInsertionThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or GuardedInsertionThresholds()
        self.thresholds.validate()
        self._last_time_s: float | None = None
        self._entry_time_s: float | None = None
        self._entry_depth_m: float | None = None
        self._progress_reference_time_s: float | None = None
        self._progress_reference_depth_m: float | None = None
        self._effort_start_s: float | None = None
        self._socket_force_start_s: float | None = None
        self._maximum_effort_fraction = 0.0
        self._maximum_socket_normal_force_n = 0.0
        self._maximum_depth_m = 0.0
        self._stop_reason: str | None = None

    def update(
        self,
        *,
        time_s: float,
        phase: str,
        insertion_depth_m: float,
        target_insertion_depth_m: float,
        maximum_effort_fraction: float,
        socket_normal_force_n: float = 0.0,
    ) -> bool:
        values = (
            time_s,
            insertion_depth_m,
            target_insertion_depth_m,
            maximum_effort_fraction,
            socket_normal_force_n,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("Guarded-insertion samples must be finite")
        if insertion_depth_m < 0.0 or target_insertion_depth_m <= 0.0:
            raise ValueError("Insertion depths must be non-negative with a positive target")
        if maximum_effort_fraction < 0.0 or socket_normal_force_n < 0.0:
            raise ValueError("Effort fraction and socket force must be non-negative")
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError("Guarded-insertion timestamps must be monotonic")
        self._last_time_s = time_s
        if self._stop_reason is not None:
            return False
        if phase != self._PHASE:
            self._effort_start_s = None
            self._socket_force_start_s = None
            return True

        self._maximum_effort_fraction = max(
            self._maximum_effort_fraction,
            maximum_effort_fraction,
        )
        self._maximum_depth_m = max(self._maximum_depth_m, insertion_depth_m)
        self._maximum_socket_normal_force_n = max(
            self._maximum_socket_normal_force_n,
            socket_normal_force_n,
        )

        if self._entry_time_s is None:
            self._entry_time_s = time_s
            self._entry_depth_m = insertion_depth_m
            self._progress_reference_time_s = time_s
            self._progress_reference_depth_m = insertion_depth_m

        if maximum_effort_fraction >= self.thresholds.maximum_effort_fraction:
            if self._effort_start_s is None:
                self._effort_start_s = time_s
            if time_s - self._effort_start_s >= self.thresholds.effort_window_s:
                self._stop_reason = "sustained_joint_effort_limit"
                return False
        else:
            self._effort_start_s = None

        if socket_normal_force_n >= self.thresholds.maximum_socket_normal_force_n:
            if self._socket_force_start_s is None:
                self._socket_force_start_s = time_s
            if (
                time_s - self._socket_force_start_s
                >= self.thresholds.socket_force_window_s
            ):
                self._stop_reason = "sustained_socket_normal_force_limit"
                return False
        else:
            self._socket_force_start_s = None

        target_reached = (
            insertion_depth_m
            >= target_insertion_depth_m - self.thresholds.target_depth_tolerance_m
        )
        assert self._progress_reference_depth_m is not None
        assert self._progress_reference_time_s is not None
        if (
            insertion_depth_m - self._progress_reference_depth_m
            >= self.thresholds.minimum_progress_m
        ):
            self._progress_reference_depth_m = insertion_depth_m
            self._progress_reference_time_s = time_s
        elif (
            not target_reached
            and time_s - self._progress_reference_time_s
            >= self.thresholds.progress_window_s
        ):
            self._stop_reason = "insertion_depth_progress_stalled"
            return False
        return True

    def summary(self) -> dict[str, object]:
        return {
            "signal_basis": (
                "Newton/MJWarp socket normal force plus joint-effort fraction "
                "and insertion-depth progress; no wrist force/torque or contact moment"
            ),
            "thresholds": asdict(self.thresholds),
            "entry_time_s": self._entry_time_s,
            "entry_depth_m": self._entry_depth_m,
            "maximum_depth_m": self._maximum_depth_m,
            "maximum_effort_fraction": self._maximum_effort_fraction,
            "maximum_socket_normal_force_n": self._maximum_socket_normal_force_n,
            "stop_reason": self._stop_reason,
            "triggered": self._stop_reason is not None,
            "passed": self._entry_time_s is not None and self._stop_reason is None,
        }


@dataclass(frozen=True)
class TaskEvidenceSample:
    time_s: float
    phase: str
    tcp_position_m: Vector3
    peg_position_m: Vector3
    peg_linear_velocity_m_s: Vector3
    peg_angular_velocity_rad_s: Vector3
    grasp_candidate: bool
    peg_lift_m: float
    insertion_depth_m: float
    lateral_offset_m: float
    tilt_deg: float
    target_insertion_depth_m: float
    lateral_tolerance_m: float
    tilt_tolerance_deg: float
    inferred_jam_candidate: bool = False

    @property
    def peg_lifted(self) -> bool:
        return self.peg_lift_m > 0.0

    @property
    def peg_linear_speed_m_s(self) -> float:
        return _norm(self.peg_linear_velocity_m_s)

    @property
    def peg_angular_speed_rad_s(self) -> float:
        return _norm(self.peg_angular_velocity_rad_s)


@dataclass(frozen=True)
class InferredTaskEvents:
    """Per-sample event estimates derived from kinematics, never raw contact."""

    inference_basis: str
    grasp_candidate: bool
    retention_candidate: bool
    sustained_grasp_retention: bool
    slip_candidate: bool
    drop_candidate: bool
    socket_contact_candidate: bool
    first_socket_contact_candidate: bool
    jam_candidate: bool
    seated_pose_candidate: bool
    stable_transport: bool
    full_transfer_candidate: bool
    full_transfer_distance_reached: bool
    full_transfer_qualified: bool
    full_transfer_violated: bool
    stable_seating: bool
    relative_pose_drift_m: float | None
    retention_duration_s: float
    transport_stable_duration_s: float
    full_transfer_displacement_m: float | None
    seating_stable_duration_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TaskEvidenceTracker:
    """Track continuous evidence windows and reject one-frame successes."""

    INFERENCE_BASIS = (
        "raw_bilateral_gripper_contact_plus_kinematic_"
        "retention_slip_drop_socket_inference"
    )
    _GRASP_REFERENCE_PHASES = {"close", "verify_grasp"}
    _TRANSPORT_PHASES = {
        "align",
        "guarded_insert",
        "verify_depth",
        "complete",
    }
    _FULL_TRANSFER_PHASES = {"lift", "move_above_socket", "align"}
    _INSERTION_PHASES = {"guarded_insert", "verify_depth", "complete"}

    def __init__(self, thresholds: EvaluationThresholds | None = None) -> None:
        self.thresholds = thresholds or EvaluationThresholds()
        self.thresholds.validate()
        self._last_time_s: float | None = None
        self._grasp_relative_reference_m: Vector3 | None = None
        self._retention_start_s: float | None = None
        self._transport_start_s: float | None = None
        self._full_transfer_start_s: float | None = None
        self._full_transfer_duration_s = 0.0
        self._full_transfer_phases_seen: set[str] = set()
        self._full_transfer_move_start_tcp_m: Vector3 | None = None
        self._maximum_full_transfer_displacement_m = 0.0
        self._full_transfer_qualified = False
        self._full_transfer_violated = False
        self._full_transfer_violation_reason: str | None = None
        self._seating_start_s: float | None = None
        self._sustained_retention_seen = False
        self._drop_seen = False
        self._slip_seen = False
        self._socket_contact_seen = False
        self._maximum_relative_drift_m: float | None = None
        self._latest: InferredTaskEvents | None = None

    def update(self, sample: TaskEvidenceSample) -> InferredTaskEvents:
        self._validate_sample(sample)
        if self._last_time_s is not None and sample.time_s < self._last_time_s:
            raise ValueError("Task evidence timestamps must be monotonic")
        self._last_time_s = sample.time_s

        relative = _subtract(sample.peg_position_m, sample.tcp_position_m)
        if (
            self._grasp_relative_reference_m is None
            and sample.grasp_candidate
            and sample.phase in self._GRASP_REFERENCE_PHASES
        ):
            self._grasp_relative_reference_m = relative

        relative_drift_m = (
            dist(relative, self._grasp_relative_reference_m)
            if self._grasp_relative_reference_m is not None
            else None
        )
        if relative_drift_m is not None:
            self._maximum_relative_drift_m = max(
                self._maximum_relative_drift_m or 0.0,
                relative_drift_m,
            )

        lifted = (
            self.thresholds.minimum_lift_m
            <= sample.peg_lift_m
            <= self.thresholds.maximum_lift_m
        )
        supported_by_insertion = bool(
            sample.phase in self._INSERTION_PHASES and sample.insertion_depth_m > 0.0
        )
        retention_candidate = bool(
            (lifted or (self._sustained_retention_seen and supported_by_insertion))
            and relative_drift_m is not None
            and relative_drift_m <= self.thresholds.relative_retention_tolerance_m
        )
        self._retention_start_s, retention_duration_s = _update_window(
            self._retention_start_s,
            sample.time_s,
            retention_candidate,
        )
        sustained_retention = (
            retention_candidate
            and retention_duration_s >= self.thresholds.retention_window_s
        )
        self._sustained_retention_seen |= sustained_retention

        slip_candidate = bool(
            lifted
            and relative_drift_m is not None
            and relative_drift_m > self.thresholds.relative_retention_tolerance_m
        )
        drop_candidate = bool(
            self._sustained_retention_seen
            and (
                (not lifted and not supported_by_insertion)
                or (
                    relative_drift_m is not None
                    and relative_drift_m > self.thresholds.drop_relative_drift_m
                )
            )
        )
        self._slip_seen |= slip_candidate
        self._drop_seen |= drop_candidate

        full_transfer_candidate = False
        full_transfer_displacement_m: float | None = None
        if (
            sample.phase == "move_above_socket"
            and self._full_transfer_move_start_tcp_m is None
        ):
            self._full_transfer_move_start_tcp_m = sample.tcp_position_m
        if (
            self._full_transfer_move_start_tcp_m is not None
            and sample.phase in {"move_above_socket", "align"}
        ):
            dx = sample.tcp_position_m[0] - self._full_transfer_move_start_tcp_m[0]
            dy = sample.tcp_position_m[1] - self._full_transfer_move_start_tcp_m[1]
            full_transfer_displacement_m = sqrt(dx * dx + dy * dy)
            self._maximum_full_transfer_displacement_m = max(
                self._maximum_full_transfer_displacement_m,
                full_transfer_displacement_m,
            )
        full_transfer_distance_reached = bool(
            self._maximum_full_transfer_displacement_m
            >= self.thresholds.minimum_transport_displacement_m
        )
        if (
            sample.phase in self._FULL_TRANSFER_PHASES
            and self._grasp_relative_reference_m is not None
        ):
            self._full_transfer_phases_seen.add(sample.phase)
            relative_safe = bool(
                relative_drift_m is not None
                and relative_drift_m <= self.thresholds.relative_retention_tolerance_m
            )
            linear_speed_safe = bool(
                sample.peg_linear_speed_m_s
                <= self.thresholds.transport_linear_speed_limit_m_s
            )
            angular_speed_safe = bool(
                sample.peg_angular_speed_rad_s
                <= self.thresholds.transport_angular_speed_limit_rad_s
            )
            full_transfer_candidate = bool(
                relative_safe
                and linear_speed_safe
                and angular_speed_safe
                and not drop_candidate
            )
            if not full_transfer_candidate:
                self._full_transfer_violated = True
                if self._full_transfer_violation_reason is None:
                    if not relative_safe:
                        self._full_transfer_violation_reason = (
                            "relative_pose_drift_exceeded"
                        )
                    elif not linear_speed_safe:
                        self._full_transfer_violation_reason = (
                            "linear_speed_exceeded"
                        )
                    elif not angular_speed_safe:
                        self._full_transfer_violation_reason = (
                            "angular_speed_exceeded"
                        )
                    else:
                        self._full_transfer_violation_reason = "drop_detected"
            if full_transfer_candidate and not self._full_transfer_violated:
                if self._full_transfer_start_s is None:
                    self._full_transfer_start_s = sample.time_s
                self._full_transfer_duration_s = (
                    sample.time_s - self._full_transfer_start_s
                )
                required_phases_seen = {
                    "lift",
                    "move_above_socket",
                    "align",
                }.issubset(
                    self._full_transfer_phases_seen
                )
                self._full_transfer_qualified |= bool(
                    required_phases_seen
                    and full_transfer_distance_reached
                    and self._sustained_retention_seen
                    and self._full_transfer_duration_s
                    >= self.thresholds.transport_window_s
                )
            else:
                self._full_transfer_start_s = None
                self._full_transfer_duration_s = 0.0

        transport_candidate = bool(
            sample.phase in self._TRANSPORT_PHASES
            and sustained_retention
            and sample.lateral_offset_m <= self.thresholds.transport_lateral_tolerance_m
            and sample.peg_linear_speed_m_s
            <= self.thresholds.transport_linear_speed_limit_m_s
            and sample.peg_angular_speed_rad_s
            <= self.thresholds.transport_angular_speed_limit_rad_s
            and not self._drop_seen
        )
        self._transport_start_s, transport_duration_s = _update_window(
            self._transport_start_s,
            sample.time_s,
            transport_candidate,
        )
        stable_transport = bool(
            transport_candidate
            and transport_duration_s >= self.thresholds.transport_window_s
        )

        seated_pose_candidate = bool(
            sample.insertion_depth_m >= sample.target_insertion_depth_m
            and sample.lateral_offset_m <= sample.lateral_tolerance_m
            and sample.tilt_deg <= sample.tilt_tolerance_deg
        )
        seating_candidate = bool(
            sample.phase in self._INSERTION_PHASES
            and seated_pose_candidate
            and sample.peg_linear_speed_m_s
            <= self.thresholds.seating_linear_speed_limit_m_s
            and sample.peg_angular_speed_rad_s
            <= self.thresholds.seating_angular_speed_limit_rad_s
        )
        self._seating_start_s, seating_duration_s = _update_window(
            self._seating_start_s,
            sample.time_s,
            seating_candidate,
        )
        stable_seating = bool(
            seating_candidate
            and seating_duration_s >= self.thresholds.seating_window_s
        )

        socket_contact_candidate = bool(
            sample.phase in self._INSERTION_PHASES and sample.insertion_depth_m > 0.0
        )
        first_socket_contact_candidate = bool(
            socket_contact_candidate and not self._socket_contact_seen
        )
        self._socket_contact_seen |= socket_contact_candidate

        self._latest = InferredTaskEvents(
            inference_basis=self.INFERENCE_BASIS,
            grasp_candidate=sample.grasp_candidate,
            retention_candidate=retention_candidate,
            sustained_grasp_retention=sustained_retention,
            slip_candidate=slip_candidate,
            drop_candidate=drop_candidate,
            socket_contact_candidate=socket_contact_candidate,
            first_socket_contact_candidate=first_socket_contact_candidate,
            jam_candidate=sample.inferred_jam_candidate,
            seated_pose_candidate=seated_pose_candidate,
            stable_transport=stable_transport,
            full_transfer_candidate=full_transfer_candidate,
            full_transfer_distance_reached=full_transfer_distance_reached,
            full_transfer_qualified=(
                self._full_transfer_qualified and not self._full_transfer_violated
            ),
            full_transfer_violated=self._full_transfer_violated,
            stable_seating=stable_seating,
            relative_pose_drift_m=relative_drift_m,
            retention_duration_s=retention_duration_s,
            transport_stable_duration_s=transport_duration_s,
            full_transfer_displacement_m=full_transfer_displacement_m,
            seating_stable_duration_s=seating_duration_s,
        )
        return self._latest

    def summary(self) -> dict[str, object]:
        latest = self._latest
        return {
            "inference_basis": self.INFERENCE_BASIS,
            "thresholds": asdict(self.thresholds),
            "grasp_relative_reference_m": (
                list(self._grasp_relative_reference_m)
                if self._grasp_relative_reference_m is not None
                else None
            ),
            "maximum_relative_pose_drift_m": self._maximum_relative_drift_m,
            "sustained_grasp_retention_seen": self._sustained_retention_seen,
            "sustained_grasp_retention_final": bool(
                latest and latest.sustained_grasp_retention and not self._drop_seen
            ),
            "stable_transport_final": bool(
                latest and latest.stable_transport and not self._drop_seen
            ),
            "full_transfer_required_phases": [
                "lift",
                "move_above_socket",
                "align",
            ],
            "full_transfer_phases_seen": sorted(self._full_transfer_phases_seen),
            "full_transfer_move_start_tcp_m": (
                list(self._full_transfer_move_start_tcp_m)
                if self._full_transfer_move_start_tcp_m is not None
                else None
            ),
            "maximum_full_transfer_displacement_m": (
                self._maximum_full_transfer_displacement_m
            ),
            "full_transfer_distance_reached": bool(
                self._maximum_full_transfer_displacement_m
                >= self.thresholds.minimum_transport_displacement_m
            ),
            "full_transfer_duration_s": self._full_transfer_duration_s,
            "full_transfer_qualified": bool(
                self._full_transfer_qualified
                and not self._full_transfer_violated
                and not self._drop_seen
            ),
            "full_transfer_violated": self._full_transfer_violated,
            "full_transfer_violation_reason": self._full_transfer_violation_reason,
            "stable_seating_final": bool(latest and latest.stable_seating),
            "inferred_slip_candidate_seen": self._slip_seen,
            "inferred_drop_candidate_seen": self._drop_seen,
            "inferred_socket_contact_candidate_seen": self._socket_contact_seen,
            "final_events": latest.to_dict() if latest is not None else None,
        }

    @staticmethod
    def _validate_sample(sample: TaskEvidenceSample) -> None:
        scalars = (
            sample.time_s,
            sample.peg_lift_m,
            sample.insertion_depth_m,
            sample.lateral_offset_m,
            sample.tilt_deg,
            sample.target_insertion_depth_m,
            sample.lateral_tolerance_m,
            sample.tilt_tolerance_deg,
        )
        vectors = (
            sample.tcp_position_m,
            sample.peg_position_m,
            sample.peg_linear_velocity_m_s,
            sample.peg_angular_velocity_rad_s,
        )
        if any(len(vector) != 3 for vector in vectors):
            raise ValueError("Task-evidence vectors must contain exactly three values")
        if any(not isfinite(value) for value in scalars):
            raise ValueError("Task-evidence scalar values must be finite")
        if any(not isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Task-evidence vector values must be finite")


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _norm(vector: Vector3) -> float:
    return sqrt(sum(value * value for value in vector))


def _update_window(
    start_s: float | None,
    now_s: float,
    condition: bool,
) -> tuple[float | None, float]:
    if not condition:
        return None, 0.0
    if start_s is None:
        start_s = now_s
    return start_s, max(0.0, now_s - start_s)
