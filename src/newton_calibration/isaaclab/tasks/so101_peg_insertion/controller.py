from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist, isfinite

from .contract import PegInsertionMode, PegInsertionSceneSpec


class ControllerPhase(str, Enum):
    SETTLE = "settle"
    APPROACH_PEG = "approach_peg"
    STABILIZE_ABOVE_PEG = "stabilize_above_peg"
    DESCEND_TO_GRASP = "descend_to_grasp"
    CLOSE = "close"
    VERIFY_GRASP = "verify_grasp"
    LIFT = "lift"
    MOVE_ABOVE_SOCKET = "move_above_socket"
    ALIGN = "align"
    GUARDED_INSERT = "guarded_insert"
    VERIFY_DEPTH = "verify_depth"
    COMPLETE = "complete"
    ABORT = "abort"


@dataclass(frozen=True)
class ControllerObservation:
    ee_position_m: tuple[float, float, float]
    peg_position_m: tuple[float, float, float]
    gripper_gap_m: float
    insertion_depth_m: float = 0.0
    peg_grasped: bool = False
    peg_dropped: bool = False
    jammed: bool = False
    force_limit_exceeded: bool = False
    guard_stop_requested: bool = False
    pre_insertion_ready: bool = False
    pre_insertion_gate_failed: bool = False


@dataclass(frozen=True)
class ControllerCommand:
    phase: ControllerPhase
    target_position_m: tuple[float, float, float]
    gripper_closed: bool
    guarded_contact: bool = False
    done: bool = False
    aborted: bool = False


class BilateralContactGripperLatch:
    """Stop closing at the first measured bilateral peg contact.

    The latch records the already rate-limited command rather than jumping to
    the measured joint position (which would remove the contact preload) or
    continuing toward a hard-coded fully-closed target (which can over-compress
    the peg and amplify unequal proxy geometry).
    """

    def __init__(self, minimum_partner_force_n: float = 0.05) -> None:
        if (
            not isfinite(minimum_partner_force_n)
            or minimum_partner_force_n <= 0.0
        ):
            raise ValueError("minimum_partner_force_n must be finite and positive")
        self.minimum_partner_force_n = float(minimum_partner_force_n)
        self.latched_command_rad: float | None = None
        self.latched_time_s: float | None = None

    def update(
        self,
        *,
        time_s: float,
        phase: ControllerPhase,
        current_command_rad: float,
        fixed_finger_force_n: float,
        moving_finger_force_n: float,
    ) -> float | None:
        values = (
            time_s,
            current_command_rad,
            fixed_finger_force_n,
            moving_finger_force_n,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("gripper latch inputs must be finite")
        if (
            self.latched_command_rad is None
            and phase is ControllerPhase.CLOSE
            and fixed_finger_force_n >= self.minimum_partner_force_n
            and moving_finger_force_n >= self.minimum_partner_force_n
        ):
            self.latched_command_rad = float(current_command_rad)
            self.latched_time_s = float(time_s)
        return self.latched_command_rad

    def target(self, default_closed_rad: float) -> float:
        return (
            float(default_closed_rad)
            if self.latched_command_rad is None
            else self.latched_command_rad
        )

    def summary(self) -> dict[str, float | bool | None]:
        return {
            "minimum_partner_force_n": self.minimum_partner_force_n,
            "latched": self.latched_command_rad is not None,
            "latched_command_rad": self.latched_command_rad,
            "latched_time_s": self.latched_time_s,
        }


class PegInsertionController:
    """Deterministic task-space state machine for environment validation.

    This controller defines the collection phases and guard conditions.  A
    robot-specific adapter converts its task-space target into joint commands.
    It is intentionally separate from any policy used to prove transfer.
    """

    def __init__(
        self,
        spec: PegInsertionSceneSpec,
        mode: PegInsertionMode = PegInsertionMode.INSERTION,
        *,
        settle_s: float = 0.25,
        approach_stabilize_s: float = 0.25,
        close_s: float = 0.45,
        verify_grasp_s: float = 0.20,
        max_phase_s: float = 6.0,
        max_lift_phase_s: float | None = None,
        abort_retract_m: float = 0.020,
    ) -> None:
        self.spec = spec
        self.mode = mode
        self.settle_s = settle_s
        if approach_stabilize_s <= 0.0:
            raise ValueError("approach_stabilize_s must be positive")
        self.approach_stabilize_s = approach_stabilize_s
        self.close_s = close_s
        if verify_grasp_s <= 0.0:
            raise ValueError("verify_grasp_s must be positive")
        self.verify_grasp_s = verify_grasp_s
        self.max_phase_s = max_phase_s
        self.max_lift_phase_s = (
            max_phase_s if max_lift_phase_s is None else max_lift_phase_s
        )
        if self.max_phase_s <= 0.0 or self.max_lift_phase_s <= 0.0:
            raise ValueError("phase timeouts must be positive")
        if abort_retract_m <= 0.0:
            raise ValueError("abort_retract_m must be positive")
        self.abort_retract_m = abort_retract_m
        self.reset()

    def reset(self) -> None:
        self.phase = ControllerPhase.SETTLE
        self.phase_time_s = 0.0
        self._initial_tcp_position_m: tuple[float, float, float] | None = None
        self._initial_peg_position_m: tuple[float, float, float] | None = None
        self._abort_target_position_m: tuple[float, float, float] | None = None

    def request_abort(self, observation: ControllerObservation) -> ControllerCommand:
        """Latch one upward retreat target and return an open-gripper command."""

        if self.phase is not ControllerPhase.ABORT:
            self.phase = ControllerPhase.ABORT
            self.phase_time_s = 0.0
        if self._abort_target_position_m is None:
            x, y, z = observation.ee_position_m
            self._abort_target_position_m = (x, y, z + self.abort_retract_m)
        return self._command(observation)

    def step(self, observation: ControllerObservation, dt_s: float) -> ControllerCommand:
        if self._initial_tcp_position_m is None:
            self._initial_tcp_position_m = observation.ee_position_m
        if self._initial_peg_position_m is None:
            self._initial_peg_position_m = observation.peg_position_m
        if (
            observation.force_limit_exceeded
            or observation.guard_stop_requested
            or observation.peg_dropped
            or observation.jammed
            or (
                self.mode is PegInsertionMode.INSERTION
                and observation.pre_insertion_gate_failed
            )
        ):
            self.request_abort(observation)
        command = self._command(observation)
        self.phase_time_s += dt_s
        self._advance(observation)
        return command

    def _advance(self, observation: ControllerObservation) -> None:
        target = self._target_for_phase(observation)
        reached = dist(observation.ee_position_m, target) <= self._position_tolerance_for_phase()
        next_phase: ControllerPhase | None = None
        if self.phase is ControllerPhase.SETTLE and self.phase_time_s >= self.settle_s:
            next_phase = ControllerPhase.APPROACH_PEG
        elif self.phase is ControllerPhase.APPROACH_PEG and reached:
            next_phase = (
                ControllerPhase.COMPLETE
                if self.mode is PegInsertionMode.ROBOT_ONLY
                else ControllerPhase.STABILIZE_ABOVE_PEG
            )
        elif self.phase is ControllerPhase.STABILIZE_ABOVE_PEG:
            # Require a continuous dynamically settled window over the peg.
            # Reaching a loose approach tolerance once is not sufficient: the
            # prior live run transitioned while 1.8 mm away and the resulting
            # lateral transient touched the fixed fingertip before closure.
            if not reached:
                self.phase_time_s = 0.0
            elif self.phase_time_s >= self.approach_stabilize_s:
                next_phase = ControllerPhase.DESCEND_TO_GRASP
        elif self.phase is ControllerPhase.DESCEND_TO_GRASP and reached:
            next_phase = ControllerPhase.CLOSE
        elif self.phase is ControllerPhase.CLOSE and self.phase_time_s >= self.close_s:
            next_phase = ControllerPhase.VERIFY_GRASP
        elif (
            self.phase is ControllerPhase.VERIFY_GRASP
            and self.phase_time_s >= self.verify_grasp_s
        ):
            next_phase = ControllerPhase.LIFT if observation.peg_grasped else ControllerPhase.ABORT
        elif self.phase is ControllerPhase.LIFT and reached:
            next_phase = ControllerPhase.MOVE_ABOVE_SOCKET
        elif self.phase is ControllerPhase.MOVE_ABOVE_SOCKET and reached:
            next_phase = ControllerPhase.ALIGN
        elif self.phase is ControllerPhase.ALIGN and reached:
            if self.mode is PegInsertionMode.GRASP_TRANSPORT:
                next_phase = ControllerPhase.COMPLETE
            elif observation.pre_insertion_gate_failed:
                next_phase = ControllerPhase.ABORT
            elif observation.pre_insertion_ready:
                next_phase = ControllerPhase.GUARDED_INSERT
        elif self.phase is ControllerPhase.GUARDED_INSERT and observation.insertion_depth_m >= self.spec.target_insertion_depth_m:
            next_phase = ControllerPhase.VERIFY_DEPTH
        elif self.phase is ControllerPhase.VERIFY_DEPTH:
            next_phase = ControllerPhase.COMPLETE
        elif (
            self.phase not in {ControllerPhase.COMPLETE, ControllerPhase.ABORT}
            and self.phase_time_s
            >= (
                self.max_lift_phase_s
                if self.phase is ControllerPhase.LIFT
                else self.max_phase_s
            )
        ):
            next_phase = ControllerPhase.ABORT
        if next_phase is not None:
            self.phase = next_phase
            self.phase_time_s = 0.0

    def _position_tolerance_for_phase(self) -> float:
        if self.phase in {
            ControllerPhase.APPROACH_PEG,
            ControllerPhase.STABILIZE_ABOVE_PEG,
        }:
            return 0.002 if self.mode is PegInsertionMode.ROBOT_ONLY else 0.00075
        if self.phase is ControllerPhase.ALIGN:
            return 0.00075
        if self.phase in {ControllerPhase.GUARDED_INSERT, ControllerPhase.VERIFY_DEPTH}:
            return 0.0005
        if self.phase is ControllerPhase.DESCEND_TO_GRASP:
            return 0.0015
        if self.phase in {ControllerPhase.LIFT, ControllerPhase.MOVE_ABOVE_SOCKET}:
            return 0.002
        return 0.008

    def _command(self, observation: ControllerObservation) -> ControllerCommand:
        target = self._target_for_phase(observation)
        closed = self.phase not in {
            ControllerPhase.SETTLE,
            ControllerPhase.APPROACH_PEG,
            ControllerPhase.STABILIZE_ABOVE_PEG,
            ControllerPhase.DESCEND_TO_GRASP,
            ControllerPhase.ABORT,
        }
        return ControllerCommand(
            phase=self.phase,
            target_position_m=target,
            gripper_closed=closed,
            guarded_contact=self.phase is ControllerPhase.GUARDED_INSERT,
            done=self.phase is ControllerPhase.COMPLETE,
            aborted=self.phase is ControllerPhase.ABORT,
        )

    def _target_for_phase(self, observation: ControllerObservation) -> tuple[float, float, float]:
        initial_tcp = self._initial_tcp_position_m or observation.ee_position_m
        px, py = self.spec.peg_start_xy_m
        sx, sy = self.spec.socket_center_xy_m
        if self.phase is ControllerPhase.SETTLE:
            return initial_tcp
        if self.phase in {
            ControllerPhase.APPROACH_PEG,
            ControllerPhase.STABILIZE_ABOVE_PEG,
        }:
            return (px, py, self.spec.transport_tcp_height_m)
        if self.phase in {ControllerPhase.DESCEND_TO_GRASP, ControllerPhase.CLOSE, ControllerPhase.VERIFY_GRASP}:
            return (px, py, self.spec.grasp_tcp_height_m)
        if self.phase is ControllerPhase.LIFT:
            return (px, py, self.spec.transport_tcp_height_m)
        if self.phase in {ControllerPhase.MOVE_ABOVE_SOCKET, ControllerPhase.ALIGN}:
            height = (
                self.spec.transport_tcp_height_m
                if self.phase is ControllerPhase.MOVE_ABOVE_SOCKET
                else self.spec.align_tcp_height_m
            )
            return (sx, sy, height)
        if self.phase is ControllerPhase.COMPLETE:
            if self.mode is PegInsertionMode.ROBOT_ONLY:
                return (px, py, self.spec.transport_tcp_height_m)
            if self.mode is PegInsertionMode.GRASP_TRANSPORT:
                return (sx, sy, self.spec.align_tcp_height_m)
            return (sx, sy, self.spec.insertion_tcp_height_m)
        if self.phase in {ControllerPhase.GUARDED_INSERT, ControllerPhase.VERIFY_DEPTH}:
            return (sx, sy, self.spec.insertion_tcp_height_m)
        if self.phase is ControllerPhase.ABORT:
            if self._abort_target_position_m is None:
                x, y, z = observation.ee_position_m
                self._abort_target_position_m = (x, y, z + self.abort_retract_m)
            return self._abort_target_position_m
        return observation.ee_position_m
