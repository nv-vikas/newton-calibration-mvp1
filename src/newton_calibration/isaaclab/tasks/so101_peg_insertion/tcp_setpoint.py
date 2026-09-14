from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from .controller import ControllerPhase


# Cartesian speed is deliberately independent of the joint-rate limit in the
# Newton adapter.  The setpoint rate protects the workpiece from a large task-
# space target discontinuity; the joint-rate limit protects the mechanism.
TCP_SETPOINT_SPEED_M_S: Mapping[ControllerPhase, float] = MappingProxyType(
    {
        ControllerPhase.APPROACH_PEG: 0.040,
        ControllerPhase.STABILIZE_ABOVE_PEG: 0.006,
        ControllerPhase.DESCEND_TO_GRASP: 0.006,
        ControllerPhase.CLOSE: 0.006,
        ControllerPhase.VERIFY_GRASP: 0.006,
        ControllerPhase.LIFT: 0.020,
        ControllerPhase.MOVE_ABOVE_SOCKET: 0.030,
        ControllerPhase.ALIGN: 0.010,
        ControllerPhase.GUARDED_INSERT: 0.004,
        ControllerPhase.VERIFY_DEPTH: 0.004,
        ControllerPhase.COMPLETE: 0.004,
        ControllerPhase.ABORT: 0.020,
    }
)


@dataclass(frozen=True)
class ProgressiveTCPSetpoint:
    """One bounded Cartesian-setpoint update toward a final phase target."""

    position_m: tuple[float, float, float]
    final_target_m: tuple[float, float, float]
    commanded_step_m: float
    remaining_distance_m: float
    max_speed_m_s: float
    limited: bool
    reached_target: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "position_m": list(self.position_m),
            "final_target_m": list(self.final_target_m),
            "commanded_step_m": self.commanded_step_m,
            "remaining_distance_m": self.remaining_distance_m,
            "max_speed_m_s": self.max_speed_m_s,
            "limited": self.limited,
            "reached_target": self.reached_target,
        }


def tcp_setpoint_speed_for_phase(phase: ControllerPhase) -> float:
    """Return the commissioned Cartesian speed bound for a servo phase."""

    try:
        return TCP_SETPOINT_SPEED_M_S[phase]
    except KeyError as exc:
        raise ValueError(f"Phase {phase.value!r} does not use the TCP servo") from exc


def resolve_tcp_setpoint_speeds(
    *,
    lift_speed_m_s: float | None = None,
) -> Mapping[ControllerPhase, float]:
    """Resolve one run's audited Cartesian speed profile.

    The default map remains immutable.  Commissioning runs may slow the lift
    to distinguish a controller/contact transient from gravity-driven slip,
    but this interface deliberately refuses to make the contact motion faster
    than the qualified default.
    """

    speeds = dict(TCP_SETPOINT_SPEED_M_S)
    if lift_speed_m_s is not None:
        if (
            not isfinite(lift_speed_m_s)
            or lift_speed_m_s <= 0.0
            or lift_speed_m_s > TCP_SETPOINT_SPEED_M_S[ControllerPhase.LIFT]
        ):
            raise ValueError(
                "lift_speed_m_s must be finite, positive, and no greater than "
                f"{TCP_SETPOINT_SPEED_M_S[ControllerPhase.LIFT]:.3f} m/s"
            )
        speeds[ControllerPhase.LIFT] = float(lift_speed_m_s)
    return MappingProxyType(speeds)


def advance_tcp_setpoint(
    *,
    current_setpoint_m: Iterable[float],
    final_target_m: Iterable[float],
    max_speed_m_s: float,
    dt_s: float,
) -> ProgressiveTCPSetpoint:
    """Advance a TCP setpoint without overshoot or a task-space target jump.

    This pure function intentionally advances from the *previous setpoint*, not
    the measured TCP.  Re-basing on a lagging measurement each frame can hide a
    tracking problem and make the commanded speed difficult to audit.
    """

    if not isfinite(max_speed_m_s) or max_speed_m_s <= 0.0:
        raise ValueError("max_speed_m_s must be finite and positive")
    if not isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    current = np.asarray(tuple(current_setpoint_m), dtype=np.float64)
    target = np.asarray(tuple(final_target_m), dtype=np.float64)
    if current.shape != (3,) or target.shape != (3,):
        raise ValueError("TCP positions must contain exactly three values")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("TCP positions must be finite")

    delta = target - current
    distance = float(np.linalg.norm(delta))
    maximum_step = max_speed_m_s * dt_s
    limited = distance > maximum_step
    if limited:
        position = current + delta * (maximum_step / distance)
        commanded_step = maximum_step
        remaining = distance - maximum_step
        reached = False
    else:
        position = target.copy()
        commanded_step = distance
        remaining = 0.0
        reached = True

    return ProgressiveTCPSetpoint(
        position_m=tuple(float(value) for value in position),
        final_target_m=tuple(float(value) for value in target),
        commanded_step_m=float(commanded_step),
        remaining_distance_m=float(remaining),
        max_speed_m_s=float(max_speed_m_s),
        limited=bool(limited),
        reached_target=bool(reached),
    )
