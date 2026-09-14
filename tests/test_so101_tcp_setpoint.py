from __future__ import annotations

import numpy as np
import pytest

from newton_calibration.isaaclab.tasks.so101_peg_insertion import (
    ControllerPhase,
    TCP_SETPOINT_SPEED_M_S,
    advance_tcp_setpoint,
    resolve_tcp_setpoint_speeds,
    tcp_setpoint_speed_for_phase,
)


def test_large_cartesian_move_advances_by_exact_speed_bound() -> None:
    result = advance_tcp_setpoint(
        current_setpoint_m=(0.0, 0.0, 0.0),
        final_target_m=(0.03, 0.04, 0.0),
        max_speed_m_s=0.02,
        dt_s=0.1,
    )

    assert result.position_m == pytest.approx((0.0012, 0.0016, 0.0))
    assert result.commanded_step_m == pytest.approx(0.002)
    assert result.remaining_distance_m == pytest.approx(0.048)
    assert result.limited
    assert not result.reached_target


def test_short_cartesian_move_lands_exactly_without_overshoot() -> None:
    result = advance_tcp_setpoint(
        current_setpoint_m=(0.1, -0.2, 0.3),
        final_target_m=(0.101, -0.2, 0.3),
        max_speed_m_s=0.02,
        dt_s=0.1,
    )

    assert result.position_m == pytest.approx(result.final_target_m)
    assert result.commanded_step_m == pytest.approx(0.001)
    assert result.remaining_distance_m == 0.0
    assert not result.limited
    assert result.reached_target


def test_phase_change_cannot_create_a_cartesian_target_jump() -> None:
    dt_s = 1.0 / 240.0
    previous = np.asarray((0.235, -0.080, 0.090))
    distant_grasp_target = (0.235, -0.080, 0.055)
    speed = tcp_setpoint_speed_for_phase(ControllerPhase.DESCEND_TO_GRASP)

    first = advance_tcp_setpoint(
        current_setpoint_m=previous,
        final_target_m=distant_grasp_target,
        max_speed_m_s=speed,
        dt_s=dt_s,
    )

    target_step = np.linalg.norm(np.asarray(first.position_m) - previous)
    assert target_step == pytest.approx(speed * dt_s)
    assert target_step == pytest.approx(0.000025)
    assert first.remaining_distance_m == pytest.approx(0.034975)


def test_phase_speed_limits_are_slowest_near_contact() -> None:
    assert tcp_setpoint_speed_for_phase(ControllerPhase.STABILIZE_ABOVE_PEG) == 0.006
    assert tcp_setpoint_speed_for_phase(ControllerPhase.DESCEND_TO_GRASP) == 0.006
    assert tcp_setpoint_speed_for_phase(ControllerPhase.LIFT) == 0.020
    assert tcp_setpoint_speed_for_phase(ControllerPhase.MOVE_ABOVE_SOCKET) == 0.030
    assert tcp_setpoint_speed_for_phase(ControllerPhase.ALIGN) == 0.010
    assert tcp_setpoint_speed_for_phase(ControllerPhase.ABORT) == 0.020
    assert tcp_setpoint_speed_for_phase(ControllerPhase.GUARDED_INSERT) == 0.004
    assert set(TCP_SETPOINT_SPEED_M_S) >= {
        ControllerPhase.DESCEND_TO_GRASP,
        ControllerPhase.STABILIZE_ABOVE_PEG,
        ControllerPhase.LIFT,
        ControllerPhase.MOVE_ABOVE_SOCKET,
        ControllerPhase.ALIGN,
        ControllerPhase.GUARDED_INSERT,
    }


def test_lift_speed_override_is_scoped_and_does_not_mutate_defaults() -> None:
    resolved = resolve_tcp_setpoint_speeds(lift_speed_m_s=0.005)

    assert resolved[ControllerPhase.LIFT] == pytest.approx(0.005)
    assert resolved[ControllerPhase.MOVE_ABOVE_SOCKET] == pytest.approx(0.030)
    assert TCP_SETPOINT_SPEED_M_S[ControllerPhase.LIFT] == pytest.approx(0.020)


@pytest.mark.parametrize("value", [0.0, -0.001, 0.021, np.nan])
def test_lift_speed_override_rejects_unsafe_values(value: float) -> None:
    with pytest.raises(ValueError, match="lift_speed_m_s"):
        resolve_tcp_setpoint_speeds(lift_speed_m_s=value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_speed_m_s": 0.0}, "max_speed_m_s"),
        ({"dt_s": 0.0}, "dt_s"),
        ({"current_setpoint_m": (0.0, 0.0)}, "exactly three"),
        ({"final_target_m": (0.0, np.nan, 0.0)}, "finite"),
    ],
)
def test_setpoint_rejects_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "current_setpoint_m": (0.0, 0.0, 0.0),
        "final_target_m": (1.0, 0.0, 0.0),
        "max_speed_m_s": 0.1,
        "dt_s": 0.01,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        advance_tcp_setpoint(**values)


def test_non_servo_phase_has_no_implicit_motion_limit() -> None:
    with pytest.raises(ValueError, match="does not use the TCP servo"):
        tcp_setpoint_speed_for_phase(ControllerPhase.SETTLE)
