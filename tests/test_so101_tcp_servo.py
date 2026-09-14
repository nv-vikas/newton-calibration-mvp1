from __future__ import annotations

import numpy as np
import pytest

from newton_calibration.isaaclab.tasks.so101_peg_insertion import (
    bounded_dls_tcp_step,
    estimate_so101_fingertip_aperture_m,
    interpolate_phase_servo_nominal,
    shift_linear_jacobian_to_point,
)


def test_usd_measured_gripper_aperture_matches_commissioning_points() -> None:
    assert estimate_so101_fingertip_aperture_m(0.0) == pytest.approx(0.0158000028)
    assert estimate_so101_fingertip_aperture_m(0.3) == pytest.approx(0.0389812996)
    contact_q = (0.018 - 0.015800002818840475) / 0.07727098939348247
    assert contact_q == pytest.approx(0.0284711921)


def test_phase_nominal_interpolates_arm_and_applies_endpoint_gripper() -> None:
    entry = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.30))
    distant_endpoint = np.asarray((-0.4, 0.8, -0.6, 1.2, -0.2, -0.04))

    nominal = interpolate_phase_servo_nominal(
        entry,
        distant_endpoint,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.25),
        (0.0, 0.0, 1.0),
    )

    assert nominal[:5] == pytest.approx(entry[:5] + 0.25 * (distant_endpoint[:5] - entry[:5]))
    assert nominal[5] == pytest.approx(distant_endpoint[5])
    assert distant_endpoint[:5] == pytest.approx((-0.4, 0.8, -0.6, 1.2, -0.2))


def test_phase_nominal_has_no_jump_at_entry_or_for_zero_length_phase() -> None:
    entry = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.30))
    endpoint = np.asarray((-0.4, 0.8, -0.6, 1.2, -0.2, -0.04))
    at_entry = interpolate_phase_servo_nominal(
        entry, endpoint, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 4.0)
    )
    zero_length = interpolate_phase_servo_nominal(
        entry, endpoint, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)
    )
    assert at_entry[:5] == pytest.approx(entry[:5])
    assert zero_length[:5] == pytest.approx(entry[:5])
    assert at_entry[5] == pytest.approx(endpoint[5])
    assert zero_length[5] == pytest.approx(endpoint[5])


def test_phase_nominal_rejects_non_finite_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="equal 1-D"):
        interpolate_phase_servo_nominal(
            (0.0,) * 6, (0.0,) * 5, (0.0,) * 3, (0.0,) * 3, (0.0,) * 3
        )
    with pytest.raises(ValueError, match="finite"):
        interpolate_phase_servo_nominal(
            (0.0,) * 5 + (np.nan,),
            (0.0,) * 6,
            (0.0,) * 3,
            (0.0,) * 3,
            (0.0,) * 3,
        )


def test_link_jacobian_is_shifted_to_tcp_with_angular_velocity() -> None:
    link = np.zeros((6, 5), dtype=np.float64)
    link[5, 0] = 1.0  # joint 0 produces +Z angular velocity

    tcp = shift_linear_jacobian_to_point(link, (1.0, 0.0, 0.0))

    # omega=(0,0,1), r=(1,0,0): omega cross r = (0,1,0).
    assert tcp[:, 0] == pytest.approx((0.0, 1.0, 0.0))


def test_dls_servo_reduces_reachable_error_without_moving_gripper() -> None:
    jacobian = np.zeros((3, 5), dtype=np.float64)
    jacobian[:3, :3] = np.eye(3)
    measured = np.zeros(6)
    previous = np.zeros(6)
    nominal = np.zeros(6)
    nominal[5] = 0.9
    result = bounded_dls_tcp_step(
        tcp_jacobian_m_per_rad=jacobian,
        current_tcp_position_m=(0.0, 0.0, 0.0),
        target_tcp_position_m=(0.001, -0.002, 0.003),
        measured_joint_position_rad=measured,
        previous_joint_command_rad=previous,
        nominal_joint_position_rad=nominal,
        joint_lower_rad=(-2.0,) * 6,
        joint_upper_rad=(2.0,) * 6,
        dt_s=0.01,
        proportional_gain_per_s=2.0,
        max_arm_rate_rad_s=1.0,
        max_gripper_rate_rad_s=1.0,
    )

    command = np.asarray(result.joint_position_rad)
    predicted_motion = jacobian @ command[:5]
    error = np.asarray(result.position_error_m)
    assert float(predicted_motion @ error) > 0.0
    assert command[5] == pytest.approx(0.01)  # separately rate-limited toward open
    assert not result.error_limited


def test_dls_servo_corrects_wrist_pitch_in_position_null_space() -> None:
    jacobian = np.zeros((3, 5), dtype=np.float64)
    jacobian[:, (0, 1, 4)] = np.eye(3)
    result = bounded_dls_tcp_step(
        tcp_jacobian_m_per_rad=jacobian,
        current_tcp_position_m=(0.0, 0.0, 0.0),
        target_tcp_position_m=(0.0, 0.0, 0.0),
        measured_joint_position_rad=(0.0,) * 6,
        previous_joint_command_rad=(0.0,) * 6,
        nominal_joint_position_rad=(0.0,) * 6,
        joint_lower_rad=(-2.0,) * 6,
        joint_upper_rad=(2.0,) * 6,
        dt_s=0.01,
        max_arm_rate_rad_s=2.0,
        target_wrist_pitch_sum_rad=0.10,
    )

    command = np.asarray(result.joint_position_rad)
    assert result.wrist_pitch_control_available
    assert result.wrist_pitch_error_rad == pytest.approx(0.10)
    assert command[2] + command[3] > 0.0
    assert jacobian @ command[:5] == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-6)


def test_dls_servo_enforces_error_rate_offset_and_joint_limits() -> None:
    jacobian = np.zeros((3, 5), dtype=np.float64)
    jacobian[:3, :3] = np.eye(3)
    result = bounded_dls_tcp_step(
        tcp_jacobian_m_per_rad=jacobian,
        current_tcp_position_m=(0.0, 0.0, 0.0),
        target_tcp_position_m=(1.0, 1.0, 1.0),
        measured_joint_position_rad=(0.0,) * 6,
        previous_joint_command_rad=(0.0,) * 6,
        nominal_joint_position_rad=(0.0,) * 6,
        joint_lower_rad=(-0.001,) * 6,
        joint_upper_rad=(0.001,) * 6,
        dt_s=1.0,
        proportional_gain_per_s=100.0,
        max_position_error_m=0.01,
        max_arm_rate_rad_s=0.1,
        max_gripper_rate_rad_s=0.1,
        max_offset_from_nominal_rad=0.05,
    )

    command = np.asarray(result.joint_position_rad)
    assert result.error_limited
    assert result.rate_limited
    assert result.offset_limited
    assert result.joint_limit_limited
    assert np.all(command <= 0.001)
    assert np.all(command >= -0.001)


def test_nominal_offset_reentry_cannot_bypass_final_rate_limit() -> None:
    jacobian = np.zeros((3, 5), dtype=np.float64)
    jacobian[:3, :3] = np.eye(3)
    previous = np.zeros(6)
    nominal = np.zeros(6)
    nominal[2] = 0.40  # previous lies outside nominal +/- 0.18 rad
    dt_s = 1.0 / 240.0
    max_rate_rad_s = 0.35
    result = bounded_dls_tcp_step(
        tcp_jacobian_m_per_rad=jacobian,
        current_tcp_position_m=(0.0, 0.0, 0.0),
        target_tcp_position_m=(0.0, 0.0, 0.0),
        measured_joint_position_rad=previous,
        previous_joint_command_rad=previous,
        nominal_joint_position_rad=nominal,
        joint_lower_rad=(-2.0,) * 6,
        joint_upper_rad=(2.0,) * 6,
        dt_s=dt_s,
        max_arm_rate_rad_s=max_rate_rad_s,
        max_gripper_rate_rad_s=1.0,
        max_offset_from_nominal_rad=0.18,
    )

    command = np.asarray(result.joint_position_rad)
    assert result.offset_limited
    assert command[2] == pytest.approx(max_rate_rad_s * dt_s)
    assert np.all(np.abs(command[:5] - previous[:5]) <= max_rate_rad_s * dt_s + 1.0e-12)


def test_dls_servo_rejects_non_finite_jacobian() -> None:
    jacobian = np.zeros((3, 5), dtype=np.float64)
    jacobian[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bounded_dls_tcp_step(
            tcp_jacobian_m_per_rad=jacobian,
            current_tcp_position_m=(0.0, 0.0, 0.0),
            target_tcp_position_m=(0.0, 0.0, 0.0),
            measured_joint_position_rad=(0.0,) * 6,
            previous_joint_command_rad=(0.0,) * 6,
            nominal_joint_position_rad=(0.0,) * 6,
            joint_lower_rad=(-1.0,) * 6,
            joint_upper_rad=(1.0,) * 6,
            dt_s=0.01,
        )
