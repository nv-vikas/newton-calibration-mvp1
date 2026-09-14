from __future__ import annotations

from dataclasses import replace

import pytest

from newton_calibration.isaaclab.tasks.so101_peg_insertion.run_evaluation import (
    BilateralGraspContactTracker,
    GuardedInsertionTracker,
    PreInsertionReadinessThresholds,
    PreInsertionReadinessTracker,
    PrecontactTCPTrackingTracker,
    PrecloseDisturbanceTracker,
    TaskEvidenceSample,
    TaskEvidenceTracker,
)


def _sample(
    time_s: float,
    *,
    phase: str = "lift",
    tcp_position_m: tuple[float, float, float] = (0.2, 0.0, 0.10),
    peg_position_m: tuple[float, float, float] = (0.2, 0.0, 0.08),
    peg_lift_m: float = 0.020,
    grasp_candidate: bool = False,
    linear_speed_m_s: float = 0.0,
    angular_speed_rad_s: float = 0.0,
    insertion_depth_m: float = 0.0,
    lateral_offset_m: float = 0.001,
    tilt_deg: float = 0.5,
) -> TaskEvidenceSample:
    return TaskEvidenceSample(
        time_s=time_s,
        phase=phase,
        tcp_position_m=tcp_position_m,
        peg_position_m=peg_position_m,
        peg_linear_velocity_m_s=(linear_speed_m_s, 0.0, 0.0),
        peg_angular_velocity_rad_s=(0.0, angular_speed_rad_s, 0.0),
        grasp_candidate=grasp_candidate,
        peg_lift_m=peg_lift_m,
        insertion_depth_m=insertion_depth_m,
        lateral_offset_m=lateral_offset_m,
        tilt_deg=tilt_deg,
        target_insertion_depth_m=0.024,
        lateral_tolerance_m=0.0015,
        tilt_tolerance_deg=5.0,
    )


def _latch_grasp_reference(tracker: TaskEvidenceTracker) -> None:
    tracker.update(
        _sample(
            0.0,
            phase="close",
            peg_lift_m=0.0,
            grasp_candidate=True,
        )
    )


def test_one_lifted_frame_is_not_sustained_grasp_evidence() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    event = tracker.update(_sample(0.10))
    assert event.retention_candidate
    assert not event.sustained_grasp_retention
    assert not tracker.summary()["sustained_grasp_retention_final"]


def test_fixed_relative_pose_over_the_window_is_sustained_grasp_evidence() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    for time_s in (0.10, 0.20, 0.30, 0.40):
        event = tracker.update(_sample(time_s))
    assert event.sustained_grasp_retention
    assert tracker.summary()["sustained_grasp_retention_final"]


def test_collision_launched_peg_does_not_count_as_retained() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    event = tracker.update(
        _sample(
            0.10,
            peg_position_m=(0.23, 0.0, 0.11),
            peg_lift_m=0.030,
        )
    )
    assert event.slip_candidate
    assert not event.retention_candidate
    assert not tracker.summary()["sustained_grasp_retention_seen"]


def test_transport_gate_requires_a_stable_window_ending_at_final_sample() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    for time_s in (0.10, 0.20, 0.30, 0.40):
        tracker.update(_sample(time_s))
    for time_s in (0.50, 0.60, 0.70, 0.80):
        event = tracker.update(_sample(time_s, phase="align"))
    assert event.stable_transport
    assert tracker.summary()["stable_transport_final"]

    moving_final = tracker.update(
        _sample(0.90, phase="complete", linear_speed_m_s=0.20)
    )
    assert not moving_final.stable_transport
    assert not tracker.summary()["stable_transport_final"]


def test_full_transfer_requires_lift_and_move_not_only_quiet_alignment() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    for time_s in (0.10, 0.20, 0.30, 0.40):
        tracker.update(_sample(time_s, phase="lift"))
    for time_s in (0.50, 0.60, 0.70, 0.80):
        event = tracker.update(_sample(time_s, phase="align"))
    assert event.stable_transport
    assert not event.full_transfer_qualified
    assert not tracker.summary()["full_transfer_qualified"]


def test_full_transfer_is_sticky_and_covers_lift_plus_move() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    for time_s in (0.10, 0.20, 0.30, 0.40):
        tracker.update(_sample(time_s, phase="lift"))
    tracker.update(_sample(0.50, phase="move_above_socket"))
    qualified = tracker.update(
        _sample(
            0.60,
            phase="align",
            tcp_position_m=(0.2, 0.15, 0.10),
            peg_position_m=(0.2, 0.15, 0.08),
        )
    )
    assert qualified.full_transfer_qualified
    assert qualified.full_transfer_distance_reached
    assert qualified.full_transfer_displacement_m == pytest.approx(0.150)
    assert tracker.summary()["full_transfer_qualified"]

    violated = tracker.update(
        _sample(
            0.70,
            phase="align",
            tcp_position_m=(0.2, 0.15, 0.10),
            peg_position_m=(0.207, 0.15, 0.08),
        )
    )
    assert violated.full_transfer_violated
    assert not violated.full_transfer_qualified
    summary = tracker.summary()
    assert not summary["full_transfer_qualified"]
    assert summary["full_transfer_violation_reason"] == (
        "relative_pose_drift_exceeded"
    )


def test_phase_labels_without_actual_150mm_move_do_not_qualify_transfer() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    for time_s in (0.10, 0.20, 0.30, 0.40):
        tracker.update(_sample(time_s, phase="lift"))
    tracker.update(_sample(0.50, phase="move_above_socket"))
    event = tracker.update(
        _sample(
            0.60,
            phase="align",
            tcp_position_m=(0.2, 0.10, 0.10),
            peg_position_m=(0.2, 0.10, 0.08),
        )
    )
    assert event.full_transfer_candidate
    assert not event.full_transfer_distance_reached
    assert not event.full_transfer_qualified
    summary = tracker.summary()
    assert summary["maximum_full_transfer_displacement_m"] == pytest.approx(0.100)
    assert not summary["full_transfer_distance_reached"]


@pytest.mark.parametrize(
    ("sample_kwargs", "reason"),
    (
        ({"linear_speed_m_s": 0.031}, "linear_speed_exceeded"),
        ({"angular_speed_rad_s": 0.51}, "angular_speed_exceeded"),
        ({"peg_lift_m": 0.0}, "drop_detected"),
    ),
)
def test_transfer_violation_during_move_is_sticky(
    sample_kwargs: dict[str, float],
    reason: str,
) -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    for time_s in (0.10, 0.20, 0.30, 0.40):
        tracker.update(_sample(time_s, phase="lift"))
    violated = tracker.update(
        _sample(0.50, phase="move_above_socket", **sample_kwargs)
    )
    assert violated.full_transfer_violated
    assert tracker.summary()["full_transfer_violation_reason"] == reason

    recovered_pose = tracker.update(
        _sample(
            0.60,
            phase="align",
            tcp_position_m=(0.2, 0.15, 0.10),
            peg_position_m=(0.2, 0.15, 0.08),
        )
    )
    assert recovered_pose.full_transfer_distance_reached
    assert recovered_pose.full_transfer_violated
    assert not recovered_pose.full_transfer_qualified


def test_retention_continues_when_a_held_peg_is_lowered_into_the_socket() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    for time_s in (0.10, 0.20, 0.30, 0.40):
        tracker.update(_sample(time_s))

    inserted = tracker.update(
        _sample(
            0.50,
            phase="guarded_insert",
            tcp_position_m=(0.2, 0.0, 0.06),
            peg_position_m=(0.2, 0.0, 0.04),
            peg_lift_m=0.005,
            insertion_depth_m=0.010,
        )
    )
    assert inserted.retention_candidate
    assert inserted.sustained_grasp_retention
    assert not inserted.drop_candidate


def test_lowering_the_peg_outside_insertion_is_inferred_as_a_drop() -> None:
    tracker = TaskEvidenceTracker()
    _latch_grasp_reference(tracker)
    for time_s in (0.10, 0.20, 0.30, 0.40):
        tracker.update(_sample(time_s))

    dropped = tracker.update(
        _sample(
            0.50,
            phase="align",
            tcp_position_m=(0.2, 0.0, 0.06),
            peg_position_m=(0.2, 0.0, 0.04),
            peg_lift_m=0.005,
        )
    )
    assert dropped.drop_candidate
    assert tracker.summary()["inferred_drop_candidate_seen"]


def test_seating_gate_requires_low_velocity_for_a_continuous_window() -> None:
    tracker = TaskEvidenceTracker()
    seated = _sample(
        0.0,
        phase="guarded_insert",
        insertion_depth_m=0.024,
    )
    for time_s in (0.0, 0.10, 0.20, 0.30):
        event = tracker.update(replace(seated, time_s=time_s))
    assert event.seated_pose_candidate
    assert event.stable_seating
    assert tracker.summary()["stable_seating_final"]

    moving_final = tracker.update(
        replace(seated, time_s=0.40, peg_linear_velocity_m_s=(0.02, 0.0, 0.0))
    )
    assert moving_final.seated_pose_candidate
    assert not moving_final.stable_seating
    assert not tracker.summary()["stable_seating_final"]


def test_contact_like_events_are_explicitly_labelled_as_inferred() -> None:
    tracker = TaskEvidenceTracker()
    first = tracker.update(
        _sample(
            0.0,
            phase="guarded_insert",
            insertion_depth_m=0.001,
        )
    )
    second = tracker.update(
        _sample(
            0.1,
            phase="guarded_insert",
            insertion_depth_m=0.002,
        )
    )
    assert first.inference_basis == (
        "raw_bilateral_gripper_contact_plus_kinematic_"
        "retention_slip_drop_socket_inference"
    )
    assert first.socket_contact_candidate
    assert first.first_socket_contact_candidate
    assert second.socket_contact_candidate
    assert not second.first_socket_contact_candidate


def test_non_monotonic_or_non_finite_evidence_is_rejected() -> None:
    tracker = TaskEvidenceTracker()
    tracker.update(_sample(1.0))
    with pytest.raises(ValueError, match="monotonic"):
        tracker.update(_sample(0.5))

    with pytest.raises(ValueError, match="finite"):
        TaskEvidenceTracker().update(
            _sample(0.0, peg_position_m=(float("inf"), 0.0, 0.08))
        )


def test_preclose_disturbance_latches_after_settle_and_accepts_small_drift() -> None:
    tracker = PrecloseDisturbanceTracker()
    assert tracker.update(phase="settle", peg_position_m=(0.2, 0.0, 0.035))
    assert tracker.summary()["reference_peg_position_m"] is None

    assert tracker.update(phase="approach_peg", peg_position_m=(0.2, 0.0, 0.0347))
    assert tracker.update(
        phase="descend_to_grasp",
        peg_position_m=(0.20015, 0.00010, 0.03450),
    )
    summary = tracker.summary()
    assert summary["passed"]
    assert summary["maximum_xy_drift_m"] < 0.00025
    assert summary["maximum_z_drift_m"] == pytest.approx(0.00020)


def test_preclose_disturbance_is_sticky_and_rejects_false_contact() -> None:
    tracker = PrecloseDisturbanceTracker()
    assert tracker.update(phase="approach_peg", peg_position_m=(0.2, 0.0, 0.0347))
    assert not tracker.update(
        phase="descend_to_grasp",
        peg_position_m=(0.20030, 0.0, 0.0347),
    )
    assert not tracker.update(phase="close", peg_position_m=(0.2, 0.0, 0.0347))
    summary = tracker.summary()
    assert summary["violated"]
    assert not summary["passed"]


def test_preclose_disturbance_rejects_non_finite_position() -> None:
    with pytest.raises(ValueError, match="three finite"):
        PrecloseDisturbanceTracker().update(
            phase="approach_peg",
            peg_position_m=(0.2, float("nan"), 0.0347),
        )


def test_preclose_disturbance_rejects_contact_even_before_visible_motion() -> None:
    tracker = PrecloseDisturbanceTracker()
    assert tracker.update(
        phase="approach_peg",
        peg_position_m=(0.2, 0.0, 0.0347),
    )
    assert not tracker.update(
        phase="descend_to_grasp",
        peg_position_m=(0.2, 0.0, 0.0347),
        fixed_finger_force_n=0.051,
    )
    assert tracker.summary()["maximum_gripper_contact_force_n"] == pytest.approx(
        0.051
    )


def test_precontact_tcp_tracking_accepts_small_error_and_rejects_sag() -> None:
    tracker = PrecontactTCPTrackingTracker()
    assert tracker.update(
        time_s=0.0,
        phase="stabilize_above_peg",
        measured_tcp_position_m=(0.2352, -0.0801, 0.0892),
        commanded_tcp_setpoint_m=(0.2350, -0.0800, 0.0900),
    )
    assert not tracker.update(
        time_s=0.1,
        phase="descend_to_grasp",
        measured_tcp_position_m=(0.2353, -0.0801, 0.0860),
        commanded_tcp_setpoint_m=(0.2350, -0.0800, 0.0880),
    )
    summary = tracker.summary()
    assert summary["violated"]
    assert summary["first_violation"]["vertical_error_m"] == pytest.approx(0.002)
    assert not tracker.update(
        time_s=0.2,
        phase="close",
        measured_tcp_position_m=(0.2350, -0.0800, 0.0880),
        commanded_tcp_setpoint_m=(0.2350, -0.0800, 0.0880),
    )


def test_precontact_tcp_tracking_ignores_approach_and_validates_samples() -> None:
    tracker = PrecontactTCPTrackingTracker()
    assert tracker.update(
        time_s=0.0,
        phase="approach_peg",
        measured_tcp_position_m=(0.2300, -0.0800, 0.0800),
        commanded_tcp_setpoint_m=(0.2350, -0.0800, 0.0900),
    )
    assert tracker.summary()["sample_count"] == 0
    with pytest.raises(ValueError, match="finite"):
        tracker.update(
            time_s=0.1,
            phase="descend_to_grasp",
            measured_tcp_position_m=(float("nan"), 0.0, 0.0),
            commanded_tcp_setpoint_m=(0.0, 0.0, 0.0),
        )


def test_pre_insertion_readiness_requires_a_stable_conservative_pose() -> None:
    tracker = PreInsertionReadinessTracker()
    assert not tracker.update(
        time_s=0.0,
        phase="move_above_socket",
        lateral_offset_m=0.010,
        tilt_deg=0.5,
        peg_bottom_clearance_m=0.005,
        socket_normal_force_n=0.0,
    )
    assert not tracker.update(
        time_s=0.10,
        phase="align",
        lateral_offset_m=0.00060,
        tilt_deg=0.8,
        peg_bottom_clearance_m=0.00060,
        socket_normal_force_n=0.0,
    )
    assert tracker.update(
        time_s=0.21,
        phase="align",
        lateral_offset_m=0.00060,
        tilt_deg=0.8,
        peg_bottom_clearance_m=0.00060,
        socket_normal_force_n=0.0,
    )
    summary = tracker.summary()
    assert summary["passed"]
    assert summary["worst_case_radial_sweep_m"] < summary["thresholds"][
        "effective_radial_clearance_m"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lateral_offset_m", 0.00066),
        ("tilt_deg", 1.01),
        ("peg_bottom_clearance_m", 0.00049),
    ),
)
def test_pre_insertion_readiness_rejects_each_geometric_limit(
    field: str,
    value: float,
) -> None:
    sample = {
        "lateral_offset_m": 0.00060,
        "tilt_deg": 0.8,
        "peg_bottom_clearance_m": 0.00060,
    }
    sample[field] = value
    tracker = PreInsertionReadinessTracker()
    assert not tracker.update(
        time_s=0.0,
        phase="align",
        socket_normal_force_n=0.0,
        **sample,
    )
    assert not tracker.summary()["passed"]


def test_pre_guard_socket_contact_is_sticky_and_fails_closed() -> None:
    tracker = PreInsertionReadinessTracker()
    assert not tracker.update(
        time_s=0.0,
        phase="move_above_socket",
        lateral_offset_m=0.010,
        tilt_deg=0.5,
        peg_bottom_clearance_m=0.005,
        socket_normal_force_n=0.051,
    )
    assert tracker.violated
    assert not tracker.update(
        time_s=0.20,
        phase="align",
        lateral_offset_m=0.00010,
        tilt_deg=0.1,
        peg_bottom_clearance_m=0.001,
        socket_normal_force_n=0.0,
    )
    summary = tracker.summary()
    assert summary["violated"]
    assert summary["stop_reason"] == "socket_contact_before_guarded_insert"


def test_pre_insertion_thresholds_reject_an_exhausted_clearance_budget() -> None:
    with pytest.raises(ValueError, match="radial clearance"):
        PreInsertionReadinessThresholds(
            maximum_lateral_offset_m=0.0010,
            maximum_tilt_deg=1.0,
        ).validate()


def test_guarded_insertion_accepts_continuing_depth_progress() -> None:
    tracker = GuardedInsertionTracker()
    for index in range(8):
        assert tracker.update(
            time_s=0.1 * index,
            phase="guarded_insert",
            insertion_depth_m=0.0004 * index,
            target_insertion_depth_m=0.024,
            maximum_effort_fraction=0.50,
        )
    summary = tracker.summary()
    assert summary["passed"]
    assert not summary["triggered"]


def test_guarded_insertion_stops_after_depth_progress_stalls() -> None:
    tracker = GuardedInsertionTracker()
    assert tracker.update(
        time_s=0.0,
        phase="guarded_insert",
        insertion_depth_m=0.010,
        target_insertion_depth_m=0.024,
        maximum_effort_fraction=0.40,
    )
    assert not tracker.update(
        time_s=0.51,
        phase="guarded_insert",
        insertion_depth_m=0.0101,
        target_insertion_depth_m=0.024,
        maximum_effort_fraction=0.40,
    )
    assert tracker.summary()["stop_reason"] == "insertion_depth_progress_stalled"


def test_guarded_insertion_stops_on_sustained_joint_effort_limit() -> None:
    tracker = GuardedInsertionTracker()
    assert tracker.update(
        time_s=0.0,
        phase="guarded_insert",
        insertion_depth_m=0.0,
        target_insertion_depth_m=0.024,
        maximum_effort_fraction=0.95,
    )
    assert not tracker.update(
        time_s=0.16,
        phase="guarded_insert",
        insertion_depth_m=0.0003,
        target_insertion_depth_m=0.024,
        maximum_effort_fraction=0.95,
    )
    summary = tracker.summary()
    assert summary["stop_reason"] == "sustained_joint_effort_limit"
    assert "no wrist force/torque" in summary["signal_basis"]


def test_guarded_insertion_stops_on_sustained_socket_force() -> None:
    tracker = GuardedInsertionTracker()
    assert tracker.update(
        time_s=0.0,
        phase="guarded_insert",
        insertion_depth_m=0.001,
        target_insertion_depth_m=0.024,
        maximum_effort_fraction=0.40,
        socket_normal_force_n=9.0,
    )
    assert not tracker.update(
        time_s=0.06,
        phase="guarded_insert",
        insertion_depth_m=0.0013,
        target_insertion_depth_m=0.024,
        maximum_effort_fraction=0.40,
        socket_normal_force_n=9.0,
    )
    assert tracker.summary()["stop_reason"] == "sustained_socket_normal_force_limit"


def test_bilateral_grasp_requires_both_partners_for_a_sustained_window() -> None:
    tracker = BilateralGraspContactTracker()
    assert not tracker.update(
        time_s=0.0,
        phase="close",
        fixed_finger_force_n=0.2,
        moving_finger_force_n=0.0,
    )
    assert not tracker.update(
        time_s=0.10,
        phase="close",
        fixed_finger_force_n=0.2,
        moving_finger_force_n=0.2,
    )
    assert tracker.update(
        time_s=0.21,
        phase="verify_grasp",
        fixed_finger_force_n=0.2,
        moving_finger_force_n=0.2,
    )
    summary = tracker.summary()
    assert summary["qualified"]
    assert summary["maximum_fixed_finger_force_n"] == pytest.approx(0.2)
