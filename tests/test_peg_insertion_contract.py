from __future__ import annotations

from dataclasses import replace

import pytest

from newton_calibration.isaaclab.tasks.so101_peg_insertion import (
    ControllerObservation,
    ControllerPhase,
    PegInsertionController,
    PegInsertionMode,
    PegInsertionSceneSpec,
    PegInsertionTaskMetrics,
    local_z_axis_from_quaternion_xyzw,
    required_evidence_channels,
)


def test_reference_geometry_has_positive_clearance() -> None:
    spec = PegInsertionSceneSpec()
    spec.validate()
    assert spec.radial_clearance_m > spec.contact_gap_m


def test_invalid_aperture_is_rejected() -> None:
    spec = replace(PegInsertionSceneSpec(), socket_aperture_m=0.0181, contact_gap_m=0.0001)
    with pytest.raises(ValueError, match="aperture"):
        spec.validate()


def test_alignment_keeps_peg_clear_until_guarded_insertion() -> None:
    spec = PegInsertionSceneSpec()
    spec.validate()
    assert spec.aligned_peg_bottom_clearance_m > spec.contact_gap_m
    aligned_metrics = PegInsertionTaskMetrics.from_pose(
        spec,
        (
            *spec.socket_center_xy_m,
            spec.align_tcp_height_m - spec.grasp_tcp_above_peg_center_m,
        ),
        (0.0, 0.0, 1.0),
        settled=False,
    )
    assert aligned_metrics.insertion_depth_m == 0.0
    assert not aligned_metrics.seated

    unsafe = replace(spec, align_tcp_height_m=0.075)
    with pytest.raises(ValueError, match="above the socket rim"):
        unsafe.validate()


def test_evidence_contract_grows_without_losing_robot_channels() -> None:
    mvp1 = set(required_evidence_channels(PegInsertionMode.ROBOT_ONLY))
    mvp2 = set(required_evidence_channels(PegInsertionMode.GRASP_TRANSPORT))
    mvp3 = set(required_evidence_channels(PegInsertionMode.INSERTION))
    assert mvp1 < mvp2 < mvp3
    assert "event.jam" in mvp3
    assert "event.jam" not in mvp2


def test_xyzw_pose_conversion_distinguishes_upright_and_horizontal_peg() -> None:
    assert local_z_axis_from_quaternion_xyzw((0.0, 0.0, 0.0, 1.0)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    # 90 degrees about world X rotates local +Z into -Y.
    half_sqrt_two = 2.0**-0.5
    horizontal = local_z_axis_from_quaternion_xyzw(
        (half_sqrt_two, 0.0, 0.0, half_sqrt_two)
    )
    assert horizontal == pytest.approx((0.0, -1.0, 0.0), abs=1.0e-12)

    spec = PegInsertionSceneSpec()
    seated_center_z = spec.table_top_z_m + 0.5 * spec.peg_height_m
    metrics = PegInsertionTaskMetrics.from_pose(
        spec,
        (*spec.socket_center_xy_m, seated_center_z),
        horizontal,
        settled=True,
    )
    assert metrics.tilt_deg == pytest.approx(90.0)
    assert not metrics.seated


def test_xyzw_pose_conversion_rejects_invalid_quaternion() -> None:
    with pytest.raises(ValueError, match="norm"):
        local_z_axis_from_quaternion_xyzw((0.0, 0.0, 0.0, 0.0))


def test_pose_metrics_distinguish_seated_and_jammed() -> None:
    spec = PegInsertionSceneSpec()
    seated = PegInsertionTaskMetrics.from_pose(
        spec,
        (
            spec.socket_center_xy_m[0],
            spec.socket_center_xy_m[1],
            spec.table_top_z_m + 0.5 * spec.peg_height_m,
        ),
        (0.0, 0.0, 1.0),
        settled=True,
    )
    jammed = PegInsertionTaskMetrics.from_pose(
        spec,
        (
            spec.socket_center_xy_m[0] + 0.02,
            spec.socket_center_xy_m[1],
            spec.socket_top_z_m + 0.5 * spec.peg_height_m,
        ),
        (0.0, 0.0, 1.0),
        settled=True,
    )
    assert seated.seated and not seated.jammed
    assert jammed.jammed and not jammed.seated


def test_controller_reaches_grasp_verification_deterministically() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(spec, settle_s=0.1, close_s=0.1)
    obs = ControllerObservation(
        ee_position_m=(*spec.peg_start_xy_m, spec.transport_tcp_height_m),
        peg_position_m=(*spec.peg_start_xy_m, spec.peg_rest_center_z_m),
        gripper_gap_m=0.04,
    )
    controller.step(obs, 0.1)
    controller.step(obs, 0.1)
    assert controller.phase is ControllerPhase.STABILIZE_ABOVE_PEG
    controller.step(obs, controller.approach_stabilize_s)
    assert controller.phase is ControllerPhase.DESCEND_TO_GRASP
    grasp_obs = replace(obs, ee_position_m=(*spec.peg_start_xy_m, spec.grasp_tcp_height_m))
    controller.step(grasp_obs, 0.1)
    controller.step(grasp_obs, 0.1)
    assert controller.phase is ControllerPhase.VERIFY_GRASP


def test_lift_target_is_fixed_when_the_grasped_peg_moves() -> None:
    spec = PegInsertionSceneSpec()
    controller = PegInsertionController(spec)
    initial = ControllerObservation(
        ee_position_m=(*spec.peg_start_xy_m, spec.grasp_tcp_height_m),
        peg_position_m=(*spec.peg_start_xy_m, spec.peg_rest_center_z_m),
        gripper_gap_m=0.01,
        peg_grasped=True,
    )
    controller.step(initial, 0.01)
    controller.phase = ControllerPhase.LIFT
    first_target = controller._target_for_phase(initial)
    moved = replace(
        initial,
        peg_position_m=(
            spec.peg_start_xy_m[0],
            spec.peg_start_xy_m[1],
            spec.peg_rest_center_z_m + 0.03,
        ),
    )
    assert controller._target_for_phase(moved) == first_target
    assert first_target[2] == spec.transport_tcp_height_m


def test_depth_and_jam_are_not_inferred_for_a_peg_away_from_the_socket() -> None:
    spec = PegInsertionSceneSpec()
    metrics = PegInsertionTaskMetrics.from_pose(
        spec,
        (*spec.peg_start_xy_m, spec.peg_rest_center_z_m),
        (0.0, 0.0, 1.0),
        settled=True,
    )
    assert metrics.insertion_depth_m == 0.0
    assert not metrics.seated
    assert not metrics.jammed
