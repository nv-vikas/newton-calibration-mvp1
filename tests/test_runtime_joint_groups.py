from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from newton_calibration.adapters.runtime.analytic import (
    AnalyticPDReplayRuntime,
    _joint_properties,
    _resolve_joint_layout,
)
from newton_calibration.adapters.runtime.newton import IsaacLabNewtonRuntime, _actuator_global_indices
from newton_calibration.core.models import EnvironmentSpec

_SO101_MAP = {
    "rotation": "shoulder_pan",
    "pitch": "shoulder_lift",
    "elbow": "elbow_flex",
    "wrist_pitch": "wrist_flex",
    "wrist_roll": "wrist_roll",
    "jaw": "gripper",
}


def _generic_environment(**updates) -> EnvironmentSpec:
    values = {
        "adapter": "analytic",
        "asset_path": "robot.usd",
        "robot_id": "five_dof_fixture",
        "profile_schema": "articulation-profile/v1",
        "dt": 0.01,
        "joint_map": {
            "shoulder": "joint_a",
            "elbow": "joint_b",
            "wrist": "joint_c",
            "finger_left": "finger_a",
            "finger_right": "finger_b",
        },
        "joint_groups": {
            "manipulator": ("shoulder", "elbow", "wrist"),
            "end_effector": ("finger_left", "finger_right"),
        },
        "base_stiffness": 2.0,
        "base_damping": 0.5,
        "base_effort_limit": 10.0,
        "base_armature": 0.0,
        "base_stiffness_by_joint": {
            "elbow": 3.0,
            "finger_left": 4.0,
            "finger_right": 5.0,
        },
        "base_damping_by_joint": {
            "finger_left": 0.25,
            "finger_right": 0.3,
        },
        "base_effort_limit_by_joint": {
            "finger_left": 2.0,
            "finger_right": 2.5,
        },
        "base_armature_by_joint": {
            "finger_left": 0.01,
            "finger_right": 0.02,
        },
        "analytic_inertia_by_joint": {
            "shoulder": 1.5,
            "elbow": 1.0,
            "wrist": 0.5,
            "finger_left": 0.1,
            "finger_right": 0.12,
        },
    }
    values.update(updates)
    return EnvironmentSpec(**values)


def test_legacy_so101_layout_and_analytic_dynamics_are_unchanged() -> None:
    environment = EnvironmentSpec(
        adapter="analytic",
        asset_path="so101.usd",
        joint_map=dict(_SO101_MAP),
    )
    runtime = AnalyticPDReplayRuntime(environment)

    assert runtime.joint_layout.legacy_so101
    assert runtime.joint_layout.logical_names == tuple(_SO101_MAP)
    assert runtime.joint_layout.runtime_names == tuple(_SO101_MAP.values())
    assert runtime.joint_layout.groups == (
        ("arm", ("rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll")),
        ("gripper", ("jaw",)),
    )
    np.testing.assert_allclose(runtime.base_inertia, [1.8, 1.6, 1.2, 0.7, 0.5, 0.35])

    properties = _joint_properties(
        environment,
        runtime.joint_layout,
        {
            "arm_stiffness_scale": 2.0,
            "arm_damping_scale": 3.0,
            "arm_effort_scale": 0.5,
            "arm_friction_nm": 0.1,
            "arm_armature": 0.02,
            "gripper_stiffness_scale": 4.0,
            "gripper_damping_scale": 5.0,
            "gripper_effort_scale": 0.25,
            "gripper_friction_nm": 0.03,
            "gripper_armature": 0.004,
        },
        analytic=True,
    )
    np.testing.assert_allclose(properties["stiffness"], [70.0] * 5 + [112.0])
    np.testing.assert_allclose(properties["damping"], [6.0] * 5 + [7.5])
    np.testing.assert_allclose(properties["effort"], [5.0] * 5 + [2.5])
    np.testing.assert_allclose(properties["friction"], [0.1] * 5 + [0.03])
    np.testing.assert_allclose(properties["armature"], [0.02] * 5 + [0.004])


def test_generic_groups_define_canonical_order_and_group_parameters() -> None:
    environment = _generic_environment()
    layout = _resolve_joint_layout(environment)

    assert not layout.legacy_so101
    assert layout.logical_names == (
        "shoulder",
        "elbow",
        "wrist",
        "finger_left",
        "finger_right",
    )
    assert layout.runtime_names == (
        "joint_a",
        "joint_b",
        "joint_c",
        "finger_a",
        "finger_b",
    )
    assert layout.group_by_joint == (
        "manipulator",
        "manipulator",
        "manipulator",
        "end_effector",
        "end_effector",
    )

    candidate = {
        "manipulator_stiffness_scale": 2.0,
        "end_effector_stiffness_scale": 0.5,
        "manipulator_damping_scale": 3.0,
        "manipulator_effort_scale": 0.5,
        "end_effector_effort_scale": 2.0,
        "manipulator_friction_nm": 0.1,
        "end_effector_friction_nm": 0.02,
        "manipulator_armature": 0.2,
    }
    properties = _joint_properties(
        environment,
        layout,
        candidate,
        analytic=False,
    )
    np.testing.assert_allclose(properties["stiffness"], [4.0, 6.0, 4.0, 2.0, 2.5])
    np.testing.assert_allclose(properties["damping"], [1.5, 1.5, 1.5, 0.25, 0.3])
    np.testing.assert_allclose(properties["effort"], [5.0, 5.0, 5.0, 4.0, 5.0])
    np.testing.assert_allclose(properties["friction"], [0.1, 0.1, 0.1, 0.02, 0.02])
    np.testing.assert_allclose(properties["armature"], [0.2, 0.2, 0.2, 0.01, 0.02])


def test_newton_adapter_uses_the_same_generic_parameter_vectors_without_isaac_imports() -> None:
    environment = _generic_environment()
    runtime = IsaacLabNewtonRuntime.__new__(IsaacLabNewtonRuntime)
    runtime.environment = environment
    runtime.joint_layout = _resolve_joint_layout(environment)

    values = runtime._parameter_vectors(
        {
            "manipulator_stiffness_scale": 1.5,
            "end_effector_damping_scale": 2.0,
            "end_effector_armature": 0.04,
        }
    )

    np.testing.assert_allclose(values["stiffness"], [3.0, 4.5, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(values["damping"], [0.5, 0.5, 0.5, 0.5, 0.6])
    np.testing.assert_allclose(values["armature"], [0.0, 0.0, 0.0, 0.04, 0.04])


def test_actuator_local_scope_normalizes_noncontiguous_global_indices() -> None:
    assert _actuator_global_indices([1, 4, 7], total_joints=9) == [1, 4, 7]
    assert _actuator_global_indices(slice(None), total_joints=4) == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="joint_indices"):
        _actuator_global_indices(None, total_joints=4)


def test_analytic_rollout_accepts_non_so101_joint_width() -> None:
    runtime = AnalyticPDReplayRuntime(_generic_environment())
    episode = SimpleNamespace(
        name="generic-step",
        time_s=np.arange(5, dtype=np.float64) * 0.01,
        command_q=np.full((5, 5), 0.1, dtype=np.float64),
        actual_q=np.zeros((5, 5), dtype=np.float64),
        actual_dq=np.zeros((5, 5), dtype=np.float64),
    )

    q, dq = runtime._rollout({}, episode)

    assert q.shape == (5, 5)
    assert dq.shape == (5, 5)
    assert np.isfinite(q).all()
    assert np.isfinite(dq).all()


def test_episode_width_mismatch_fails_before_broadcasting() -> None:
    runtime = AnalyticPDReplayRuntime(_generic_environment())
    episode = SimpleNamespace(
        name="wrong-width",
        time_s=np.arange(3, dtype=np.float64) * 0.01,
        command_q=np.zeros((3, 4), dtype=np.float64),
        actual_q=np.zeros((3, 4), dtype=np.float64),
        actual_dq=np.zeros((3, 4), dtype=np.float64),
    )

    with pytest.raises(ValueError, match=r"command_q must have shape \[samples, 5\]"):
        runtime._rollout({}, episode)


@pytest.mark.parametrize(
    ("joint_groups", "joint_map", "exception", "message"),
    [
        (
            {"arm": ("joint",), "tool": ("joint",)},
            {"joint": "axis"},
            ValueError,
            "only one joint group",
        ),
        (
            {"arm": ("missing",)},
            {"joint": "axis"},
            ValueError,
            "absent from joint_map",
        ),
        (
            {"bad-group": ("joint",)},
            {"joint": "axis"},
            ValueError,
            "valid calibration parameter prefix",
        ),
        (
            {"arm": "joint"},
            {"joint": "axis"},
            ValueError,
            "must map non-empty names to joint sequences",
        ),
    ],
)
def test_invalid_joint_group_contract_fails_closed(joint_groups, joint_map, exception, message) -> None:
    with pytest.raises(exception, match=message):
        environment = EnvironmentSpec(
            adapter="analytic",
            asset_path="robot.usd",
            profile_schema="articulation-profile/v1",
            joint_groups=joint_groups,
            joint_map=joint_map,
        )
        _resolve_joint_layout(environment)


def test_generic_analytic_runtime_requires_explicit_positive_inertia() -> None:
    environment = _generic_environment(analytic_inertia_by_joint={"shoulder": 1.0})

    with pytest.raises(ValueError, match="missing analytic_inertia_by_joint"):
        AnalyticPDReplayRuntime(environment)

    invalid = dict(_generic_environment().analytic_inertia_by_joint)
    invalid["finger_right"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        AnalyticPDReplayRuntime(_generic_environment(analytic_inertia_by_joint=invalid))
