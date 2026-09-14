import json

import numpy as np
import pytest

from newton_calibration.actuators import load_residual


def test_bounded_linear_residual(tmp_path):
    joints = ["rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw"]
    path = tmp_path / "residual.json"
    path.write_text(
        json.dumps(
            {
                "schema": "newton.calibration.actuator-residual/v1",
                "joint_names": joints,
                "weights": [[10.0, 0.0, 0.0, 0.0]] * 6,
                "clip_nm": [0.2] * 6,
            }
        ),
        encoding="utf-8",
    )
    residual = load_residual(path, joints)
    residual.reset(np.zeros(6))
    torque = residual.compute(np.ones(6), np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(torque, np.full(6, 0.2))


@pytest.mark.parametrize(
    ("weights", "clip_nm", "message"),
    [
        ([[float("nan"), 0.0, 0.0, 0.0]], [0.2], "finite"),
        ([[0.0, 0.0, 0.0, 0.0]], [float("inf")], "finite"),
        ([[0.0, 0.0, 0.0, 0.0]], [-0.1], "non-negative"),
    ],
)
def test_residual_rejects_unsafe_numeric_values(tmp_path, weights, clip_nm, message):
    path = tmp_path / "unsafe-residual.json"
    path.write_text(
        json.dumps(
            {
                "schema": "newton.calibration.actuator-residual/v1",
                "joint_names": ["joint"],
                "weights": weights,
                "clip_nm": clip_nm,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_residual(path, ["joint"])
