import json

import numpy as np

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
