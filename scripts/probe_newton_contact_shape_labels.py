#!/usr/bin/env python3
"""Capture finalized Newton labels before contact-sensor matching.

This diagnostic launches the real Isaac Lab/Newton scene, wraps the contact
sensor registration point, and durably records the model labels that the
Newton matcher actually sees.  It does not step physics or alter the task.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if os.environ.get("ACCEPT_EULA") != "Y":
    raise RuntimeError("Review the NVIDIA Isaac Sim license and export ACCEPT_EULA=Y before running.")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", required=True)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

from isaaclab_newton.physics.newton_manager import NewtonManager

from newton_calibration.isaaclab.tasks.so101_peg_insertion import (
    PegInsertionMode,
)
from newton_calibration.isaaclab.tasks.so101_peg_insertion.scene import (
    SO101PegInsertionScene,
)


def main() -> None:
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = NewtonManager.add_contact_sensor.__func__
    captured = False

    @classmethod
    def capture_labels(cls, *sensor_args, **sensor_kwargs):
        nonlocal captured
        if not captured:
            shape_labels = [str(value) for value in cls._model.shape_label]
            body_labels = [str(value) for value in cls._model.body_label]
            payload = {
                "schema": "newton.calibration/contact-label-probe@1.0",
                "asset": str(Path(args.asset).expanduser().resolve()),
                "shape_label_count": len(shape_labels),
                "body_label_count": len(body_labels),
                "collision_proxy_shape_labels": [
                    value
                    for value in shape_labels
                    if "newton_collision" in value or "part_" in value
                ],
                "gripper_shape_labels": [
                    value
                    for value in shape_labels
                    if "gripper" in value or "jaw" in value or "follower" in value
                ],
                "socket_shape_labels": [
                    value for value in shape_labels if "/Socket" in value
                ],
                "gripper_body_labels": [
                    value
                    for value in body_labels
                    if "gripper" in value or "jaw" in value
                ],
                "first_sensor_args_repr": repr(sensor_args),
                "first_sensor_kwargs_repr": repr(sensor_kwargs),
            }
            with destination.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            captured = True
            print(f"LABELS={destination}", flush=True)
            print(
                json.dumps(
                    payload["collision_proxy_shape_labels"],
                    indent=2,
                    allow_nan=False,
                ),
                flush=True,
            )
        return original(cls, *sensor_args, **sensor_kwargs)

    NewtonManager.add_contact_sensor = capture_labels
    try:
        SO101PegInsertionScene(
            usd_path=args.asset,
            mode=PegInsertionMode.GRASP_TRANSPORT,
        )
    except RuntimeError:
        if not destination.is_file():
            raise
        # The configured sensor may intentionally fail after labels are
        # captured; the durable label artifact is this probe's result.
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
