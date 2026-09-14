from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from newton_calibration.adapters.surface import SO101EnvCfg
from newton_calibration.isaaclab import tuning

JOINTS = ["rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw"]
TRAIN = ["step-response", "chirp-sweep", "static-holding", "gripper-cycles"]
HELDOUT = ["frequency-sweep", "friction-gravity", "hold-under-gravity", "backlash-detection"]


def _simulate(command: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    q = command[0].copy()
    dq = np.zeros_like(q)
    qs, dqs = [], []
    for target in command:
        torque = 70.0 * (target - q) - 4.0 * dq - 0.03 * np.tanh(dq / 0.01)
        dq += dt * torque / np.array([1.8, 1.6, 1.2, 0.7, 0.5, 0.35])
        q += dt * dq
        qs.append(q.copy())
        dqs.append(dq.copy())
    return np.asarray(qs), np.asarray(dqs)


def _write_episode(path: Path, phase: float):
    state_dt = 0.01
    command_dt = 0.05
    state_time = np.arange(0.0, 2.5, state_dt)
    command_time = np.arange(0.0, 2.5, command_dt)
    command_state = np.stack(
        [0.2 * np.sin((joint + 1) * 0.55 * state_time + phase) for joint in range(6)], axis=1
    )
    actual, velocity = _simulate(command_state, state_dt)
    command_sparse = np.stack(
        [0.2 * np.sin((joint + 1) * 0.55 * command_time + phase) for joint in range(6)], axis=1
    )
    origin = 1_777_000_000_000_000_000
    records = []
    experiment = path.stem
    for joint_index, joint in enumerate(JOINTS):
        for time_s, value in zip(command_time, command_sparse[:, joint_index]):
            records.append((origin + int(time_s * 1e9), experiment, f"{joint}/command_q", value))
        for signal, values in (("actual_q", actual[:, joint_index]), ("dq", velocity[:, joint_index])):
            for time_s, value in zip(state_time, values):
                records.append((origin + int(time_s * 1e9), experiment, f"{joint}/{signal}", value))
    frame = pd.DataFrame(records, columns=["time_ns", "experiment", "field", "value"])
    frame["time_utc"] = pd.to_datetime(frame["time_ns"], utc=True)
    frame[["time_ns", "time_utc", "experiment", "field", "value"]].to_parquet(path)


def test_five_call_workflow(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for index, name in enumerate(TRAIN):
        _write_episode(evidence / f"so101-sysid-50motion-train-{name}.parquet", 0.1 * index)
    for index, name in enumerate(HELDOUT):
        _write_episode(evidence / f"so101-sysid-50motion-heldout-{name}.parquet", 0.15 * index)
    asset = tmp_path / "so101.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    env = SO101EnvCfg(usd_path=str(asset), runtime="analytic", device="cpu", dt=state_dt())

    analysis = tuning.analyze(env=env, evidence=evidence, workdir=tmp_path / "runs")
    calibration_plan = tuning.plan(analysis)
    fit_run = tuning.fit(calibration_plan, generations=2, population=4, resume=False)
    validation = tuning.validate(fit_run)
    package = tuning.write(validation, output=tmp_path / "package")

    assert analysis.readiness["heldout_split_present"]
    assert len(analysis.identifiability) == 11
    assert fit_run.completed_generations == 2
    assert Path(package.manifest_path).exists()
    assert Path(package.overlay_path).exists()
    assert "heldout_only" in validation.gates


def state_dt() -> float:
    return 0.01
