from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from newton_calibration.adapters.surface import SO101EnvCfg
from newton_calibration.isaaclab import tuning
from newton_calibration.optimizers import OptimizerInit, register_optimizer

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
    command_state = np.stack([0.2 * np.sin((joint + 1) * 0.55 * state_time + phase) for joint in range(6)], axis=1)
    actual, velocity = _simulate(command_state, state_dt)
    command_sparse = np.stack([0.2 * np.sin((joint + 1) * 0.55 * command_time + phase) for joint in range(6)], axis=1)
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


def _analyze_synthetic_job(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for index, name in enumerate(TRAIN):
        _write_episode(evidence / f"so101-sysid-50motion-train-{name}.parquet", 0.1 * index)
    for index, name in enumerate(HELDOUT):
        _write_episode(evidence / f"so101-sysid-50motion-heldout-{name}.parquet", 0.15 * index)
    asset = tmp_path / "so101.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    env = SO101EnvCfg(usd_path=str(asset), runtime="analytic", device="cpu", dt=state_dt())
    return tuning.analyze(env=env, evidence=evidence, workdir=tmp_path / "runs")


def test_five_call_workflow(tmp_path):
    analysis = _analyze_synthetic_job(tmp_path)
    calibration_plan = tuning.plan(analysis)
    fit_run = tuning.fit(calibration_plan, generations=2, population=4, resume=False)
    validation = tuning.validate(fit_run)
    package = tuning.write(validation, output=tmp_path / "package")

    assert analysis.readiness["heldout_split_present"]
    assert len(analysis.identifiability) == 11
    assert fit_run.completed_generations == 2
    assert fit_run.optimizer["name"] == "diagonal-cma-es"
    assert fit_run.optimizer["version"] == "1"
    assert fit_run.optimizer["provider"] == "newton-calibration"
    assert Path(package.manifest_path).exists()
    assert Path(package.overlay_path).exists()
    assert (Path(package.output_dir) / "job" / "fit-generations" / "000000.json").exists()
    manifest = json.loads(Path(package.manifest_path).read_text())
    assert manifest["optimizer"]["config_fingerprint"] == fit_run.optimizer["config_fingerprint"]
    assert manifest["inputs"]["asset_sha256"] == analysis.asset_fingerprint
    assert manifest["activation_allowed"] is False  # analytic is contract-only, never production-activatable
    assert manifest["status"] == "contract-validated-nonactivatable"
    assert "Activation: **NOT ALLOWED**" in Path(package.report_path).read_text(encoding="utf-8")
    assert "heldout_only" in validation.gates
    with pytest.raises(FileExistsError, match="not empty"):
        tuning.write(validation, output=tmp_path / "package")


class _MinjaeStyleOptimizer:
    """Contract-only stand-in proving an installed Minjae provider can drive fit()."""

    instances: ClassVar[list[_MinjaeStyleOptimizer]] = []

    def __init__(self, initialization: OptimizerInit):
        self.initialization = initialization
        self._generation = 0
        self._best = None
        self.ask_count = 0
        self.tell_count = 0
        self.instances.append(self)

    def ask(self):
        self.ask_count += 1
        fractions = np.linspace(0.2, 0.8, self.initialization.population)
        return [
            {
                parameter.name: float(parameter.lower + fraction * (parameter.upper - parameter.lower))
                for parameter in self.initialization.parameters
            }
            for fraction in fractions
        ]

    def tell(self, candidates, scores):
        self.tell_count += 1
        winner = int(np.argmin(scores))
        if self._best is None or scores[winner] < self._best[1]:
            self._best = (dict(candidates[winner]), float(scores[winner]))
        self._generation += 1

    @property
    def generation(self):
        return self._generation

    @property
    def best(self):
        return self._best

    def state_dict(self):
        return {"generation": self._generation, "best": self._best}

    def load_state_dict(self, state):
        self._generation = int(state["generation"])
        best = state.get("best")
        self._best = (dict(best[0]), float(best[1])) if best else None


def test_external_optimizer_drives_fit_and_resumes_from_versioned_checkpoint(tmp_path):
    optimizer_name = "test.minjae-style.v1"
    _MinjaeStyleOptimizer.instances.clear()
    register_optimizer(optimizer_name, _MinjaeStyleOptimizer, version="2026.09", replace=True)
    analysis = _analyze_synthetic_job(tmp_path)
    calibration_plan = tuning.plan(
        analysis,
        optimizer=optimizer_name,
        optimizer_options={"strategy": "adaptive-search"},
    )

    first = tuning.fit(calibration_plan, generations=1, population=4, resume=False)
    resumed = tuning.fit(calibration_plan, generations=2, population=4, resume=True)

    assert first.optimizer["name"] == optimizer_name
    assert first.optimizer["version"] == "2026.09"
    assert first.optimizer["provider"] == "in-process"
    assert first.optimizer["options"] == {"strategy": "adaptive-search"}
    assert resumed.completed_generations == 2
    assert len(_MinjaeStyleOptimizer.instances) == 2
    assert _MinjaeStyleOptimizer.instances[0].tell_count == 1
    assert _MinjaeStyleOptimizer.instances[1].generation == 2
    checkpoint = json.loads((Path(calibration_plan.workdir) / "fit-checkpoint.json").read_text())
    assert checkpoint["optimizer_name"] == optimizer_name
    assert checkpoint["optimizer_version"] == "2026.09"
    assert checkpoint["optimizer_config_fingerprint"] == resumed.optimizer["config_fingerprint"]


def test_resume_rejects_changed_external_optimizer_configuration(tmp_path):
    optimizer_name = "test.minjae-resume-guard.v1"
    register_optimizer(optimizer_name, _MinjaeStyleOptimizer, version="1", replace=True)
    analysis = _analyze_synthetic_job(tmp_path)
    calibration_plan = tuning.plan(analysis, optimizer=optimizer_name, optimizer_options={"mode": "first"})
    tuning.fit(calibration_plan, generations=1, population=4, resume=False)
    changed_plan = replace(
        calibration_plan,
        optimizer={**calibration_plan.optimizer, "options": {"mode": "changed"}},
    )

    with pytest.raises(RuntimeError, match="execution fingerprint mismatch"):
        tuning.fit(changed_plan, generations=2, population=4, resume=True)


def test_resume_rejects_changed_evidence_or_run_identity(tmp_path):
    analysis = _analyze_synthetic_job(tmp_path)
    calibration_plan = tuning.plan(analysis)
    tuning.fit(calibration_plan, generations=1, population=4, resume=False)

    for changed_plan, message in (
        (replace(calibration_plan, evidence_fingerprint="different-evidence"), "Evidence fingerprint changed"),
        (replace(calibration_plan, run_id="different-run"), "run_id mismatch"),
    ):
        with pytest.raises(RuntimeError, match=message):
            tuning.fit(changed_plan, generations=2, population=4, resume=True)


def test_fit_rejects_evidence_bytes_changed_after_plan(tmp_path):
    analysis = _analyze_synthetic_job(tmp_path)
    calibration_plan = tuning.plan(analysis)
    evidence_file = max(Path(analysis.evidence_uri).glob("*.parquet"), key=lambda path: path.name)
    evidence_file.write_bytes(evidence_file.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="Evidence fingerprint changed"):
        tuning.fit(calibration_plan, generations=1, population=4, resume=False)


def test_fit_rejects_usd_bytes_changed_after_plan(tmp_path):
    analysis = _analyze_synthetic_job(tmp_path)
    calibration_plan = tuning.plan(analysis)
    asset = Path(calibration_plan.environment.asset_path)
    asset.write_text("#usda 1.0\n# changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="USD asset fingerprint changed"):
        tuning.fit(calibration_plan, generations=1, population=4, resume=False)


def test_fit_rejects_overwrite_and_generation_budget_regression(tmp_path):
    analysis = _analyze_synthetic_job(tmp_path)
    calibration_plan = tuning.plan(analysis)
    tuning.fit(calibration_plan, generations=2, population=4, resume=False)

    with pytest.raises(RuntimeError, match="not empty"):
        tuning.fit(calibration_plan, generations=2, population=4, resume=False)
    with pytest.raises(ValueError, match="below the 2 committed generations"):
        tuning.fit(calibration_plan, generations=1, population=4, resume=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"generations": 0, "population": 4},
        {"generations": -1, "population": 4},
        {"generations": 1, "population": 0},
        {"generations": 1.5, "population": 4},
    ],
)
def test_fit_rejects_invalid_budget_overrides(tmp_path, overrides):
    analysis = _analyze_synthetic_job(tmp_path)
    calibration_plan = tuning.plan(analysis)

    with pytest.raises((TypeError, ValueError), match="positive integer"):
        tuning.fit(calibration_plan, resume=False, **overrides)


def state_dt() -> float:
    return 0.01
