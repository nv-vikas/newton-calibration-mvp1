from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from newton_calibration.adapters.evidence import TabularJointEvidence
from newton_calibration.adapters.runtime.analytic import (
    AnalyticPDReplayRuntime,
    _joint_properties,
    _resolve_joint_layout,
)
from newton_calibration.collection import CalibrationRequest
from newton_calibration.core import JointBinding, LongFormSchema, SignalBinding, bind_evidence_files
from newton_calibration.core.attestation import numeric_surface_fingerprint, parameter_fingerprint
from newton_calibration.core.io import sha256_file, write_json
from newton_calibration.isaaclab import (
    ArticulationEnvCfg,
    CalibrationPackageLoadError,
    VerifiedArticulationPackage,
    tuning,
)


class _AuthoritativeContractRuntime(AnalyticPDReplayRuntime):
    """Emit the Newton attestation shape for a kitless package-contract test."""

    def evaluate(self, candidate, episodes, objective_weights, **context):
        self._candidate = dict(candidate)
        return super().evaluate(candidate, episodes, objective_weights, **context)

    def attestation(self):
        layout = _resolve_joint_layout(self.environment)
        values = _joint_properties(self.environment, layout, self._candidate, analytic=False)
        result = super().attestation()
        result.update(
            {
                "backend": "isaaclab_newton",
                "authoritative": True,
                "asset_sha256": sha256_file(self.environment.asset_path),
                "readback_parameters": [
                    "stiffness",
                    "damping",
                    "effort_limit",
                    "armature",
                    "friction_nm",
                ],
                "full_state_reset_per_episode": True,
                "candidate_sha256": parameter_fingerprint(self._candidate),
                "readback_values_sha256": numeric_surface_fingerprint(
                    {
                        "stiffness": values["stiffness"].tolist(),
                        "damping": values["damping"].tolist(),
                        "effort_limit": values["effort"].tolist(),
                        "armature": values["armature"].tolist(),
                        "friction_nm": values["friction"].tolist(),
                    }
                ),
            }
        )
        return result


def _write_episode(path: Path, joints: tuple[str, ...], *, phase: float) -> None:
    dt = 0.02
    time_s = np.arange(0.0, 1.0, dt)
    rows = []
    for index, joint in enumerate(joints):
        command = 0.1 * np.sin(2.0 * np.pi * (1.0 + 0.2 * index) * time_s + phase)
        actual = np.empty_like(command)
        actual[0] = command[0]
        for step in range(1, len(time_s)):
            actual[step] = actual[step - 1] + 0.15 * (command[step - 1] - actual[step - 1])
        velocity = np.gradient(actual, dt)
        for signal, values in (("target", command), ("position", actual), ("velocity", velocity)):
            rows.extend(
                {
                    "time_s": timestamp,
                    "joint": joint,
                    "signal": signal,
                    "value": value,
                }
                for timestamp, value in zip(time_s, values)
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_authoritative_contract_fit(tmp_path: Path, monkeypatch, *, targets=()):
    logical_joints = ("joint_0", "joint_1")
    runtime_joints = ("axis_0", "axis_1")
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    episodes = []
    for index, split in enumerate(("train", "heldout")):
        name = f"{split}-0"
        _write_episode(evidence_root / f"{name}.csv", logical_joints, phase=0.3 * index)
        episodes.append(
            {
                "name": name,
                "path": f"{name}.csv",
                "split": split,
                "trial_id": f"capture-{index}",
            }
        )
    asset = tmp_path / "robot.usda"
    asset.write_text(
        '#usda 1.0\n\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])\n'
        '{\n    def PhysicsRevoluteJoint "axis_0" {}\n'
        '    def PhysicsRevoluteJoint "axis_1" {}\n}\n',
        encoding="utf-8",
    )
    evidence_spec = bind_evidence_files(
        root=evidence_root,
        episodes=episodes,
        schema=LongFormSchema(
            time_column="time_s",
            time_unit="s",
            value_column="value",
            joint_column="joint",
            signal_column="signal",
            field_column=None,
        ),
        # Deliberately reverse the evidence/runtime order relative to group insertion.
        joint_bindings=(
            JointBinding("joint_1", "axis_1", "rad", "rad", transform_confirmed=True),
            JointBinding("joint_0", "axis_0", "rad", "rad", transform_confirmed=True),
        ),
        signal_bindings=(
            SignalBinding("target", "command_q"),
            SignalBinding("position", "actual_q"),
            SignalBinding("velocity", "actual_dq"),
        ),
        revision="fixture-r1",
        clock_synchronized=True,
        effort_saturation_joints=logical_joints,
    )
    bounds = {
        f"{group}_{suffix}": values
        for group in ("first", "second")
        for suffix, values in (
            ("effort_scale", (0.1, 2.0, 1.0)),
            ("armature", (0.0, 0.5, 0.02)),
            ("friction_nm", (0.0, 0.5, 0.01)),
        )
    }
    env = ArticulationEnvCfg(
        usd_path=str(asset),
        robot_id="ledger-fixture",
        joint_groups={"first": ("joint_0",), "second": ("joint_1",)},
        joint_order=("joint_1", "joint_0"),
        joint_map=dict(zip(logical_joints, runtime_joints)),
        profile_confirmed=True,
        controller_profile_confirmed=True,
        controller_profile_source="fixture controller configuration sha256:abc",
        runtime="isaaclab_newton",
        device="cpu",
        dt=0.02,
        base_stiffness_by_joint={name: 20.0 for name in logical_joints},
        base_damping_by_joint={name: 1.0 for name in logical_joints},
        base_effort_limit_by_joint={name: 10.0 for name in logical_joints},
        analytic_inertia_by_joint={name: 1.0 for name in logical_joints},
        parameter_bounds=bounds,
    )
    monkeypatch.setattr(tuning, "create_runtime", lambda environment: _AuthoritativeContractRuntime(environment))
    analysis = tuning.analyze(
        env=env,
        evidence=TabularJointEvidence(evidence_spec),
        workdir=tmp_path / "runs",
        request=CalibrationRequest(target_parameters=targets),
    )
    plan = tuning.plan(analysis)
    fit = tuning.fit(plan, generations=8, population=12, resume=False)
    validation = tuning.validate(fit)
    assert validation.passed, (validation.improvement_pct, validation.regressions, validation.gates)
    return plan, validation


def test_scoped_generic_package_round_trip_preserves_unselected_baselines(tmp_path, monkeypatch):
    # Synthetic attestation fixture, solely for package/loader contract testing.
    targets = ("first_stiffness_scale", "first_damping_scale", "second_stiffness_scale", "second_damping_scale")
    _plan, validation = _run_authoritative_contract_fit(tmp_path, monkeypatch, targets=targets)
    package = tuning.write(validation, output=tmp_path / "scoped-package")
    verified = VerifiedArticulationPackage.open(package.output_dir)
    assert set(verified.parameters) == set(targets)
    restored = verified.to_env_cfg(device="cpu").describe()
    assert restored.tuning_targets == targets
    verified.assert_matches_environment(restored)
    assert verified.actuator.requested_command_delay_s == 0.0
    import yaml

    patch = yaml.safe_load(Path(package.output_dir, "actuator_patch.yaml").read_text())
    for joint in patch["ordered_joints"]:
        assert joint["effort_limit"] == 10.0 and joint["armature"] == 0.0 and joint["friction_nm"] == 0.0

    # Even internally consistent edited YAML cannot silently add a parameter
    # that was not selected and validated in the recorded recipe.
    manifest_path = Path(package.manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["first_effort_scale"] = 1.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CalibrationPackageLoadError, match="must contain exactly"):
        VerifiedArticulationPackage.open(package.output_dir)


def test_generic_v2_round_trip_and_generation_tamper_rejection(tmp_path: Path, monkeypatch) -> None:
    plan, validation = _run_authoritative_contract_fit(tmp_path, monkeypatch)
    package = tuning.write(validation, output=tmp_path / "package")

    verified = VerifiedArticulationPackage.open(package.output_dir)
    assert verified.to_env_cfg(device="cpu").joint_order == ("joint_1", "joint_0")

    run_generation = Path(plan.workdir) / "fit-generations" / "000000.json"
    corrupted = json.loads(run_generation.read_text(encoding="utf-8"))
    corrupted["run_id"] = "contradicts-root-run-id"
    run_generation.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="optimizer journal.*hash mismatch"):
        tuning.write(validation, output=tmp_path / "corrupt-source-package")

    packaged_generation = Path(package.output_dir) / "job" / "fit-generations" / "000000.json"
    corrupted = json.loads(packaged_generation.read_text(encoding="utf-8"))
    corrupted["run_id"] = "contradicts-root-run-id"
    packaged_generation.write_text(json.dumps(corrupted), encoding="utf-8")
    manifest_path = Path(package.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["job_record_sha256"]["job/fit-generations/000000.json"] = sha256_file(packaged_generation)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(CalibrationPackageLoadError, match="optimizer journal.*hash mismatch"):
        VerifiedArticulationPackage.open(package.output_dir)


def test_generic_v2_rejects_analysis_run_id_tamper_at_write_and_open(tmp_path: Path, monkeypatch) -> None:
    plan, validation = _run_authoritative_contract_fit(tmp_path, monkeypatch)
    package = tuning.write(validation, output=tmp_path / "package")

    analysis_path = Path(plan.workdir) / "analysis.json"
    corrupted = json.loads(analysis_path.read_text(encoding="utf-8"))
    corrupted["run_id"] = "contradicts-root-run-id"
    analysis_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="analysis record is not the predecessor"):
        tuning.write(validation, output=tmp_path / "corrupt-source-package")

    packaged_analysis = Path(package.output_dir) / "job" / "analysis.json"
    corrupted = json.loads(packaged_analysis.read_text(encoding="utf-8"))
    corrupted["run_id"] = "contradicts-root-run-id"
    packaged_analysis.write_text(json.dumps(corrupted), encoding="utf-8")
    manifest_path = Path(package.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["job_record_sha256"]["job/analysis.json"] = sha256_file(packaged_analysis)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(CalibrationPackageLoadError, match="analysis record is not the predecessor"):
        VerifiedArticulationPackage.open(package.output_dir)


def test_generic_v2_rejects_selected_evaluation_bookkeeping_tamper(tmp_path: Path, monkeypatch) -> None:
    plan, validation = _run_authoritative_contract_fit(tmp_path, monkeypatch)
    package = tuning.write(validation, output=tmp_path / "package")

    validation.fit.best.candidate_id = 999_999
    validation.fit.best.generation = 999_999
    write_json(Path(plan.workdir) / "fit.json", validation.fit)
    write_json(Path(plan.workdir) / "validation.json", validation)
    with pytest.raises(RuntimeError, match="evaluation bookkeeping does not follow"):
        tuning.write(validation, output=tmp_path / "corrupt-source-package")

    package_root = Path(package.output_dir)
    for relative_path in ("validation.json", "job/fit.json", "job/validation.json"):
        path = package_root / relative_path
        corrupted = json.loads(path.read_text(encoding="utf-8"))
        target = corrupted["fit"]["best"] if relative_path != "job/fit.json" else corrupted["best"]
        target["candidate_id"] = 999_999
        target["generation"] = 999_999
        path.write_text(json.dumps(corrupted), encoding="utf-8")
    manifest_path = Path(package.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path in ("job/fit.json", "job/validation.json"):
        manifest["artifacts"]["job_record_sha256"][relative_path] = sha256_file(package_root / relative_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(CalibrationPackageLoadError, match="evaluation bookkeeping does not follow"):
        VerifiedArticulationPackage.open(package.output_dir)
