from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from newton_calibration.adapters.surface import (
    CalibrationPackageLoadError,
    SO101EnvCfg,
    VerifiedCalibrationPackage,
    VerifiedSO101Package,
)

CANONICAL_JOINTS = ("rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw")
PARAMETERS = {
    "arm_armature": 0.006491857443545953,
    "arm_damping_scale": 20.0,
    "arm_effort_scale": 0.2,
    "arm_friction_nm": 0.3328502442500357,
    "arm_stiffness_scale": 8.0,
    "command_delay_s": 0.028078889494155723,
    "gripper_armature": 0.002139518124074991,
    "gripper_damping_scale": 0.25,
    "gripper_effort_scale": 0.012715129200380332,
    "gripper_friction_nm": 0.04357827196319325,
    "gripper_stiffness_scale": 2.523279682647735,
}
APPLICATION = {
    "isaaclab_explicit_pd": [
        "arm_stiffness_scale",
        "arm_damping_scale",
        "arm_effort_scale",
        "gripper_stiffness_scale",
        "gripper_damping_scale",
        "gripper_effort_scale",
    ],
    "newton_runtime": [
        "arm_armature",
        "arm_friction_nm",
        "gripper_armature",
        "gripper_friction_nm",
    ],
    "toolkit_replay": ["command_delay_s"],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "portable-package"
    root.mkdir(parents=True)
    asset = root / "so101.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    (root / "calibration.usda").write_text("#usda 1.0\n", encoding="utf-8")
    runtime = {
        "adapter": "isaaclab_newton",
        "asset_path": "/stale/build/path/so101.usda",
        "base_armature": 0.0,
        "base_damping": 0.017453292,
        "base_effort_limit": 10.0,
        "base_stiffness": 1.7453293,
        "device": "cuda:0",
        "dt": 1.0 / 120.0,
        "gravity": [0.0, 0.0, -9.81],
        # Deliberately alphabetical, matching write_json(sort_keys=True).
        "joint_map": {
            "elbow": "elbow_flex",
            "jaw": "gripper",
            "pitch": "shoulder_lift",
            "rotation": "shoulder_pan",
            "wrist_pitch": "wrist_flex",
            "wrist_roll": "wrist_roll",
        },
        "num_substeps": 1,
        "residual_model_path": None,
        "solver_iterations": 100,
        "solver_tolerance": 1e-6,
    }
    actual = {
        "arm_stiffness": runtime["base_stiffness"] * PARAMETERS["arm_stiffness_scale"],
        "arm_damping": runtime["base_damping"] * PARAMETERS["arm_damping_scale"],
        "arm_armature": PARAMETERS["arm_armature"],
        "arm_friction_nm": PARAMETERS["arm_friction_nm"],
        "arm_effort_limit": runtime["base_effort_limit"] * PARAMETERS["arm_effort_scale"],
        "gripper_stiffness": runtime["base_stiffness"] * PARAMETERS["gripper_stiffness_scale"],
        "gripper_damping": runtime["base_damping"] * PARAMETERS["gripper_damping_scale"],
        "gripper_armature": PARAMETERS["gripper_armature"],
        "gripper_friction_nm": PARAMETERS["gripper_friction_nm"],
        "gripper_effort_limit": runtime["base_effort_limit"] * PARAMETERS["gripper_effort_scale"],
    }
    (root / "isaaclab_actuator.yaml").write_text(
        yaml.safe_dump(
            {
                "recipe": "so101_actuator_dynamics.v1",
                "source_asset": "./so101.usda",
                "actuators": actual,
                "command_delay_s": PARAMETERS["command_delay_s"],
                "command_delay_steps_at_runtime_dt": 3,
                "command_delay_effective_s_at_runtime_dt": 0.025,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    validation = {
        "run_id": "so101-test",
        "passed": True,
        "improvement_pct": 92.1,
        "regressions": [],
        "gates": {
            "stable": True,
            "minimum_improvement": True,
            "no_large_episode_regression": True,
            "heldout_only": True,
        },
        "fit": {
            "plan": {
                "run_id": "so101-test",
                "recipe": "so101_actuator_dynamics.v1",
                "asset_fingerprint": _sha256(asset),
                "evidence_fingerprint": "a" * 64,
                "evidence_revision": "test-revision",
                "validation_gates": {
                    "minimum_improvement_pct": 30.0,
                    "maximum_episode_regression_pct": 10.0,
                },
                "environment": runtime,
            },
            "best": {"parameters": PARAMETERS},
        },
    }
    (root / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    manifest = {
        "schema": "newton.calibration.package/v1",
        "run_id": "so101-test",
        "status": "validated",
        "activation_allowed": True,
        "scope": "SO-101 free-space arm and unloaded gripper actuation",
        "claims": {
            "heldout_passed": True,
            "heldout_improvement_pct": 92.1,
            "contact_or_grip_force_validated": False,
        },
        "inputs": {
            "asset_sha256": _sha256(asset),
            "residual_sha256": None,
            "evidence_fingerprint": "a" * 64,
            "evidence_revision": "test-revision",
        },
        "runtime": runtime,
        "recipe": "so101_actuator_dynamics.v1",
        "parameters": PARAMETERS,
        "parameter_application": APPLICATION,
        "timing_quantization": {
            "requested_command_delay_s": PARAMETERS["command_delay_s"],
            "runtime_dt_s": runtime["dt"],
            "applied_delay_steps": 3,
            "effective_command_delay_s": 0.025,
        },
        "artifacts": {
            "source_asset": "so101.usda",
            "usd_provenance_layer": "calibration.usda",
            "isaaclab_actuator_config": "isaaclab_actuator.yaml",
            "validation": "validation.json",
            "actuator_residual": None,
        },
        "runtime_build": {"newton": "1.2.1", "isaaclab_newton_extension": "0.13.6"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _replace_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _open(root: Path) -> VerifiedSO101Package:
    return VerifiedSO101Package.open(root, expected_manifest_sha256=_sha256(root / "manifest.json"))


def test_loads_relocated_package_and_reconstructs_canonical_joint_order(tmp_path):
    root = _write_fixture(tmp_path)

    package = _open(root)
    env = package.to_env_cfg(device="cuda:7")
    vectors = package.actuator.joint_vectors()

    assert package.asset_path == root / "so101.usda"
    assert package.verification_level == "complete-v1"
    assert tuple(package.joint_map) == CANONICAL_JOINTS
    assert tuple(env.joint_map) == CANONICAL_JOINTS
    assert list(vectors["stiffness"]) == pytest.approx([13.9626344] * 5 + [4.403953962])
    assert list(vectors["damping"]) == pytest.approx([0.34906584] * 5 + [0.004363323])
    assert list(vectors["armature"]) == pytest.approx([0.006491857444] * 5 + [0.002139518124])
    assert list(vectors["friction_nm"]) == pytest.approx([0.3328502443] * 5 + [0.04357827196])
    assert list(vectors["effort_limit"]) == pytest.approx([2.0] * 5 + [0.127151292])
    assert package.actuator.command_delay_steps == 3
    assert package.actuator.effective_command_delay_s == pytest.approx(0.025)
    assert env.device == "cuda:7"
    assert env.usd_path == str(root / "so101.usda")
    assert env.calibrated_candidate() == PARAMETERS


def test_env_classmethod_is_kitless_and_preserves_locked_physics(tmp_path):
    root = _write_fixture(tmp_path)
    before = {name for name in sys.modules if name == "isaaclab" or name.startswith("isaaclab.")}

    env = SO101EnvCfg.from_calibration(
        root,
        device="cuda:1",
        expected_manifest_sha256=_sha256(root / "manifest.json"),
    )

    after = {name for name in sys.modules if name == "isaaclab" or name.startswith("isaaclab.")}
    assert after == before
    assert env.runtime == "isaaclab_newton"
    assert env.dt == pytest.approx(1.0 / 120.0)
    assert env.num_substeps == 1
    assert env.solver_iterations == 100
    assert env.solver_tolerance == pytest.approx(1e-6)
    assert env.device == "cuda:1"


def test_unattested_v1_requires_trust_on_every_public_loading_surface(tmp_path):
    root = _write_fixture(tmp_path)
    for loader in (VerifiedSO101Package.open, VerifiedCalibrationPackage.open, SO101EnvCfg.from_calibration):
        with pytest.raises(CalibrationPackageLoadError, match="Legacy unattested v1"):
            loader(root)


def test_environment_reverification_rejects_locked_field_override(tmp_path):
    root = _write_fixture(tmp_path)
    package = _open(root)
    environment = package.to_env_cfg().describe()
    package.assert_matches_environment(environment)

    with pytest.raises(CalibrationPackageLoadError, match="loaded dt mismatch"):
        package.assert_matches_environment(replace(environment, dt=1.0 / 60.0))

    reordered = dict(sorted(environment.joint_map.items()))
    assert reordered == environment.joint_map  # Same pairs; only semantic order differs.
    with pytest.raises(CalibrationPackageLoadError, match="joint map was changed"):
        package.assert_matches_environment(replace(environment, joint_map=reordered))


def test_trusted_manifest_digest_detects_manifest_replacement(tmp_path):
    root = _write_fixture(tmp_path)
    original = _open(root)
    manifest = _manifest(root)
    manifest["scope"] = "rewritten scope"
    _replace_manifest(root, manifest)

    with pytest.raises(CalibrationPackageLoadError, match="trusted digest"):
        VerifiedSO101Package.open(root, expected_manifest_sha256=original.manifest_sha256)


def test_legacy_portable_package_requires_external_trust_anchor(tmp_path):
    root = _write_fixture(tmp_path)
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["fit"]["plan"].pop("asset_fingerprint")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    trusted_digest = _sha256(root / "manifest.json")

    with pytest.raises(CalibrationPackageLoadError, match="Legacy unattested v1"):
        VerifiedSO101Package.open(root)

    package = VerifiedSO101Package.open(root, expected_manifest_sha256=trusted_digest)
    assert package.verification_level == "legacy-trusted-manifest"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update(activation_allowed=False), "not approved"),
        (lambda manifest: manifest["claims"].update(heldout_passed=False), "held-out"),
        (
            lambda manifest: manifest["claims"].update(contact_or_grip_force_validated=True),
            "disclaim contact",
        ),
        (lambda manifest: manifest["runtime"].update(adapter="analytic"), "runtime.adapter"),
        (lambda manifest: manifest.update(schema="newton.calibration.package/v2"), "manifest.schema"),
        (lambda manifest: manifest.update(scope="insertion transfer"), "manifest.scope"),
    ],
)
def test_rejects_nonactivatable_or_incompatible_manifest(tmp_path, mutation, message):
    root = _write_fixture(tmp_path)
    manifest = _manifest(root)
    mutation(manifest)
    _replace_manifest(root, manifest)

    with pytest.raises(CalibrationPackageLoadError, match=message):
        _open(root)


def test_rejects_asset_drift(tmp_path):
    root = _write_fixture(tmp_path)
    (root / "so101.usda").write_text("#usda 1.0\n# modified\n", encoding="utf-8")

    with pytest.raises(CalibrationPackageLoadError, match="source USD"):
        _open(root)


def test_rejects_yaml_projection_drift(tmp_path):
    root = _write_fixture(tmp_path)
    config = yaml.safe_load((root / "isaaclab_actuator.yaml").read_text(encoding="utf-8"))
    config["actuators"]["gripper_damping"] *= 2.0
    (root / "isaaclab_actuator.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(CalibrationPackageLoadError, match="gripper_damping mismatch"):
        _open(root)


def test_rejects_validation_drift(tmp_path):
    root = _write_fixture(tmp_path)
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    validation["gates"]["stable"] = False
    (root / "validation.json").write_text(json.dumps(validation), encoding="utf-8")

    with pytest.raises(CalibrationPackageLoadError, match="Validation must contain exactly"):
        _open(root)


def test_rejects_manifest_runtime_not_used_for_validation(tmp_path):
    root = _write_fixture(tmp_path)
    manifest = _manifest(root)
    manifest["runtime"]["solver_iterations"] = 50
    _replace_manifest(root, manifest)

    with pytest.raises(CalibrationPackageLoadError, match="runtime/environment"):
        _open(root)


def test_rejects_coordinated_asset_and_self_hash_substitution(tmp_path):
    root = _write_fixture(tmp_path)
    asset = root / "so101.usda"
    asset.write_text("#usda 1.0\n# substituted robot\n", encoding="utf-8")
    manifest = _manifest(root)
    manifest["inputs"]["asset_sha256"] = _sha256(asset)
    _replace_manifest(root, manifest)

    with pytest.raises(CalibrationPackageLoadError, match="Validation asset fingerprint"):
        _open(root)


def test_rejects_missing_fit_or_validation_gate(tmp_path):
    root = _write_fixture(tmp_path)
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.pop("fit")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(CalibrationPackageLoadError, match="missing its fit record"):
        _open(root)

    root = _write_fixture(tmp_path / "second")
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["gates"].pop("minimum_improvement")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(CalibrationPackageLoadError, match="exactly the passing MVP1 gates"):
        _open(root)


def test_rejects_artifact_traversal_and_symlinks(tmp_path):
    root = _write_fixture(tmp_path)
    manifest = _manifest(root)
    manifest["artifacts"]["source_asset"] = "../outside.usda"
    _replace_manifest(root, manifest)
    (tmp_path / "outside.usda").write_text("#usda 1.0\n", encoding="utf-8")
    with pytest.raises(CalibrationPackageLoadError, match="confined"):
        _open(root)

    root = _write_fixture(tmp_path / "second")
    (root / "so101.usda").unlink()
    (root / "so101.usda").symlink_to(tmp_path / "outside.usda")
    with pytest.raises(CalibrationPackageLoadError, match="symbolic link"):
        _open(root)


def test_rejects_duplicate_yaml_keys(tmp_path):
    root = _write_fixture(tmp_path)
    yaml_path = root / "isaaclab_actuator.yaml"
    yaml_path.write_text(yaml_path.read_text(encoding="utf-8") + "recipe: duplicate\n", encoding="utf-8")

    with pytest.raises(CalibrationPackageLoadError, match="Duplicate YAML key"):
        _open(root)


def test_rejects_parameter_assigned_to_two_surfaces(tmp_path):
    root = _write_fixture(tmp_path)
    manifest = _manifest(root)
    manifest["parameter_application"]["toolkit_replay"].append("arm_armature")
    _replace_manifest(root, manifest)

    with pytest.raises(CalibrationPackageLoadError, match="does not match the supported package binding"):
        _open(root)


def test_rejects_legacy_nonportable_v1_package(tmp_path):
    root = _write_fixture(tmp_path)
    manifest = _manifest(root)
    manifest["artifacts"].pop("source_asset")
    manifest["artifacts"]["usd_overlay"] = "/old/absolute/path/calibration.usda"
    _replace_manifest(root, manifest)

    with pytest.raises(CalibrationPackageLoadError, match="migrate legacy packages"):
        _open(root)


def test_loads_and_hash_checks_optional_actuator_residual(tmp_path):
    root = _write_fixture(tmp_path)
    residual_path = root / "actuator_residual.json"
    residual_path.write_text(
        json.dumps(
            {
                "schema": "newton.calibration.actuator-residual/v1",
                "joint_names": list(CANONICAL_JOINTS),
                "weights": [[0.0, 0.0, 0.0, 0.0] for _ in CANONICAL_JOINTS],
                "clip_nm": [0.1 for _ in CANONICAL_JOINTS],
            }
        ),
        encoding="utf-8",
    )
    manifest = _manifest(root)
    manifest["artifacts"]["actuator_residual"] = residual_path.name
    manifest["inputs"]["residual_sha256"] = _sha256(residual_path)
    manifest["runtime"]["residual_model_path"] = "/stale/original/residual.json"
    _replace_manifest(root, manifest)
    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["fit"]["plan"]["environment"]["residual_model_path"] = "/stale/original/residual.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    package = _open(root)
    assert package.residual_model_path == residual_path
    assert package.to_env_cfg().residual_model_path == str(residual_path)

    residual_path.write_text("{}", encoding="utf-8")
    with pytest.raises(CalibrationPackageLoadError, match="residual does not match"):
        _open(root)
