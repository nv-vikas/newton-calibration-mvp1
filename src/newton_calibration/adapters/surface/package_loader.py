from __future__ import annotations

import hmac
import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from newton_calibration.actuators import load_residual
from newton_calibration.adapters.evidence.anchor_lab_so101 import SO101_JOINTS
from newton_calibration.adapters.runtime.analytic import _joint_properties, _resolve_joint_layout
from newton_calibration.core.attestation import (
    evaluation_result_fingerprint,
    numeric_surface_fingerprint,
    parameter_fingerprint,
    record_fingerprint,
)
from newton_calibration.core.evidence_spec import BoundEvidenceSpec
from newton_calibration.core.fit_journal import FitJournal, FitJournalError
from newton_calibration.core.io import sha256_file
from newton_calibration.core.models import EnvironmentSpec, jsonable
from newton_calibration.recipes import get_recipe

_PACKAGE_SCHEMA = "newton.calibration.package/v1"
_RECIPE = "so101_actuator_dynamics.v1"
_RUNTIME = "isaaclab_newton"
_SCOPE = "SO-101 free-space arm and unloaded gripper actuation"
_VALIDATION_GATES = {
    "stable",
    "minimum_improvement",
    "no_large_episode_regression",
    "heldout_only",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_VALIDATION_BYTES = 32 * 1024 * 1024
_MAX_YAML_BYTES = 256 * 1024
_ACTUATOR_KEYS = (
    "arm_stiffness",
    "arm_damping",
    "arm_armature",
    "arm_friction_nm",
    "arm_effort_limit",
    "gripper_stiffness",
    "gripper_damping",
    "gripper_armature",
    "gripper_friction_nm",
    "gripper_effort_limit",
)
_EXPECTED_APPLICATION = {
    "isaaclab_explicit_pd": {
        "arm_stiffness_scale",
        "arm_damping_scale",
        "arm_effort_scale",
        "gripper_stiffness_scale",
        "gripper_damping_scale",
        "gripper_effort_scale",
    },
    "newton_runtime": {
        "arm_armature",
        "arm_friction_nm",
        "gripper_armature",
        "gripper_friction_nm",
    },
    "toolkit_replay": {"command_delay_s"},
}


class CalibrationPackageLoadError(ValueError):
    """Raised when a package cannot safely configure an Isaac Lab run."""


@dataclass(frozen=True)
class SO101ActuatorSettings:
    """The fully resolved values Isaac Lab/Newton must apply for this package."""

    arm_stiffness: float
    arm_damping: float
    arm_armature: float
    arm_friction_nm: float
    arm_effort_limit: float
    gripper_stiffness: float
    gripper_damping: float
    gripper_armature: float
    gripper_friction_nm: float
    gripper_effort_limit: float
    requested_command_delay_s: float
    command_delay_steps: int
    effective_command_delay_s: float

    def joint_vectors(self) -> dict[str, tuple[float, ...]]:
        """Return values in canonical evidence/robot order, with jaw last."""

        return {
            "stiffness": (self.arm_stiffness,) * 5 + (self.gripper_stiffness,),
            "damping": (self.arm_damping,) * 5 + (self.gripper_damping,),
            "armature": (self.arm_armature,) * 5 + (self.gripper_armature,),
            "friction_nm": (self.arm_friction_nm,) * 5 + (self.gripper_friction_nm,),
            "effort_limit": (self.arm_effort_limit,) * 5 + (self.gripper_effort_limit,),
        }


@dataclass(frozen=True)
class VerifiedSO101Package:
    """A validated, internally consistent, relocatable SO-101 package.

    Opening is intentionally kitless: Isaac Lab is imported only if the caller
    subsequently constructs an environment/runtime.
    """

    root: Path
    manifest_path: Path
    manifest_sha256: str
    verification_level: str
    run_id: str
    recipe: str
    scope: str
    asset_path: Path
    provenance_path: Path | None
    actuator_config_path: Path
    validation_path: Path
    residual_model_path: Path | None
    runtime_adapter: str
    runtime_device: str
    dt: float
    gravity: tuple[float, float, float]
    joint_map_items: tuple[tuple[str, str], ...]
    base_stiffness: float
    base_damping: float
    base_effort_limit: float
    base_armature: float
    num_substeps: int
    solver_iterations: int
    solver_tolerance: float
    parameter_items: tuple[tuple[str, float], ...]
    actuator: SO101ActuatorSettings
    heldout_improvement_pct: float
    runtime_build_items: tuple[tuple[str, str], ...]

    @classmethod
    def open(
        cls,
        package: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> VerifiedSO101Package:
        root = Path(package).expanduser().resolve()
        if not root.is_dir():
            raise CalibrationPackageLoadError(f"Calibration package directory does not exist: {root}")
        manifest_path = _confined_file(root, "manifest.json", "manifest")
        manifest_sha256 = sha256_file(manifest_path)
        if expected_manifest_sha256 is not None:
            expected = _sha256_text(expected_manifest_sha256, "expected manifest SHA-256")
            if not hmac.compare_digest(manifest_sha256, expected):
                raise CalibrationPackageLoadError("Calibration manifest SHA-256 does not match the trusted digest")
        manifest = _read_json_object(manifest_path, maximum_bytes=_MAX_MANIFEST_BYTES)
        has_execution_attestation = (
            "activation_checks" in manifest and "runtime_attestation" in manifest
        )
        if not has_execution_attestation and expected_manifest_sha256 is None:
            raise CalibrationPackageLoadError(
                "Legacy unattested v1 packages require a separately trusted manifest SHA-256; "
                "regenerate them for execution-bound verification"
            )
        _require_equal(manifest, "schema", _PACKAGE_SCHEMA, "manifest")
        _require_equal(manifest, "recipe", _RECIPE, "manifest")
        _require_equal(manifest, "status", "validated", "manifest")
        if manifest.get("activation_allowed") is not True:
            raise CalibrationPackageLoadError("Package is not approved for activation")
        run_id = _text(manifest.get("run_id"), "manifest.run_id")
        _require_equal(manifest, "scope", _SCOPE, "manifest")
        scope = _SCOPE

        claims = _mapping(manifest.get("claims"), "manifest.claims")
        if claims.get("heldout_passed") is not True:
            raise CalibrationPackageLoadError("Package does not claim passed held-out validation")
        if claims.get("contact_or_grip_force_validated") is not False:
            raise CalibrationPackageLoadError("MVP1 package must explicitly disclaim contact and grip-force validation")
        improvement = _finite_number(claims.get("heldout_improvement_pct"), "claims.heldout_improvement_pct")

        runtime = _mapping(manifest.get("runtime"), "manifest.runtime")
        _require_equal(runtime, "adapter", _RUNTIME, "manifest.runtime")
        dt = _positive_number(runtime.get("dt"), "runtime.dt")
        gravity_value = runtime.get("gravity")
        if not isinstance(gravity_value, list) or len(gravity_value) != 3:
            raise CalibrationPackageLoadError("runtime.gravity must contain exactly three values")
        gravity = tuple(_finite_number(value, f"runtime.gravity[{index}]") for index, value in enumerate(gravity_value))
        raw_joint_map = _mapping(runtime.get("joint_map"), "runtime.joint_map")
        if set(raw_joint_map) != set(SO101_JOINTS):
            raise CalibrationPackageLoadError(
                f"runtime.joint_map must contain exactly the canonical SO-101 joints: {SO101_JOINTS}"
            )
        joint_map_items = tuple(
            (joint, _text(raw_joint_map[joint], f"runtime.joint_map.{joint}")) for joint in SO101_JOINTS
        )
        if len({target for _, target in joint_map_items}) != len(joint_map_items):
            raise CalibrationPackageLoadError("runtime.joint_map contains duplicate physical joints")

        base_stiffness = _positive_number(runtime.get("base_stiffness"), "runtime.base_stiffness")
        base_damping = _nonnegative_number(runtime.get("base_damping"), "runtime.base_damping")
        base_effort_limit = _positive_number(runtime.get("base_effort_limit"), "runtime.base_effort_limit")
        base_armature = _nonnegative_number(runtime.get("base_armature"), "runtime.base_armature")
        num_substeps = _positive_integer(runtime.get("num_substeps"), "runtime.num_substeps")
        solver_iterations = _positive_integer(runtime.get("solver_iterations"), "runtime.solver_iterations")
        solver_tolerance = _positive_number(runtime.get("solver_tolerance"), "runtime.solver_tolerance")
        runtime_device = _text(runtime.get("device"), "runtime.device")

        parameters = _verified_parameters(manifest.get("parameters"))
        parameter_application = _mapping(manifest.get("parameter_application"), "manifest.parameter_application")
        _verify_parameter_application(parameter_application, set(parameters))

        artifacts = _mapping(manifest.get("artifacts"), "manifest.artifacts")
        asset_path = _artifact(root, artifacts, "source_asset")
        if asset_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise CalibrationPackageLoadError(f"Packaged source asset has unsupported extension: {asset_path.suffix}")
        inputs = _mapping(manifest.get("inputs"), "manifest.inputs")
        expected_asset_sha256 = _sha256_text(inputs.get("asset_sha256"), "inputs.asset_sha256")
        evidence_fingerprint = _sha256_text(inputs.get("evidence_fingerprint"), "inputs.evidence_fingerprint")
        evidence_revision = _text(inputs.get("evidence_revision"), "inputs.evidence_revision")
        if not hmac.compare_digest(sha256_file(asset_path), expected_asset_sha256):
            raise CalibrationPackageLoadError("Packaged source USD does not match inputs.asset_sha256")

        actuator_config_path = _artifact(root, artifacts, "isaaclab_actuator_config")
        validation_path = _artifact(root, artifacts, "validation")
        provenance_path = _optional_artifact(root, artifacts, "usd_provenance_layer")
        residual_model_path = _optional_artifact(root, artifacts, "actuator_residual")
        expected_residual_sha256 = inputs.get("residual_sha256")
        if residual_model_path is None:
            if expected_residual_sha256 is not None:
                raise CalibrationPackageLoadError("inputs.residual_sha256 is set but no residual artifact is present")
        else:
            residual_sha256 = _sha256_text(expected_residual_sha256, "inputs.residual_sha256")
            if not hmac.compare_digest(sha256_file(residual_model_path), residual_sha256):
                raise CalibrationPackageLoadError("Actuator residual does not match inputs.residual_sha256")
            try:
                residual = load_residual(residual_model_path, list(SO101_JOINTS))
            except (KeyError, TypeError, ValueError) as exc:
                raise CalibrationPackageLoadError(f"Invalid packaged actuator residual: {exc}") from exc
            if residual is None or not np.isfinite(residual.weights).all() or not np.isfinite(residual.clip_nm).all():
                raise CalibrationPackageLoadError("Packaged actuator residual must contain finite values")
            if np.any(residual.clip_nm < 0.0):
                raise CalibrationPackageLoadError("Packaged actuator residual clip values must be non-negative")

        timing = _mapping(manifest.get("timing_quantization"), "manifest.timing_quantization")
        requested_delay = _nonnegative_number(
            timing.get("requested_command_delay_s"),
            "timing_quantization.requested_command_delay_s",
        )
        timing_dt = _positive_number(timing.get("runtime_dt_s"), "timing_quantization.runtime_dt_s")
        delay_steps = _nonnegative_integer(
            timing.get("applied_delay_steps"),
            "timing_quantization.applied_delay_steps",
        )
        effective_delay = _nonnegative_number(
            timing.get("effective_command_delay_s"),
            "timing_quantization.effective_command_delay_s",
        )
        _close(timing_dt, dt, "timing runtime dt", relative_tolerance=1e-12)
        _close(requested_delay, parameters["command_delay_s"], "requested command delay")
        expected_steps = max(0, round(requested_delay / dt))
        if delay_steps != expected_steps:
            raise CalibrationPackageLoadError(
                f"Applied command delay is {delay_steps} steps; expected {expected_steps} at the locked dt"
            )
        _close(effective_delay, delay_steps * dt, "effective command delay")

        expected_actuators = {
            "arm_stiffness": base_stiffness * parameters["arm_stiffness_scale"],
            "arm_damping": base_damping * parameters["arm_damping_scale"],
            "arm_armature": parameters["arm_armature"],
            "arm_friction_nm": parameters["arm_friction_nm"],
            "arm_effort_limit": base_effort_limit * parameters["arm_effort_scale"],
            "gripper_stiffness": base_stiffness * parameters["gripper_stiffness_scale"],
            "gripper_damping": base_damping * parameters["gripper_damping_scale"],
            "gripper_armature": parameters["gripper_armature"],
            "gripper_friction_nm": parameters["gripper_friction_nm"],
            "gripper_effort_limit": base_effort_limit * parameters["gripper_effort_scale"],
        }
        yaml_payload = _read_yaml_object(actuator_config_path)
        expected_yaml_keys = {
            "recipe",
            "source_asset",
            "actuators",
            "command_delay_s",
            "command_delay_steps_at_runtime_dt",
            "command_delay_effective_s_at_runtime_dt",
        }
        if set(yaml_payload) != expected_yaml_keys:
            raise CalibrationPackageLoadError(f"Actuator YAML must contain exactly: {sorted(expected_yaml_keys)}")
        _require_equal(yaml_payload, "recipe", _RECIPE, "actuator YAML")
        yaml_asset = _confined_file(root, yaml_payload.get("source_asset"), "actuator YAML source_asset")
        if yaml_asset != asset_path:
            raise CalibrationPackageLoadError("Actuator YAML source_asset does not match the manifest source asset")
        yaml_actuators = _mapping(yaml_payload.get("actuators"), "actuator YAML actuators")
        if set(yaml_actuators) != set(_ACTUATOR_KEYS):
            raise CalibrationPackageLoadError(f"Actuator YAML must contain exactly: {list(_ACTUATOR_KEYS)}")
        for name, expected_value in expected_actuators.items():
            _close(
                _nonnegative_number(yaml_actuators.get(name), f"actuator YAML actuators.{name}"),
                expected_value,
                f"actuator YAML {name}",
            )
        _close(
            _nonnegative_number(yaml_payload.get("command_delay_s"), "actuator YAML command_delay_s"),
            requested_delay,
            "actuator YAML requested command delay",
        )
        yaml_steps = _nonnegative_integer(
            yaml_payload.get("command_delay_steps_at_runtime_dt"),
            "actuator YAML command_delay_steps_at_runtime_dt",
        )
        if yaml_steps != delay_steps:
            raise CalibrationPackageLoadError("Actuator YAML command-delay step count does not match the manifest")
        _close(
            _nonnegative_number(
                yaml_payload.get("command_delay_effective_s_at_runtime_dt"),
                "actuator YAML command_delay_effective_s_at_runtime_dt",
            ),
            effective_delay,
            "actuator YAML effective command delay",
        )

        validation = _read_json_object(validation_path, maximum_bytes=_MAX_VALIDATION_BYTES)
        if validation.get("run_id") != run_id:
            raise CalibrationPackageLoadError("Validation and manifest run IDs do not match")
        if validation.get("passed") is not True:
            raise CalibrationPackageLoadError("Packaged validation did not pass")
        gates = _mapping(validation.get("gates"), "validation.gates")
        if set(gates) != _VALIDATION_GATES or any(value is not True for value in gates.values()):
            raise CalibrationPackageLoadError(
                f"Validation must contain exactly the passing MVP1 gates: {sorted(_VALIDATION_GATES)}"
            )
        regressions = validation.get("regressions")
        if regressions != []:
            raise CalibrationPackageLoadError("Packaged validation must contain no held-out episode regressions")
        _close(
            _finite_number(validation.get("improvement_pct"), "validation.improvement_pct"),
            improvement,
            "validation improvement",
        )
        verification_level = _verify_validation_fit(
            validation,
            run_id=run_id,
            recipe=_RECIPE,
            manifest_runtime=runtime,
            parameters=parameters,
            asset_sha256=expected_asset_sha256,
            evidence_fingerprint=evidence_fingerprint,
            evidence_revision=evidence_revision,
            improvement=improvement,
            trusted_manifest=expected_manifest_sha256 is not None,
        )
        if "activation_checks" in manifest or "runtime_attestation" in manifest:
            if "activation_checks" not in manifest or "runtime_attestation" not in manifest:
                raise CalibrationPackageLoadError("Attested v1 packages require both activation checks and attestations")
            locked_environment = _generic_environment(runtime)
            attested_residual = (
                _sha256_text(expected_residual_sha256, "inputs.residual_sha256")
                if expected_residual_sha256 is not None
                else None
            )
            bundle = _mapping(manifest.get("runtime_attestation"), "manifest.runtime_attestation")
            if set(bundle) != {"fit", "heldout_validation"}:
                raise CalibrationPackageLoadError("Runtime attestation must record fit and held-out validation")
            fit_attestation = _mapping(bundle.get("fit"), "runtime_attestation.fit")
            heldout_attestation = _mapping(
                bundle.get("heldout_validation"),
                "runtime_attestation.heldout_validation",
            )
            validation_fit = _mapping(validation.get("fit"), "validation.fit")
            if validation_fit.get("backend") != "isaaclab_newton":
                raise CalibrationPackageLoadError("Validation fit was not produced by the Newton runtime")
            if _mapping(validation_fit.get("runtime_attestation"), "validation.fit.runtime_attestation") != fit_attestation:
                raise CalibrationPackageLoadError("Validation fit attestation does not match the manifest")
            if (
                _mapping(validation.get("runtime_attestation"), "validation.runtime_attestation")
                != heldout_attestation
            ):
                raise CalibrationPackageLoadError("Held-out validation attestation does not match the manifest")
            locked_plan = _mapping(validation_fit.get("plan"), "validation.fit.plan")
            plan_sha256 = record_fingerprint(locked_plan)
            mapping_fingerprint = str(
                _mapping(locked_plan.get("evidence_spec", {}), "validation.fit.plan.evidence_spec").get(
                    "mapping_fingerprint", ""
                )
            )
            train_names = _text_list(locked_plan.get("train_episodes"), "validation.fit.plan.train_episodes")
            heldout_names = _text_list(
                locked_plan.get("heldout_episodes"), "validation.fit.plan.heldout_episodes"
            )
            fit_inputs = _verified_attested_episode_inputs(fit_attestation, train_names, "train")
            heldout_bundle = _mapping(heldout_attestation, "runtime_attestation.heldout_validation")
            if set(heldout_bundle) != {"baseline", "calibrated"}:
                raise CalibrationPackageLoadError("Held-out attestation must contain baseline and calibrated runs")
            heldout_baseline = _mapping(heldout_bundle["baseline"], "heldout attestation baseline")
            heldout_calibrated = _mapping(heldout_bundle["calibrated"], "heldout attestation calibrated")
            heldout_inputs = _verified_attested_episode_inputs(heldout_calibrated, heldout_names, "heldout")
            if _verified_attested_episode_inputs(heldout_baseline, heldout_names, "heldout") != heldout_inputs:
                raise CalibrationPackageLoadError("Held-out baseline and calibrated runs used different evidence")
            fit_best = _mapping(validation_fit.get("best"), "validation.fit.best")
            fit_baseline = _mapping(validation_fit.get("baseline"), "validation.fit.baseline")
            per_episode = _mapping(validation.get("per_episode"), "validation.per_episode")
            before = {
                name: _mapping(_mapping(per_episode[name], f"per_episode.{name}").get("baseline"),
                               f"per_episode.{name}.baseline")
                for name in heldout_names
            }
            after = {
                name: _mapping(_mapping(per_episode[name], f"per_episode.{name}").get("calibrated"),
                               f"per_episode.{name}.calibrated")
                for name in heldout_names
            }
            baseline_metrics = _mapping(validation.get("baseline_metrics"), "validation.baseline_metrics")
            calibrated_metrics = _mapping(validation.get("calibrated_metrics"), "validation.calibrated_metrics")
            baseline_stable = validation.get("baseline_stable")
            calibrated_stable = validation.get("calibrated_stable")
            if not isinstance(baseline_stable, bool) or not isinstance(calibrated_stable, bool):
                raise CalibrationPackageLoadError("Attested validation must record both stability values")
            common = {
                "environment": locked_environment,
                "asset_sha256": expected_asset_sha256,
                "residual_sha256": attested_residual,
                "run_id": run_id,
                "plan_sha256": plan_sha256,
                "evidence_fingerprint": evidence_fingerprint,
                "mapping_fingerprint": mapping_fingerprint,
            }
            expected_fit_attestation = _expected_articulation_attestation(
                parameters=_mapping(fit_best.get("parameters"), "validation.fit.best.parameters"),
                phase="fit-selected",
                evidence_episodes=fit_inputs,
                result_sha256=evaluation_result_fingerprint(
                    score=fit_best.get("score"),
                    metrics=_mapping(fit_best.get("metrics"), "validation.fit.best.metrics"),
                    episodes=_mapping(fit_best.get("episodes"), "validation.fit.best.episodes"),
                    stable=fit_best.get("stable"),
                ),
                **common,
            )
            expected_heldout = {
                "baseline": _expected_articulation_attestation(
                    parameters=_mapping(fit_baseline.get("parameters"), "validation.fit.baseline.parameters"),
                    phase="heldout-baseline",
                    evidence_episodes=heldout_inputs,
                    result_sha256=evaluation_result_fingerprint(
                        score=baseline_metrics.get("score"), metrics=baseline_metrics,
                        episodes=before, stable=baseline_stable,
                    ),
                    **common,
                ),
                "calibrated": _expected_articulation_attestation(
                    parameters=parameters,
                    phase="heldout-validation",
                    evidence_episodes=heldout_inputs,
                    result_sha256=evaluation_result_fingerprint(
                        score=calibrated_metrics.get("score"), metrics=calibrated_metrics,
                        episodes=after, stable=calibrated_stable,
                    ),
                    **common,
                ),
            }
            if fit_attestation != expected_fit_attestation or heldout_attestation != expected_heldout:
                raise CalibrationPackageLoadError("Package lacks run-bound Newton evaluation attestations")
            checks = _mapping(manifest.get("activation_checks"), "manifest.activation_checks")
            for check in (
                "heldout_validation_passed",
                "authoritative_fit_attestation",
                "authoritative_validation_attestation",
                "self_contained_asset",
            ):
                if checks.get(check) is not True:
                    raise CalibrationPackageLoadError(f"Activation check failed: {check}")
            self_contained, _ = _asset_is_self_contained(asset_path)
            if not self_contained:
                raise CalibrationPackageLoadError("Packaged USD is not provably self-contained")

        actuator = SO101ActuatorSettings(
            **expected_actuators,
            requested_command_delay_s=requested_delay,
            command_delay_steps=delay_steps,
            effective_command_delay_s=effective_delay,
        )
        runtime_build = manifest.get("runtime_build", {})
        runtime_build_mapping = _mapping(runtime_build, "manifest.runtime_build") if runtime_build else {}
        runtime_build_items = tuple(
            sorted((str(key), _text(value, f"runtime_build.{key}")) for key, value in runtime_build_mapping.items())
        )
        return cls(
            root=root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            verification_level=verification_level,
            run_id=run_id,
            recipe=_RECIPE,
            scope=scope,
            asset_path=asset_path,
            provenance_path=provenance_path,
            actuator_config_path=actuator_config_path,
            validation_path=validation_path,
            residual_model_path=residual_model_path,
            runtime_adapter=_RUNTIME,
            runtime_device=runtime_device,
            dt=dt,
            gravity=gravity,
            joint_map_items=joint_map_items,
            base_stiffness=base_stiffness,
            base_damping=base_damping,
            base_effort_limit=base_effort_limit,
            base_armature=base_armature,
            num_substeps=num_substeps,
            solver_iterations=solver_iterations,
            solver_tolerance=solver_tolerance,
            parameter_items=tuple(
                (parameter.name, parameters[parameter.name]) for parameter in get_recipe(_RECIPE).parameters
            ),
            actuator=actuator,
            heldout_improvement_pct=improvement,
            runtime_build_items=runtime_build_items,
        )

    @property
    def parameters(self) -> dict[str, float]:
        return dict(self.parameter_items)

    @property
    def joint_map(self) -> dict[str, str]:
        return dict(self.joint_map_items)

    @property
    def runtime_build(self) -> dict[str, str]:
        return dict(self.runtime_build_items)

    def reverify(self) -> None:
        """Close the common verify/use gap immediately before creating an environment."""

        VerifiedSO101Package.open(
            self.root,
            expected_manifest_sha256=self.manifest_sha256,
        )

    def assert_matches_environment(self, environment: Any) -> None:
        """Reject mutation of any validated field after package loading."""

        checks = {
            "runtime adapter": (getattr(environment, "adapter", None), self.runtime_adapter),
            "asset": (
                str(Path(getattr(environment, "asset_path", "")).expanduser().resolve()),
                str(self.asset_path),
            ),
            "gravity": (tuple(getattr(environment, "gravity", ())), self.gravity),
            "joint map": (tuple(getattr(environment, "joint_map", {}).items()), self.joint_map_items),
            "substeps": (getattr(environment, "num_substeps", None), self.num_substeps),
            "solver iterations": (getattr(environment, "solver_iterations", None), self.solver_iterations),
            "calibration run": (getattr(environment, "calibration_run_id", None), self.run_id),
            "calibration manifest": (
                str(Path(getattr(environment, "calibration_manifest_path", "")).expanduser().resolve()),
                str(self.manifest_path),
            ),
            "calibration parameters": (
                dict(getattr(environment, "calibration_parameters", {})),
                self.parameters,
            ),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                raise CalibrationPackageLoadError(f"Loaded {label} was changed after package verification")
        numeric_checks = {
            "dt": (getattr(environment, "dt", None), self.dt),
            "base stiffness": (getattr(environment, "base_stiffness", None), self.base_stiffness),
            "base damping": (getattr(environment, "base_damping", None), self.base_damping),
            "base effort limit": (getattr(environment, "base_effort_limit", None), self.base_effort_limit),
            "base armature": (getattr(environment, "base_armature", None), self.base_armature),
            "solver tolerance": (getattr(environment, "solver_tolerance", None), self.solver_tolerance),
        }
        for label, (actual, expected) in numeric_checks.items():
            _close(
                _finite_number(actual, f"environment {label}"), expected, f"loaded {label}", relative_tolerance=1e-12
            )
        manifest_digest = getattr(environment, "calibration_manifest_sha256", None)
        if manifest_digest != self.manifest_sha256:
            raise CalibrationPackageLoadError("Loaded manifest digest was changed after package verification")
        expected_residual = str(self.residual_model_path) if self.residual_model_path else None
        if getattr(environment, "residual_model_path", None) != expected_residual:
            raise CalibrationPackageLoadError("Loaded residual path was changed after package verification")
        expected_residual_sha256 = sha256_file(self.residual_model_path) if self.residual_model_path else None
        if getattr(environment, "residual_model_sha256", None) != expected_residual_sha256:
            raise CalibrationPackageLoadError("Loaded residual fingerprint was changed after package verification")

    def to_env_cfg(self, *, device: str | None = None):
        """Construct the complete serializable surface used by the Newton adapter."""

        from .isaac_lab import SO101EnvCfg

        self.reverify()
        return SO101EnvCfg(
            usd_path=str(self.asset_path),
            runtime=self.runtime_adapter,
            device=device or self.runtime_device,
            dt=self.dt,
            gravity=self.gravity,
            joint_map=self.joint_map,
            base_stiffness=self.base_stiffness,
            base_damping=self.base_damping,
            base_effort_limit=self.base_effort_limit,
            base_armature=self.base_armature,
            num_substeps=self.num_substeps,
            solver_iterations=self.solver_iterations,
            solver_tolerance=self.solver_tolerance,
            residual_model_path=str(self.residual_model_path) if self.residual_model_path else None,
            residual_model_sha256=(sha256_file(self.residual_model_path) if self.residual_model_path else None),
            calibration_run_id=self.run_id,
            calibration_manifest_path=str(self.manifest_path),
            calibration_manifest_sha256=self.manifest_sha256,
            calibration_parameters=self.parameters,
        )


@dataclass(frozen=True)
class ArticulationActuatorSettings:
    """Resolved per-coordinate settings from a generic v2 package."""

    ordered_joint_items: tuple[tuple[str, str, str, float, float, float, float, float], ...]
    requested_command_delay_s: float
    command_delay_steps: int
    effective_command_delay_s: float

    def joint_vectors(self) -> dict[str, tuple[float, ...]]:
        return {
            "stiffness": tuple(item[3] for item in self.ordered_joint_items),
            "damping": tuple(item[4] for item in self.ordered_joint_items),
            "effort_limit": tuple(item[5] for item in self.ordered_joint_items),
            "armature": tuple(item[6] for item in self.ordered_joint_items),
            "friction_nm": tuple(item[7] for item in self.ordered_joint_items),
        }


@dataclass(frozen=True)
class VerifiedArticulationPackage:
    """Verified generic articulation package; Isaac Lab import remains lazy."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    run_id: str
    recipe: str
    scope: str
    asset_path: Path
    validation_path: Path
    residual_model_path: Path | None
    environment: EnvironmentSpec
    parameters: dict[str, float]
    actuator: ArticulationActuatorSettings
    heldout_improvement_pct: float

    @classmethod
    def open(
        cls,
        package: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> VerifiedArticulationPackage:
        root = Path(package).expanduser().resolve()
        if not root.is_dir():
            raise CalibrationPackageLoadError(f"Calibration package directory does not exist: {root}")
        manifest_path = _confined_file(root, "manifest.json", "manifest")
        manifest_sha256 = sha256_file(manifest_path)
        if expected_manifest_sha256 is not None:
            expected = _sha256_text(expected_manifest_sha256, "expected manifest SHA-256")
            if not hmac.compare_digest(manifest_sha256, expected):
                raise CalibrationPackageLoadError("Calibration manifest SHA-256 does not match the trusted digest")
        manifest = _read_json_object(manifest_path, maximum_bytes=_MAX_MANIFEST_BYTES)
        _require_equal(manifest, "schema", "newton.calibration.package/v2", "manifest")
        _require_equal(manifest, "status", "validated", "manifest")
        if manifest.get("activation_allowed") is not True:
            raise CalibrationPackageLoadError("Package is not approved for activation")
        run_id = _text(manifest.get("run_id"), "manifest.run_id")
        recipe_name = _text(manifest.get("recipe"), "manifest.recipe")
        scope = _text(manifest.get("scope"), "manifest.scope")

        runtime_payload = _mapping(manifest.get("runtime"), "manifest.runtime")
        environment = _generic_environment(runtime_payload)
        if environment.adapter != "isaaclab_newton":
            raise CalibrationPackageLoadError("Generic activation requires the isaaclab_newton runtime")
        if not environment.joint_groups or not environment.profile_confirmed:
            raise CalibrationPackageLoadError("Generic package requires a confirmed non-empty robot profile")
        if not environment.controller_profile_confirmed or not environment.controller_profile_source.strip():
            raise CalibrationPackageLoadError("Generic package requires a confirmed controller profile with provenance")
        expected_scope = f"{environment.robot_id} free-space position-controlled articulation dynamics"
        if scope != expected_scope:
            raise CalibrationPackageLoadError("Generic package scope does not match its robot profile")
        logical_order = list(environment.joint_order) or [
            joint for members in environment.joint_groups.values() for joint in members
        ]
        if set(logical_order) != set(environment.joint_map) or len(logical_order) != len(set(logical_order)):
            raise CalibrationPackageLoadError("Runtime joint groups and real-to-USD joint map are inconsistent")
        controlled = set(logical_order)
        for field_name in (
            "base_stiffness_by_joint",
            "base_damping_by_joint",
            "base_effort_limit_by_joint",
        ):
            if set(getattr(environment, field_name)) != controlled:
                raise CalibrationPackageLoadError(
                    f"Runtime controller profile {field_name} must cover every controlled joint"
                )

        recipe_cfg = get_recipe(recipe_name, environment)
        expected_parameters = {item.name: item for item in recipe_cfg.parameters}
        if set(recipe_cfg.required_parameter_names) != set(expected_parameters):
            raise CalibrationPackageLoadError("Robot profile does not declare safe bounds for the full recipe surface")
        parameter_payload = _mapping(manifest.get("parameters"), "manifest.parameters")
        if set(parameter_payload) != set(expected_parameters):
            raise CalibrationPackageLoadError(f"manifest.parameters must contain exactly: {list(expected_parameters)}")
        parameters: dict[str, float] = {}
        for name, spec in expected_parameters.items():
            value = _finite_number(parameter_payload[name], f"parameters.{name}")
            if not spec.lower <= value <= spec.upper:
                raise CalibrationPackageLoadError(
                    f"parameters.{name}={value} is outside recipe bounds [{spec.lower}, {spec.upper}]"
                )
            parameters[name] = value
        expected_application: dict[str, list[str]] = {}
        for parameter in recipe_cfg.parameters:
            expected_application.setdefault(parameter.owner, []).append(parameter.name)
        application = _mapping(manifest.get("parameter_application"), "manifest.parameter_application")
        if application != expected_application:
            raise CalibrationPackageLoadError("Generic parameter_application does not match recipe ownership")

        artifacts = _mapping(manifest.get("artifacts"), "manifest.artifacts")
        _verify_job_record_integrity(root, artifacts)
        asset_path = _artifact(root, artifacts, "source_asset")
        inputs = _mapping(manifest.get("inputs"), "manifest.inputs")
        asset_sha256 = _sha256_text(inputs.get("asset_sha256"), "inputs.asset_sha256")
        if not hmac.compare_digest(sha256_file(asset_path), asset_sha256):
            raise CalibrationPackageLoadError("Packaged source USD does not match inputs.asset_sha256")
        attested_residual_sha256 = inputs.get("residual_sha256")
        if attested_residual_sha256 is not None:
            attested_residual_sha256 = _sha256_text(attested_residual_sha256, "inputs.residual_sha256")
        if environment.residual_model_sha256 != attested_residual_sha256:
            raise CalibrationPackageLoadError("Runtime residual fingerprint does not match package inputs")
        attestation_bundle = _mapping(manifest.get("runtime_attestation"), "manifest.runtime_attestation")
        if set(attestation_bundle) != {"fit", "heldout_validation"}:
            raise CalibrationPackageLoadError("Generic runtime attestation must record fit and held-out validation")
        fit_attestation = _mapping(attestation_bundle.get("fit"), "runtime_attestation.fit")
        validation_attestation = _mapping(
            attestation_bundle.get("heldout_validation"),
            "runtime_attestation.heldout_validation",
        )
        activation_checks = _mapping(manifest.get("activation_checks"), "manifest.activation_checks")
        expected_check_keys = {
            "heldout_validation_passed",
            "authoritative_fit_attestation",
            "authoritative_validation_attestation",
            "self_contained_asset",
            "asset_dependency_reason",
        }
        if set(activation_checks) != expected_check_keys:
            raise CalibrationPackageLoadError("Generic activation checks are incomplete or unsupported")
        for check in (
            "heldout_validation_passed",
            "authoritative_fit_attestation",
            "authoritative_validation_attestation",
            "self_contained_asset",
        ):
            if activation_checks.get(check) is not True:
                raise CalibrationPackageLoadError(f"Generic activation check failed: {check}")
        if not isinstance(activation_checks.get("asset_dependency_reason"), str):
            raise CalibrationPackageLoadError("Generic asset dependency check must include a diagnostic reason")
        self_contained, _ = _asset_is_self_contained(asset_path)
        if not self_contained:
            raise CalibrationPackageLoadError("Packaged generic USD is not provably self-contained")
        profile_path = _artifact(root, artifacts, "robot_profile")
        profile = _read_json_object(profile_path, maximum_bytes=_MAX_MANIFEST_BYTES)
        _require_equal(profile, "schema", "newton.calibration.robot-profile/v1", "robot profile")
        if profile.get("robot_id") != environment.robot_id:
            raise CalibrationPackageLoadError("Robot profile ID does not match runtime.robot_id")
        if profile.get("asset_sha256") != asset_sha256:
            raise CalibrationPackageLoadError("Robot profile asset fingerprint does not match the manifest")
        if profile.get("joint_map") != runtime_payload.get("joint_map"):
            raise CalibrationPackageLoadError("Robot profile joint map does not match the locked runtime")
        if profile.get("joint_groups") != runtime_payload.get("joint_groups"):
            raise CalibrationPackageLoadError("Robot profile groups do not match the locked runtime")
        if profile.get("joint_order") != runtime_payload.get("joint_order"):
            raise CalibrationPackageLoadError("Robot profile joint order does not match the locked runtime")
        if profile.get("profile_confirmed") is not True:
            raise CalibrationPackageLoadError("Robot profile was not explicitly confirmed")
        if profile.get("controller_profile_confirmed") is not True:
            raise CalibrationPackageLoadError("Controller profile was not explicitly confirmed")
        if profile.get("controller_profile_source") != environment.controller_profile_source:
            raise CalibrationPackageLoadError("Controller profile provenance does not match the runtime")
        profile_fields = {
            "base_stiffness_by_joint": environment.base_stiffness_by_joint,
            "base_damping_by_joint": environment.base_damping_by_joint,
            "base_effort_limit_by_joint": environment.base_effort_limit_by_joint,
            "base_armature_by_joint": environment.base_armature_by_joint,
        }
        for field_name, expected_value in profile_fields.items():
            if profile.get(field_name) != expected_value:
                raise CalibrationPackageLoadError(f"Robot profile {field_name} does not match the runtime")

        evidence_path = _artifact(root, artifacts, "evidence_spec")
        evidence_payload = _read_json_object(evidence_path, maximum_bytes=_MAX_VALIDATION_BYTES)
        evidence_spec = BoundEvidenceSpec.from_dict(evidence_payload)
        non_radian_targets = sorted(
            item.source_joint for item in evidence_spec.joint_bindings if item.usd_unit != "rad"
        )
        if non_radian_targets:
            raise CalibrationPackageLoadError(
                "Generic MVP1 Newton revolute-joint bindings must target radians: "
                f"{non_radian_targets}"
            )
        evidence_fingerprint = _sha256_text(inputs.get("evidence_fingerprint"), "inputs.evidence_fingerprint")
        if evidence_spec.fingerprint != evidence_fingerprint:
            raise CalibrationPackageLoadError("Evidence specification does not match the locked evidence fingerprint")
        mapping_fingerprint = _sha256_text(inputs.get("mapping_fingerprint"), "inputs.mapping_fingerprint")
        if evidence_spec.mapping_fingerprint != mapping_fingerprint:
            raise CalibrationPackageLoadError("Evidence mapping does not match the locked mapping fingerprint")
        binding_map = {item.source_joint: item.usd_joint for item in evidence_spec.joint_bindings}
        if binding_map != environment.joint_map:
            raise CalibrationPackageLoadError("Evidence bindings do not match runtime.joint_map")

        mapping_path = _artifact(root, artifacts, "joint_mapping")
        mapping = _read_json_object(mapping_path, maximum_bytes=_MAX_YAML_BYTES)
        _require_equal(mapping, "schema", "newton.calibration.joint-mapping/v1", "joint mapping")
        if mapping.get("mapping_fingerprint") != mapping_fingerprint:
            raise CalibrationPackageLoadError("Joint mapping artifact fingerprint does not match the manifest")
        if mapping.get("bindings") != evidence_payload.get("joint_bindings"):
            raise CalibrationPackageLoadError("Joint mapping artifact does not match evidence bindings")

        patch_path = _artifact(root, artifacts, "actuator_patch")
        patch = _read_yaml_object(patch_path)
        expected_patch_keys = {"schema", "recipe", "source_asset", "ordered_joints", "command_delay"}
        if set(patch) != expected_patch_keys:
            raise CalibrationPackageLoadError(f"Actuator patch must contain exactly: {sorted(expected_patch_keys)}")
        _require_equal(patch, "schema", "newton.calibration.actuator-patch/v1", "actuator patch")
        _require_equal(patch, "recipe", recipe_name, "actuator patch")
        if _confined_file(root, patch.get("source_asset"), "actuator patch source_asset") != asset_path:
            raise CalibrationPackageLoadError("Actuator patch source asset does not match the manifest")
        raw_entries = patch.get("ordered_joints")
        if not isinstance(raw_entries, list) or len(raw_entries) != len(logical_order):
            raise CalibrationPackageLoadError("Actuator patch must contain one ordered entry per controlled joint")
        group_by_joint = {joint: group for group, members in environment.joint_groups.items() for joint in members}
        settings: list[tuple[str, str, str, float, float, float, float, float]] = []
        for index, logical_joint in enumerate(logical_order):
            entry = _mapping(raw_entries[index], f"actuator patch ordered_joints[{index}]")
            expected_keys = {
                "logical_joint",
                "usd_joint",
                "group",
                "stiffness",
                "damping",
                "effort_limit",
                "armature",
                "friction_nm",
            }
            if set(entry) != expected_keys:
                raise CalibrationPackageLoadError(f"Actuator patch entry {index} has unsupported keys")
            group = group_by_joint[logical_joint]
            if (
                entry.get("logical_joint") != logical_joint
                or entry.get("usd_joint") != environment.joint_map[logical_joint]
            ):
                raise CalibrationPackageLoadError("Actuator patch joint order/mapping does not match the robot profile")
            if entry.get("group") != group:
                raise CalibrationPackageLoadError("Actuator patch group does not match the robot profile")
            expected_values = (
                environment.base_stiffness_by_joint.get(logical_joint, environment.base_stiffness)
                * parameters.get(f"{group}_stiffness_scale", 1.0),
                environment.base_damping_by_joint.get(logical_joint, environment.base_damping)
                * parameters.get(f"{group}_damping_scale", 1.0),
                environment.base_effort_limit_by_joint.get(logical_joint, environment.base_effort_limit)
                * parameters.get(f"{group}_effort_scale", 1.0),
                parameters.get(f"{group}_armature", environment.base_armature_by_joint.get(logical_joint, environment.base_armature)),
                parameters.get(f"{group}_friction_nm", 0.0),
            )
            actual_values = tuple(
                _nonnegative_number(entry[key], f"actuator patch {logical_joint}.{key}")
                for key in ("stiffness", "damping", "effort_limit", "armature", "friction_nm")
            )
            for label, actual, expected in zip(
                ("stiffness", "damping", "effort_limit", "armature", "friction_nm"),
                actual_values,
                expected_values,
            ):
                _close(actual, expected, f"actuator patch {logical_joint}.{label}")
            settings.append((logical_joint, environment.joint_map[logical_joint], group, *actual_values))

        timing = _mapping(manifest.get("timing_quantization"), "manifest.timing_quantization")
        requested_delay = _nonnegative_number(timing.get("requested_command_delay_s"), "requested command delay")
        _close(
            _positive_number(timing.get("runtime_dt_s"), "timing runtime dt"),
            environment.dt,
            "timing runtime dt",
            relative_tolerance=1e-12,
        )
        delay_steps = _nonnegative_integer(timing.get("applied_delay_steps"), "applied delay steps")
        effective_delay = _nonnegative_number(timing.get("effective_command_delay_s"), "effective command delay")
        _close(requested_delay, parameters.get("command_delay_s", 0.0), "requested command delay")
        if delay_steps != max(0, round(requested_delay / environment.dt)):
            raise CalibrationPackageLoadError("Applied command-delay steps do not match the locked runtime dt")
        _close(effective_delay, delay_steps * environment.dt, "effective command delay")
        patch_delay = _mapping(patch.get("command_delay"), "actuator patch command_delay")
        _close(
            _nonnegative_number(patch_delay.get("requested_s"), "patch requested delay"),
            requested_delay,
            "patch requested delay",
        )
        if _nonnegative_integer(patch_delay.get("steps_at_runtime_dt"), "patch delay steps") != delay_steps:
            raise CalibrationPackageLoadError("Actuator patch delay steps do not match the manifest")
        _close(
            _nonnegative_number(patch_delay.get("effective_s"), "patch effective delay"),
            effective_delay,
            "patch effective delay",
        )

        validation_path = _artifact(root, artifacts, "validation")
        validation = _read_json_object(validation_path, maximum_bytes=_MAX_VALIDATION_BYTES)
        if validation.get("run_id") != run_id or validation.get("passed") is not True:
            raise CalibrationPackageLoadError("Packaged held-out validation did not pass for this run")
        gates = _mapping(validation.get("gates"), "validation.gates")
        if set(gates) != _VALIDATION_GATES or any(value is not True for value in gates.values()):
            raise CalibrationPackageLoadError("All generic MVP1 held-out validation gates must pass")
        if validation.get("regressions") != []:
            raise CalibrationPackageLoadError("Generic MVP1 package contains held-out regressions")
        claims = _mapping(manifest.get("claims"), "manifest.claims")
        if claims.get("heldout_passed") is not True or claims.get("contact_or_task_transfer_validated") is not False:
            raise CalibrationPackageLoadError("Generic MVP1 package has an invalid claim boundary")
        improvement = _finite_number(claims.get("heldout_improvement_pct"), "claims.heldout_improvement_pct")
        if improvement < float(recipe_cfg.validation_gates["minimum_improvement_pct"]):
            raise CalibrationPackageLoadError("Held-out improvement is below the locked recipe gate")
        _close(
            _finite_number(validation.get("improvement_pct"), "validation.improvement_pct"),
            improvement,
            "validation improvement",
        )
        fit = _mapping(validation.get("fit"), "validation.fit")
        if fit.get("backend") != "isaaclab_newton":
            raise CalibrationPackageLoadError("Generic validation fit was not produced by the Newton runtime")
        if fit.get("run_id") != run_id:
            raise CalibrationPackageLoadError("Validation fit run ID does not match the manifest")
        if _mapping(fit.get("runtime_attestation"), "validation.fit.runtime_attestation") != fit_attestation:
            raise CalibrationPackageLoadError("Validation fit attestation does not match the manifest")
        if (
            _mapping(validation.get("runtime_attestation"), "validation.runtime_attestation")
            != validation_attestation
        ):
            raise CalibrationPackageLoadError("Held-out validation attestation does not match the manifest")
        locked_plan = _mapping(fit.get("plan"), "validation.fit.plan")
        if locked_plan.get("run_id") != run_id:
            raise CalibrationPackageLoadError("Locked calibration plan run ID does not match the manifest")
        job_analysis = _read_json_object(
            _confined_file(root, "job/analysis.json", "durable analysis record"),
            maximum_bytes=_MAX_VALIDATION_BYTES,
        )
        job_plan = _read_json_object(
            _confined_file(root, "job/plan.json", "durable plan record"),
            maximum_bytes=_MAX_VALIDATION_BYTES,
        )
        job_fit = _read_json_object(
            _confined_file(root, "job/fit.json", "durable fit record"),
            maximum_bytes=_MAX_VALIDATION_BYTES,
        )
        job_validation = _read_json_object(
            _confined_file(root, "job/validation.json", "durable validation record"),
            maximum_bytes=_MAX_VALIDATION_BYTES,
        )
        if job_plan != locked_plan or job_fit != fit or job_validation != validation:
            raise CalibrationPackageLoadError(
                "Durable job ledger does not match the packaged plan, fit, and validation result"
            )
        _verify_packaged_analysis_predecessor(job_analysis, locked_plan)
        if locked_plan.get("environment") != runtime_payload or locked_plan.get("recipe") != recipe_name:
            raise CalibrationPackageLoadError("Validation plan does not match the package runtime and recipe")
        if locked_plan.get("asset_fingerprint") != asset_sha256:
            raise CalibrationPackageLoadError("Validation plan asset fingerprint does not match the packaged USD")
        if locked_plan.get("evidence_fingerprint") != evidence_fingerprint:
            raise CalibrationPackageLoadError("Validation plan evidence fingerprint does not match the package")
        locked_evidence_spec = _mapping(
            locked_plan.get("evidence_spec"),
            "validation.fit.plan.evidence_spec",
        )
        if locked_evidence_spec != evidence_payload:
            raise CalibrationPackageLoadError("Validation plan evidence contract does not match the packaged contract")
        if locked_plan.get("evidence_revision") != evidence_spec.revision:
            raise CalibrationPackageLoadError("Validation plan evidence revision does not match the package")
        if inputs.get("evidence_revision") != evidence_spec.revision:
            raise CalibrationPackageLoadError("Manifest evidence revision does not match the evidence specification")
        expected_train = [item.name for item in evidence_spec.episodes if item.split == "train"]
        expected_heldout = [item.name for item in evidence_spec.episodes if item.split == "heldout"]
        locked_train = _text_list(locked_plan.get("train_episodes"), "validation.fit.plan.train_episodes")
        locked_heldout = _text_list(
            locked_plan.get("heldout_episodes"),
            "validation.fit.plan.heldout_episodes",
        )
        if locked_train != expected_train or locked_heldout != expected_heldout:
            raise CalibrationPackageLoadError("Locked train/held-out episodes do not match the evidence split manifest")
        if not locked_train or not locked_heldout or set(locked_train) & set(locked_heldout):
            raise CalibrationPackageLoadError("Generic calibration requires non-empty disjoint train and held-out trials")
        per_episode = _mapping(validation.get("per_episode"), "validation.per_episode")
        if list(per_episode) != locked_heldout:
            raise CalibrationPackageLoadError("Validation results must contain exactly the locked held-out episodes")
        expected_parameter_specs = [jsonable(item) for item in recipe_cfg.parameters]
        if locked_plan.get("parameters") != expected_parameter_specs:
            raise CalibrationPackageLoadError("Locked parameter specifications do not match the selected recipe")
        if locked_plan.get("objective_weights") != recipe_cfg.objective_weights:
            raise CalibrationPackageLoadError("Locked objective weights do not match the selected recipe")
        locked_optimizer = _mapping(locked_plan.get("optimizer"), "validation.fit.plan.optimizer")
        if recipe_cfg.max_episode_duration_s is None:
            if "max_episode_duration_s" not in locked_optimizer or locked_optimizer["max_episode_duration_s"] is not None:
                raise CalibrationPackageLoadError("This recipe requires complete episodes, without truncation")
        else:
            _close(
                _positive_number(locked_optimizer.get("max_episode_duration_s"), "validation.fit.plan.optimizer.max_episode_duration_s"),
                recipe_cfg.max_episode_duration_s, "recipe max episode duration", relative_tolerance=1e-12,
            )
        fit_optimizer = _mapping(fit.get("optimizer"), "validation.fit.optimizer")
        manifest_optimizer = _mapping(manifest.get("optimizer"), "manifest.optimizer")
        if manifest_optimizer != fit_optimizer:
            raise CalibrationPackageLoadError("Manifest optimizer record does not match the validated fit")
        for field_name in ("name", "version", "provider", "seed", "options"):
            if fit_optimizer.get(field_name) != locked_optimizer.get(field_name):
                raise CalibrationPackageLoadError(
                    f"Validated optimizer {field_name} does not match the locked calibration plan"
                )
        optimizer_population = _positive_integer(
            fit_optimizer.get("population"), "validation.fit.optimizer.population"
        )
        optimizer_generation_budget = _positive_integer(
            fit_optimizer.get("generation_budget"), "validation.fit.optimizer.generation_budget"
        )
        optimizer_config_fingerprint = _sha256_text(
            fit_optimizer.get("config_fingerprint"), "optimizer config fingerprint"
        )
        optimizer_execution_fingerprint = _sha256_text(
            fit_optimizer.get("execution_fingerprint"), "optimizer execution fingerprint"
        )
        locked_gates = _mapping(locked_plan.get("validation_gates"), "validation.fit.plan.validation_gates")
        if set(locked_gates) != set(recipe_cfg.validation_gates):
            raise CalibrationPackageLoadError("Validation plan gates do not match the recipe")
        for name, expected in recipe_cfg.validation_gates.items():
            _close(
                _nonnegative_number(locked_gates[name], f"validation gate {name}"),
                float(expected),
                f"validation gate {name}",
                relative_tolerance=1e-12,
            )
        best = _mapping(fit.get("best"), "validation.fit.best")
        if best.get("parameters") != parameter_payload:
            raise CalibrationPackageLoadError("Validation best parameters do not match the manifest")
        baseline = _mapping(fit.get("baseline"), "validation.fit.baseline")
        baseline_parameters = {
            name: _finite_number(value, f"validation.fit.baseline.parameters.{name}")
            for name, value in _mapping(
                baseline.get("parameters"), "validation.fit.baseline.parameters"
            ).items()
        }
        if set(baseline_parameters) != set(parameters):
            raise CalibrationPackageLoadError("Validation baseline parameters do not match the recipe surface")
        canonical_baseline = {item.name: item.initial for item in recipe_cfg.parameters}
        if baseline_parameters != canonical_baseline:
            raise CalibrationPackageLoadError("Validation baseline does not match the locked recipe initials")
        _verify_packaged_fit_journal(
            root,
            run_id=run_id,
            fit=fit,
            locked_plan=locked_plan,
            baseline=baseline,
            best=best,
            optimizer=fit_optimizer,
            optimizer_population=optimizer_population,
            optimizer_generation_budget=optimizer_generation_budget,
            optimizer_config_fingerprint=optimizer_config_fingerprint,
            optimizer_execution_fingerprint=optimizer_execution_fingerprint,
        )
        episode_by_name = {item.name: item for item in evidence_spec.episodes}

        def expected_inputs(names: list[str]) -> list[dict[str, str]]:
            return [
                {
                    "name": episode_by_name[name].name,
                    "split": episode_by_name[name].split,
                    "trial_id": episode_by_name[name].trial_id,
                    "source_sha256": episode_by_name[name].sha256,
                }
                for name in names
            ]

        fit_result_sha256 = evaluation_result_fingerprint(
            score=best.get("score"),
            metrics=_mapping(best.get("metrics"), "validation.fit.best.metrics"),
            episodes=_mapping(best.get("episodes"), "validation.fit.best.episodes"),
            stable=best.get("stable"),
        )
        baseline_metrics = _mapping(validation.get("baseline_metrics"), "validation.baseline_metrics")
        calibrated_metrics = _mapping(validation.get("calibrated_metrics"), "validation.calibrated_metrics")
        baseline_stable = validation.get("baseline_stable")
        calibrated_stable = validation.get("calibrated_stable")
        if not isinstance(baseline_stable, bool) or not isinstance(calibrated_stable, bool):
            raise CalibrationPackageLoadError("Validation must record baseline and calibrated stability")
        if calibrated_stable is not gates.get("stable"):
            raise CalibrationPackageLoadError("Calibrated stability does not match the validation gate")
        baseline_episodes = {
            name: _mapping(_mapping(per_episode[name], f"validation.per_episode.{name}").get("baseline"),
                           f"validation.per_episode.{name}.baseline")
            for name in locked_heldout
        }
        calibrated_episodes = {
            name: _mapping(_mapping(per_episode[name], f"validation.per_episode.{name}").get("calibrated"),
                           f"validation.per_episode.{name}.calibrated")
            for name in locked_heldout
        }
        validation_result_sha256 = evaluation_result_fingerprint(
            score=calibrated_metrics.get("score"),
            metrics=calibrated_metrics,
            episodes=calibrated_episodes,
            stable=gates.get("stable"),
        )
        expected_fit_attestation = _expected_articulation_attestation(
            environment,
            asset_sha256,
            parameters,
            residual_sha256=attested_residual_sha256,
            phase="fit-selected",
            evidence_episodes=expected_inputs(locked_train),
            result_sha256=fit_result_sha256,
            run_id=run_id,
            plan_sha256=record_fingerprint(locked_plan),
            evidence_fingerprint=evidence_fingerprint,
            mapping_fingerprint=mapping_fingerprint,
        )
        baseline_result_sha256 = evaluation_result_fingerprint(
            score=baseline_metrics.get("score"),
            metrics=baseline_metrics,
            episodes=baseline_episodes,
            stable=baseline_stable,
        )
        expected_validation_attestation = {
            "baseline": _expected_articulation_attestation(
                environment,
                asset_sha256,
                baseline_parameters,
                residual_sha256=attested_residual_sha256,
                phase="heldout-baseline",
                evidence_episodes=expected_inputs(locked_heldout),
                result_sha256=baseline_result_sha256,
                run_id=run_id,
                plan_sha256=record_fingerprint(locked_plan),
                evidence_fingerprint=evidence_fingerprint,
                mapping_fingerprint=mapping_fingerprint,
            ),
            "calibrated": _expected_articulation_attestation(
                environment,
                asset_sha256,
                parameters,
                residual_sha256=attested_residual_sha256,
                phase="heldout-validation",
                evidence_episodes=expected_inputs(locked_heldout),
                result_sha256=validation_result_sha256,
                run_id=run_id,
                plan_sha256=record_fingerprint(locked_plan),
                evidence_fingerprint=evidence_fingerprint,
                mapping_fingerprint=mapping_fingerprint,
            ),
        }
        if fit_attestation != expected_fit_attestation:
            raise CalibrationPackageLoadError("Fit attestation is not bound to the selected Newton training run")
        if validation_attestation != expected_validation_attestation:
            raise CalibrationPackageLoadError("Validation attestation is not bound to the held-out Newton run")
        baseline_score = _finite_number(baseline_metrics.get("score"), "validation baseline score")
        calibrated_score = _finite_number(calibrated_metrics.get("score"), "validation calibrated score")
        recomputed_improvement = 100.0 * (baseline_score - calibrated_score) / max(abs(baseline_score), 1e-12)
        _close(recomputed_improvement, improvement, "validation improvement", relative_tolerance=1e-10)
        maximum_regression = float(recipe_cfg.validation_gates["maximum_episode_regression_pct"])
        for name in locked_heldout:
            before_score = _finite_number(baseline_episodes[name].get("score"), f"{name} baseline score")
            after_score = _finite_number(calibrated_episodes[name].get("score"), f"{name} calibrated score")
            regression = 100.0 * (after_score - before_score) / max(abs(before_score), 1e-12)
            if regression > maximum_regression:
                raise CalibrationPackageLoadError(f"Held-out episode {name!r} exceeds the regression gate")

        residual_model_path = _optional_artifact(root, artifacts, "actuator_residual")
        expected_residual_sha256 = inputs.get("residual_sha256")
        if residual_model_path is None:
            if expected_residual_sha256 is not None:
                raise CalibrationPackageLoadError("Residual digest is present but its artifact is missing")
        else:
            residual_sha256 = _sha256_text(expected_residual_sha256, "inputs.residual_sha256")
            if not hmac.compare_digest(sha256_file(residual_model_path), residual_sha256):
                raise CalibrationPackageLoadError("Actuator residual does not match inputs.residual_sha256")
            try:
                residual = load_residual(residual_model_path, logical_order)
            except (KeyError, TypeError, ValueError) as exc:
                raise CalibrationPackageLoadError(f"Invalid packaged actuator residual: {exc}") from exc
            if residual is None or not np.isfinite(residual.weights).all() or not np.isfinite(residual.clip_nm).all():
                raise CalibrationPackageLoadError("Packaged actuator residual must contain finite values")
            if np.any(residual.clip_nm < 0.0):
                raise CalibrationPackageLoadError("Packaged actuator residual clip values must be non-negative")

        return cls(
            root=root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            recipe=recipe_name,
            scope=scope,
            asset_path=asset_path,
            validation_path=validation_path,
            residual_model_path=residual_model_path,
            environment=environment,
            parameters=parameters,
            actuator=ArticulationActuatorSettings(
                ordered_joint_items=tuple(settings),
                requested_command_delay_s=requested_delay,
                command_delay_steps=delay_steps,
                effective_command_delay_s=effective_delay,
            ),
            heldout_improvement_pct=improvement,
        )

    def reverify(self) -> None:
        VerifiedArticulationPackage.open(self.root, expected_manifest_sha256=self.manifest_sha256)

    def assert_matches_environment(self, environment: EnvironmentSpec) -> None:
        expected = jsonable(self.environment)
        actual = jsonable(environment)
        for mutable in (
            "asset_path",
            "device",
            "residual_model_path",
            "calibration_run_id",
            "calibration_manifest_path",
            "calibration_manifest_sha256",
            "calibration_parameters",
        ):
            expected.pop(mutable, None)
            actual.pop(mutable, None)
        if actual != expected:
            raise CalibrationPackageLoadError("Loaded generic environment was changed after package verification")
        if Path(environment.asset_path).expanduser().resolve() != self.asset_path:
            raise CalibrationPackageLoadError("Loaded generic package asset was changed after verification")
        if environment.calibration_run_id != self.run_id or environment.calibration_parameters != self.parameters:
            raise CalibrationPackageLoadError("Loaded generic calibration identity or parameters were changed")
        if environment.calibration_manifest_sha256 != self.manifest_sha256:
            raise CalibrationPackageLoadError("Loaded generic manifest digest was changed")
        if Path(environment.calibration_manifest_path or "").expanduser().resolve() != self.manifest_path:
            raise CalibrationPackageLoadError("Loaded generic manifest path was changed")
        expected_residual = str(self.residual_model_path) if self.residual_model_path else None
        if environment.residual_model_path != expected_residual:
            raise CalibrationPackageLoadError("Loaded generic actuator residual path was changed")

    def to_env_cfg(self, *, device: str | None = None):
        from .isaac_lab import ArticulationEnvCfg

        self.reverify()
        env = self.environment
        return ArticulationEnvCfg(
            usd_path=str(self.asset_path),
            robot_id=env.robot_id,
            joint_groups={name: tuple(members) for name, members in env.joint_groups.items()},
            joint_map=dict(env.joint_map),
            joint_order=tuple(env.joint_order),
            profile_confirmed=True,
            controller_profile_confirmed=env.controller_profile_confirmed,
            controller_profile_source=env.controller_profile_source,
            runtime=env.adapter,
            device=device or env.device,
            dt=env.dt,
            gravity=tuple(env.gravity),
            base_stiffness=env.base_stiffness,
            base_damping=env.base_damping,
            base_effort_limit=env.base_effort_limit,
            base_armature=env.base_armature,
            base_stiffness_by_joint=dict(env.base_stiffness_by_joint),
            base_damping_by_joint=dict(env.base_damping_by_joint),
            base_effort_limit_by_joint=dict(env.base_effort_limit_by_joint),
            base_armature_by_joint=dict(env.base_armature_by_joint),
            analytic_inertia_by_joint=dict(env.analytic_inertia_by_joint),
            parameter_bounds={name: tuple(bounds) for name, bounds in env.parameter_bounds.items()},
            tuning_targets=tuple(env.tuning_targets),
            num_substeps=env.num_substeps,
            solver_iterations=env.solver_iterations,
            solver_tolerance=env.solver_tolerance,
            residual_model_path=str(self.residual_model_path) if self.residual_model_path else None,
            residual_model_sha256=(
                sha256_file(self.residual_model_path) if self.residual_model_path else None
            ),
            calibration_run_id=self.run_id,
            calibration_manifest_path=str(self.manifest_path),
            calibration_manifest_sha256=self.manifest_sha256,
            calibration_parameters=dict(self.parameters),
        )


class VerifiedCalibrationPackage:
    """Schema-dispatch entry point for legacy v1 and generic v2 packages."""

    @classmethod
    def open(cls, package: str | Path, *, expected_manifest_sha256: str | None = None):
        root = Path(package).expanduser().resolve()
        manifest_path = _confined_file(root, "manifest.json", "manifest")
        manifest = _read_json_object(manifest_path, maximum_bytes=_MAX_MANIFEST_BYTES)
        schema = manifest.get("schema")
        if schema == _PACKAGE_SCHEMA:
            if expected_manifest_sha256 is None and not (
                "activation_checks" in manifest and "runtime_attestation" in manifest
            ):
                raise CalibrationPackageLoadError(
                    "Legacy unattested v1 packages require a separately trusted manifest SHA-256; "
                    "regenerate them for execution-bound verification"
                )
            return VerifiedSO101Package.open(root, expected_manifest_sha256=expected_manifest_sha256)
        if schema == "newton.calibration.package/v2":
            return VerifiedArticulationPackage.open(root, expected_manifest_sha256=expected_manifest_sha256)
        raise CalibrationPackageLoadError(f"Unsupported calibration package schema: {schema!r}")


def _verified_attested_episode_inputs(
    attestation: dict[str, Any], expected_names: list[str], expected_split: str
) -> list[dict[str, str]]:
    value = attestation.get("evidence_episodes")
    if not isinstance(value, list) or len(value) != len(expected_names):
        raise CalibrationPackageLoadError("Runtime attestation evidence episode list has the wrong size")
    result: list[dict[str, str]] = []
    for index, expected_name in enumerate(expected_names):
        item = _mapping(value[index], f"runtime attestation evidence_episodes[{index}]")
        if set(item) != {"name", "split", "trial_id", "source_sha256"}:
            raise CalibrationPackageLoadError("Runtime attestation evidence identity has unsupported fields")
        if item.get("name") != expected_name or item.get("split") != expected_split:
            raise CalibrationPackageLoadError("Runtime attestation evidence does not match the locked split")
        result.append(
            {
                "name": expected_name,
                "split": expected_split,
                "trial_id": _text(item.get("trial_id"), "attested episode trial_id"),
                "source_sha256": _sha256_text(
                    item.get("source_sha256"), "attested episode source_sha256"
                ),
            }
        )
    return result


def _expected_articulation_attestation(
    environment: EnvironmentSpec,
    asset_sha256: str,
    parameters: dict[str, float],
    *,
    residual_sha256: str | None,
    phase: str,
    evidence_episodes: list[dict[str, str]],
    result_sha256: str,
    run_id: str,
    plan_sha256: str,
    evidence_fingerprint: str,
    mapping_fingerprint: str,
) -> dict[str, Any]:
    layout = _resolve_joint_layout(environment)
    logical_order = list(layout.logical_names)
    properties = _joint_properties(
        environment,
        layout,
        parameters,
        analytic=False,
    )
    return {
        "schema": "newton.calibration.runtime-attestation/v2",
        "backend": "isaaclab_newton",
        "authoritative": True,
        "robot_id": environment.robot_id,
        "asset_sha256": asset_sha256,
        "logical_joints": logical_order,
        "runtime_joints": list(layout.runtime_names),
        "runtime_dt_s": environment.dt,
        "gravity": list(environment.gravity),
        "num_substeps": environment.num_substeps,
        "solver_iterations": environment.solver_iterations,
        "solver_tolerance": environment.solver_tolerance,
        "selected_joint_scoped": True,
        "full_state_reset_per_episode": True,
        "readback_parameters": ["stiffness", "damping", "effort_limit", "armature", "friction_nm"],
        "candidate_sha256": parameter_fingerprint(parameters),
        "readback_values_sha256": numeric_surface_fingerprint(
            {
                "stiffness": properties["stiffness"].tolist(),
                "damping": properties["damping"].tolist(),
                "effort_limit": properties["effort"].tolist(),
                "armature": properties["armature"].tolist(),
                "friction_nm": properties["friction"].tolist(),
            }
        ),
        "residual_sha256": residual_sha256,
        "evaluation_phase": phase,
        "run_id": run_id,
        "plan_sha256": plan_sha256,
        "evidence_fingerprint": evidence_fingerprint,
        "mapping_fingerprint": mapping_fingerprint,
        "evidence_episodes": evidence_episodes,
        "result_sha256": result_sha256,
    }


def _asset_is_self_contained(path: Path) -> tuple[bool, str]:
    """Recompute the generic package's fail-closed dependency assertion."""

    if path.suffix.lower() == ".usda":
        try:
            references = re.findall(r"@([^@]+)@", path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return False, "USDA text could not be decoded"
        if references:
            return False, "USDA contains external asset references"
        return True, "root layer has no external asset references"
    try:
        from pxr import UsdUtils

        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - exercised in the Newton image
        return False, f"dependency closure could not be proven: {type(exc).__name__}"
    external_layers = [layer for layer in layers if Path(layer.realPath).resolve() != path]
    if external_layers or assets or unresolved:
        return False, "external USD dependency closure is present"
    return True, "OpenUSD dependency closure contains only the root layer"


def _generic_environment(value: dict[str, Any]) -> EnvironmentSpec:
    allowed = {item.name for item in fields(EnvironmentSpec)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CalibrationPackageLoadError(f"Generic runtime contains unsupported fields: {unknown}")
    normalized = dict(value)
    if "gravity" in normalized:
        normalized["gravity"] = tuple(normalized["gravity"])
    if "joint_order" in normalized:
        normalized["joint_order"] = tuple(normalized["joint_order"])
    normalized["joint_groups"] = {
        str(name): tuple(members)
        for name, members in _mapping(normalized.get("joint_groups"), "runtime.joint_groups").items()
    }
    for field_name in ("parameter_bounds",):
        normalized[field_name] = {
            str(name): tuple(bounds)
            for name, bounds in _mapping(normalized.get(field_name, {}), f"runtime.{field_name}").items()
        }
    if bool(normalized.get("residual_model_path")) != bool(normalized.get("residual_model_sha256")):
        raise CalibrationPackageLoadError(
            "Generic runtime residual_model_path and residual_model_sha256 must both be present or both be absent"
        )
    try:
        return EnvironmentSpec(**normalized)
    except (TypeError, ValueError) as exc:
        raise CalibrationPackageLoadError(f"Invalid generic runtime configuration: {exc}") from exc


def _verify_job_record_integrity(root: Path, artifacts: dict[str, Any]) -> None:
    if artifacts.get("job_records") != "job/":
        raise CalibrationPackageLoadError("Generic package job_records must identify the job/ directory")
    claimed = _mapping(artifacts.get("job_record_sha256"), "artifacts.job_record_sha256")
    normalized = {
        _text(name, "job record path"): _sha256_text(digest, f"job record digest {name}")
        for name, digest in claimed.items()
    }
    required = {
        "job/analysis.json",
        "job/plan.json",
        "job/baseline.json",
        "job/fit-checkpoint.json",
        "job/candidate-history.jsonl",
        "job/fit.json",
        "job/validation.json",
    }
    if not required.issubset(normalized):
        raise CalibrationPackageLoadError("Generic package durable job ledger is incomplete")
    generation_names = sorted(name for name in normalized if name.startswith("job/fit-generations/"))
    if not generation_names:
        raise CalibrationPackageLoadError("Generic package has no hashed optimizer generation records")
    job_dir = root / "job"
    if job_dir.is_symlink() or not job_dir.is_dir():
        raise CalibrationPackageLoadError("Generic package job ledger is missing or symbolic")
    actual_names = {
        path.relative_to(root).as_posix()
        for path in job_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_names != set(normalized):
        raise CalibrationPackageLoadError("Generic package job ledger file set does not match its digest index")
    for name, expected in normalized.items():
        path = _confined_file(root, name, f"job record {name}")
        if not hmac.compare_digest(sha256_file(path), expected):
            raise CalibrationPackageLoadError(f"Generic job record digest mismatch: {name}")


def _verify_packaged_fit_journal(
    root: Path,
    *,
    run_id: str,
    fit: dict[str, Any],
    locked_plan: dict[str, Any],
    baseline: dict[str, Any],
    best: dict[str, Any],
    optimizer: dict[str, Any],
    optimizer_population: int,
    optimizer_generation_budget: int,
    optimizer_config_fingerprint: str,
    optimizer_execution_fingerprint: str,
) -> None:
    job_baseline = _read_json_object(
        _confined_file(root, "job/baseline.json", "durable baseline record"),
        maximum_bytes=_MAX_VALIDATION_BYTES,
    )
    if job_baseline != baseline:
        raise CalibrationPackageLoadError("Durable baseline record does not match validation.fit.baseline")

    checkpoint_metadata = {
        "optimizer_name": _text(optimizer.get("name"), "optimizer name"),
        "optimizer_version": _text(optimizer.get("version"), "optimizer version"),
        "optimizer_provider": _text(optimizer.get("provider"), "optimizer provider"),
        "optimizer_config_fingerprint": optimizer_config_fingerprint,
    }
    try:
        state = FitJournal(
            root / "job",
            run_id=run_id,
            execution_fingerprint=optimizer_execution_fingerprint,
            checkpoint_metadata=checkpoint_metadata,
        ).verify_snapshot()
    except (FitJournalError, TypeError, ValueError) as exc:
        raise CalibrationPackageLoadError(f"Durable optimizer journal is inconsistent: {exc}") from exc

    completed_generations = _positive_integer(
        fit.get("completed_generations"), "validation.fit.completed_generations"
    )
    if state.completed_generations != completed_generations:
        raise CalibrationPackageLoadError("Fit completed_generations does not match its durable optimizer journal")
    if optimizer_generation_budget != completed_generations:
        raise CalibrationPackageLoadError("Optimizer generation budget does not match the completed fit journal")
    if baseline.get("candidate_id") != -1 or baseline.get("generation") != -1:
        raise CalibrationPackageLoadError("Fit baseline bookkeeping does not identify the pre-search evaluation")
    if best.get("candidate_id") != state.next_candidate_id or best.get("generation") != completed_generations:
        raise CalibrationPackageLoadError("Selected fit evaluation bookkeeping does not follow the optimizer journal")

    planned_history = Path(_text(locked_plan.get("workdir"), "validation.fit.plan.workdir")) / "candidate-history.jsonl"
    fit_history = Path(_text(fit.get("history_path"), "validation.fit.history_path"))
    if fit_history.expanduser().resolve() != planned_history.expanduser().resolve():
        raise CalibrationPackageLoadError("Fit history path does not identify the locked candidate-history projection")

    best_parameters = _mapping(best.get("parameters"), "validation.fit.best.parameters")
    candidate_was_evaluated = False
    for generation_path in state.generation_paths:
        generation = _read_json_object(generation_path, maximum_bytes=_MAX_VALIDATION_BYTES)
        candidates = generation.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != optimizer_population:
            raise CalibrationPackageLoadError("Optimizer generation population does not match the validated fit")
        if best_parameters in candidates:
            candidate_was_evaluated = True
    if not candidate_was_evaluated:
        raise CalibrationPackageLoadError("Selected best parameters do not occur in the durable optimizer journal")


def _verify_packaged_analysis_predecessor(analysis: dict[str, Any], plan: dict[str, Any]) -> None:
    copied_fields = (
        "run_id",
        "recipe",
        "evidence_uri",
        "evidence_revision",
        "evidence_fingerprint",
        "asset_fingerprint",
        "environment",
        "workdir",
        "evidence_spec",
    )
    if any(analysis.get(name) != plan.get(name) for name in copied_fields):
        raise CalibrationPackageLoadError("Durable analysis record is not the predecessor of the locked plan")
    if analysis.get("identifiable_parameters") != plan.get("parameters"):
        raise CalibrationPackageLoadError("Durable analysis parameter surface does not match the locked plan")
    readiness = analysis.get("readiness")
    if not isinstance(readiness, dict) or not readiness or any(value is not True for value in readiness.values()):
        raise CalibrationPackageLoadError("Durable analysis does not record a successful readiness decision")
    for field_name in ("train_episodes", "heldout_episodes"):
        analyzed = analysis.get(field_name)
        selected = plan.get(field_name)
        if (
            not isinstance(analyzed, list)
            or not isinstance(selected, list)
            or any(not isinstance(name, str) for name in analyzed)
            or any(not isinstance(name, str) for name in selected)
            or not set(selected).issubset(set(analyzed))
        ):
            raise CalibrationPackageLoadError(
                f"Locked {field_name} are not derived from the durable analysis"
            )


class _StrictSafeLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise CalibrationPackageLoadError("YAML aliases are not permitted in calibration packages")
        return super().compose_node(parent, index)


def _construct_unique_mapping(loader: _StrictSafeLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise CalibrationPackageLoadError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_yaml_object(path: Path) -> dict[str, Any]:
    payload = _read_bounded_text(path, _MAX_YAML_BYTES)
    try:
        value = yaml.load(payload, Loader=_StrictSafeLoader)
    except CalibrationPackageLoadError:
        raise
    except yaml.YAMLError as exc:
        raise CalibrationPackageLoadError(f"Invalid actuator YAML in {path}: {exc}") from exc
    return dict(_mapping(value, f"YAML {path}"))


def _read_json_object(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    payload = _read_bounded_text(path, maximum_bytes)

    def reject_constant(value: str):
        raise CalibrationPackageLoadError(f"Non-finite JSON value {value!r} in {path}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CalibrationPackageLoadError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=unique_object)
    except CalibrationPackageLoadError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationPackageLoadError(f"Invalid JSON in {path}: {exc}") from exc
    return dict(_mapping(value, f"JSON {path}"))


def _read_bounded_text(path: Path, maximum_bytes: int) -> str:
    size = path.stat().st_size
    if size > maximum_bytes:
        raise CalibrationPackageLoadError(f"Calibration artifact is unexpectedly large ({size} bytes): {path}")
    return path.read_text(encoding="utf-8")


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise CalibrationPackageLoadError(f"{label} must be an array")
    result = [_text(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise CalibrationPackageLoadError(f"{label} must not contain duplicates")
    return result


def _artifact(root: Path, artifacts: dict[str, Any], name: str) -> Path:
    if name not in artifacts:
        raise CalibrationPackageLoadError(
            f"Portable package is missing artifacts.{name}; migrate legacy packages with scripts/upgrade_mvp1_package.py"
        )
    return _confined_file(root, artifacts[name], f"artifacts.{name}")


def _optional_artifact(root: Path, artifacts: dict[str, Any], name: str) -> Path | None:
    value = artifacts.get(name)
    return None if value is None else _confined_file(root, value, f"artifacts.{name}")


def _confined_file(root: Path, value: Any, label: str) -> Path:
    relative = Path(_text(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise CalibrationPackageLoadError(f"{label} must be a relative path confined to the package")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CalibrationPackageLoadError(f"{label} may not traverse a symbolic link: {current}")
    try:
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise CalibrationPackageLoadError(f"{label} is missing or escapes the package: {relative}") from exc
    if not resolved.is_file():
        raise CalibrationPackageLoadError(f"{label} is not a regular file: {resolved}")
    return resolved


def _verified_parameters(value: Any) -> dict[str, float]:
    payload = _mapping(value, "manifest.parameters")
    recipe = get_recipe(_RECIPE)
    expected = {parameter.name: parameter for parameter in recipe.parameters}
    if set(payload) != set(expected):
        raise CalibrationPackageLoadError(f"manifest.parameters must contain exactly: {list(expected)}")
    result = {}
    for name, specification in expected.items():
        number = _finite_number(payload[name], f"parameters.{name}")
        if not specification.lower <= number <= specification.upper:
            raise CalibrationPackageLoadError(
                f"parameters.{name}={number} is outside recipe bounds [{specification.lower}, {specification.upper}]"
            )
        result[name] = number
    return result


def _verify_parameter_application(value: dict[str, Any], parameters: set[str]) -> None:
    if set(value) != set(_EXPECTED_APPLICATION):
        raise CalibrationPackageLoadError(f"parameter_application must contain exactly: {list(_EXPECTED_APPLICATION)}")
    consumed: list[str] = []
    for surface, expected in _EXPECTED_APPLICATION.items():
        actual = value[surface]
        if not isinstance(actual, list) or any(not isinstance(name, str) for name in actual):
            raise CalibrationPackageLoadError(f"parameter_application.{surface} must be a list of names")
        if set(actual) != expected or len(actual) != len(expected):
            raise CalibrationPackageLoadError(
                f"parameter_application.{surface} does not match the supported package binding"
            )
        consumed.extend(actual)
    if set(consumed) != parameters or len(consumed) != len(parameters):
        raise CalibrationPackageLoadError("Every calibrated parameter must be consumed exactly once")


def _verify_validation_fit(
    validation: dict[str, Any],
    *,
    run_id: str,
    recipe: str,
    manifest_runtime: dict[str, Any],
    parameters: dict[str, float],
    asset_sha256: str,
    evidence_fingerprint: str,
    evidence_revision: str,
    improvement: float,
    trusted_manifest: bool,
) -> str:
    fit = validation.get("fit")
    if fit is None:
        raise CalibrationPackageLoadError("Packaged validation is missing its fit record")
    fit_mapping = _mapping(fit, "validation.fit")
    plan = _mapping(fit_mapping.get("plan"), "validation.fit.plan")
    if plan.get("run_id") != run_id:
        raise CalibrationPackageLoadError("Validation plan and manifest run IDs do not match")
    if plan.get("recipe") != recipe:
        raise CalibrationPackageLoadError("Validation recipe does not match the manifest")
    environment = _mapping(plan.get("environment"), "validation.fit.plan.environment")
    if environment != manifest_runtime:
        raise CalibrationPackageLoadError("Validation runtime/environment does not match the manifest")
    plan_asset_fingerprint = plan.get("asset_fingerprint")
    if plan_asset_fingerprint is None:
        if not trusted_manifest:
            raise CalibrationPackageLoadError(
                "Legacy portable package has no run-side asset fingerprint; supply its separately trusted "
                "manifest SHA-256 or regenerate the package"
            )
        verification_level = "legacy-trusted-manifest"
    elif plan_asset_fingerprint != asset_sha256:
        raise CalibrationPackageLoadError("Validation asset fingerprint does not match the packaged source asset")
    else:
        verification_level = "complete-v1"
    if plan.get("evidence_fingerprint") != evidence_fingerprint:
        raise CalibrationPackageLoadError("Validation evidence fingerprint does not match the manifest")
    if plan.get("evidence_revision") != evidence_revision:
        raise CalibrationPackageLoadError("Validation evidence revision does not match the manifest")
    validation_gates = _mapping(plan.get("validation_gates"), "validation.fit.plan.validation_gates")
    recipe_gates = get_recipe(recipe).validation_gates
    if set(validation_gates) != set(recipe_gates):
        raise CalibrationPackageLoadError("Validation plan gates do not match the supported recipe")
    for name, expected in recipe_gates.items():
        _close(
            _nonnegative_number(validation_gates[name], f"validation gate {name}"),
            expected,
            f"validation gate {name}",
            relative_tolerance=1e-12,
        )
    minimum_improvement = float(recipe_gates["minimum_improvement_pct"])
    if improvement < minimum_improvement:
        raise CalibrationPackageLoadError(
            f"Held-out improvement {improvement} is below the locked minimum {minimum_improvement}"
        )
    best = _mapping(fit_mapping.get("best"), "validation.fit.best")
    best_parameters = _mapping(best.get("parameters"), "validation.fit.best.parameters")
    if set(best_parameters) != set(parameters):
        raise CalibrationPackageLoadError("Validation best parameters do not match the manifest")
    for name, expected in parameters.items():
        _close(_finite_number(best_parameters[name], f"validation best {name}"), expected, f"validation best {name}")
    return verification_level


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationPackageLoadError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationPackageLoadError(f"{label} must be non-empty text")
    return value.strip()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationPackageLoadError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationPackageLoadError(f"{label} must be finite")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise CalibrationPackageLoadError(f"{label} must be positive")
    return result


def _nonnegative_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0.0:
        raise CalibrationPackageLoadError(f"{label} must be non-negative")
    return result


def _positive_integer(value: Any, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result == 0:
        raise CalibrationPackageLoadError(f"{label} must be positive")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationPackageLoadError(f"{label} must be a non-negative integer")
    return value


def _sha256_text(value: Any, label: str) -> str:
    text = _text(value, label).lower()
    if not _SHA256.fullmatch(text):
        raise CalibrationPackageLoadError(f"{label} must contain 64 hexadecimal characters")
    return text


def _require_equal(mapping: dict[str, Any], key: str, expected: Any, label: str) -> None:
    if mapping.get(key) != expected:
        raise CalibrationPackageLoadError(f"Unsupported {label}.{key}: {mapping.get(key)!r}; expected {expected!r}")


def _close(actual: float, expected: float, label: str, *, relative_tolerance: float = 1e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=relative_tolerance, abs_tol=1e-10):
        raise CalibrationPackageLoadError(f"{label} mismatch: {actual} != {expected}")
