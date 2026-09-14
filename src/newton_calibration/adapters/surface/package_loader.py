from __future__ import annotations

import hmac
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from newton_calibration.actuators import load_residual
from newton_calibration.adapters.evidence.anchor_lab_so101 import SO101_JOINTS
from newton_calibration.core.io import sha256_file
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
            calibration_run_id=self.run_id,
            calibration_manifest_path=str(self.manifest_path),
            calibration_manifest_sha256=self.manifest_sha256,
            calibration_parameters=self.parameters,
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
