from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    initial: float
    unit: str
    owner: str
    rationale: str

    def __post_init__(self) -> None:
        values = (self.lower, self.upper, self.initial)
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"Parameter {self.name!r} bounds and initial value must be finite")
        if self.lower >= self.upper:
            raise ValueError(f"Parameter {self.name!r} lower bound must be below its upper bound")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError(f"Parameter {self.name!r} initial value must lie inside its bounds")


@dataclass(frozen=True)
class EnvironmentSpec:
    adapter: str
    asset_path: str
    robot_id: str = "articulation"
    profile_schema: str = "legacy-so101/v1"
    device: str = "cuda:0"
    dt: float = 1.0 / 120.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    joint_map: dict[str, str] = field(default_factory=dict)
    # Logical evidence coordinates grouped by actuator role.  The flattened
    # insertion order is the canonical evidence/runtime vector order; values in
    # ``joint_map`` are the corresponding USD/runtime DOF names.
    joint_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    joint_order: tuple[str, ...] = ()
    profile_confirmed: bool = False
    # The controller baselines are part of the calibrated system, not portable
    # robot defaults. Generic profiles must record who/what supplied them and
    # explicitly confirm the values before a fit can start.
    controller_profile_confirmed: bool = False
    controller_profile_source: str = ""
    base_stiffness: float = 1.7453293
    base_damping: float = 0.017453292
    base_effort_limit: float = 10.0
    base_armature: float = 0.0
    base_stiffness_by_joint: dict[str, float] = field(default_factory=dict)
    base_damping_by_joint: dict[str, float] = field(default_factory=dict)
    base_effort_limit_by_joint: dict[str, float] = field(default_factory=dict)
    base_armature_by_joint: dict[str, float] = field(default_factory=dict)
    analytic_inertia_by_joint: dict[str, float] = field(default_factory=dict)
    # Optional robot-profile bounds as ``name -> (lower, upper, initial)``.
    # Required for absolute physical quantities whose safe range cannot be
    # inferred portably from a USD.
    parameter_bounds: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    num_substeps: int = 1
    solver_iterations: int = 100
    solver_tolerance: float = 1e-6
    residual_model_path: str | None = None
    residual_model_sha256: str | None = None
    calibration_run_id: str | None = None
    calibration_manifest_path: str | None = None
    calibration_manifest_sha256: str | None = None
    calibration_parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.adapter or not self.asset_path:
            raise ValueError("Environment adapter and asset_path must be non-empty")
        if self.profile_schema not in {"legacy-so101/v1", "articulation-profile/v1"}:
            raise ValueError(f"Unsupported robot profile schema: {self.profile_schema!r}")
        if self.profile_schema == "legacy-so101/v1" and self.joint_groups:
            raise ValueError("Legacy SO-101 profiles must not declare generic joint_groups")
        if not isinstance(self.controller_profile_confirmed, bool):
            raise TypeError("controller_profile_confirmed must be a boolean")
        if not isinstance(self.controller_profile_source, str):
            raise TypeError("controller_profile_source must be a string")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Environment dt must be finite and positive")
        if len(self.gravity) != 3 or any(not math.isfinite(float(value)) for value in self.gravity):
            raise ValueError("Environment gravity must contain three finite values")
        numeric = {
            "base_stiffness": self.base_stiffness,
            "base_damping": self.base_damping,
            "base_effort_limit": self.base_effort_limit,
            "base_armature": self.base_armature,
            "solver_tolerance": self.solver_tolerance,
        }
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in numeric.values()):
            raise ValueError("Environment actuator and solver baselines must be finite and non-negative")
        if self.base_effort_limit <= 0.0 or self.solver_tolerance <= 0.0:
            raise ValueError("Environment effort limit and solver tolerance must be positive")
        if isinstance(self.num_substeps, bool) or self.num_substeps <= 0:
            raise ValueError("Environment num_substeps must be a positive integer")
        if isinstance(self.solver_iterations, bool) or self.solver_iterations <= 0:
            raise ValueError("Environment solver_iterations must be a positive integer")
        for group, members in self.joint_groups.items():
            if not isinstance(group, str) or not group or isinstance(members, (str, bytes)):
                raise ValueError("Joint groups must map non-empty names to joint sequences")
            if any(not isinstance(member, str) or not member for member in members):
                raise ValueError(f"Joint group {group!r} contains an invalid logical joint")
        for name, bounds in self.parameter_bounds.items():
            if len(bounds) != 3:
                raise ValueError(f"Parameter bounds for {name!r} must be (lower, upper, initial)")
            lower, upper, initial = (float(value) for value in bounds)
            if not all(math.isfinite(value) for value in (lower, upper, initial)):
                raise ValueError(f"Parameter bounds for {name!r} must be finite")
            if lower >= upper or not lower <= initial <= upper:
                raise ValueError(f"Parameter bounds for {name!r} are inconsistent")
        if self.residual_model_sha256 is not None:
            digest = self.residual_model_sha256
            if self.residual_model_path is None:
                raise ValueError("residual_model_sha256 requires residual_model_path")
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("residual_model_sha256 must be a lowercase SHA-256 digest")


@dataclass
class AnalysisResult:
    run_id: str
    created_at: str
    evidence_uri: str
    evidence_revision: str
    evidence_fingerprint: str
    asset_fingerprint: str
    environment: EnvironmentSpec
    train_episodes: list[str]
    heldout_episodes: list[str]
    joints: list[str]
    signals: list[str]
    sample_rates_hz: dict[str, float]
    identifiable_parameters: list[ParameterSpec]
    identifiability: dict[str, str]
    warnings: list[str]
    readiness: dict[str, bool]
    workdir: str
    recipe: str = ""
    evidence_spec: dict[str, Any] = field(default_factory=dict)
    mapping_report: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationPlan:
    run_id: str
    created_at: str
    recipe: str
    evidence_uri: str
    evidence_revision: str
    evidence_fingerprint: str
    asset_fingerprint: str
    environment: EnvironmentSpec
    parameters: list[ParameterSpec]
    train_episodes: list[str]
    heldout_episodes: list[str]
    objective_weights: dict[str, float]
    optimizer: dict[str, Any]
    validation_gates: dict[str, float]
    workdir: str
    evidence_spec: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateEvaluation:
    candidate_id: int
    generation: int
    parameters: dict[str, float]
    score: float
    metrics: dict[str, float]
    episodes: dict[str, dict[str, float]]
    stable: bool = True
    error: str | None = None


@dataclass
class FitResult:
    run_id: str
    created_at: str
    plan: CalibrationPlan
    baseline: CandidateEvaluation
    best: CandidateEvaluation
    history_path: str
    completed_generations: int
    backend: str
    optimizer: dict[str, Any] = field(default_factory=dict)
    runtime_attestation: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    run_id: str
    created_at: str
    fit: FitResult
    baseline_metrics: dict[str, float]
    calibrated_metrics: dict[str, float]
    per_episode: dict[str, dict[str, dict[str, float]]]
    improvement_pct: float
    regressions: list[str]
    passed: bool
    gates: dict[str, bool]
    runtime_attestation: dict[str, Any] = field(default_factory=dict)
    baseline_stable: bool = True
    calibrated_stable: bool = True


@dataclass
class CalibrationPackage:
    run_id: str
    output_dir: str
    overlay_path: str
    isaaclab_cfg_path: str
    manifest_path: str
    validation_path: str
    report_path: str


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
