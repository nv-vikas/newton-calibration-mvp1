from __future__ import annotations

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


@dataclass(frozen=True)
class EnvironmentSpec:
    adapter: str
    asset_path: str
    device: str = "cuda:0"
    dt: float = 1.0 / 120.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    joint_map: dict[str, str] = field(default_factory=dict)
    base_stiffness: float = 1.7453293
    base_damping: float = 0.017453292
    base_effort_limit: float = 10.0
    base_armature: float = 0.0
    num_substeps: int = 1
    solver_iterations: int = 100
    solver_tolerance: float = 1e-6
    residual_model_path: str | None = None


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
