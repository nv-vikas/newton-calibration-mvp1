from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from newton_calibration.adapters.asset import supported_parameter_names, validate_so101_asset
from newton_calibration.adapters.evidence import AnchorLabSO101Evidence
from newton_calibration.adapters.runtime import create_runtime
from newton_calibration.core.fit_journal import FitJournal
from newton_calibration.core.io import sha256_file, utc_now, write_json
from newton_calibration.core.models import (
    AnalysisResult,
    CalibrationPackage,
    CalibrationPlan,
    CandidateEvaluation,
    EnvironmentSpec,
    FitResult,
    ValidationResult,
    jsonable,
)
from newton_calibration.optimizers import (
    OptimizerContractError,
    OptimizerInit,
    create_optimizer,
    get_optimizer_registration,
    optimizer_config_fingerprint,
    validate_candidates,
    validate_scores,
)
from newton_calibration.packaging.writer import write_package
from newton_calibration.recipes import get_recipe


def analyze(
    *,
    env: Any,
    evidence: str | Path | AnchorLabSO101Evidence,
    evidence_revision: str = "local",
    workdir: str | Path = "runs",
) -> AnalysisResult:
    """Call 1/5: inventory evidence and determine recipe/evidence readiness.

    The signal/name checks below establish that a parameter is reasonable to
    include in this recipe.  They are not a numerical identifiability proof.
    """
    environment = _environment_spec(env)
    adapter = _evidence_adapter(evidence, evidence_revision)
    inventory = adapter.inventory()
    recipe = get_recipe("so101_actuator_dynamics.v1")
    asset_errors = validate_so101_asset(environment)
    exposed = supported_parameter_names(environment)
    identifiability = _identify_parameters(inventory)
    identifiable = [
        parameter for parameter in recipe.parameters if parameter.name in exposed and parameter.name in identifiability
    ]
    required_joints = set(environment.joint_map)
    evidence_joints = set(inventory["joints"])
    readiness = {
        "asset_exists": not any(error.startswith("USD asset does not exist") for error in asset_errors),
        "joint_map_complete": required_joints.issubset(evidence_joints)
        and not any("joint map" in error for error in asset_errors),
        "required_signals_present": bool(inventory["required_signals_present"]),
        "train_split_present": bool(inventory["train_episodes"]),
        "heldout_split_present": bool(inventory["heldout_episodes"]),
        "requested_parameter_surface_exposed": all(parameter.name in exposed for parameter in recipe.parameters),
        "requested_parameters_identifiable": len(identifiable) == len(recipe.parameters),
    }
    warnings = list(asset_errors)
    warnings.extend(
        [
            "Anchor-Lab does not publish a complete units/controller dictionary; q and dq are treated as radians and radians/s.",
            "present_load_raw and tau_abs are excluded from the primary objective because raw-load calibration and torque sign are unavailable.",
            "MVP1 validates free-space arm and unloaded gripper actuation; it does not claim absolute gripping force or contact fidelity.",
            "The released SO-101 USD is already calibrated; use an explicitly declared untuned asset for unbiased improvement claims.",
        ]
    )
    run_id = f"so101-{uuid.uuid4().hex[:10]}"
    run_dir = Path(workdir).expanduser().resolve() / run_id
    result = AnalysisResult(
        run_id=run_id,
        created_at=utc_now(),
        evidence_uri=adapter.uri,
        evidence_revision=adapter.revision,
        evidence_fingerprint=inventory["fingerprint"],
        asset_fingerprint=(
            sha256_file(environment.asset_path) if Path(environment.asset_path).expanduser().resolve().is_file() else ""
        ),
        environment=environment,
        train_episodes=inventory["train_episodes"],
        heldout_episodes=inventory["heldout_episodes"],
        joints=inventory["joints"],
        signals=inventory["signals"],
        sample_rates_hz=inventory["sample_rates_hz"],
        identifiable_parameters=identifiable,
        identifiability=identifiability,
        warnings=warnings,
        readiness=readiness,
        workdir=str(run_dir),
    )
    write_json(run_dir / "analysis.json", result)
    return result


def plan(
    analysis: AnalysisResult,
    *,
    recipe: str = "so101_actuator_dynamics.v1",
    optimizer: str | None = None,
    optimizer_options: dict[str, Any] | None = None,
) -> CalibrationPlan:
    """Call 2/5: freeze recipe, bounds, splits, objective, runtime, and gates."""
    failed = [name for name, ready in analysis.readiness.items() if not ready]
    if failed:
        raise ValueError(f"Cannot plan calibration; readiness checks failed: {failed}")
    recipe_cfg = get_recipe(recipe)
    optimizer_config = dict(recipe_cfg.optimizer)
    optimizer_name = optimizer or str(optimizer_config["name"])
    optimizer_registration = get_optimizer_registration(optimizer_name)
    optimizer_config["name"] = optimizer_registration.name
    optimizer_config["version"] = optimizer_registration.version
    optimizer_config["provider"] = optimizer_registration.provider
    optimizer_config["options"] = dict(
        optimizer_config.get("options", {}) if optimizer_options is None else optimizer_options
    )
    train = _select_episodes(analysis.train_episodes, recipe_cfg.train_selectors)
    heldout = _select_episodes(analysis.heldout_episodes, recipe_cfg.heldout_selectors)
    result = CalibrationPlan(
        run_id=analysis.run_id,
        created_at=utc_now(),
        recipe=recipe_cfg.name,
        evidence_uri=analysis.evidence_uri,
        evidence_revision=analysis.evidence_revision,
        evidence_fingerprint=analysis.evidence_fingerprint,
        asset_fingerprint=analysis.asset_fingerprint,
        environment=analysis.environment,
        parameters=list(analysis.identifiable_parameters),
        train_episodes=train,
        heldout_episodes=heldout,
        objective_weights=dict(recipe_cfg.objective_weights),
        optimizer={**optimizer_config, "max_episode_duration_s": recipe_cfg.max_episode_duration_s},
        validation_gates=dict(recipe_cfg.validation_gates),
        workdir=analysis.workdir,
    )
    write_json(Path(result.workdir) / "plan.json", result)
    return result


def fit(
    calibration_plan: CalibrationPlan,
    *,
    generations: int | None = None,
    population: int | None = None,
    resume: bool = True,
) -> FitResult:
    """Call 3/5: replay evidence, search parameters, and checkpoint every generation."""
    run_dir = Path(calibration_plan.workdir)
    history_path = run_dir / "candidate-history.jsonl"
    evidence = _evidence_adapter(calibration_plan.evidence_uri, calibration_plan.evidence_revision)
    _assert_locked_inputs_unchanged(calibration_plan, evidence=evidence)
    duration = float(calibration_plan.optimizer["max_episode_duration_s"])
    episodes = [
        evidence.load_episode(name, dt=calibration_plan.environment.dt, max_duration_s=duration)
        for name in calibration_plan.train_episodes
    ]
    population_size = _positive_integer(
        "population",
        calibration_plan.optimizer["population"] if population is None else population,
    )
    generation_count = _positive_integer(
        "generations",
        calibration_plan.optimizer["generations"] if generations is None else generations,
    )
    optimizer_name = str(calibration_plan.optimizer["name"])
    registration = get_optimizer_registration(optimizer_name)
    planned_version = calibration_plan.optimizer.get("version")
    if planned_version is not None and str(planned_version) != registration.version:
        raise RuntimeError(
            f"Optimizer {optimizer_name!r} was planned at version {planned_version!r}, "
            f"but version {registration.version!r} is installed; create a new plan"
        )
    planned_provider = calibration_plan.optimizer.get("provider")
    if planned_provider is not None and str(planned_provider) != registration.provider:
        raise RuntimeError(
            f"Optimizer {optimizer_name!r} was planned from provider {planned_provider!r}, "
            f"but provider {registration.provider!r} is installed; create a new plan"
        )
    optimizer_options = calibration_plan.optimizer.get("options", {})
    if not isinstance(optimizer_options, dict):
        raise TypeError("locked optimizer options must be a JSON object")
    optimizer_initialization = OptimizerInit(
        parameters=calibration_plan.parameters,
        population=population_size,
        seed=int(calibration_plan.optimizer["seed"]),
        options=optimizer_options,
    )
    optimizer_fingerprint = optimizer_config_fingerprint(
        registration.name,
        registration.version,
        optimizer_initialization,
        provider=registration.provider,
    )
    execution_fingerprint = _fit_execution_fingerprint(
        calibration_plan,
        optimizer_name=registration.name,
        optimizer_version=registration.version,
        optimizer_provider=registration.provider,
        initialization=optimizer_initialization,
    )
    optimizer_record = {
        "name": registration.name,
        "version": registration.version,
        "provider": registration.provider,
        "population": population_size,
        "seed": optimizer_initialization.seed,
        "options": dict(optimizer_initialization.options),
        "generation_budget": generation_count,
        "config_fingerprint": optimizer_fingerprint,
        "execution_fingerprint": execution_fingerprint,
    }
    journal = FitJournal(
        run_dir,
        run_id=calibration_plan.run_id,
        execution_fingerprint=execution_fingerprint,
        checkpoint_metadata={
            "optimizer_name": registration.name,
            "optimizer_version": registration.version,
            "optimizer_provider": registration.provider,
            "optimizer_config_fingerprint": optimizer_fingerprint,
        },
    )
    if resume:
        journal_state = journal.recover()
    else:
        journal.assert_empty()
        journal_state = journal.scan()
    if generation_count < journal_state.completed_generations:
        raise ValueError(
            f"generation target {generation_count} is below the {journal_state.completed_generations} "
            "committed generations; increase the target or create a new run"
        )

    optimizer = create_optimizer(registration.name, optimizer_initialization)
    if journal_state.optimizer_state is not None:
        optimizer.load_state_dict(journal_state.optimizer_state)
    if optimizer.generation != journal_state.completed_generations:
        raise OptimizerContractError(
            f"Optimizer {registration.name!r} restored generation {optimizer.generation}, "
            f"but the fit journal committed {journal_state.completed_generations}"
        )
    candidate_id = journal_state.next_candidate_id

    runtime = create_runtime(calibration_plan.environment)
    try:
        initial = {parameter.name: parameter.initial for parameter in calibration_plan.parameters}
        baseline_eval = _evaluate(runtime, initial, episodes, calibration_plan, candidate_id=-1, generation=-1)
        write_json(run_dir / "baseline.json", baseline_eval)
        for generation in range(optimizer.generation, generation_count):
            candidate_id_start = candidate_id
            candidates = validate_candidates(
                optimizer.ask(),
                calibration_plan.parameters,
                expected_count=population_size,
            )
            evaluations: list[CandidateEvaluation] = []
            for candidate in candidates:
                try:
                    evaluation = _evaluate(
                        runtime, candidate, episodes, calibration_plan, candidate_id=candidate_id, generation=generation
                    )
                # A bad physics candidate is data, not a reason to lose a long-running job.
                except Exception as exc:  # noqa: BLE001
                    evaluation = CandidateEvaluation(
                        candidate_id=candidate_id,
                        generation=generation,
                        parameters=dict(candidate),
                        score=1_000_000.0,
                        metrics={},
                        episodes={},
                        stable=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                evaluations.append(evaluation)
                candidate_id += 1
            scores = validate_scores(
                [evaluation.score for evaluation in evaluations],
                expected_count=len(candidates),
            )
            generation_before_tell = optimizer.generation
            optimizer.tell(candidates, scores)
            if optimizer.generation != generation_before_tell + 1:
                raise OptimizerContractError(
                    f"Optimizer {registration.name!r} must advance generation by exactly one in tell(); "
                    f"observed {generation_before_tell} -> {optimizer.generation}"
                )
            journal_state = journal.commit_generation(
                generation=generation,
                candidate_id_start=candidate_id_start,
                candidates=candidates,
                evaluations=evaluations,
                optimizer_state=optimizer.state_dict(),
                optimizer_generation=optimizer.generation,
            )
            candidate_id = journal_state.next_candidate_id
        if optimizer.best is None:
            raise RuntimeError("Optimizer completed without a valid candidate")
        best_parameters, best_score = optimizer.best
        best_parameters = validate_candidates(
            [best_parameters],
            calibration_plan.parameters,
            expected_count=1,
        )[0]
        validate_scores([best_score], expected_count=1)
        best_eval = _evaluate(
            runtime,
            best_parameters,
            episodes,
            calibration_plan,
            candidate_id=candidate_id,
            generation=optimizer.generation,
        )
    finally:
        runtime.close()

    result = FitResult(
        run_id=calibration_plan.run_id,
        created_at=utc_now(),
        plan=calibration_plan,
        baseline=baseline_eval,
        best=best_eval,
        history_path=str(history_path),
        completed_generations=optimizer.generation,
        backend=calibration_plan.environment.adapter,
        optimizer=optimizer_record,
    )
    write_json(run_dir / "fit.json", result)
    return result


def validate(fit_run: FitResult) -> ValidationResult:
    """Call 4/5: compare baseline and calibrated parameters on data excluded from fitting."""
    plan_cfg = fit_run.plan
    evidence = _evidence_adapter(plan_cfg.evidence_uri, plan_cfg.evidence_revision)
    _assert_locked_inputs_unchanged(plan_cfg, evidence=evidence)
    duration = float(plan_cfg.optimizer["max_episode_duration_s"])
    episodes = [
        evidence.load_episode(name, dt=plan_cfg.environment.dt, max_duration_s=duration)
        for name in plan_cfg.heldout_episodes
    ]
    runtime = create_runtime(plan_cfg.environment)
    try:
        baseline = _evaluate(runtime, fit_run.baseline.parameters, episodes, plan_cfg, candidate_id=-1, generation=-1)
        calibrated = _evaluate(runtime, fit_run.best.parameters, episodes, plan_cfg, candidate_id=-2, generation=-1)
    finally:
        runtime.close()
    improvement = 100.0 * (baseline.score - calibrated.score) / max(abs(baseline.score), 1e-12)
    max_regression = float(plan_cfg.validation_gates["maximum_episode_regression_pct"])
    regressions: list[str] = []
    per_episode: dict[str, dict[str, dict[str, float]]] = {}
    for name in plan_cfg.heldout_episodes:
        before = baseline.episodes[name]
        after = calibrated.episodes[name]
        per_episode[name] = {"baseline": before, "calibrated": after}
        regression_pct = 100.0 * (after["score"] - before["score"]) / max(abs(before["score"]), 1e-12)
        if regression_pct > max_regression:
            regressions.append(f"{name}: {regression_pct:.1f}%")
    gates = {
        "stable": calibrated.stable,
        "minimum_improvement": improvement >= float(plan_cfg.validation_gates["minimum_improvement_pct"]),
        "no_large_episode_regression": not regressions,
        "heldout_only": all("-heldout-" in name for name in plan_cfg.heldout_episodes),
    }
    result = ValidationResult(
        run_id=fit_run.run_id,
        created_at=utc_now(),
        fit=fit_run,
        baseline_metrics=baseline.metrics,
        calibrated_metrics=calibrated.metrics,
        per_episode=per_episode,
        improvement_pct=improvement,
        regressions=regressions,
        passed=all(gates.values()),
        gates=gates,
    )
    write_json(Path(plan_cfg.workdir) / "validation.json", result)
    return result


def write(validation: ValidationResult, *, output: str | Path) -> CalibrationPackage:
    """Call 5/5: emit the setup-scoped package and complete job record."""
    _assert_locked_inputs_unchanged(validation.fit.plan)
    return write_package(validation, output)


def _environment_spec(env: Any) -> EnvironmentSpec:
    if isinstance(env, EnvironmentSpec):
        return env
    if hasattr(env, "describe"):
        spec = env.describe()
        if isinstance(spec, EnvironmentSpec):
            return spec
    raise TypeError("env must be an EnvironmentSpec or expose describe() -> EnvironmentSpec")


def _evidence_adapter(value: str | Path | AnchorLabSO101Evidence, revision: str) -> AnchorLabSO101Evidence:
    return value if isinstance(value, AnchorLabSO101Evidence) else AnchorLabSO101Evidence(value, revision=revision)


def _select_episodes(available: list[str], selectors: tuple[str, ...]) -> list[str]:
    selected: list[str] = []
    for selector in selectors:
        matches = [name for name in available if selector in name]
        if len(matches) != 1:
            raise ValueError(f"Recipe selector {selector!r} resolved to {len(matches)} episodes")
        selected.append(matches[0])
    return selected


def _positive_integer(label: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if normalized <= 0 or normalized != value:
        raise ValueError(f"{label} must be a positive integer")
    return normalized


def _assert_locked_inputs_unchanged(
    calibration_plan: CalibrationPlan,
    *,
    evidence: AnchorLabSO101Evidence | None = None,
) -> None:
    """Reject a fit or validation if evidence or USD bytes drift after planning."""

    evidence_adapter = evidence or _evidence_adapter(
        calibration_plan.evidence_uri,
        calibration_plan.evidence_revision,
    )
    current_evidence_fingerprint = evidence_adapter.inventory()["fingerprint"]
    if current_evidence_fingerprint != calibration_plan.evidence_fingerprint:
        raise RuntimeError(
            "Evidence fingerprint changed after the calibration plan was locked; run analyze() and plan() again"
        )
    if not calibration_plan.asset_fingerprint:
        raise RuntimeError("Calibration plan has no locked USD fingerprint; run analyze() and plan() again")
    current_asset_fingerprint = sha256_file(calibration_plan.environment.asset_path)
    if current_asset_fingerprint != calibration_plan.asset_fingerprint:
        raise RuntimeError(
            "USD asset fingerprint changed after the calibration plan was locked; run analyze() and plan() again"
        )


def _evaluate(runtime, candidate, episodes, plan_cfg, candidate_id, generation) -> CandidateEvaluation:
    score, metrics, per_episode, stable = runtime.evaluate(candidate, episodes, plan_cfg.objective_weights)
    return CandidateEvaluation(
        candidate_id=candidate_id,
        generation=generation,
        parameters=dict(candidate),
        score=float(score),
        metrics=metrics,
        episodes=per_episode,
        stable=stable,
    )


def _identify_parameters(inventory: dict[str, Any]) -> dict[str, str]:
    names = inventory["train_episodes"]
    signals = set(inventory["signals"])
    joints = set(inventory["joints"])
    reasons: dict[str, str] = {}
    dynamic = any(
        any(label in name for label in ("step-response", "chirp-sweep", "prbs", "multisine")) for name in names
    )
    holding = any(any(label in name for label in ("static-holding", "friction", "gravity")) for name in names)
    gripper = "jaw" in joints and any("gripper-cycles" in name for name in names)
    timed = {"command_q", "actual_q", "dq"}.issubset(signals) and bool(inventory["sample_rates_hz"])
    if dynamic:
        reasons.update(
            {
                "arm_stiffness_scale": "step/chirp/PRBS excitation exposes command-to-position gain",
                "arm_damping_scale": "dynamic excitation exposes overshoot and settling",
                "arm_armature": "dynamic excitation exposes reflected motor and gearbox inertia",
                "arm_effort_scale": "large/fast excitation exposes saturation behavior",
            }
        )
    if holding:
        reasons["arm_friction_nm"] = "static/gravity evidence exposes low-speed holding loss"
    if gripper:
        reasons.update(
            {
                "gripper_stiffness_scale": "jaw command cycles expose gripper response gain",
                "gripper_damping_scale": "jaw command cycles expose gripper settling",
                "gripper_armature": "jaw reversals expose reflected gripper motor inertia",
                "gripper_friction_nm": "jaw reversals expose a bounded hysteresis/friction proxy",
                "gripper_effort_scale": "jaw command cycles expose gripper saturation behavior",
            }
        )
    if timed:
        reasons["command_delay_s"] = "independently timestamped command and state streams expose latency"
    return reasons


def _fit_execution_fingerprint(
    calibration_plan: CalibrationPlan,
    *,
    optimizer_name: str,
    optimizer_version: str,
    optimizer_provider: str,
    initialization: OptimizerInit,
) -> str:
    """Fingerprint every locked fit input except the extendable generation ceiling."""
    plan_payload = jsonable(calibration_plan)
    plan_payload.pop("created_at", None)
    plan_payload.pop("workdir", None)
    optimizer_payload = dict(plan_payload["optimizer"])
    optimizer_payload.pop("generations", None)
    optimizer_payload.update(
        {
            "name": optimizer_name,
            "version": optimizer_version,
            "provider": optimizer_provider,
            "population": initialization.population,
            "seed": initialization.seed,
            "options": dict(initialization.options),
        }
    )
    plan_payload["optimizer"] = optimizer_payload
    try:
        encoded = json.dumps(
            {"schema": "newton.calibration.fit-execution/v1", "plan": plan_payload},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OptimizerContractError("fit execution configuration must be JSON serializable and finite") from exc
    return hashlib.sha256(encoded).hexdigest()
