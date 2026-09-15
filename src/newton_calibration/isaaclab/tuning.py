from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from newton_calibration.adapters.asset import (
    inspect_usd as inspect_usd_asset,
)
from newton_calibration.adapters.asset import (
    supported_parameter_names,
    validate_articulation_asset,
    validate_so101_asset,
)
from newton_calibration.adapters.evidence import AnchorLabSO101Evidence, TabularJointEvidence
from newton_calibration.adapters.runtime import create_runtime
from newton_calibration.collection import CollectionPlan, MotionSpec, create_collection_plan, prepare_assistance
from newton_calibration.collection.contracts import CalibrationRequest
from newton_calibration.collection.planning import ScenePreview
from newton_calibration.collection.registry import get_catalog
from newton_calibration.core.attestation import record_fingerprint
from newton_calibration.core.evidence_spec import BoundEvidenceSpec
from newton_calibration.core.fit_journal import FitJournal
from newton_calibration.core.io import sha256_file, utc_now, write_json
from newton_calibration.core.joint_mapping import JointMappingReport, propose_joint_mapping
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
    evidence: str | Path | AnchorLabSO101Evidence | TabularJointEvidence | BoundEvidenceSpec | None = None,
    recipe: str | None = None,
    request: CalibrationRequest | None = None,
    evidence_revision: str = "local",
    workdir: str | Path = "runs",
) -> AnalysisResult:
    """Call 1/5: inventory evidence and determine recipe/evidence readiness.

    The signal/name checks below establish that a parameter is reasonable to
    include in this recipe.  They are not a numerical identifiability proof.
    With no evidence, emit an asset/readiness report and agent-assisted next
    actions. plan() routes to collection, never to an unqualified fitting job.
    """
    environment = _lock_residual_fingerprint(_environment_spec(env))
    request = request or CalibrationRequest()
    if request.target_parameters:
        if environment.profile_schema != "articulation-profile/v1":
            raise ValueError("Parameter-scoped requests require a generic articulation profile")
        environment = replace(environment, tuning_targets=request.target_parameters)
    adapter = _evidence_adapter(evidence, evidence_revision)
    inventory = adapter.inventory()
    generic = environment.profile_schema == "articulation-profile/v1"
    recipe_name = recipe or ("articulation.position_pd.free_space@3" if generic else "so101_actuator_dynamics.v1")
    recipe_cfg = get_recipe(recipe_name, environment if generic else None)
    if generic:
        asset_report = validate_articulation_asset(environment)
        asset_errors = list(asset_report.blockers)
        asset_warnings = list(asset_report.warnings)
    else:
        asset_errors = validate_so101_asset(environment)
        asset_warnings = []
    exposed = supported_parameter_names(environment)
    identifiability = _identify_parameters(inventory, environment if generic else None)
    identifiable = [
        parameter
        for parameter in recipe_cfg.parameters
        if parameter.name in exposed and parameter.name in identifiability
    ]
    required_joint_order = _ordered_joints(environment) if generic else list(environment.joint_map)
    required_joints = set(required_joint_order)
    evidence_joints = set(inventory.get("source_joints", inventory["joints"]))
    mapping_report = _mapping_report(environment, adapter, required_joints) if generic else {}
    residual_errors = _residual_errors(environment, required_joint_order)
    requested_names = set(recipe_cfg.required_parameter_names)
    declared_names = {parameter.name for parameter in recipe_cfg.parameters}
    readiness = {
        "asset_exists": not any(error.startswith("USD asset does not exist") for error in asset_errors),
        "asset_profile_valid": not asset_errors,
        "residual_model_valid": not residual_errors,
        "joint_map_complete": required_joints.issubset(evidence_joints)
        and not any("joint map" in error.lower() or "mapping" in error.lower() for error in asset_errors)
        and (not generic or bool(mapping_report.get("ready"))),
        "required_signals_present": bool(inventory["required_signals_present"]),
        "train_split_present": bool(inventory["train_episodes"]),
        "heldout_split_present": bool(inventory["heldout_episodes"]),
        "requested_parameter_surface_exposed": requested_names.issubset(exposed),
        "requested_parameter_bounds_declared": requested_names == declared_names,
        "requested_parameters_identifiable": {parameter.name for parameter in identifiable} == requested_names,
    }
    warnings = list(asset_errors) + asset_warnings + residual_errors
    if isinstance(adapter, _MissingEvidence):
        warnings.append(
            "No real evidence supplied: this is asset/evidence-readiness analysis, not measured-data analysis or calibration."
        )
    if generic:
        missing_bounds = sorted(requested_names - declared_names)
        if missing_bounds:
            warnings.append(
                "Robot-specific safe bounds are required for absolute armature/friction parameters: "
                + ", ".join(missing_bounds)
            )
        unqualified = sorted(requested_names - set(identifiability))
        if unqualified:
            warnings.append(
                "Evidence does not qualify these parameters for fitting: "
                + ", ".join(unqualified)
                + ". Delay requires synchronized clocks; dynamic terms require measured motion excitation; "
                "friction requires bidirectional reversals; effort scale requires an explicit saturation observation."
            )
        warnings.extend(
            [
                "The mapping is a locked coordinate transform, not an optimizer variable; ambiguous names, units, signs, or offsets must be confirmed before planning.",
                "MVP1 validates free-space position-controlled articulation dynamics only; it does not claim contact, grasp force, or task transfer.",
            ]
        )
    else:
        warnings.extend(
            [
                "Anchor-Lab does not publish a complete units/controller dictionary; q and dq are treated as radians and radians/s.",
                "present_load_raw and tau_abs are excluded from the primary objective because raw-load calibration and torque sign are unavailable.",
                "MVP1 validates free-space arm and unloaded gripper actuation; it does not claim absolute gripping force or contact fidelity.",
                "The released SO-101 USD is already calibrated; use an explicitly declared untuned asset for unbiased improvement claims.",
            ]
        )
    prefix = _run_prefix(environment.robot_id if generic else "so101")
    run_id = f"{prefix}-{uuid.uuid4().hex[:10]}"
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
        recipe=recipe_cfg.name,
        evidence_spec=_describe_evidence(adapter),
        mapping_report=mapping_report,
        collection_request=jsonable(request),
        evidence_needs=get_catalog(recipe_cfg.collection_catalog).assess(
            environment, recipe_cfg.required_parameter_names, inventory, identifiability, request
        ),
    )
    result.assistance = prepare_assistance(result)
    write_json(run_dir / "analysis.json", result)
    return result


def assist(
    *,
    env: Any,
    evidence: Any = None,
    recipe: str | None = None,
    request: CalibrationRequest | None = None,
    collection: MotionSpec | None = None,
    preview: ScenePreview | None = None,
    video: bool = True,
    workdir: str | Path = "runs",
) -> CalibrationPlan | CollectionPlan:
    """Agent-facing analyze → plan convenience; no LLM or hardware execution.

    A scene surface may expose describe_collection() and preview_collection().
    External agents (including a Minjae integration) use this same deterministic
    contract; this helper does not pretend to install or run such an agent.
    """
    if collection is None and hasattr(env, "describe_collection"):
        collection = env.describe_collection()
    if preview is None and hasattr(env, "preview_collection"):
        preview = env.preview_collection
    analysis = analyze(env=env, evidence=evidence, recipe=recipe, request=request, workdir=workdir)
    return plan(analysis, collection=collection, preview=preview, video=video)


def plan(
    analysis: AnalysisResult,
    *,
    recipe: str | None = None,
    optimizer: str | None = None,
    optimizer_options: dict[str, Any] | None = None,
    intent: str = "auto",
    collection: MotionSpec | None = None,
    preview: ScenePreview | None = None,
    video: bool = True,
) -> CalibrationPlan | CollectionPlan:
    """Call 2/5: plan fitting, or generate evidence-collection commands.

    Missing evidence routes to collection by default. With a scene motion
    specification, CSVs are generated and the bound scene preview is called
    automatically (video=True). An unavailable renderer is a durable pending
    action, not a fabricated video or a fit-ready calibration plan.
    """
    if intent not in {"auto", "fit", "collect"}:
        raise ValueError("intent must be auto, fit or collect")
    recorded_analysis = json.loads((Path(analysis.workdir) / "analysis.json").read_text())
    if recorded_analysis != jsonable(analysis):
        raise ValueError("Analysis changed after analyze(); create a new analysis revision")
    if recipe is not None and analysis.recipe and recipe != analysis.recipe:
        raise ValueError("Rerun analyze() before changing the recipe")
    evidence_checks = (
        "joint_map_complete",
        "required_signals_present",
        "train_split_present",
        "heldout_split_present",
        "requested_parameters_identifiable",
    )
    evidence_gaps = any(not analysis.readiness.get(key, False) for key in evidence_checks)
    if intent == "collect" or (
        intent == "auto"
        and (analysis.evidence_spec.get("adapter") == "missing" or (collection is not None and evidence_gaps))
    ):
        return create_collection_plan(analysis, motion=collection, preview=preview, video=video)
    failed = [name for name, ready in analysis.readiness.items() if not ready]
    if failed:
        raise ValueError(f"Cannot plan calibration; readiness checks failed: {failed}")
    selected_recipe = recipe or analysis.recipe or "so101_actuator_dynamics.v1"
    if analysis.recipe and selected_recipe != analysis.recipe:
        raise ValueError(
            f"Analysis used recipe {analysis.recipe!r}; rerun analyze() before changing to {selected_recipe!r}"
        )
    recipe_cfg = get_recipe(
        selected_recipe,
        analysis.environment if analysis.environment.profile_schema == "articulation-profile/v1" else None,
    )
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
        evidence_spec=dict(analysis.evidence_spec),
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
    if not isinstance(calibration_plan, CalibrationPlan):
        raise TypeError(
            "fit requires a CalibrationPlan backed by real evidence, not a CollectionPlan; collect and re-analyze first"
        )
    run_dir = Path(calibration_plan.workdir)
    history_path = run_dir / "candidate-history.jsonl"
    evidence = _evidence_adapter_from_plan(calibration_plan)
    _assert_locked_inputs_unchanged(calibration_plan, evidence=evidence)
    duration = calibration_plan.optimizer["max_episode_duration_s"]
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
    runtime_attestation: dict[str, Any] = {}
    try:
        initial = {parameter.name: parameter.initial for parameter in calibration_plan.parameters}
        baseline_eval = _evaluate(
            runtime, initial, episodes, calibration_plan, candidate_id=-1, generation=-1, phase="fit-baseline"
        )
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
                        runtime,
                        candidate,
                        episodes,
                        calibration_plan,
                        candidate_id=candidate_id,
                        generation=generation,
                        phase="fit-search",
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
            phase="fit-selected",
        )
        # Capture the attestation only after the selected result has actually
        # executed.  A successfully constructed runtime is not sufficient
        # evidence that its parameter surface was exercised by the fit.
        runtime_attestation = runtime.attestation()
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
        runtime_attestation=runtime_attestation,
    )
    write_json(run_dir / "fit.json", result)
    return result


def validate(fit_run: FitResult) -> ValidationResult:
    """Call 4/5: compare baseline and calibrated parameters on data excluded from fitting."""
    plan_cfg = fit_run.plan
    canonical_baseline = {parameter.name: parameter.initial for parameter in plan_cfg.parameters}
    if fit_run.baseline.parameters != canonical_baseline:
        raise ValueError("Fit baseline parameters do not match the locked recipe initials")
    evidence = _evidence_adapter_from_plan(plan_cfg)
    _assert_locked_inputs_unchanged(plan_cfg, evidence=evidence)
    duration = plan_cfg.optimizer["max_episode_duration_s"]
    episodes = [
        evidence.load_episode(name, dt=plan_cfg.environment.dt, max_duration_s=duration)
        for name in plan_cfg.heldout_episodes
    ]
    runtime = create_runtime(plan_cfg.environment)
    runtime_attestation: dict[str, Any] = {}
    try:
        baseline = _evaluate(
            runtime,
            canonical_baseline,
            episodes,
            plan_cfg,
            candidate_id=-1,
            generation=-1,
            phase="heldout-baseline",
        )
        baseline_attestation = runtime.attestation()
        calibrated = _evaluate(
            runtime,
            fit_run.best.parameters,
            episodes,
            plan_cfg,
            candidate_id=-2,
            generation=-1,
            phase="heldout-validation",
        )
        runtime_attestation = {
            "baseline": baseline_attestation,
            "calibrated": runtime.attestation(),
        }
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
        "heldout_only": all(episode.split == "heldout" for episode in episodes),
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
        runtime_attestation=runtime_attestation,
        baseline_stable=baseline.stable,
        calibrated_stable=calibrated.stable,
    )
    write_json(Path(plan_cfg.workdir) / "validation.json", result)
    return result


def write(validation: ValidationResult, *, output: str | Path) -> CalibrationPackage:
    """Call 5/5: emit the setup-scoped package and complete job record."""
    _assert_locked_inputs_unchanged(validation.fit.plan)
    _assert_result_records_unchanged(validation)
    return write_package(validation, output)


def _environment_spec(env: Any) -> EnvironmentSpec:
    if isinstance(env, EnvironmentSpec):
        return env
    if hasattr(env, "describe"):
        spec = env.describe()
        if isinstance(spec, EnvironmentSpec):
            return spec
    raise TypeError("env must be an EnvironmentSpec or expose describe() -> EnvironmentSpec")


def _lock_residual_fingerprint(environment: EnvironmentSpec) -> EnvironmentSpec:
    if not environment.residual_model_path or environment.residual_model_sha256:
        return environment
    path = Path(environment.residual_model_path).expanduser().resolve()
    if not path.is_file():
        return environment
    return replace(environment, residual_model_sha256=sha256_file(path))


def _residual_errors(environment: EnvironmentSpec, logical_joints: list[str]) -> list[str]:
    if not environment.residual_model_path:
        return []
    path = Path(environment.residual_model_path).expanduser().resolve()
    if not path.is_file():
        return [f"Actuator residual does not exist: {path}"]
    current = sha256_file(path)
    if environment.residual_model_sha256 != current:
        return ["Actuator residual fingerprint does not match the locked environment"]
    try:
        from newton_calibration.actuators import load_residual

        load_residual(path, logical_joints)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"Actuator residual is invalid for the controlled joint order: {exc}"]
    return []


class _MissingEvidence:
    """Explicit absence, never a synthetic episode or a fit-capable adapter."""

    uri = "evidence:not-collected"
    revision = "not-collected"

    def inventory(self) -> dict[str, Any]:
        return {
            "fingerprint": hashlib.sha256(b"newton.calibration:no-evidence/v1").hexdigest(),
            "joints": [],
            "source_joints": [],
            "signals": [],
            "train_episodes": [],
            "heldout_episodes": [],
            "sample_rates_hz": {},
            "required_signals_present": False,
            "clock_synchronized": False,
        }


def _evidence_adapter(value: Any, revision: str):
    if value is None:
        return _MissingEvidence()
    if isinstance(value, (AnchorLabSO101Evidence, TabularJointEvidence)):
        return value
    if isinstance(value, BoundEvidenceSpec):
        return TabularJointEvidence(value)
    if isinstance(value, dict) and value.get("adapter") == "tabular_joint.v1":
        return TabularJointEvidence(BoundEvidenceSpec.from_dict(value))
    if isinstance(value, (str, Path)) and Path(value).suffix.lower() == ".json":
        # JSON is the serialized generic evidence contract.  Never reinterpret
        # a malformed contract as a legacy evidence locator: doing so could
        # silently discard its split, mapping, or unit semantics.
        return TabularJointEvidence(BoundEvidenceSpec.read(value))
    return AnchorLabSO101Evidence(value, revision=revision)


def _evidence_adapter_from_plan(plan_cfg: CalibrationPlan):
    if plan_cfg.evidence_spec.get("adapter") == "tabular_joint.v1":
        return TabularJointEvidence(BoundEvidenceSpec.from_dict(plan_cfg.evidence_spec))
    return _evidence_adapter(plan_cfg.evidence_uri, plan_cfg.evidence_revision)


def _describe_evidence(adapter: Any) -> dict[str, Any]:
    if isinstance(adapter, _MissingEvidence):
        return {
            "adapter": "missing",
            "root": adapter.uri,
            "revision": adapter.revision,
            "real_samples": 0,
            "fit_allowed": False,
        }
    if isinstance(adapter, TabularJointEvidence):
        payload = adapter.spec.to_dict()
        payload["fingerprint"] = adapter.spec.fingerprint
        payload["mapping_fingerprint"] = adapter.spec.mapping_fingerprint
        return payload
    return {
        "adapter": "anchor_lab_so101.v1",
        "root": adapter.uri,
        "revision": adapter.revision,
    }


def _select_episodes(available: list[str], selectors: tuple[str, ...]) -> list[str]:
    if not selectors:
        return list(available)
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
    evidence: Any | None = None,
) -> None:
    """Reject a fit or validation if evidence or USD bytes drift after planning."""

    plan_path = Path(calibration_plan.workdir).expanduser().resolve() / "plan.json"
    if not plan_path.is_file():
        raise RuntimeError("Locked calibration plan record is missing")
    try:
        recorded_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Locked calibration plan record is unreadable") from exc
    if recorded_plan != jsonable(calibration_plan):
        raise RuntimeError("Calibration plan changed after plan() locked it; create a new plan")

    evidence_adapter = evidence or _evidence_adapter_from_plan(calibration_plan)
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
    residual_path = calibration_plan.environment.residual_model_path
    residual_sha256 = calibration_plan.environment.residual_model_sha256
    if residual_path:
        path = Path(residual_path).expanduser().resolve()
        if not path.is_file() or residual_sha256 is None or sha256_file(path) != residual_sha256:
            raise RuntimeError(
                "Actuator residual changed after the calibration plan was locked; run analyze() and plan() again"
            )
    elif residual_sha256 is not None:
        raise RuntimeError("Calibration plan contains a residual fingerprint without a residual model path")


def _assert_result_records_unchanged(validation: ValidationResult) -> None:
    run_dir = Path(validation.fit.plan.workdir).expanduser().resolve()
    for name, expected in (
        ("fit.json", jsonable(validation.fit)),
        ("validation.json", jsonable(validation)),
    ):
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"Durable calibration result is missing: {name}")
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Durable calibration result is unreadable: {name}") from exc
        if actual != expected:
            raise RuntimeError(f"In-memory calibration result differs from durable {name}")


def _evaluate(runtime, candidate, episodes, plan_cfg, candidate_id, generation, *, phase) -> CandidateEvaluation:
    score, metrics, per_episode, stable = runtime.evaluate(
        candidate,
        episodes,
        plan_cfg.objective_weights,
        phase=phase,
        run_id=plan_cfg.run_id,
        plan_sha256=record_fingerprint(jsonable(plan_cfg)),
        evidence_fingerprint=plan_cfg.evidence_fingerprint,
        mapping_fingerprint=str(plan_cfg.evidence_spec.get("mapping_fingerprint", "")),
    )
    return CandidateEvaluation(
        candidate_id=candidate_id,
        generation=generation,
        parameters=dict(candidate),
        score=float(score),
        metrics=metrics,
        episodes=per_episode,
        stable=stable,
    )


def _identify_parameters(inventory: dict[str, Any], environment: EnvironmentSpec | None = None) -> dict[str, str]:
    if environment is not None and environment.profile_schema == "articulation-profile/v1":
        signals = set(inventory["signals"])
        evidence_joints = set(inventory.get("source_joints", inventory["joints"]))
        dynamically_excited = set(inventory.get("dynamic_excitation_joints", ()))
        reversed_joints = set(inventory.get("reversal_joints", ()))
        saturation_joints = set(inventory.get("effort_saturation_joints", ()))
        reasons: dict[str, str] = {}
        complete_signals = {"command_q", "actual_q", "actual_dq"}.issubset(signals)
        for group, members in environment.joint_groups.items():
            group_joints = set(members)
            covered = group_joints.issubset(evidence_joints) and bool(inventory["train_episodes"])
            if complete_signals and covered and group_joints.issubset(dynamically_excited):
                reasons.update(
                    {
                        f"{group}_stiffness_scale": f"measured dynamic excitation covers every {group} coordinate",
                        f"{group}_damping_scale": f"measured dynamic excitation covers every {group} coordinate",
                        f"{group}_armature": f"measured dynamic excitation covers every {group} coordinate",
                    }
                )
            if complete_signals and covered and group_joints.issubset(reversed_joints):
                reasons[f"{group}_friction_nm"] = f"measured bidirectional reversals cover every {group} coordinate"
            if complete_signals and covered and group_joints.issubset(saturation_joints):
                reasons[f"{group}_effort_scale"] = (
                    f"independently declared effort saturation covers every {group} coordinate"
                )
        timing_rates = inventory.get("sample_rates_hz", {})
        timed_streams = {"command_q", "actual_q"}.issubset(timing_rates)
        if (
            complete_signals
            and inventory.get("clock_synchronized") is True
            and timed_streams
            and bool(dynamically_excited)
        ):
            reasons["command_delay_s"] = (
                "synchronized command/state clocks and measured excitation qualify latency scoring"
            )
        return reasons
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


def inspect_usd(path: str | Path):
    """Agent preflight: enumerate supported USD joints and typed blockers."""

    return inspect_usd_asset(path)


def propose_mapping(*, usd: str | Path | Any, source_joints: list[str] | tuple[str, ...]) -> JointMappingReport:
    """Agent preflight: propose only deterministic exact/normalized joint matches."""

    report = inspect_usd_asset(usd) if isinstance(usd, (str, Path)) else usd
    targets = [joint.name for joint in report.joints]
    return propose_joint_mapping(source_joints, targets)


def inspect_evidence(evidence: Any, *, evidence_revision: str = "local") -> dict[str, Any]:
    """Agent preflight: inventory a configured evidence source without fitting."""

    return _evidence_adapter(evidence, evidence_revision).inventory()


def _mapping_report(environment: EnvironmentSpec, adapter: Any, required_joints: set[str]) -> dict[str, Any]:
    if not isinstance(adapter, TabularJointEvidence):
        return {
            "ready": False,
            "blockers": ["Generic articulation calibration requires a bound evidence manifest"],
        }
    bindings = adapter.spec.joint_bindings
    binding_map = {item.source_joint: item.usd_joint for item in bindings}
    expected_order = _ordered_joints(environment)
    binding_order = [item.source_joint for item in bindings]
    blockers: list[str] = []
    if binding_order != expected_order:
        blockers.append(
            "Bound evidence joint order does not match the robot profile: "
            f"expected {expected_order}, found {binding_order}"
        )
    expected_map = {name: environment.joint_map.get(name) for name in expected_order}
    if binding_map != expected_map:
        blockers.append("Bound evidence mapping does not exactly match the confirmed robot profile")
    if set(binding_map) != required_joints:
        blockers.append("Bound evidence does not cover exactly the controlled logical joints")
    non_radian_targets = sorted(item.source_joint for item in bindings if item.usd_unit != "rad")
    if non_radian_targets:
        blockers.append(
            "The MVP1 Newton revolute-joint runtime requires target units in radians; "
            f"non-radian bindings: {non_radian_targets}"
        )
    return {
        "ready": not blockers,
        "blockers": blockers,
        "mapping_fingerprint": adapter.spec.mapping_fingerprint,
        "bindings": [item.to_dict() for item in bindings],
    }


def _run_prefix(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "-" for character in value.casefold())
    return normalized.strip("-") or "articulation"


def _ordered_joints(environment: EnvironmentSpec) -> list[str]:
    return list(environment.joint_order) or [
        joint for members in environment.joint_groups.values() for joint in members
    ]


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
