from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml

from newton_calibration.adapters.runtime.analytic import _joint_properties, _resolve_joint_layout
from newton_calibration.core.attestation import (
    episode_inputs,
    evaluation_result_fingerprint,
    numeric_surface_fingerprint,
    parameter_fingerprint,
    record_fingerprint,
)
from newton_calibration.core.evidence_spec import BoundEvidenceSpec
from newton_calibration.core.fit_journal import FitJournal, FitJournalError
from newton_calibration.core.io import sha256_file, write_json
from newton_calibration.core.models import CalibrationPackage, ValidationResult, jsonable


def write_package(validation: ValidationResult, output: str | Path) -> CalibrationPackage:
    _verify_durable_result_records(validation)
    if validation.fit.plan.environment.profile_schema == "articulation-profile/v1":
        return _write_articulation_package(validation, output)
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Calibration package destination is not empty: {output_dir}; use a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = validation.fit.plan
    canonical_baseline = {parameter.name: parameter.initial for parameter in plan.parameters}
    if validation.fit.baseline.parameters != canonical_baseline:
        raise ValueError("Fit baseline parameters do not match the locked recipe initials")
    params = validation.fit.best.parameters
    source_asset = Path(plan.environment.asset_path).expanduser().resolve()
    if not source_asset.is_file():
        raise FileNotFoundError(f"Cannot package missing source asset: {source_asset}")
    current_asset_fingerprint = sha256_file(source_asset)
    if current_asset_fingerprint != plan.asset_fingerprint:
        raise RuntimeError("USD asset fingerprint changed after calibration; validate a new plan before packaging")
    self_contained, dependency_reason = _self_contained_asset(source_asset)
    fit_attestation_valid = _valid_newton_attestation(
        validation.fit.runtime_attestation,
        environment=plan.environment,
        asset_sha256=current_asset_fingerprint,
        candidate=params,
        phase="fit-selected",
        evidence_episodes=_expected_episode_inputs(plan, plan.train_episodes),
        result_sha256=_candidate_result_sha256(validation.fit.best),
        run_id=plan.run_id,
        plan_sha256=record_fingerprint(jsonable(plan)),
        evidence_fingerprint=plan.evidence_fingerprint,
        mapping_fingerprint=str(plan.evidence_spec.get("mapping_fingerprint", "")),
    )
    validation_attestation_valid = _valid_heldout_attestation(
        validation.runtime_attestation,
        environment=plan.environment,
        asset_sha256=current_asset_fingerprint,
        evidence_episodes=_expected_episode_inputs(plan, plan.heldout_episodes),
        validation=validation,
        plan=plan,
    )
    activation_allowed = (
        validation.passed
        and plan.environment.adapter == "isaaclab_newton"
        and fit_attestation_valid
        and validation_attestation_valid
        and self_contained
    )
    if activation_allowed:
        status = "validated"
    elif validation.passed:
        status = "contract-validated-nonactivatable"
    else:
        status = "rejected"
    packaged_asset = output_dir / source_asset.name
    shutil.copy2(source_asset, packaged_asset)
    if sha256_file(packaged_asset) != current_asset_fingerprint:
        raise RuntimeError("USD asset changed while the calibration package was being created")

    job_dir = output_dir / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(plan.workdir).expanduser().resolve()
    job_files = (
        "analysis.json",
        "plan.json",
        "baseline.json",
        "fit-checkpoint.json",
        "candidate-history.jsonl",
        "fit.json",
        "validation.json",
    )
    for name in job_files:
        source = run_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Cannot package missing durable job record: {source}")
        shutil.copy2(source, job_dir / name)
    generation_records = run_dir / "fit-generations"
    if not generation_records.is_dir() or not any(generation_records.glob("*.json")):
        raise FileNotFoundError(f"Cannot package missing authoritative fit generations: {generation_records}")
    shutil.copytree(generation_records, job_dir / "fit-generations")

    delay_steps = max(0, round(params["command_delay_s"] / plan.environment.dt))
    effective_delay_s = delay_steps * plan.environment.dt
    overlay = output_dir / ("calibration.usda" if activation_allowed else "nonactivatable_candidate.usda")
    overlay.write_text(
        "#usda 1.0\n"
        "(\n"
        f"    subLayers = [@./{packaged_asset.name}@]\n"
        "    customLayerData = {\n"
        f'        string newtonCalibrationRun = "{validation.run_id}"\n'
        '        string newtonCalibrationManifest = "manifest.json"\n'
        '        string newtonCalibrationScope = "SO-101 free-space arm and unloaded gripper actuation"\n'
        "    }\n"
        ")\n",
        encoding="utf-8",
    )
    isaaclab_cfg = output_dir / "isaaclab_actuator.yaml"
    isaaclab_cfg.write_text(
        "recipe: so101_actuator_dynamics.v1\n"
        f"source_asset: {json.dumps('./' + packaged_asset.name)}\n"
        "actuators:\n"
        f"  arm_stiffness: {plan.environment.base_stiffness * params['arm_stiffness_scale']:.10g}\n"
        f"  arm_damping: {plan.environment.base_damping * params['arm_damping_scale']:.10g}\n"
        f"  arm_armature: {params['arm_armature']:.10g}\n"
        f"  arm_friction_nm: {params['arm_friction_nm']:.10g}\n"
        f"  arm_effort_limit: {plan.environment.base_effort_limit * params['arm_effort_scale']:.10g}\n"
        f"  gripper_stiffness: {plan.environment.base_stiffness * params['gripper_stiffness_scale']:.10g}\n"
        f"  gripper_damping: {plan.environment.base_damping * params['gripper_damping_scale']:.10g}\n"
        f"  gripper_armature: {params['gripper_armature']:.10g}\n"
        f"  gripper_friction_nm: {params['gripper_friction_nm']:.10g}\n"
        f"  gripper_effort_limit: {plan.environment.base_effort_limit * params['gripper_effort_scale']:.10g}\n"
        f"command_delay_s: {params['command_delay_s']:.10g}\n"
        f"command_delay_steps_at_runtime_dt: {delay_steps}\n"
        f"command_delay_effective_s_at_runtime_dt: {effective_delay_s:.10g}\n",
        encoding="utf-8",
    )
    validation_path = write_json(output_dir / "validation.json", validation)
    residual_artifact = None
    residual_sha256 = None
    if plan.environment.residual_model_path:
        residual_source = Path(plan.environment.residual_model_path).expanduser().resolve()
        residual_target = output_dir / "actuator_residual.json"
        residual_sha256 = sha256_file(residual_source)
        shutil.copy2(residual_source, residual_target)
        if sha256_file(residual_target) != residual_sha256:
            raise RuntimeError("Actuator residual changed while the calibration package was being created")
        residual_artifact = residual_target.name
    manifest = {
        "schema": "newton.calibration.package/v1",
        "run_id": validation.run_id,
        "created_at": validation.created_at,
        "status": status,
        "activation_allowed": activation_allowed,
        "activation_checks": {
            "heldout_validation_passed": validation.passed,
            "authoritative_fit_attestation": fit_attestation_valid,
            "authoritative_validation_attestation": validation_attestation_valid,
            "self_contained_asset": self_contained,
            "asset_dependency_reason": dependency_reason,
        },
        "scope": "SO-101 free-space arm and unloaded gripper actuation",
        "claims": {
            "heldout_passed": validation.passed,
            "heldout_improvement_pct": validation.improvement_pct,
            "contact_or_grip_force_validated": False,
        },
        "inputs": {
            "asset": plan.environment.asset_path,
            "asset_sha256": current_asset_fingerprint,
            "residual_sha256": residual_sha256,
            "evidence": plan.evidence_uri,
            "evidence_revision": plan.evidence_revision,
            "evidence_fingerprint": plan.evidence_fingerprint,
        },
        "runtime": jsonable(plan.environment),
        "runtime_attestation": {
            "fit": validation.fit.runtime_attestation,
            "heldout_validation": validation.runtime_attestation,
        },
        "recipe": plan.recipe,
        "optimizer": validation.fit.optimizer,
        "parameters": params,
        "parameter_application": {
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
        },
        "timing_quantization": {
            "requested_command_delay_s": params["command_delay_s"],
            "runtime_dt_s": plan.environment.dt,
            "applied_delay_steps": delay_steps,
            "effective_command_delay_s": effective_delay_s,
        },
        "artifacts": {
            "source_asset": packaged_asset.name,
            "usd_provenance_layer": overlay.name,
            "isaaclab_actuator_config": isaaclab_cfg.name,
            "validation": validation_path.name,
            "job_records": "job/",
            "candidate_history": "job/candidate-history.jsonl",
            "optimizer_generation_records": "job/fit-generations/",
            "actuator_residual": residual_artifact,
        },
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    report = output_dir / "report.md"
    if activation_allowed:
        claim = (
            "This package is validated for SO-101 free-space arm motion and unloaded gripper tracking on the "
            "recorded conditions represented by the held-out Anchor-Lab trajectories."
        )
    elif validation.passed:
        claim = (
            "This analytic-backend run passed contract tests only. It is not a Newton physics validation and must "
            "not be activated as a calibrated asset."
        )
    else:
        claim = "This candidate failed held-out validation and must not be activated as a calibrated asset."
    report.write_text(
        f"# SO-101 Newton calibration {'package' if activation_allowed else 'non-activatable candidate'}\n\n"
        f"- Run: `{validation.run_id}`\n"
        f"- Held-out result: **{'PASS' if validation.passed else 'FAIL'}**\n"
        f"- Activation: **{'ALLOWED' if activation_allowed else 'NOT ALLOWED'}**\n"
        f"- Weighted-error improvement: **{validation.improvement_pct:.1f}%**\n"
        f"- Runtime: `{plan.environment.adapter}`\n"
        f"- Optimizer: `{validation.fit.optimizer.get('name', plan.optimizer.get('name', 'unknown'))}` "
        f"(`{validation.fit.optimizer.get('version', 'unknown')}`, provider "
        f"`{validation.fit.optimizer.get('provider', 'unknown')}`)\n"
        f"- Evidence revision: `{plan.evidence_revision}`\n\n"
        "## Product claim\n\n"
        f"{claim} It does not validate absolute grasp force, "
        "object contact, or transfer to a different robot setup.\n\n"
        "## What the package applies\n\n"
        "- Isaac Lab explicit-PD: grouped arm/jaw stiffness, damping, and effort limits.\n"
        "- Newton runtime: grouped arm/jaw armature and joint friction.\n"
        f"- Toolkit replay: requested command delay {params['command_delay_s'] * 1000.0:.3f} ms; "
        f"applied as {delay_steps} steps = {effective_delay_s * 1000.0:.3f} ms at the locked dt.\n"
        f"- `{overlay.name}`: relative source-asset provenance layer; parameter values are in "
        "`isaaclab_actuator.yaml` and `manifest.json`.\n\n"
        "## Gates\n\n"
        + "\n".join(f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in validation.gates.items())
        + "\n",
        encoding="utf-8",
    )
    package = CalibrationPackage(
        run_id=validation.run_id,
        output_dir=str(output_dir),
        overlay_path=str(overlay),
        isaaclab_cfg_path=str(isaaclab_cfg),
        manifest_path=str(manifest_path),
        validation_path=str(validation_path),
        report_path=str(report),
    )
    write_json(output_dir / "package.json", package)
    return package


def _write_articulation_package(validation: ValidationResult, output: str | Path) -> CalibrationPackage:
    """Write the robot-agnostic v2 result without changing the proven v1 format."""

    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Calibration package destination is not empty: {output_dir}; use a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = validation.fit.plan
    canonical_baseline = {parameter.name: parameter.initial for parameter in plan.parameters}
    if validation.fit.baseline.parameters != canonical_baseline:
        raise ValueError("Fit baseline parameters do not match the locked recipe initials")
    environment = plan.environment
    params = validation.fit.best.parameters
    source_asset = Path(environment.asset_path).expanduser().resolve()
    if not source_asset.is_file():
        raise FileNotFoundError(f"Cannot package missing source asset: {source_asset}")
    asset_sha256 = sha256_file(source_asset)
    if asset_sha256 != plan.asset_fingerprint:
        raise RuntimeError("USD asset fingerprint changed after calibration; validate a new plan before packaging")
    self_contained, dependency_reason = _self_contained_asset(source_asset)
    fit_attestation_valid = _valid_newton_attestation(
        validation.fit.runtime_attestation,
        environment=environment,
        asset_sha256=asset_sha256,
        candidate=params,
        phase="fit-selected",
        evidence_episodes=_expected_episode_inputs(plan, plan.train_episodes),
        result_sha256=_candidate_result_sha256(validation.fit.best),
        run_id=plan.run_id,
        plan_sha256=record_fingerprint(jsonable(plan)),
        evidence_fingerprint=plan.evidence_fingerprint,
        mapping_fingerprint=str(plan.evidence_spec.get("mapping_fingerprint", "")),
    )
    validation_attestation_valid = _valid_heldout_attestation(
        validation.runtime_attestation,
        environment=environment,
        asset_sha256=asset_sha256,
        evidence_episodes=_expected_episode_inputs(plan, plan.heldout_episodes),
        validation=validation,
        plan=plan,
    )
    activation_allowed = (
        validation.passed
        and environment.adapter == "isaaclab_newton"
        and fit_attestation_valid
        and validation_attestation_valid
        and self_contained
    )
    status = (
        "validated" if activation_allowed else "contract-validated-nonactivatable" if validation.passed else "rejected"
    )
    assets_dir = output_dir / "assets"
    assets_dir.mkdir()
    packaged_asset = assets_dir / source_asset.name
    shutil.copy2(source_asset, packaged_asset)

    job_dir = output_dir / "job"
    job_dir.mkdir()
    run_dir = Path(plan.workdir).expanduser().resolve()
    for name in (
        "analysis.json",
        "plan.json",
        "baseline.json",
        "fit-checkpoint.json",
        "candidate-history.jsonl",
        "fit.json",
        "validation.json",
    ):
        source = run_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Cannot package missing durable job record: {source}")
        shutil.copy2(source, job_dir / name)
    generations = run_dir / "fit-generations"
    if not generations.is_dir() or not any(generations.glob("*.json")):
        raise FileNotFoundError(f"Cannot package missing authoritative fit generations: {generations}")
    shutil.copytree(generations, job_dir / "fit-generations")
    job_record_sha256 = _job_record_sha256(output_dir, job_dir)

    profile = {
        "schema": "newton.calibration.robot-profile/v1",
        "robot_id": environment.robot_id,
        "asset_sha256": asset_sha256,
        "joint_groups": {name: list(members) for name, members in environment.joint_groups.items()},
        "joint_order": list(environment.joint_order),
        "joint_map": dict(environment.joint_map),
        "profile_confirmed": environment.profile_confirmed,
        "controller_profile_confirmed": environment.controller_profile_confirmed,
        "controller_profile_source": environment.controller_profile_source,
        "base_stiffness_by_joint": dict(environment.base_stiffness_by_joint),
        "base_damping_by_joint": dict(environment.base_damping_by_joint),
        "base_effort_limit_by_joint": dict(environment.base_effort_limit_by_joint),
        "base_armature_by_joint": dict(environment.base_armature_by_joint),
    }
    profile_path = write_json(output_dir / "robot_profile.json", profile)
    evidence_path = write_json(output_dir / "evidence_spec.json", plan.evidence_spec)
    mapping_payload = {
        "schema": "newton.calibration.joint-mapping/v1",
        "mapping_fingerprint": plan.evidence_spec.get("mapping_fingerprint"),
        "bindings": plan.evidence_spec.get("joint_bindings", []),
    }
    # The serialized evidence spec intentionally omits its derived fingerprint;
    # carry the locked analysis fingerprint in the manifest below.
    mapping_path = write_json(output_dir / "joint_mapping.json", mapping_payload)

    ordered_joints = list(environment.joint_order) or [
        joint for members in environment.joint_groups.values() for joint in members
    ]
    group_by_joint = {joint: group for group, members in environment.joint_groups.items() for joint in members}
    patch_entries = []
    for logical_joint in ordered_joints:
        group = group_by_joint[logical_joint]
        patch_entries.append(
            {
                "logical_joint": logical_joint,
                "usd_joint": environment.joint_map[logical_joint],
                "group": group,
                "stiffness": environment.base_stiffness_by_joint.get(logical_joint, environment.base_stiffness)
                * params[f"{group}_stiffness_scale"],
                "damping": environment.base_damping_by_joint.get(logical_joint, environment.base_damping)
                * params[f"{group}_damping_scale"],
                "effort_limit": environment.base_effort_limit_by_joint.get(logical_joint, environment.base_effort_limit)
                * params[f"{group}_effort_scale"],
                "armature": params[f"{group}_armature"],
                "friction_nm": params[f"{group}_friction_nm"],
            }
        )
    delay_steps = max(0, round(params["command_delay_s"] / environment.dt))
    effective_delay_s = delay_steps * environment.dt
    actuator_patch = output_dir / "actuator_patch.yaml"
    actuator_patch.write_text(
        yaml.safe_dump(
            {
                "schema": "newton.calibration.actuator-patch/v1",
                "recipe": plan.recipe,
                "source_asset": f"./assets/{packaged_asset.name}",
                "ordered_joints": patch_entries,
                "command_delay": {
                    "requested_s": params["command_delay_s"],
                    "steps_at_runtime_dt": delay_steps,
                    "effective_s": effective_delay_s,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    overlay = output_dir / ("calibration.usda" if activation_allowed else "nonactivatable_candidate.usda")
    overlay.write_text(
        "#usda 1.0\n"
        "(\n"
        f"    subLayers = [@./assets/{packaged_asset.name}@]\n"
        "    customLayerData = {\n"
        f'        string newtonCalibrationRun = "{validation.run_id}"\n'
        f'        string newtonCalibrationRobot = "{environment.robot_id}"\n'
        '        string newtonCalibrationManifest = "manifest.json"\n'
        "    }\n"
        ")\n",
        encoding="utf-8",
    )
    validation_path = write_json(output_dir / "validation.json", validation)
    residual_artifact = None
    residual_sha256 = None
    if environment.residual_model_path:
        residual_source = Path(environment.residual_model_path).expanduser().resolve()
        residual_target = output_dir / "actuator_residual.json"
        residual_sha256 = sha256_file(residual_source)
        if residual_sha256 != environment.residual_model_sha256:
            raise RuntimeError("Actuator residual fingerprint changed after calibration")
        shutil.copy2(residual_source, residual_target)
        if sha256_file(residual_target) != residual_sha256:
            raise RuntimeError("Actuator residual changed while the calibration package was being created")
        residual_artifact = residual_target.name
    application: dict[str, list[str]] = {}
    for parameter in plan.parameters:
        application.setdefault(parameter.owner, []).append(parameter.name)
    manifest = {
        "schema": "newton.calibration.package/v2",
        "run_id": validation.run_id,
        "created_at": validation.created_at,
        "status": status,
        "activation_allowed": activation_allowed,
        "activation_checks": {
            "heldout_validation_passed": validation.passed,
            "authoritative_fit_attestation": fit_attestation_valid,
            "authoritative_validation_attestation": validation_attestation_valid,
            "self_contained_asset": self_contained,
            "asset_dependency_reason": dependency_reason,
        },
        "scope": f"{environment.robot_id} free-space position-controlled articulation dynamics",
        "claims": {
            "heldout_passed": validation.passed,
            "heldout_improvement_pct": validation.improvement_pct,
            "contact_or_task_transfer_validated": False,
        },
        "inputs": {
            "asset": environment.asset_path,
            "asset_sha256": asset_sha256,
            "residual_sha256": residual_sha256,
            "evidence": plan.evidence_uri,
            "evidence_revision": plan.evidence_revision,
            "evidence_fingerprint": plan.evidence_fingerprint,
            "mapping_fingerprint": plan.evidence_spec.get("mapping_fingerprint"),
        },
        "runtime": jsonable(environment),
        "runtime_attestation": {
            "fit": validation.fit.runtime_attestation,
            "heldout_validation": validation.runtime_attestation,
        },
        "recipe": plan.recipe,
        "optimizer": validation.fit.optimizer,
        "parameters": params,
        "parameter_application": application,
        "timing_quantization": {
            "requested_command_delay_s": params["command_delay_s"],
            "runtime_dt_s": environment.dt,
            "applied_delay_steps": delay_steps,
            "effective_command_delay_s": effective_delay_s,
        },
        "artifacts": {
            "source_asset": f"assets/{packaged_asset.name}",
            "usd_provenance_layer": overlay.name,
            "actuator_patch": actuator_patch.name,
            "robot_profile": profile_path.name,
            "evidence_spec": evidence_path.name,
            "joint_mapping": mapping_path.name,
            "validation": validation_path.name,
            "job_records": "job/",
            "job_record_sha256": job_record_sha256,
            "actuator_residual": residual_artifact,
        },
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    report = output_dir / "report.md"
    report.write_text(
        f"# {environment.robot_id} Newton calibration result\n\n"
        f"- Held-out result: **{'PASS' if validation.passed else 'FAIL'}**\n"
        f"- Activation: **{'ALLOWED' if activation_allowed else 'NOT ALLOWED'}**\n"
        f"- Weighted-error improvement: **{validation.improvement_pct:.1f}%**\n"
        f"- Controlled coordinates: **{len(ordered_joints)}** across "
        f"**{len(environment.joint_groups)}** groups\n"
        f"- Evidence-to-USD mapping: `{plan.evidence_spec.get('mapping_fingerprint', 'recorded in joint_mapping.json')}`\n\n"
        "## Claim boundary\n\n"
        "This setup-scoped package records free-space joint-command replay against held-out real evidence. "
        "It does not validate grasp force, contact, insertion, or end-to-end task transfer.\n",
        encoding="utf-8",
    )
    package = CalibrationPackage(
        run_id=validation.run_id,
        output_dir=str(output_dir),
        overlay_path=str(overlay),
        isaaclab_cfg_path=str(actuator_patch),
        manifest_path=str(manifest_path),
        validation_path=str(validation_path),
        report_path=str(report),
    )
    write_json(output_dir / "package.json", package)
    return package


def _valid_newton_attestation(
    value: dict,
    *,
    environment,
    asset_sha256: str,
    candidate: dict[str, float],
    phase: str,
    evidence_episodes: list[dict[str, str]],
    result_sha256: str,
    run_id: str,
    plan_sha256: str,
    evidence_fingerprint: str,
    mapping_fingerprint: str,
) -> bool:
    layout = _resolve_joint_layout(environment)
    logical_order = list(layout.logical_names)
    runtime_order = list(layout.runtime_names)
    properties = _joint_properties(
        environment,
        layout,
        candidate,
        analytic=False,
    )
    readback_values_sha256 = numeric_surface_fingerprint(
        {
            "stiffness": properties["stiffness"].tolist(),
            "damping": properties["damping"].tolist(),
            "effort_limit": properties["effort"].tolist(),
            "armature": properties["armature"].tolist(),
            "friction_nm": properties["friction"].tolist(),
        }
    )
    residual_sha256 = sha256_file(environment.residual_model_path) if environment.residual_model_path else None
    return value == {
        "schema": "newton.calibration.runtime-attestation/v2",
        "backend": "isaaclab_newton",
        "authoritative": True,
        "robot_id": environment.robot_id,
        "asset_sha256": asset_sha256,
        "logical_joints": logical_order,
        "runtime_joints": runtime_order,
        "runtime_dt_s": environment.dt,
        "gravity": list(environment.gravity),
        "num_substeps": environment.num_substeps,
        "solver_iterations": environment.solver_iterations,
        "solver_tolerance": environment.solver_tolerance,
        "selected_joint_scoped": True,
        "full_state_reset_per_episode": True,
        "readback_parameters": ["stiffness", "damping", "effort_limit", "armature", "friction_nm"],
        "candidate_sha256": parameter_fingerprint(candidate),
        "readback_values_sha256": readback_values_sha256,
        "residual_sha256": residual_sha256,
        "evaluation_phase": phase,
        "run_id": run_id,
        "plan_sha256": plan_sha256,
        "evidence_fingerprint": evidence_fingerprint,
        "mapping_fingerprint": mapping_fingerprint,
        "evidence_episodes": evidence_episodes,
        "result_sha256": result_sha256,
    }


def _candidate_result_sha256(candidate) -> str:
    return evaluation_result_fingerprint(
        score=candidate.score,
        metrics=candidate.metrics,
        episodes=candidate.episodes,
        stable=candidate.stable,
    )


def _valid_heldout_attestation(
    value: dict,
    *,
    environment,
    asset_sha256: str,
    evidence_episodes: list[dict[str, str]],
    validation: ValidationResult,
    plan,
) -> bool:
    if set(value) != {"baseline", "calibrated"}:
        return False
    before = {name: values["baseline"] for name, values in validation.per_episode.items()}
    after = {name: values["calibrated"] for name, values in validation.per_episode.items()}
    baseline_sha256 = evaluation_result_fingerprint(
        score=validation.baseline_metrics["score"],
        metrics=validation.baseline_metrics,
        episodes=before,
        stable=validation.baseline_stable,
    )
    calibrated_sha256 = evaluation_result_fingerprint(
        score=validation.calibrated_metrics["score"],
        metrics=validation.calibrated_metrics,
        episodes=after,
        stable=validation.calibrated_stable,
    )
    return _valid_newton_attestation(
        value["baseline"],
        environment=environment,
        asset_sha256=asset_sha256,
        candidate=validation.fit.baseline.parameters,
        phase="heldout-baseline",
        evidence_episodes=evidence_episodes,
        result_sha256=baseline_sha256,
        run_id=plan.run_id,
        plan_sha256=record_fingerprint(jsonable(plan)),
        evidence_fingerprint=plan.evidence_fingerprint,
        mapping_fingerprint=str(plan.evidence_spec.get("mapping_fingerprint", "")),
    ) and _valid_newton_attestation(
        value["calibrated"],
        environment=environment,
        asset_sha256=asset_sha256,
        candidate=validation.fit.best.parameters,
        phase="heldout-validation",
        evidence_episodes=evidence_episodes,
        result_sha256=calibrated_sha256,
        run_id=plan.run_id,
        plan_sha256=record_fingerprint(jsonable(plan)),
        evidence_fingerprint=plan.evidence_fingerprint,
        mapping_fingerprint=str(plan.evidence_spec.get("mapping_fingerprint", "")),
    )


def _expected_episode_inputs(plan, names: list[str]) -> list[dict[str, str]]:
    if plan.evidence_spec.get("adapter") == "tabular_joint.v1":
        spec = BoundEvidenceSpec.from_dict(plan.evidence_spec)
        by_name = {item.name: item for item in spec.episodes}
        try:
            selected = [by_name[name] for name in names]
        except KeyError as exc:
            raise RuntimeError(f"Locked plan references an undeclared evidence episode: {exc.args[0]}") from exc
        return [
            {
                "name": item.name,
                "split": item.split,
                "trial_id": item.trial_id,
                "source_sha256": item.sha256,
            }
            for item in selected
        ]

    # Legacy SO-101/Anchor-Lab path. Resolve the files again so the package
    # check is anchored to the locked evidence source, not to attestation text.
    from newton_calibration.adapters.evidence import AnchorLabSO101Evidence

    adapter = AnchorLabSO101Evidence(plan.evidence_uri, revision=plan.evidence_revision)
    loaded = [
        adapter.load_episode(name, dt=plan.environment.dt, max_duration_s=0.001)
        for name in names
    ]
    return episode_inputs(loaded)


def _self_contained_asset(path: Path) -> tuple[bool, str]:
    """Return whether copying only the root layer preserves the asset.

    Generic v2 deliberately fails closed until full dependency vendoring is
    implemented. Self-contained USDA files can be proven with a text scan;
    binary layers require OpenUSD's dependency resolver.
    """

    if path.suffix.lower() == ".usda":
        try:
            references = re.findall(r"@([^@]+)@", path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return False, "USDA text could not be decoded"
        if references:
            return False, f"external USD dependencies are not yet vendored: {sorted(set(references))}"
        return True, "root layer has no external asset references"
    try:
        from pxr import UsdUtils

        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - exercised in the Newton image
        return False, f"dependency closure could not be proven: {type(exc).__name__}"
    external_layers = [layer for layer in layers if Path(layer.realPath).resolve() != path]
    if external_layers or assets or unresolved:
        return False, "external USD dependency closure is not yet vendored"
    return True, "OpenUSD dependency closure contains only the root layer"


def _job_record_sha256(package_root: Path, job_dir: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(job_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        records[path.relative_to(package_root).as_posix()] = sha256_file(path)
    if not records:
        raise RuntimeError("Calibration package contains no durable job records")
    return records


def _verify_durable_result_records(validation: ValidationResult) -> None:
    run_dir = Path(validation.fit.plan.workdir).expanduser().resolve()
    analysis_path = run_dir / "analysis.json"
    try:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot package unreadable durable job record: {analysis_path}") from exc
    _verify_analysis_predecessor(analysis, jsonable(validation.fit.plan))

    expected_records = {
        "plan.json": jsonable(validation.fit.plan),
        "baseline.json": jsonable(validation.fit.baseline),
        "fit.json": jsonable(validation.fit),
        "validation.json": jsonable(validation),
    }
    for name, expected in expected_records.items():
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"Cannot package missing durable job record: {path}")
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot package unreadable durable job record: {path}") from exc
        if actual != expected:
            raise RuntimeError(f"In-memory calibration result differs from durable {name}")

    fit = validation.fit
    if validation.run_id != fit.run_id or fit.run_id != fit.plan.run_id:
        raise RuntimeError("Calibration result run IDs are inconsistent")
    optimizer = fit.optimizer
    required_optimizer_fields = {
        "name",
        "version",
        "provider",
        "population",
        "generation_budget",
        "config_fingerprint",
        "execution_fingerprint",
    }
    if not required_optimizer_fields.issubset(optimizer):
        raise RuntimeError("Calibration fit lacks the optimizer identity required to verify its journal")
    checkpoint_metadata = {
        "optimizer_name": optimizer["name"],
        "optimizer_version": optimizer["version"],
        "optimizer_provider": optimizer["provider"],
        "optimizer_config_fingerprint": optimizer["config_fingerprint"],
    }
    try:
        state = FitJournal(
            run_dir,
            run_id=fit.run_id,
            execution_fingerprint=str(optimizer["execution_fingerprint"]),
            checkpoint_metadata=checkpoint_metadata,
        ).verify_snapshot()
    except (FitJournalError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Durable optimizer journal is inconsistent: {exc}") from exc
    if state.completed_generations != fit.completed_generations:
        raise RuntimeError("Fit completed_generations does not match the durable optimizer journal")
    if fit.baseline.candidate_id != -1 or fit.baseline.generation != -1:
        raise RuntimeError("Fit baseline bookkeeping does not identify the pre-search evaluation")
    if fit.best.candidate_id != state.next_candidate_id or fit.best.generation != state.completed_generations:
        raise RuntimeError("Selected fit evaluation bookkeeping does not follow the optimizer journal")
    generation_budget = optimizer["generation_budget"]
    if (
        isinstance(generation_budget, bool)
        or not isinstance(generation_budget, int)
        or generation_budget != fit.completed_generations
    ):
        raise RuntimeError("Optimizer generation budget does not match the completed fit journal")
    population = optimizer["population"]
    if isinstance(population, bool) or not isinstance(population, int) or population <= 0:
        raise RuntimeError("Optimizer population must be a positive integer")
    if Path(fit.history_path).expanduser().resolve() != (run_dir / "candidate-history.jsonl").resolve():
        raise RuntimeError("Fit history path does not identify the durable candidate-history projection")
    best_parameters = jsonable(fit.best.parameters)
    candidate_was_evaluated = False
    for generation_path in state.generation_paths:
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if len(generation["candidates"]) != population:
            raise RuntimeError("Optimizer generation population does not match the validated fit")
        if best_parameters in generation["candidates"]:
            candidate_was_evaluated = True
            break
    if not candidate_was_evaluated:
        raise RuntimeError("Selected best parameters do not occur in the durable optimizer journal")


def _verify_analysis_predecessor(analysis: object, plan: dict) -> None:
    if not isinstance(analysis, dict):
        raise TypeError("Durable analysis record must be a JSON object")
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
        raise RuntimeError("Durable analysis record is not the predecessor of the locked plan")
    if analysis.get("identifiable_parameters") != plan.get("parameters"):
        raise RuntimeError("Durable analysis parameter surface does not match the locked plan")
    readiness = analysis.get("readiness")
    if not isinstance(readiness, dict) or not readiness or any(value is not True for value in readiness.values()):
        raise RuntimeError("Durable analysis does not record a successful readiness decision")
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
            raise RuntimeError(f"Locked {field_name} are not derived from the durable analysis")
