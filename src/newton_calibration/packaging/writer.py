from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

from newton_calibration.core.io import write_json
from newton_calibration.core.models import CalibrationPackage, ValidationResult, jsonable


def write_package(validation: ValidationResult, output: str | Path) -> CalibrationPackage:
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = validation.fit.plan
    params = validation.fit.best.parameters
    source_asset = Path(plan.environment.asset_path).expanduser().resolve()
    if not source_asset.is_file():
        raise FileNotFoundError(f"Cannot package missing source asset: {source_asset}")
    packaged_asset = output_dir / source_asset.name
    shutil.copy2(source_asset, packaged_asset)

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

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    delay_steps = max(0, round(params["command_delay_s"] / plan.environment.dt))
    effective_delay_s = delay_steps * plan.environment.dt
    overlay = output_dir / ("calibration.usda" if validation.passed else "rejected_candidate.usda")
    overlay.write_text(
        "#usda 1.0\n"
        "(\n"
        f"    subLayers = [@./{packaged_asset.name}@]\n"
        "    customLayerData = {\n"
        f"        string newtonCalibrationRun = \"{validation.run_id}\"\n"
        "        string newtonCalibrationManifest = \"manifest.json\"\n"
        "        string newtonCalibrationScope = \"SO-101 free-space arm and unloaded gripper actuation\"\n"
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
    if plan.environment.residual_model_path:
        residual_source = Path(plan.environment.residual_model_path).expanduser().resolve()
        residual_target = output_dir / "actuator_residual.json"
        shutil.copy2(residual_source, residual_target)
        residual_artifact = residual_target.name
    manifest = {
        "schema": "newton.calibration.package/v1",
        "run_id": validation.run_id,
        "created_at": validation.created_at,
        "status": "validated" if validation.passed else "rejected",
        "activation_allowed": validation.passed,
        "scope": "SO-101 free-space arm and unloaded gripper actuation",
        "claims": {
            "heldout_passed": validation.passed,
            "heldout_improvement_pct": validation.improvement_pct,
            "contact_or_grip_force_validated": False,
        },
        "inputs": {
            "asset": plan.environment.asset_path,
            "asset_sha256": sha256(source_asset),
            "evidence": plan.evidence_uri,
            "evidence_revision": plan.evidence_revision,
            "evidence_fingerprint": plan.evidence_fingerprint,
        },
        "runtime": jsonable(plan.environment),
        "recipe": plan.recipe,
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
            "actuator_residual": residual_artifact,
        },
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    report = output_dir / "report.md"
    claim = (
        "This package is validated for SO-101 free-space arm motion and unloaded gripper tracking on the recorded "
        "conditions represented by the held-out Anchor-Lab trajectories."
        if validation.passed
        else "This candidate failed held-out validation and must not be activated as a calibrated asset."
    )
    report.write_text(
        f"# SO-101 Newton calibration {'package' if validation.passed else 'candidate'}\n\n"
        f"- Run: `{validation.run_id}`\n"
        f"- Held-out result: **{'PASS' if validation.passed else 'FAIL'}**\n"
        f"- Weighted-error improvement: **{validation.improvement_pct:.1f}%**\n"
        f"- Runtime: `{plan.environment.adapter}`\n"
        f"- Evidence revision: `{plan.evidence_revision}`\n\n"
        "## Product claim\n\n"
        f"{claim} It does not validate absolute grasp force, "
        "object contact, or transfer to a different robot setup.\n\n"
        "## What the package applies\n\n"
        "- Isaac Lab explicit-PD: grouped arm/jaw stiffness, damping, and effort limits.\n"
        "- Newton runtime: grouped arm/jaw armature and joint friction.\n"
        f"- Toolkit replay: requested command delay {params['command_delay_s'] * 1000.0:.3f} ms; "
        f"applied as {delay_steps} steps = {effective_delay_s * 1000.0:.3f} ms at the locked dt.\n"
        "- `calibration.usda`: relative source-asset reference plus package provenance; parameter values are in "
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
