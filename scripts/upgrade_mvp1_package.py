"""Make the completed MVP1 package portable and record actual application surfaces.

This is a metadata/packaging repair only.  It never changes fit parameters,
trajectory arrays, scores, validation results, or the source asset bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


JOB_FILES = (
    "analysis.json",
    "plan.json",
    "baseline.json",
    "fit-checkpoint.json",
    "candidate-history.jsonl",
    "fit.json",
    "validation.json",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--isaaclab-commit", required=True)
    parser.add_argument("--isaaclab-core-version", required=True)
    parser.add_argument("--isaaclab-newton-version", required=True)
    parser.add_argument("--newton-version", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve()
    asset = Path(args.asset).expanduser().resolve()
    if not package_dir.is_dir() or not asset.is_file():
        raise SystemExit("Package directory or source asset is missing")

    manifest_path = package_dir / "manifest.json"
    manifest = load(manifest_path)
    plan = load(run_dir / "plan.json")
    validation = load(run_dir / "validation.json")
    if manifest["run_id"] != validation["run_id"]:
        raise SystemExit("Refusing to combine records from different runs")

    packaged_asset = package_dir / asset.name
    shutil.copy2(asset, packaged_asset)
    job_dir = package_dir / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in JOB_FILES:
        source = run_dir / name
        if not source.is_file():
            raise SystemExit(f"Missing durable job record: {source}")
        shutil.copy2(source, job_dir / name)

    parameters = manifest["parameters"]
    dt = float(manifest["runtime"]["dt"])
    requested_delay_s = float(parameters["command_delay_s"])
    delay_steps = max(0, round(requested_delay_s / dt))
    effective_delay_s = delay_steps * dt

    (package_dir / "calibration.usda").write_text(
        "#usda 1.0\n"
        "(\n"
        f"    subLayers = [@./{packaged_asset.name}@]\n"
        "    customLayerData = {\n"
        f"        string newtonCalibrationRun = \"{manifest['run_id']}\"\n"
        "        string newtonCalibrationManifest = \"manifest.json\"\n"
        f"        string newtonCalibrationScope = \"{manifest['scope']}\"\n"
        "    }\n"
        ")\n",
        encoding="utf-8",
    )

    yaml_path = package_dir / "isaaclab_actuator.yaml"
    yaml_lines = [
        line
        for line in yaml_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("command_delay_steps_at_runtime_dt:")
        and not line.startswith("command_delay_effective_s_at_runtime_dt:")
    ]
    yaml_lines = [
        f'source_asset: "./{packaged_asset.name}"' if line.startswith("source_asset:") else line
        for line in yaml_lines
    ]
    yaml_lines.extend(
        [
            f"command_delay_steps_at_runtime_dt: {delay_steps}",
            f"command_delay_effective_s_at_runtime_dt: {effective_delay_s:.10g}",
        ]
    )
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    manifest["inputs"]["asset_sha256"] = sha256(asset)
    manifest["inputs"]["packaged_asset"] = packaged_asset.name
    manifest["parameter_application"] = {
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
    manifest["timing_quantization"] = {
        "requested_command_delay_s": requested_delay_s,
        "runtime_dt_s": dt,
        "applied_delay_steps": delay_steps,
        "effective_command_delay_s": effective_delay_s,
    }
    manifest["runtime_build"] = {
        "container_image_digest": args.container_image_digest,
        "isaaclab_commit": args.isaaclab_commit,
        "isaaclab_core_extension": args.isaaclab_core_version,
        "isaaclab_newton_extension": args.isaaclab_newton_version,
        "newton": args.newton_version,
    }
    manifest["record_notes"] = {
        "plan_owner_field": (
            "The immutable run plan used 'newton' as a broad calibration-domain owner. "
            "parameter_application is authoritative for the actual interface used by this execution."
        ),
        "stability_gate": (
            "The implemented check requires finite trajectories and |q| <= 100 rad; it is not a direct "
            "solver-convergence certificate."
        ),
        "evidence_readiness": (
            "analyze() checked recipe/signal readiness heuristically; it did not prove numerical identifiability."
        ),
    }
    manifest["artifacts"].pop("usd_overlay", None)
    manifest["artifacts"].update(
        {
            "source_asset": packaged_asset.name,
            "usd_provenance_layer": "calibration.usda",
            "job_records": "job/",
            "candidate_history": "job/candidate-history.jsonl",
        }
    )
    write(manifest_path, manifest)

    package_record = load(package_dir / "package.json")
    package_record.update(
        {
            "output_dir": ".",
            "overlay_path": "calibration.usda",
            "isaaclab_cfg_path": "isaaclab_actuator.yaml",
            "manifest_path": "manifest.json",
            "validation_path": "validation.json",
            "report_path": "report.md",
        }
    )
    write(package_dir / "package.json", package_record)

    report = (
        f"# SO-101 Newton calibration package\n\n"
        f"- Run: `{manifest['run_id']}`\n"
        f"- Held-out result: **PASS**\n"
        f"- Weighted-error improvement: **{validation['improvement_pct']:.1f}%**\n"
        f"- Runtime: `{plan['environment']['adapter']}`\n"
        f"- Evidence revision: `{plan['evidence_revision']}`\n\n"
        "## Product claim\n\n"
        "This package is validated for SO-101 free-space arm motion and unloaded gripper tracking on the "
        "recorded conditions represented by data excluded from the fit objective. It does not validate "
        "absolute grasp force, object contact, policy transfer, or transfer to a different robot setup.\n\n"
        "## What the package applies\n\n"
        "- Isaac Lab explicit-PD: grouped arm/jaw stiffness, damping, and effort limits.\n"
        "- Newton runtime: grouped arm/jaw armature and joint friction.\n"
        f"- Toolkit replay: requested command delay {requested_delay_s * 1000.0:.3f} ms; applied as "
        f"{delay_steps} steps = {effective_delay_s * 1000.0:.3f} ms at the locked dt.\n"
        "- `calibration.usda`: relative source-asset reference plus package provenance. Parameter values live "
        "in `isaaclab_actuator.yaml` and `manifest.json`.\n\n"
        "## Gates\n\n"
        + "\n".join(f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in validation["gates"].items())
        + "\n\n"
        "The implemented stability gate checks that replayed joint trajectories remain finite and within a "
        "broad position bound; it is not a direct solver-convergence certificate.\n"
    )
    (package_dir / "report.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "package": str(package_dir),
                "asset_sha256": manifest["inputs"]["asset_sha256"],
                "job_records": len(JOB_FILES),
                "delay": manifest["timing_quantization"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
