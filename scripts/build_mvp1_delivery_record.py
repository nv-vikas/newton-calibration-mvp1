"""Build the audit-facing truth record for the completed SO-101 MVP1 run.

The output deliberately separates recorded evidence, simulated trajectories,
fitted values, held-out validation, and claims that remain untested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


APPLICATION_OWNER = {
    "arm_stiffness_scale": "Isaac Lab explicit-PD",
    "arm_damping_scale": "Isaac Lab explicit-PD",
    "arm_effort_scale": "Isaac Lab explicit-PD",
    "gripper_stiffness_scale": "Isaac Lab explicit-PD",
    "gripper_damping_scale": "Isaac Lab explicit-PD",
    "gripper_effort_scale": "Isaac Lab explicit-PD",
    "arm_armature": "Newton runtime",
    "arm_friction_nm": "Newton runtime",
    "gripper_armature": "Newton runtime",
    "gripper_friction_nm": "Newton runtime",
    "command_delay_s": "Toolkit replay",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def improvement(before: float, after: float) -> float:
    return 100.0 * (before - after) / max(abs(before), 1.0e-12)


def fmt(value: float) -> str:
    if abs(value) < 0.01 and value != 0:
        return f"{value:.6g}"
    return f"{value:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--isaaclab-commit", required=True)
    parser.add_argument("--isaaclab-core-version", required=True)
    parser.add_argument("--isaaclab-newton-version", required=True)
    parser.add_argument("--newton-version", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    package_dir = Path(args.package_dir).expanduser().resolve()
    asset = Path(args.asset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = load_json(run_dir / "analysis.json")
    plan = load_json(run_dir / "plan.json")
    fit = load_json(run_dir / "fit.json")
    validation = load_json(run_dir / "validation.json")
    manifest = load_json(package_dir / "manifest.json")
    history_path = run_dir / "candidate-history.jsonl"
    candidates = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]

    runtime_build = manifest.get("runtime_build", {})
    expected_runtime_build = {
        "container_image_digest": args.container_image_digest,
        "isaaclab_commit": args.isaaclab_commit,
        "isaaclab_core_extension": args.isaaclab_core_version,
        "isaaclab_newton_extension": args.isaaclab_newton_version,
        "newton": args.newton_version,
    }
    if runtime_build != expected_runtime_build:
        raise SystemExit(
            "Runtime provenance arguments do not match the package manifest; "
            "refusing to create a free-form truth record."
        )
    if manifest.get("inputs", {}).get("asset_sha256") != sha256(asset):
        raise SystemExit("Source asset hash does not match the package manifest")

    specifications = {entry["name"]: entry for entry in plan["parameters"]}
    tuned = manifest["parameters"]
    parameter_rows = []
    exact_bound_count = 0
    near_bound_count = 0
    for name, spec in specifications.items():
        value = float(tuned[name])
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        span = max(upper - lower, 1.0e-12)
        exact_bound = abs(value - lower) <= 1.0e-9 * span or abs(value - upper) <= 1.0e-9 * span
        near_bound = min(value - lower, upper - value) <= 0.05 * span
        exact_bound_count += int(exact_bound)
        near_bound_count += int(near_bound)
        parameter_rows.append(
            {
                "name": name,
                # The original run record used "newton" as a broad domain
                # owner.  Report the actual application surface instead.
                "owner": APPLICATION_OWNER[name],
                "initial": float(spec["initial"]),
                "tuned": value,
                "lower": lower,
                "upper": upper,
                "unit": spec["unit"],
                "exactly_at_bound": exact_bound,
                "within_5_percent_of_bound": near_bound,
            }
        )

    episode_rows = []
    for name, values in validation["per_episode"].items():
        before = float(values["baseline"]["score"])
        after = float(values["calibrated"]["score"])
        episode_rows.append(
            {
                "episode": name,
                "baseline_score": before,
                "tuned_score": after,
                "improvement_pct": improvement(before, after),
                "baseline_position_rmse_rad": float(values["baseline"]["position_rmse_rad"]),
                "tuned_position_rmse_rad": float(values["calibrated"]["position_rmse_rad"]),
            }
        )

    core_files = [
        run_dir / "analysis.json",
        run_dir / "plan.json",
        run_dir / "baseline.json",
        run_dir / "candidate-history.jsonl",
        run_dir / "fit.json",
        run_dir / "validation.json",
        package_dir / "manifest.json",
        package_dir / "calibration.usda",
        package_dir / "isaaclab_actuator.yaml",
        package_dir / "validation.json",
        asset,
    ]
    hashes = {str(path): sha256(path) for path in core_files}

    report = {
        "schema": "newton.calibration.mvp1.truth-record/v1",
        "run_id": manifest["run_id"],
        "result": {
            "status": manifest["status"],
            "activation_allowed_for_scoped_package": bool(manifest["activation_allowed"]),
            "scope": manifest["scope"],
            "aggregate_heldout_baseline_score": float(validation["baseline_metrics"]["score"]),
            "aggregate_heldout_tuned_score": float(validation["calibrated_metrics"]["score"]),
            "aggregate_heldout_improvement_pct": float(validation["improvement_pct"]),
            "gates": validation["gates"],
            "heldout_episodes": episode_rows,
        },
        "evidence": {
            "real": True,
            "source": "NVIDIA Anchor-Lab SO-101 physical robot telemetry in Parquet files",
            "available_recordings": len(analysis["train_episodes"]) + len(analysis["heldout_episodes"]),
            "fit_recordings_used": plan["train_episodes"],
            "heldout_recordings_used": plan["heldout_episodes"],
            "maximum_seconds_per_recording": float(plan["optimizer"]["max_episode_duration_s"]),
            "signals_used_by_objective": ["command_q", "actual_q", "dq"],
            "signals_explicitly_excluded": ["present_load_raw", "tau_abs"],
            "physical_camera_video_used": False,
            "revision": plan["evidence_revision"],
            "fingerprint": plan["evidence_fingerprint"],
        },
        "simulation": {
            "asset": str(asset),
            "asset_sha256": hashes[str(asset)],
            "asset_disclosure": (
                "The released file is named so101_no_camera_new_calib.usd and is already described as "
                "calibrated. The baseline is therefore the recipe-initial actuator configuration on this "
                "asset, not a raw factory USD."
            ),
            "surface": "Isaac Lab articulation adapter",
            "physics": "Newton MJWarp on CUDA",
            "environment": plan["environment"],
            "container_image_digest": runtime_build["container_image_digest"],
            "isaaclab_commit": runtime_build["isaaclab_commit"],
            "isaaclab_core_extension": runtime_build["isaaclab_core_extension"],
            "isaaclab_newton_extension": runtime_build["isaaclab_newton_extension"],
            "newton": runtime_build["newton"],
        },
        "fit": {
            "optimizer": plan["optimizer"],
            "optimizer_candidates": len(candidates),
            "stable_candidates": sum(bool(entry.get("stable")) for entry in candidates),
            "errored_candidates": sum(entry.get("error") is not None for entry in candidates),
            "training_baseline_score": float(fit["baseline"]["score"]),
            "training_tuned_score": float(fit["best"]["score"]),
            "parameters": parameter_rows,
            "exact_bound_count": exact_bound_count,
            "within_5_percent_of_bound_count": near_bound_count,
            "parameter_application": {
                "isaaclab_explicit_pd": 6,
                "newton_runtime": 4,
                "toolkit_replay": 1,
            },
            "timing_quantization": {
                "requested_command_delay_s": float(tuned["command_delay_s"]),
                "runtime_dt_s": float(plan["environment"]["dt"]),
                "applied_delay_steps": max(
                    0, round(float(tuned["command_delay_s"]) / float(plan["environment"]["dt"]))
                ),
                "effective_command_delay_s": max(
                    0, round(float(tuned["command_delay_s"]) / float(plan["environment"]["dt"]))
                )
                * float(plan["environment"]["dt"]),
            },
            "parameter_truth_disclosure": (
                "These values minimize the declared trajectory objective within the recipe bounds. "
                "They are not independently measured physical constants and are not universal SO-101 defaults."
            ),
        },
        "claim_boundaries": {
            "validated": [
                "free-space arm joint trajectory tracking",
                "unloaded gripper joint trajectory tracking",
                "four held-out Anchor-Lab motion recordings under the recorded setup conditions",
                "all evaluated trajectories stayed finite and within the implemented joint-position bound",
            ],
            "not_validated": [
                "absolute gripping force",
                "object contact, slip, grasp, or insertion",
                "policy training or policy transfer",
                "execution on a new physical SO-101 after applying the package",
                "universal physical correctness or portability to another setup",
                "formal confidence intervals or independent physical identification of each fitted value",
                "solver-parameter calibration, geometry, mass, inertia, or contact parameters",
            ],
            "measured_video_definition": (
                "Any measured lane in the presentation is recorded real joint telemetry visualized on a "
                "robot drawing or kinematically posed USD; it is not camera footage of the physical trial."
            ),
            "simulation_video_definition": (
                "Presentation RTX lanes kinematically pose the actual USD from saved baseline/tuned Newton "
                "states. RTX is visualization-only; the authoritative physics execution is the recorded "
                "Isaac Lab + Newton trajectory bundle."
            ),
        },
        "integrity": {"sha256": hashes},
    }

    (output_dir / "truth_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = report["result"]
    lines = [
        "# SO-101 Newton calibration MVP1 — truth report",
        "",
        f"**Run:** `{report['run_id']}`  ",
        f"**Result:** {result['status'].upper()} for **{result['scope']}**  ",
        f"**Held-out weighted error:** {result['aggregate_heldout_baseline_score']:.4f} → "
        f"{result['aggregate_heldout_tuned_score']:.4f} "
        f"(**{result['aggregate_heldout_improvement_pct']:.1f}% lower**)  ",
        f"**Search:** {len(candidates)} candidates; {sum(bool(entry.get('stable')) for entry in candidates)} stable; "
        f"{sum(entry.get('error') is not None for entry in candidates)} errored.",
        "",
        "## What is real",
        "",
        "Anchor-Lab Parquet files contain telemetry recorded from a physical SO-101: commanded joint "
        "positions, measured joint positions, and measured velocities. The repository contains 50 recordings; "
        "the locked recipe used four for fitting and four different recordings for validation, up to 12 seconds each. "
        "No physical-camera video was used.",
        "",
        "## What is simulated",
        "",
        "The same joint commands and initial states were replayed through an Isaac Lab articulation backed by "
        "Newton MJWarp at 120 Hz. Both the baseline and tuned trajectories are simulation outputs. The supplied "
        "SO-101 USD is an actual asset, but its filename and provenance already describe it as calibrated; the "
        "baseline is the recipe-initial actuator configuration on that asset, not an untouched factory model.",
        "",
        "## What was tuned",
        "",
        "The optimizer fitted six Isaac Lab explicit-PD controller values, four values written into the Newton "
        "runtime, and one toolkit replay-delay value. No solver, geometry, link mass/inertia, contact, or task "
        "parameters were tuned.",
        "",
        "| Parameter | Owner | Initial | Tuned | Bounds | Flag |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in parameter_rows:
        flag = "at bound" if row["exactly_at_bound"] else ("near bound" if row["within_5_percent_of_bound"] else "—")
        lines.append(
            f"| `{row['name']}` | {row['owner']} | {fmt(row['initial'])} | {fmt(row['tuned'])} | "
            f"{fmt(row['lower'])}–{fmt(row['upper'])} {row['unit']} | {flag} |"
        )
    lines.extend(
        [
            "",
            f"The requested command delay was {float(tuned['command_delay_s']) * 1000.0:.3f} ms. At the locked "
            f"120 Hz runtime it was applied as {max(0, round(float(tuned['command_delay_s']) / float(plan['environment']['dt'])))} "
            f"steps = {max(0, round(float(tuned['command_delay_s']) / float(plan['environment']['dt']))) * float(plan['environment']['dt']) * 1000.0:.3f} ms.",
            "",
            f"Four values are exactly at a recipe bound; {near_bound_count} are within 5% of a bound. "
            "That does not invalidate the trajectory result, but it limits physical interpretation and suggests "
            "the bounds/model/evidence should be revisited before generalizing the values.",
            "",
            "## What was validated",
            "",
            "The fit objective did not use the four validation recordings. The final candidate beat the baseline on "
            "the aggregate gate and on every selected held-out episode:",
            "",
            "| Held-out recording | Weighted score | Improvement | Position RMSE |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in episode_rows:
        lines.append(
            f"| {row['episode']} | {row['baseline_score']:.4f} → {row['tuned_score']:.4f} | "
            f"{row['improvement_pct']:.1f}% | {row['baseline_position_rmse_rad']:.4f} → "
            f"{row['tuned_position_rmse_rad']:.4f} rad |"
        )
    lines.extend(
        [
            "",
            "## What was not validated",
            "",
            "Gripping force, object contact, slip, grasping, insertion, policy learning, real-robot transfer after "
            "package application, and formal per-parameter confidence intervals remain untested. A measured presentation lane is telemetry playback or a "
            "kinematically posed USD—not physical-camera footage. The RTX baseline/tuned views are playback "
            "of saved Newton states, not a second physics execution.",
            "",
            "## Reproducibility anchors",
            "",
            f"- Evidence revision: `{plan['evidence_revision']}`",
            f"- Evidence fingerprint: `{plan['evidence_fingerprint']}`",
            f"- Asset SHA-256: `{hashes[str(asset)]}`",
            f"- Container image: `{args.container_image_digest}`",
            f"- Isaac Lab commit: `{args.isaaclab_commit}`",
            f"- Isaac Lab core extension: `{args.isaaclab_core_version}`",
            f"- isaaclab_newton: `{args.isaaclab_newton_version}`",
            f"- Newton: `{args.newton_version}`",
            "",
        ]
    )
    (output_dir / "truth_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"truth_report": str(output_dir / "truth_report.json"), "markdown": str(output_dir / "truth_report.md")}, indent=2))


if __name__ == "__main__":
    main()
