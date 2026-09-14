"""Record synchronized SO-101 evidence, baseline Newton, and tuned Newton motion.

This is a presentation artifact generator, not a sixth public calibration API.
It deliberately reuses the same evidence adapter, recipe, environment surface,
and Newton runtime as the five-call MVP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from newton_calibration.adapters.evidence import AnchorLabSO101Evidence
from newton_calibration.adapters.runtime import create_runtime
from newton_calibration.adapters.surface import SO101EnvCfg
from newton_calibration.recipes import get_recipe
from newton_calibration.validation.metrics import compare_trajectories


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = predicted - reference
    arm = error[:, :5]
    jaw = error[:, 5]
    return {
        "all_rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        "arm_rmse_rad": float(np.sqrt(np.mean(np.square(arm)))),
        "jaw_rmse_rad": float(np.sqrt(np.mean(np.square(jaw)))),
        "all_p95_rad": float(np.percentile(np.abs(error), 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode", default="heldout-frequency-sweep")
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    asset_path = Path(args.asset).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tuned = {name: float(value) for name, value in manifest["parameters"].items()}
    recipe = get_recipe(manifest.get("recipe", "so101_actuator_dynamics.v1"))
    baseline = {parameter.name: float(parameter.initial) for parameter in recipe.parameters}

    runtime_record = manifest.get("runtime", {})
    # JSON is written with sorted keys, so reconstructing the joint map from the
    # manifest would silently reorder the six evidence columns.  Preserve the
    # canonical SO-101 evidence order owned by SO101EnvCfg instead.
    canonical = SO101EnvCfg(usd_path=str(asset_path))
    recorded_joint_map = dict(runtime_record.get("joint_map", {}))
    if recorded_joint_map and recorded_joint_map != canonical.joint_map:
        if set(recorded_joint_map.items()) != set(canonical.joint_map.items()):
            raise ValueError("Package joint map differs from the supported SO-101 joint map")
    env = SO101EnvCfg(
        usd_path=str(asset_path),
        runtime="isaaclab_newton",
        device=args.device,
        dt=float(runtime_record.get("dt", 1.0 / 120.0)),
        gravity=tuple(runtime_record.get("gravity", (0.0, 0.0, -9.81))),
        joint_map=dict(canonical.joint_map),
        base_stiffness=float(runtime_record.get("base_stiffness", 1.7453293)),
        base_damping=float(runtime_record.get("base_damping", 0.017453292)),
        base_effort_limit=float(runtime_record.get("base_effort_limit", 10.0)),
        base_armature=float(runtime_record.get("base_armature", 0.0)),
        num_substeps=int(runtime_record.get("num_substeps", 1)),
        solver_iterations=int(runtime_record.get("solver_iterations", 100)),
        solver_tolerance=float(runtime_record.get("solver_tolerance", 1.0e-6)),
        residual_model_path=runtime_record.get("residual_model_path"),
    ).describe()
    evidence = AnchorLabSO101Evidence(args.evidence)
    episode = evidence.load_episode(
        args.episode,
        dt=env.dt,
        max_duration_s=args.duration,
    )

    runtime = create_runtime(env)
    try:
        baseline_q, baseline_dq, baseline_stable = runtime._rollout(baseline, episode)
        tuned_q, tuned_dq, tuned_stable = runtime._rollout(tuned, episode)
    finally:
        runtime.close()

    baseline_score, baseline_objective = compare_trajectories(
        baseline_q,
        baseline_dq,
        episode.actual_q,
        episode.actual_dq,
        episode.command_q,
        env.dt,
        recipe.objective_weights,
    )
    tuned_score, tuned_objective = compare_trajectories(
        tuned_q,
        tuned_dq,
        episode.actual_q,
        episode.actual_dq,
        episode.command_q,
        env.dt,
        recipe.objective_weights,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time_s=episode.time_s,
        command_q=episode.command_q,
        measured_q=episode.actual_q,
        measured_dq=episode.actual_dq,
        baseline_q=baseline_q,
        baseline_dq=baseline_dq,
        tuned_q=tuned_q,
        tuned_dq=tuned_dq,
        joint_names=np.asarray(episode.joints),
    )
    metadata = {
        "episode": episode.name,
        "split": episode.split,
        "duration_s": float(episode.time_s[-1]) if len(episode.time_s) else 0.0,
        "steps": int(len(episode.time_s)),
        "dt": env.dt,
        "runtime": env.adapter,
        "source_path": episode.source_path,
        "asset_path": str(asset_path),
        "asset_sha256": _sha256(asset_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_run_id": manifest.get("run_id"),
        "manifest_status": manifest.get("status"),
        "activation_allowed": bool(manifest.get("activation_allowed", False)),
        "scope": manifest.get("scope"),
        "evidence_revision": manifest.get("inputs", {}).get("evidence_revision"),
        "evidence_fingerprint": manifest.get("inputs", {}).get("evidence_fingerprint"),
        "aggregate_heldout_improvement_pct": manifest.get("claims", {}).get("heldout_improvement_pct"),
        "aggregate_heldout_passed": manifest.get("claims", {}).get("heldout_passed"),
        "baseline_stable": bool(baseline_stable),
        "tuned_stable": bool(tuned_stable),
        "baseline_score": baseline_score,
        "tuned_score": tuned_score,
        "score_improvement_pct": 100.0 * (baseline_score - tuned_score) / max(abs(baseline_score), 1e-12),
        "baseline_objective": baseline_objective,
        "tuned_objective": tuned_objective,
        "baseline_error": _metrics(baseline_q, episode.actual_q),
        "tuned_error": _metrics(tuned_q, episode.actual_q),
        "baseline_parameters": baseline,
        "tuned_parameters": tuned,
        "measured_lane_definition": (
            "Recorded Anchor-Lab SO-101 joint telemetry. A robot drawing or USD view is a "
            "kinematic visualization of those measurements, not physical-camera footage."
        ),
        "baseline_lane_definition": (
            "Newton simulation using the recipe-initial actuator settings on the released "
            "so101_no_camera_new_calib.usd asset. The released USD is already described as calibrated; "
            "this is not a raw factory-asset baseline."
        ),
        "tuned_lane_definition": (
            "Newton simulation using the optimizer-selected actuator settings and command delay "
            "from the validated, setup-scoped calibration package."
        ),
        "presentation_disclosure": (
            "The package passed the full MVP1 recipe for free-space arm and unloaded-gripper "
            "trajectory tracking. It does not validate grasp force, contact, insertion, a policy, "
            "or transfer on a new physical robot."
        ),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"npz": str(output), "metadata": str(metadata_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
