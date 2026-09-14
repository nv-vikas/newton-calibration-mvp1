"""Fail closed if the MVP1 presentation artifacts drift from the validated run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VIDEO_EPISODES = {
    "heldout_frequency_sweep": "so101-sysid-50motion-heldout-frequency-sweep",
    "heldout_friction_gravity": "so101-sysid-50motion-heldout-friction-gravity",
    "heldout_hold_under_gravity": "so101-sysid-50motion-heldout-hold-under-gravity",
    "heldout_backlash_detection": "so101-sysid-50motion-heldout-backlash-detection",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, *, tolerance: float = 1.0e-5) -> bool:
    """Compare replay metrics with a declared non-zero GPU tolerance."""
    return abs(actual - expected) <= tolerance * max(1.0, abs(actual), abs(expected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--video-data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    package_dir = Path(args.package_dir).resolve()
    video_data_dir = Path(args.video_data_dir).resolve()
    output = Path(args.output).resolve()
    validation = load(run_dir / "validation.json")
    fit = load(run_dir / "fit.json")
    manifest = load(package_dir / "manifest.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("run_id", validation["run_id"] == manifest["run_id"], f"{validation['run_id']} == {manifest['run_id']}")
    check("validation_passed", bool(validation["passed"]), f"passed={validation['passed']}")
    check(
        "package_activatable",
        manifest["status"] == "validated" and bool(manifest["activation_allowed"]),
        f"status={manifest['status']}; activation_allowed={manifest['activation_allowed']}",
    )
    check(
        "package_parameters_match_fit",
        set(manifest["parameters"]) == set(fit["best"]["parameters"])
        and all(close(float(manifest["parameters"][name]), float(value)) for name, value in fit["best"]["parameters"].items()),
        "manifest parameters equal the selected fit candidate",
    )
    check(
        "package_validation_matches_run",
        load(package_dir / "validation.json") == validation,
        "packaged validation.json equals the durable run record",
    )
    check(
        "package_claim_matches_validation",
        close(float(manifest["claims"]["heldout_improvement_pct"]), float(validation["improvement_pct"])),
        f"{manifest['claims']['heldout_improvement_pct']} == {validation['improvement_pct']}",
    )
    declared_artifacts = [
        manifest["artifacts"]["source_asset"],
        manifest["artifacts"]["usd_provenance_layer"],
        manifest["artifacts"]["isaaclab_actuator_config"],
        manifest["artifacts"]["validation"],
        manifest["artifacts"]["candidate_history"],
    ]
    check(
        "package_artifacts_present",
        all((package_dir / name).is_file() for name in declared_artifacts),
        ", ".join(declared_artifacts),
    )
    overlay_text = (package_dir / manifest["artifacts"]["usd_provenance_layer"]).read_text(encoding="utf-8")
    check(
        "overlay_identifies_run",
        manifest["run_id"] in overlay_text
        and "manifest.json" in overlay_text
        and f"@./{manifest['artifacts']['source_asset']}@" in overlay_text,
        "provenance layer points to the selected run, manifest, and packaged source asset",
    )
    check(
        "asset_hash",
        sha256(package_dir / manifest["artifacts"]["source_asset"]) == manifest["inputs"]["asset_sha256"],
        manifest["inputs"]["asset_sha256"],
    )
    expected_application = {
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
    check(
        "parameter_application",
        {key: set(value) for key, value in manifest["parameter_application"].items()} == expected_application,
        "6 Isaac Lab explicit-PD; 4 Newton runtime; 1 toolkit replay",
    )
    timing = manifest["timing_quantization"]
    expected_delay_steps = max(0, round(float(manifest["parameters"]["command_delay_s"]) / float(manifest["runtime"]["dt"])))
    check(
        "delay_quantization",
        int(timing["applied_delay_steps"]) == expected_delay_steps
        and close(float(timing["effective_command_delay_s"]), expected_delay_steps * float(manifest["runtime"]["dt"])),
        f"requested={timing['requested_command_delay_s']}; applied={expected_delay_steps} steps",
    )

    history = [
        json.loads(line)
        for line in (run_dir / "candidate-history.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_candidates = int(fit["plan"]["optimizer"]["population"]) * int(fit["plan"]["optimizer"]["generations"])
    check("candidate_count", len(history) == expected_candidates, f"{len(history)} == {expected_candidates}")
    check("candidate_stability", all(row["stable"] and row.get("error") is None for row in history), "all stable; no errors")

    files: dict[str, str] = {}
    for stem, episode in VIDEO_EPISODES.items():
        metadata_path = video_data_dir / f"{stem}.json"
        trajectory_path = video_data_dir / f"{stem}.npz"
        metadata = load(metadata_path)
        arrays = np.load(trajectory_path)
        expected = validation["per_episode"][episode]
        check(f"{stem}.episode", metadata["episode"] == episode and metadata["split"] == "heldout", metadata["episode"])
        check(
            f"{stem}.baseline_score",
            close(float(metadata["baseline_score"]), float(expected["baseline"]["score"])),
            f"{metadata['baseline_score']:.12g} vs {expected['baseline']['score']:.12g}",
        )
        check(
            f"{stem}.tuned_score",
            close(float(metadata["tuned_score"]), float(expected["calibrated"]["score"])),
            f"{metadata['tuned_score']:.12g} vs {expected['calibrated']['score']:.12g}",
        )
        expected_shape = arrays["measured_q"].shape
        check(
            f"{stem}.shape",
            expected_shape == arrays["baseline_q"].shape == arrays["tuned_q"].shape and expected_shape[1] == 6,
            str(expected_shape),
        )
        check(f"{stem}.finite", all(np.isfinite(arrays[key]).all() for key in ("measured_q", "baseline_q", "tuned_q")), "finite")
        files[str(metadata_path)] = sha256(metadata_path)
        files[str(trajectory_path)] = sha256(trajectory_path)

    gripper_metadata = load(video_data_dir / "train_gripper_cycles.json")
    gripper_arrays = np.load(video_data_dir / "train_gripper_cycles.npz")
    expected_gripper = fit["best"]["episodes"]["so101-sysid-50motion-train-gripper-cycles"]
    expected_gripper_baseline = fit["baseline"]["episodes"]["so101-sysid-50motion-train-gripper-cycles"]
    check("gripper_split", gripper_metadata["split"] == "train", "presentation comparison is calibration evidence, not holdout")
    check(
        "gripper_baseline_score",
        close(float(gripper_metadata["baseline_score"]), float(expected_gripper_baseline["score"])),
        f"{gripper_metadata['baseline_score']:.12g} vs {expected_gripper_baseline['score']:.12g}",
    )
    check(
        "gripper_tuned_score",
        close(float(gripper_metadata["tuned_score"]), float(expected_gripper["score"])),
        f"{gripper_metadata['tuned_score']:.12g} vs {expected_gripper['score']:.12g}",
    )
    check("gripper_shape", gripper_arrays["measured_q"].shape[1] == 6, str(gripper_arrays["measured_q"].shape))
    for path in (video_data_dir / "train_gripper_cycles.json", video_data_dir / "train_gripper_cycles.npz"):
        files[str(path)] = sha256(path)

    passed = all(item["passed"] for item in checks)
    result = {
        "schema": "newton.calibration.mvp1.delivery-verification/v1",
        "run_id": manifest["run_id"],
        "passed": passed,
        "checks": checks,
        "sha256": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": passed, "checks": len(checks)}, indent=2))
    if not passed:
        failures = [item for item in checks if not item["passed"]]
        raise SystemExit("MVP1 delivery verification failed: " + "; ".join(item["name"] for item in failures))


if __name__ == "__main__":
    main()
