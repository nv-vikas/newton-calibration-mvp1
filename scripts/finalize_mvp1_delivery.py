"""Create one immutable index for the canonical MVP1 delivery artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUN_ID = "so101-8f2830f962"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery-dir", default="deliverables/mvp1_20260913")
    parser.add_argument("--package-dir", default="packages/so101-mvp1-full-20260913")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.delivery_dir).expanduser().resolve()
    package_root = Path(args.package_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / "delivery_manifest.json"

    required = [
        root / "README.md",
        root / "evidence_analysis.md",
        root / "truth_report.json",
        root / "truth_report.md",
        root / "delivery_verification.json",
        root / "videos" / "video_manifest.json",
        root / "videos" / "so101_mvp1_five_call_overview.mp4",
        root / "videos" / "so101_mvp1_arm_synchronized.mp4",
        root / "videos" / "so101_mvp1_gripper_synchronized.mp4",
        root / "viewport" / "live_result.json",
        root / "viewport" / "live_trajectories.npz",
        root / "viewport" / "so101_mvp1_verified_actual_usd.mp4",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Cannot finalize MVP1 delivery; missing: " + ", ".join(missing))

    verification = load(root / "delivery_verification.json")
    truth = load(root / "truth_report.json")
    video = load(root / "videos" / "video_manifest.json")
    viewport = load(root / "viewport" / "live_result.json")
    package_manifest = load(package_root / "manifest.json")
    if not verification.get("passed"):
        raise SystemExit("Delivery verification is not passing")
    for label, record in (("truth", truth), ("verification", verification), ("video", video)):
        if record.get("run_id") != RUN_ID:
            raise SystemExit(f"{label} record belongs to {record.get('run_id')!r}, expected {RUN_ID!r}")
    viewport_run_id = viewport.get("package_run_id") or viewport.get("calibration_run_id") or viewport.get("run_id")
    if viewport_run_id != RUN_ID:
        raise SystemExit(f"viewport record belongs to {viewport_run_id!r}, expected {RUN_ID!r}")
    if package_manifest.get("run_id") != RUN_ID:
        raise SystemExit(f"package belongs to {package_manifest.get('run_id')!r}, expected {RUN_ID!r}")
    if package_manifest.get("status") != "validated" or not package_manifest.get("activation_allowed"):
        raise SystemExit("package is not validated and activation-allowed")
    for product in video.get("products", []):
        video_path = root / "videos" / Path(product["video"]).name
        poster_path = root / "videos" / Path(product["poster"]).name
        if sha256(video_path) != product["video_sha256"]:
            raise SystemExit(f"video hash mismatch: {video_path}")
        if sha256(poster_path) != product["poster_sha256"]:
            raise SystemExit(f"poster hash mismatch: {poster_path}")

    canonical = []
    excluded = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(root)
        if any(
            part.startswith(("rejected_", "draft_", "videos_pre_", "viewport_rejected_"))
            or "_rejected_" in part
            for part in relative.parts
        ):
            excluded.append(str(relative))
            continue
        canonical.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    result = {
        "schema": "newton.calibration.mvp1.delivery-manifest/v1",
        "run_id": RUN_ID,
        "scope": truth["result"]["scope"],
        "validation_passed": bool(truth["result"]["status"] == "validated"),
        "aggregate_heldout_improvement_pct": truth["result"]["aggregate_heldout_improvement_pct"],
        "delivery_verification_passed": True,
        "canonical_files": canonical,
        "calibration_package": {
            "path": str(package_root),
            "manifest_sha256": sha256(package_root / "manifest.json"),
            "files": [
                {
                    "path": str(path.relative_to(package_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(package_root.rglob("*"))
                if path.is_file()
            ],
        },
        "excluded_noncanonical_files": excluded,
        "claim_boundary": truth["claim_boundaries"],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "canonical_files": len(canonical), "excluded_files": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
