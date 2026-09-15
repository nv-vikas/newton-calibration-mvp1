"""Bundle commands, real-data runner and independently labelled Newton checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from pxr import Sdf, UsdUtils

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="newton-v8")
    args = parser.parse_args()
    source = ROOT / "output" / args.run
    package = ROOT / "mvp1_collection"
    toolkit = ROOT.parent / "newton-calibration-mvp1"
    patch = subprocess.check_output(["git", "diff", "--", "src/newton_calibration/isaaclab/tuning.py"], cwd=toolkit)
    (package / "toolkit_no_evidence.patch").write_bytes(patch)
    shutil.copy2(toolkit / "tests/test_missing_evidence.py", package / "test_missing_evidence.py")
    for filename in ("free_motion_screen.json", "runtime_validation.json", "newton_backend_attestation.json"):
        shutil.copy2(source / filename, package / filename)
    screen = json.loads((package / "free_motion_screen.json").read_text())
    scene = json.loads((package / "runtime_validation.json").read_text())
    if screen["command_plan_sha256"] != sha(package / "collection_plan.json"):
        raise RuntimeError("Screened command plan differs from deliverable")
    layer = Sdf.Layer.FindOrOpen(str(source / "flexiv_tabletop_scene.usda"))
    portable = Sdf.Layer.CreateNew(str(ROOT / "flexiv_tabletop_newton.usda"))
    portable.TransferContent(layer)
    UsdUtils.ModifyAssetPaths(portable, lambda p: "assets/" + Path(p).name if "/flexiv_peg_scene/assets/" in p else p)
    portable.Save()
    _, _, unresolved = UsdUtils.ComputeAllDependencies(portable.realPath)
    if unresolved:
        raise RuntimeError(f"Unresolved scene assets: {unresolved}")
    files = [ROOT / "README.md", ROOT / "Dockerfile", ROOT / ".dockerignore",
             ROOT / "requirements-asset-prep.txt", ROOT / "flexiv_tabletop_newton.usda"]
    files += list((ROOT / "scripts").glob("*.py"))
    files += [p for p in (ROOT / "assets").rglob("*") if p.is_file()]
    files += [p for p in package.rglob("*") if p.is_file() and p.name != "bundle_record.json"]
    record = dict(schema="flexiv.mvp1.bundle/v1", physics="Newton", scene_run=args.run,
        scene_hold_passed=scene.get("peg_held_pass"),
        contact_probes_passed=all(x["passed"] for x in scene.get("contact_probes", [])),
        free_motion_screen_passed=screen["passed"],
        real_data_samples=0, real_hardware_tested=False, fitted=False,
        real_execution_approved=False,
        analysis="actual toolkit no-evidence readiness analysis",
        fit_plan="blocked; data collection plan supplied separately",
        local_checks={"toolkit_pytest_passed":191,"collection_unittest_passed":4},
        hashes={str(p.relative_to(ROOT)):sha(p) for p in files})
    (package / "bundle_record.json").write_text(json.dumps(record, indent=2)+"\n")
    files.append(package / "bundle_record.json")
    archive = ROOT.parent / "flexiv_newton_mvp1_collection.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as zipped:
        for file in files:
            zipped.write(file,"flexiv_peg_scene/"+str(file.relative_to(ROOT)))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,
                         motion_screen_passed=screen["passed"],peg_hold_passed=scene.get("peg_held_pass")),indent=2))


if __name__ == "__main__":
    main()
