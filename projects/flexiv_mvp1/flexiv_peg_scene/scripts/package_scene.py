"""Make the captured stage portable; validate and bundle only deliverable files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from pxr import Sdf, Usd, UsdPhysics, UsdUtils
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run", default="verified-v2")
args = parser.parse_args()
source = ROOT / "output" / args.run
layer = Sdf.Layer.FindOrOpen(str(source / "flexiv_tabletop_scene.usda"))
portable_path = ROOT / "flexiv_tabletop_scene.usda"
portable = Sdf.Layer.CreateNew(str(portable_path))
portable.TransferContent(layer)
def rebase(path):
    if "/flexiv_peg_scene/assets/" in path:
        return "assets/" + Path(path).name
    return path
UsdUtils.ModifyAssetPaths(portable, rebase)
portable.Save()
stage = Usd.Stage.Open(str(portable_path))
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
# Exported reset state includes the held target used by the runtime.
manifest = json.loads((ROOT / "assets/asset_manifest.json").read_text())
angles = manifest["scene"]["joint_positions_rad"]
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in angles:
        name = prim.GetName()
        target = angles[name]
        if name in {"finger_joint", "right_outer_knuckle_joint", "left_inner_knuckle_joint", "right_inner_knuckle_joint"}:
            target -= .012
        elif name in {"left_outer_finger_joint", "right_outer_finger_joint"}:
            target += .012
        UsdPhysics.DriveAPI.Apply(prim, "angular").CreateTargetPositionAttr(float(np.rad2deg(target)))
stage.GetRootLayer().Save()
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(portable_path))
assert not unresolved, unresolved
assert all(str(ROOT) in l.realPath for l in layers), [(l.realPath) for l in layers]
report = {"scene": portable_path.name, "unresolved_dependencies": [],
          "layer_count": len(layers), "portable": True,
          "scene_sha256": hashlib.sha256(portable_path.read_bytes()).hexdigest()}
(ROOT / "output/package_validation.json").write_text(json.dumps(report, indent=2) + "\n")
archive = ROOT.parent / "flexiv_peg_scene.zip"
paths = [ROOT / "README.md", ROOT / "Dockerfile", ROOT / ".dockerignore", ROOT / "requirements-asset-prep.txt", portable_path]
paths += list((ROOT / "assets").rglob("*"))
paths += list((ROOT / "scripts").glob("*.py"))
paths += list(source.glob("*.png")) + list(source.glob("*.json"))
paths += [ROOT / "output/asset_validation.json", ROOT / "output/package_validation.json"]
video_dir = ROOT / "output/video-v1"
if video_dir.is_dir():
    paths += list(video_dir.glob("*.mp4"))
    paths += list(video_dir.glob("video_*.json"))
    paths += list(video_dir.glob("video_contact_sheet.jpg"))
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in paths:
        if path.is_file() and not path.name.startswith("._"):
            bundle.write(path, "flexiv_peg_scene/" + str(path.relative_to(ROOT)))
print(json.dumps({**report, "archive": str(archive)}, indent=2))
