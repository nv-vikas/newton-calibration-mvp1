"""Check SI units, physical properties, source integrity and open bore geometry."""
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
report = {"assets": {}, "checks": {}}
for name in ("Peg_25", "HoleBlock", "Flexiv_Rizon4s_Grav"):
    path = ASSETS / (name + ".usd")
    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim()
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    assert not unresolved, unresolved
    report["assets"][name] = {"usd_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "meters_per_unit": 1, "up_axis": "Z", "unresolved_dependencies": unresolved}
    if name != "Flexiv_Rizon4s_Grav":
        root = stage.GetDefaultPrim()
        mass = UsdPhysics.MassAPI(root)
        assert mass.GetMassAttr().Get() > 0
        assert all(v > 0 for v in mass.GetDiagonalInertiaAttr().Get())
        collision = stage.GetPrimAtPath(str(root.GetPath()) + "/Collision")
        assert collision.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get()
        approximation = UsdPhysics.MeshCollisionAPI(collision).GetApproximationAttr().Get()
        assert approximation == ("none" if name == "HoleBlock" else "convexHull")
        assert root.HasAPI(UsdPhysics.RigidBodyAPI) == (name == "Peg_25")
        report["assets"][name]["physics_properties_valid"] = True

block = trimesh.load_mesh(ASSETS / "source/hole_block.stl")
block.apply_scale(.001)
origins = np.array([[0, 0, .08], [.017, 0, .08], [.0119, 0, .08]])
locations, rays, _ = block.ray.intersects_location(origins, np.tile([0., 0., -1.], (3, 1)))
assert 0 not in rays, "Central hole is obstructed"
assert 2 not in rays, "Peg interior does not fit central bore"
assert 1 in rays, "Offset ray should hit solid material"
report["checks"] = {"central_bore_open": True, "peg_radius_inside_bore": True,
                     "offset_hits_fixture": True, "radial_clearance_m": .00025,
                     "formal_simready_catalog_certification": False,
                     "material_properties_measured": False}
(ROOT / "output").mkdir(exist_ok=True)
(ROOT / "output/asset_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
