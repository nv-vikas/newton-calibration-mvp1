"""Inspect user meshes and the vendor robot without modifying the sources."""
from pathlib import Path
import json

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
for name in ("Peg_25.stl", "hole_block.stl"):
    mesh = trimesh.load_mesh(ROOT / "assets/source" / name)
    print(json.dumps({"file": name, "bounds_source_units": mesh.bounds.tolist(),
                      "extents": mesh.extents.tolist(), "vertices": len(mesh.vertices),
                      "faces": len(mesh.faces), "watertight": mesh.is_watertight,
                      "volume": float(mesh.volume), "center_mass": mesh.center_mass.tolist(),
                      "components": len(mesh.split(only_watertight=False))}, indent=2))
    for axis in range(3):
        values, counts = np.unique(np.round(mesh.vertices[:, axis], 4), return_counts=True)
        top = np.argsort(counts)[-12:]
        print("axis", axis, "most common planes", sorted(zip(values[top].tolist(), counts[top].tolist())))

stage = Usd.Stage.Open(str(ROOT / "assets/source/Rizon4s_with_Grav.usd"))
print("USD", stage.GetDefaultPrim().GetPath(), "metersPerUnit", UsdGeom.GetStageMetersPerUnit(stage),
      "up", UsdGeom.GetStageUpAxis(stage))
print("LAYERS", [x.identifier for x in stage.GetUsedLayers()])
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        print("JOINT", prim.GetPath(), prim.GetTypeName(),
              {a.GetName(): str(a.Get()) for a in prim.GetAttributes() if a.HasAuthoredValueOpinion()},
              {r.GetName(): [str(p) for p in r.GetTargets()] for r in prim.GetRelationships()})
    elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print("BODY", prim.GetPath(), cache.ComputeWorldBound(prim).ComputeAlignedRange(),
              "mass", UsdPhysics.MassAPI(prim).GetMassAttr().Get())
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print("ARTICULATION", prim.GetPath())
    for attr in prim.GetAttributes():
        if str(attr.GetTypeName()) in {"asset", "asset[]"} and attr.HasAuthoredValueOpinion():
            print("ASSET", prim.GetPath(), attr.GetName(), attr.Get())
