#!/usr/bin/env python3
"""Audit SO-101 gripper collision topology and convex-hull inflation.

This is an asset-readiness diagnostic.  It does not modify the USD.  Run it
with the Isaac Lab Python environment so that both USD and trimesh are
available.  The report makes it explicit when a single authored convex hull
can fill concavities that are important to a grasp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics


GRIPPER_BODY_NAMES = {"gripper_link", "moving_jaw_so101_v1_link"}


def _triangles(mesh: UsdGeom.Mesh) -> np.ndarray:
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        polygon = indices[cursor : cursor + count]
        cursor += int(count)
        if count < 3:
            continue
        for offset in range(1, int(count) - 1):
            triangles.append((int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1])))
    return np.asarray(triangles, dtype=np.int64)


def _points_in_body(stage: Usd.Stage, prim: Usd.Prim, body: Usd.Prim) -> np.ndarray:
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_to_body, _ = cache.ComputeRelativeTransform(prim, body)
    return np.asarray(
        [mesh_to_body.Transform(Gf.Vec3d(*point)) for point in points],
        dtype=np.float64,
    )


def _bounds(points: np.ndarray) -> dict[str, list[float]]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return {
        "min_m": minimum.tolist(),
        "max_m": maximum.tolist(),
        "size_m": (maximum - minimum).tolist(),
    }


def _component_report(component: trimesh.Trimesh) -> dict[str, object]:
    report: dict[str, object] = {
        "vertex_count": int(len(component.vertices)),
        "triangle_count": int(len(component.faces)),
        "watertight": bool(component.is_watertight),
        "convex": bool(component.is_convex),
        "bounds_body": _bounds(np.asarray(component.vertices)),
    }
    try:
        hull = component.convex_hull
        report["convex_hull_vertex_count"] = int(len(hull.vertices))
        report["convex_hull_triangle_count"] = int(len(hull.faces))
        report["convex_hull_volume_m3"] = float(abs(hull.volume))
        if component.is_watertight and abs(component.volume) > 1.0e-12:
            report["mesh_volume_m3"] = float(abs(component.volume))
            report["hull_to_mesh_volume_ratio"] = float(abs(hull.volume / component.volume))
    except Exception as error:  # diagnostic must retain the rest of the audit
        report["convex_hull_error"] = f"{type(error).__name__}: {error}"
    return report


def inspect(asset: Path) -> dict[str, object]:
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {asset}")
    meshes: list[dict[str, object]] = []
    for body in stage.Traverse():
        if body.GetName() not in GRIPPER_BODY_NAMES:
            continue
        descendants = Usd.PrimRange(
            body,
            Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate),
        )
        for prim in descendants:
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = _points_in_body(stage, prim, body)
            faces = _triangles(mesh)
            # STL imports commonly duplicate all three vertices per triangle.
            # Process/merge them before connectivity analysis; otherwise every
            # triangle is incorrectly reported as its own component.
            topology = trimesh.Trimesh(
                vertices=points,
                faces=faces,
                process=True,
                validate=True,
            )
            components = sorted(
                topology.split(only_watertight=False),
                key=lambda component: len(component.faces),
                reverse=True,
            )
            approximation = None
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            entry: dict[str, object] = {
                "path": str(prim.GetPath()),
                "body_path": str(body.GetPath()),
                "body_is_instance": bool(body.IsInstance()),
                "mesh_is_instance_proxy": bool(prim.IsInstanceProxy()),
                "mesh_prim_in_prototype": (
                    str(prim.GetPrimInPrototype().GetPath()) if prim.IsInstanceProxy() else None
                ),
                "composition_layers": [spec.layer.identifier for spec in prim.GetPrimStack()],
                "approximation": str(approximation) if approximation is not None else None,
                "collision_enabled": bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
                "relationships": {
                    relationship.GetName(): [str(target) for target in relationship.GetTargets()]
                    for relationship in prim.GetRelationships()
                },
                "material_targets": [
                    {
                        "path": str(target),
                        "valid": bool(stage.GetPrimAtPath(target)),
                        "applied_schemas": list(stage.GetPrimAtPath(target).GetAppliedSchemas()),
                        "physics_and_newton_attributes": {
                            attribute.GetName(): str(attribute.Get())
                            for attribute in stage.GetPrimAtPath(target).GetAttributes()
                            if attribute.GetName().startswith(("physics:", "newton:", "mjc:", "physx"))
                        },
                    }
                    for relationship in prim.GetRelationships()
                    if relationship.GetName().startswith("material:binding")
                    for target in relationship.GetTargets()
                ],
                "vertex_count": int(len(points)),
                "triangle_count": int(len(faces)),
                "connected_component_count": int(len(components)),
                "bounds_body": _bounds(points),
                "whole_mesh": _component_report(topology),
                "components": [_component_report(component) for component in components[:64]],
                "components_omitted": max(0, len(components) - 64),
            }
            entry["risk"] = (
                "single_convex_hull_fills_concavities"
                if str(approximation).lower() == "convexhull" and not topology.is_convex
                else "no_single_hull_concavity_detected"
            )
            meshes.append(entry)
    return {
        "schema": "newton.calibration/so101-collision-audit@0.1",
        "scope": "read-only gripper collision asset-readiness audit",
        "asset": str(asset.resolve()),
        "mesh_count": len(meshes),
        "meshes": meshes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.asset)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"RESULT={args.output}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
