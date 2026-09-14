#!/usr/bin/env python3
"""Build a versioned SO-101 USD with pre-baked convex gripper colliders.

The source SO-101 asset authors each gripper collision mesh as one convex
hull.  That fills grasp-critical concavities.  This asset-preparation tool
deactivates the two grasp-critical collider instance roots and adds
deterministic, pre-baked CoACD parts in the same rigid-body frames.  The servo
housing hull is preserved because the live path-clearance audit keeps it far
outside the grasp envelope.  Visuals, articulation, joint transforms, and
authored mass properties continue to come directly from the source USD through
a sublayer.

CoACD is an authoring dependency only.  The resulting USD contains ordinary
``convexHull`` meshes, so Newton runtime jobs neither import CoACD nor risk its
silent convex-hull fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pxr import Gf, Plug, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from newton_calibration.isaaclab.tasks.so101_peg_insertion.robot_adapter import (
    SO101_TASK_TCP_OFFSET_GRIPPER_M,
)
from newton_calibration.isaaclab.tasks.so101_peg_insertion.contract import (
    PegInsertionSceneSpec,
)


GENERATOR_SCHEMA = "newton.calibration/so101-gripper-collision-overlay@1.3"
NEWTON_IMPORT_VERIFICATION_SCHEMA = "newton.calibration/newton-import-verification@1.0"
COLLISION_PROFILE = "so101_gripper_prebaked_v4"
COLLISION_SCOPE_PREFIX = "newton_collision_v4_"
EXPECTED_COACD_VERSION = "1.0.9"
MAX_HULL_VERTICES = 64
MCTS_NODES = 20
MCTS_ITERATIONS = 5
MCTS_MAX_DEPTH = 1
# Task-space gate for the grasp-critical fixed-follower opening.  This is the
# 18 mm peg envelope about the commissioned task TCP imported from the task
# adapter, over the span where the previous monolithic hull caused contact.
# The task TCP is intentionally not the open-gap midpoint: one jaw moves, so
# the 18 mm contact midpoint is different from the open-jaw visual midpoint.
APERTURE_PEG_RADIUS_M = 0.009
APERTURE_Z_RANGE_M = (-0.100, -0.070)
FIXED_FOLLOWER_MIN_CONTACT_GAP_M = 0.0001
PRESERVED_SERVO_MIN_CLEARANCE_M = 0.002
TARGETS = {
    "fixed_servo": ("gripper_link", "sts3215_03a_v1"),
    "fixed_follower": ("gripper_link", "wrist_roll_follower_so101_v1"),
    "moving_jaw": ("moving_jaw_so101_v1_link", "moving_jaw_so101_v1"),
}
DECOMPOSE_KEYS = {"fixed_follower", "moving_jaw"}
COPY_SOURCE_HULL_KEYS = set(TARGETS) - DECOMPOSE_KEYS
REPLACE_KEYS = set(TARGETS)


def _newton_schema_candidates(explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    configured = os.environ.get("NEWTON_USD_SCHEMA_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path("/isaac-sim/exts/omni.usd.schema.newton/usd/schema/newton/newton_usd_schemas"),
            Path(
                "/workspace/isaaclab/_isaac_sim/exts/omni.usd.schema.newton/"
                "usd/schema/newton/newton_usd_schemas"
            ),
        ]
    )
    isaaclab_path = os.environ.get("ISAACLAB_PATH")
    if isaaclab_path:
        candidates.append(
            Path(isaaclab_path)
            / "_isaac_sim/exts/omni.usd.schema.newton/usd/schema/newton/newton_usd_schemas"
        )
    # Keep order while removing aliases/duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def _register_newton_schema(explicit: Path | None) -> Path:
    """Register and prove the typed Newton mesh-collision schema up front."""

    candidates = _newton_schema_candidates(explicit)
    schema_path = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
    if schema_path is None:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(
            "Newton USD schema plug-in was not found. Set NEWTON_USD_SCHEMA_PATH or pass "
            f"--newton-schema-path. Checked: {checked}"
        )
    Plug.Registry().RegisterPlugins(str(schema_path))
    probe_stage = Usd.Stage.CreateInMemory()
    probe = UsdGeom.Mesh.Define(probe_stage, "/NewtonSchemaProbe").GetPrim()
    if not probe.ApplyAPI("NewtonMeshCollisionAPI"):
        raise RuntimeError(
            "Newton USD schema directory exists but NewtonMeshCollisionAPI could not be applied: "
            f"{schema_path}"
        )
    if "NewtonMeshCollisionAPI" not in probe.GetAppliedSchemas():
        raise RuntimeError("NewtonMeshCollisionAPI application was not recorded by OpenUSD")
    return schema_path


@dataclass(frozen=True)
class SourceCollider:
    key: str
    body_path: Sdf.Path
    mesh_path: Sdf.Path
    instance_root_path: Sdf.Path
    vertices_body_m: np.ndarray
    faces: np.ndarray
    collision_attributes: dict[str, str]
    material_semantics: dict[str, Any]
    subtree_semantics: dict[str, Any]
    source_collision_groups: list[str]
    source_filtered_pairs: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _proxy_geometry_sha256(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Fingerprint the exact float32/int32 geometry authored into one proxy."""

    point_bytes = np.ascontiguousarray(np.asarray(vertices, dtype="<f4"))
    face_bytes = np.ascontiguousarray(np.asarray(faces, dtype="<i4"))
    digest = hashlib.sha256()
    digest.update(b"newton.calibration/convex-proxy-geometry@1\0")
    digest.update(np.asarray(point_bytes.shape, dtype="<u8").tobytes())
    digest.update(point_bytes.tobytes())
    digest.update(np.asarray(face_bytes.shape, dtype="<u8").tobytes())
    digest.update(face_bytes.tobytes())
    return digest.hexdigest()


def _triangles(mesh: UsdGeom.Mesh) -> np.ndarray:
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count_value in counts:
        count = int(count_value)
        polygon = indices[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        for offset in range(1, count - 1):
            triangles.append((int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1])))
    result = np.asarray(triangles, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise RuntimeError(f"Could not triangulate {mesh.GetPath()}")
    return result


def _points_in_body(prim: Usd.Prim, body: Usd.Prim) -> np.ndarray:
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_to_body, _ = cache.ComputeRelativeTransform(prim, body)
    return np.asarray(
        [mesh_to_body.Transform(Gf.Vec3d(*point)) for point in points],
        dtype=np.float64,
    )


def _instance_root(prim: Usd.Prim, body: Usd.Prim) -> Usd.Prim:
    candidate = prim
    result: Usd.Prim | None = None
    while candidate and candidate != body:
        if candidate.IsInstance() or (candidate.IsInstanceProxy() and not candidate.GetParent().IsInstanceProxy()):
            result = candidate
        candidate = candidate.GetParent()
    if result is None:
        raise RuntimeError(f"Expected collider {prim.GetPath()} beneath an instance root")
    if result.IsInstanceProxy():
        result = result.GetParent()
    if not result.IsInstance():
        raise RuntimeError(f"Resolved collider root is not authorable instance prim: {result.GetPath()}")
    return result


def _physics_material_semantics(stage: Usd.Stage, prim: Usd.Prim) -> dict[str, Any]:
    """Resolve the effective physics-purpose binding, including inheritance."""

    material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    material_prim = material.GetPrim() if material else Usd.Prim()
    physics_attributes = (
        {
            attribute.GetName(): str(attribute.Get())
            for attribute in material_prim.GetAttributes()
            if attribute.GetName().startswith(("physics:", "newton:", "mjc:", "physx"))
        }
        if material_prim.IsValid()
        else {}
    )
    has_physics_material = bool(
        material_prim and material_prim.HasAPI(UsdPhysics.MaterialAPI)
    )
    if has_physics_material or physics_attributes:
        raise RuntimeError(
            f"Collider {prim.GetPath()} resolves an authored physics material at "
            f"{material_prim.GetPath()}. This generator refuses replacement until "
            "that effective binding is explicitly cloned and verified."
        )
    direct_bindings = [
        {
            "relationship": direct.GetName(),
            "targets": [str(target) for target in direct.GetTargets()],
        }
        for direct in prim.GetRelationships()
        if direct.GetName().startswith("material:binding")
    ]
    return {
        "resolution": "UsdShade.MaterialBindingAPI.ComputeBoundMaterial('physics')",
        "effective_material_path": str(material_prim.GetPath()) if material_prim.IsValid() else None,
        "effective_binding_relationship": (
            relationship.GetName() if relationship and relationship.IsValid() else None
        ),
        "effective_material_has_physics_api": has_physics_material,
        "effective_material_physics_attributes": physics_attributes,
        "source_direct_bindings": direct_bindings,
        "new_parts_use_newton_default_physics_material": True,
        "reason": (
            "effective physics-purpose material has neither UsdPhysics.MaterialAPI nor "
            "physics/newton material attributes; source and replacement therefore use "
            "Newton's default physics material semantics"
        ),
    }


def _instance_subtree_semantics(instance_root: Usd.Prim) -> dict[str, Any]:
    """Prove deactivation removes collision representation only."""

    prim_paths: list[str] = []
    mesh_paths: list[str] = []
    target_keys: list[str] = []
    unsafe_paths: list[str] = []
    for descendant in Usd.PrimRange(
        instance_root,
        Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate),
    ):
        prim_paths.append(str(descendant.GetPath()))
        if descendant.IsA(UsdGeom.Mesh):
            mesh_paths.append(str(descendant.GetPath()))
            if not descendant.HasAPI(UsdPhysics.CollisionAPI):
                unsafe_paths.append(str(descendant.GetPath()))
            matching_keys = [
                key for key, (_, token) in TARGETS.items() if token in str(descendant.GetPath())
            ]
            if len(matching_keys) != 1:
                unsafe_paths.append(str(descendant.GetPath()))
            else:
                target_keys.extend(matching_keys)
        if (
            descendant.HasAPI(UsdPhysics.RigidBodyAPI)
            or descendant.HasAPI(UsdPhysics.MassAPI)
            or descendant.GetTypeName().endswith("Joint")
        ):
            unsafe_paths.append(str(descendant.GetPath()))
    if not mesh_paths or len(mesh_paths) != len(set(target_keys)) or unsafe_paths:
        raise RuntimeError(
            f"Refusing to deactivate non-collision-only instance subtree {instance_root.GetPath()}: "
            f"meshes={mesh_paths}, unsafe={unsafe_paths}"
        )
    return {
        "instance_root": str(instance_root.GetPath()),
        "prim_count": len(prim_paths),
        "mesh_paths": mesh_paths,
        "target_keys": sorted(target_keys),
        "collision_only": True,
    }


def _filtered_pair_targets(prim: Usd.Prim) -> list[str]:
    if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        return []
    return sorted(
        str(target)
        for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
    )


def _collision_group_memberships(stage: Usd.Stage, path: Sdf.Path) -> list[str]:
    memberships: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.CollisionGroup):
            continue
        collection = Usd.CollectionAPI.Get(prim, "colliders")
        if not collection:
            continue
        query = collection.ComputeMembershipQuery()
        if query.IsPathIncluded(path):
            memberships.append(str(prim.GetPath()))
    return sorted(memberships)


def _nearest_rigid_body_path(prim: Usd.Prim) -> Sdf.Path | None:
    candidate = prim.GetParent()
    while candidate:
        if candidate.HasAPI(UsdPhysics.RigidBodyAPI):
            return candidate.GetPath()
        candidate = candidate.GetParent()
    return None


def _collect_source_colliders(stage: Usd.Stage) -> list[SourceCollider]:
    found: dict[str, SourceCollider] = {}
    for body in stage.Traverse():
        matching = [key for key, (body_name, _) in TARGETS.items() if body.GetName() == body_name]
        if not matching:
            continue
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path_text = str(prim.GetPath())
            for key in matching:
                _, token = TARGETS[key]
                if token not in path_text:
                    continue
                approximation = None
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if str(approximation) != "convexHull":
                    raise RuntimeError(
                        f"Source collider {prim.GetPath()} changed approximation: expected convexHull, got {approximation}"
                    )
                vertices = _points_in_body(prim, body)
                faces = _triangles(UsdGeom.Mesh(prim))
                topology = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
                if not topology.is_watertight or not topology.is_winding_consistent:
                    raise RuntimeError(f"Source collider is not a closed consistently wound mesh: {prim.GetPath()}")
                if key in found:
                    raise RuntimeError(f"Found multiple collision meshes for target {key!r}")
                collision_attributes = {
                    attribute.GetName(): str(attribute.Get())
                    for attribute in prim.GetAttributes()
                    if attribute.GetName().startswith(("physics:", "newton:", "mjc:", "physx"))
                }
                unexpected_collision_attributes = set(collision_attributes) - {
                    "physics:approximation",
                    "physics:collisionEnabled",
                }
                if key in REPLACE_KEYS and unexpected_collision_attributes:
                    raise RuntimeError(
                        f"Collider {prim.GetPath()} has collision attributes that must be "
                        f"explicitly propagated before replacement: {sorted(unexpected_collision_attributes)}"
                    )
                instance_root = _instance_root(prim, body)
                source_filtered_pairs = sorted(
                    set(_filtered_pair_targets(prim))
                    | set(_filtered_pair_targets(instance_root))
                )
                if key in REPLACE_KEYS and source_filtered_pairs:
                    raise RuntimeError(
                        f"Collider {prim.GetPath()} has filtered-pair targets that must be "
                        f"explicitly propagated before replacement: {source_filtered_pairs}"
                    )
                found[key] = SourceCollider(
                    key=key,
                    body_path=body.GetPath(),
                    mesh_path=prim.GetPath(),
                    instance_root_path=instance_root.GetPath(),
                    vertices_body_m=np.asarray(topology.vertices, dtype=np.float64),
                    faces=np.asarray(topology.faces, dtype=np.int64),
                    collision_attributes=collision_attributes,
                    material_semantics=_physics_material_semantics(stage, prim),
                    subtree_semantics=_instance_subtree_semantics(instance_root),
                    source_collision_groups=_collision_group_memberships(stage, prim.GetPath()),
                    source_filtered_pairs=source_filtered_pairs,
                )
    missing = sorted(set(TARGETS) - set(found))
    if missing:
        raise RuntimeError(f"Missing expected SO-101 gripper colliders: {missing}")
    return [found[key] for key in TARGETS]


def _canonical_convex_part(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True).convex_hull
    # USD Mesh points are point3f.  Re-hull after the exact float32
    # quantization that will be authored so the saved topology is convex in
    # the representation Newton actually reads.
    quantized_vertices = np.asarray(mesh.vertices, dtype=np.float32).astype(np.float64)
    mesh = trimesh.Trimesh(
        vertices=quantized_vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=True,
        validate=True,
    ).convex_hull
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not mesh.is_watertight or not mesh.is_convex:
        raise RuntimeError("CoACD returned a part that could not be canonicalized as a closed convex hull")
    if len(vertices) > MAX_HULL_VERTICES:
        raise RuntimeError(
            f"Convex part has {len(vertices)} vertices; expected <= {MAX_HULL_VERTICES}. "
            "Do not allow Newton to decimate it implicitly."
        )

    # Stable vertex and face ordering makes the authored topology reproducible.
    rounded = np.round(vertices, decimals=12)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order), dtype=np.int64)
    vertices = rounded[order]
    faces = remap[faces]
    normalized_faces: list[tuple[int, int, int]] = []
    for face in faces:
        values = [int(value) for value in face]
        minimum = values.index(min(values))
        values = values[minimum:] + values[:minimum]
        normalized_faces.append(tuple(values))
    faces = np.asarray(sorted(normalized_faces), dtype=np.int64)
    return vertices, faces, float(abs(mesh.volume))


def _decompose(
    collider: SourceCollider,
    *,
    threshold_m: float,
    seed: int,
    max_parts: int,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    try:
        import coacd
    except ImportError as error:
        raise RuntimeError(
            "CoACD is required to author this asset. Build/run Dockerfile.asset-prep; "
            "runtime fallback to one convex hull is forbidden."
        ) from error
    installed = importlib.metadata.version("coacd")
    if installed != EXPECTED_COACD_VERSION:
        raise RuntimeError(
            f"Expected coacd=={EXPECTED_COACD_VERSION}, found {installed}; "
            "refusing an unversioned decomposition."
        )
    result = coacd.run_coacd(
        coacd.Mesh(collider.vertices_body_m, collider.faces),
        threshold=threshold_m,
        max_convex_hull=max_parts,
        preprocess_mode="auto",
        preprocess_resolution=50,
        resolution=2000,
        mcts_nodes=MCTS_NODES,
        mcts_iterations=MCTS_ITERATIONS,
        mcts_max_depth=MCTS_MAX_DEPTH,
        merge=True,
        decimate=True,
        max_ch_vertex=MAX_HULL_VERTICES,
        extrude=False,
        seed=seed,
        real_metric=True,
    )
    if len(result) < 2:
        raise RuntimeError(
            f"CoACD produced {len(result)} part(s) for concave collider {collider.key}; "
            "a monolithic hull is not an acceptable correction."
        )
    parts = [_canonical_convex_part(vertices, faces) for vertices, faces in result]
    parts.sort(
        key=lambda part: (
            tuple(np.round(part[0].mean(axis=0), decimals=9)),
            round(part[2], 15),
            len(part[0]),
        )
    )
    return parts


def _convex_plane_clearance_m(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Return a conservative outside clearance for points versus one hull.

    For an outward-oriented convex hull, a point is inside iff all plane
    distances are non-positive.  The maximum plane distance is therefore
    positive for a point outside the hull.
    """

    hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True).convex_hull
    origins = np.asarray(hull.vertices, dtype=np.float64)[np.asarray(hull.faces)[:, 0]]
    normals = np.asarray(hull.face_normals, dtype=np.float64)
    offsets = np.einsum("ij,ij->i", normals, origins)
    return np.max(points @ normals.T - offsets[None, :], axis=1)


def _verify_fixed_follower_aperture(
    parts: list[tuple[np.ndarray, np.ndarray, float]],
) -> dict[str, Any]:
    """Prove the pre-baked fixed side does not bridge the 18 mm peg path."""

    radii = np.linspace(0.0, APERTURE_PEG_RADIUS_M, 5)
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    z_values = np.linspace(*APERTURE_Z_RANGE_M, 13)
    points: list[tuple[float, float, float]] = []
    for z_value in z_values:
        points.append((SO101_TASK_TCP_OFFSET_GRIPPER_M[0], 0.0, float(z_value)))
        for radius in radii[1:]:
            for angle in angles:
                points.append(
                    (
                        SO101_TASK_TCP_OFFSET_GRIPPER_M[0] + float(radius * np.cos(angle)),
                        float(radius * np.sin(angle)),
                        float(z_value),
                    )
                )
    samples = np.asarray(points, dtype=np.float64)
    per_part = [
        _convex_plane_clearance_m(samples, vertices, faces)
        for vertices, faces, _ in parts
    ]
    # A sample is outside the union only when it is outside every part; its
    # nearest conservative outside margin is the minimum across parts.
    union_clearance = np.min(np.stack(per_part, axis=1), axis=1)
    minimum_index = int(np.argmin(union_clearance))
    minimum_clearance = float(union_clearance[minimum_index])
    if minimum_clearance < FIXED_FOLLOWER_MIN_CONTACT_GAP_M:
        raise RuntimeError(
            "Pre-baked fixed-follower parts penetrate or consume the commissioned "
            "fixed-side contact gap: "
            f"minimum sampled clearance {minimum_clearance * 1000.0:.3f} mm at "
            f"{samples[minimum_index].tolist()}, required >= "
            f"{FIXED_FOLLOWER_MIN_CONTACT_GAP_M * 1000.0:.3f} mm"
        )
    return {
        "frame": "gripper_link",
        "peg_diameter_m": 2.0 * APERTURE_PEG_RADIUS_M,
        "peg_center_x_m": SO101_TASK_TCP_OFFSET_GRIPPER_M[0],
        "z_range_m": list(APERTURE_Z_RANGE_M),
        "sample_count": int(len(samples)),
        "minimum_conservative_clearance_m": minimum_clearance,
        "minimum_clearance_sample_m": samples[minimum_index].tolist(),
        "required_minimum_contact_gap_m": FIXED_FOLLOWER_MIN_CONTACT_GAP_M,
        "passed": True,
        "scope": (
            "static fixed-follower nonpenetration/contact-gap gate; the open-jaw approach/descent "
            "run remains the authoritative whole-gripper dynamic gate"
        ),
    }


def _verify_preserved_servo_clearance(
    collider: SourceCollider,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    """Qualify the preserved housing hull against the complete descent sweep."""

    spec = PegInsertionSceneSpec()
    # At grasp, the TCP is this far above the peg centre.  Approach moves the
    # gripper upward while the peg remains on the table, so the centre moves
    # farther down in the gripper frame by the TCP-height delta.
    # This intentionally uses both the adapter Z offset and the task's measured
    # peg/TCP relation; the pinned result is -104.5 mm.
    centre_z_at_grasp = -abs(float(SO101_TASK_TCP_OFFSET_GRIPPER_M[2])) - spec.grasp_tcp_above_peg_center_m
    centre_z_at_approach = centre_z_at_grasp - (
        spec.transport_tcp_height_m - spec.grasp_tcp_height_m
    )
    centre_z_values = np.linspace(centre_z_at_approach, centre_z_at_grasp, 15)
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    radii = np.linspace(0.0, spec.peg_radius_m, 5)
    heights = np.linspace(-0.5 * spec.peg_height_m, 0.5 * spec.peg_height_m, 13)
    points: list[tuple[float, float, float]] = []
    for centre_z in centre_z_values:
        for height in heights:
            points.append((SO101_TASK_TCP_OFFSET_GRIPPER_M[0], 0.0, float(centre_z + height)))
            for radius in radii[1:]:
                for angle in angles:
                    points.append(
                        (
                            SO101_TASK_TCP_OFFSET_GRIPPER_M[0] + float(radius * np.cos(angle)),
                            float(radius * np.sin(angle)),
                            float(centre_z + height),
                        )
                    )
    samples = np.asarray(points, dtype=np.float64)
    clearance = _convex_plane_clearance_m(
        samples,
        collider.vertices_body_m,
        collider.faces,
    )
    minimum_index = int(np.argmin(clearance))
    minimum_clearance = float(clearance[minimum_index])
    if minimum_clearance < PRESERVED_SERVO_MIN_CLEARANCE_M:
        raise RuntimeError(
            "The preserved fixed-servo convex hull is not clearance-qualified for the "
            f"commissioned approach/descent sweep: {minimum_clearance * 1000.0:.3f} mm"
        )
    return {
        "source_usd_sha256": source_sha256,
        "source_mesh_path": str(collider.mesh_path),
        "frame": "gripper_link",
        "task_id": spec.task_id,
        "task_tcp_offset_gripper_m": list(SO101_TASK_TCP_OFFSET_GRIPPER_M),
        "open_gripper_rad": 0.30,
        "approach_tcp_height_m": spec.transport_tcp_height_m,
        "grasp_tcp_height_m": spec.grasp_tcp_height_m,
        "peg_diameter_m": spec.peg_diameter_m,
        "peg_height_m": spec.peg_height_m,
        "sample_count": int(len(samples)),
        "minimum_conservative_clearance_m": minimum_clearance,
        "minimum_clearance_sample_m": samples[minimum_index].tolist(),
        "required_minimum_clearance_m": PRESERVED_SERVO_MIN_CLEARANCE_M,
        "passed": True,
        "method": (
            "dense swept-cylinder samples against the exact preserved source convex hull; "
            "source hash, mesh, TCP, gripper opening and path heights are locked above"
        ),
    }


def _write_mesh(stage: Usd.Stage, path: Sdf.Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*point) for point in vertices])
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(faces.reshape(-1).astype(int).tolist())
    mesh.CreateDoubleSidedAttr().Set(False)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    mesh.CreateExtentAttr().Set([Gf.Vec3f(*minimum), Gf.Vec3f(*maximum)])
    mesh.MakeInvisible()
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr().Set(True)
    approximation = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    approximation.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
    prim = mesh.GetPrim()
    if not prim.ApplyAPI("NewtonMeshCollisionAPI"):
        raise RuntimeError(f"Could not apply NewtonMeshCollisionAPI to {path}")
    max_hull_vertices = prim.GetAttribute("newton:maxHullVertices")
    if not max_hull_vertices:
        max_hull_vertices = prim.CreateAttribute(
            "newton:maxHullVertices",
            Sdf.ValueTypeNames.Int,
            custom=False,
        )
    max_hull_vertices.Set(MAX_HULL_VERTICES)


def _attribute_value(prim: Usd.Prim, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute else None
    return str(value) if value is not None else None


def _physics_signature(stage: Usd.Stage) -> dict[str, Any]:
    joints: dict[str, dict[str, Any]] = {}
    masses: dict[str, dict[str, Any]] = {}
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name.endswith("Joint"):
            joints[str(prim.GetPath())] = {
                attribute.GetName(): str(attribute.Get())
                for attribute in prim.GetAttributes()
                if attribute.GetName().startswith(("physics:", "drive:", "limit:"))
            }
        if prim.HasAPI(UsdPhysics.MassAPI):
            masses[str(prim.GetPath())] = {
                name: _attribute_value(prim, name)
                for name in (
                    "physics:mass",
                    "physics:centerOfMass",
                    "physics:diagonalInertia",
                    "physics:principalAxes",
                )
            }
    return {"joints": joints, "masses": masses}


def _verify_composed_asset(
    source_stage: Usd.Stage,
    derived_stage: Usd.Stage,
    colliders: list[SourceCollider],
    expected_part_counts: dict[str, int],
) -> dict[str, Any]:
    if _physics_signature(source_stage) != _physics_signature(derived_stage):
        raise RuntimeError("Derived USD changed joint or authored mass/inertia opinions")
    original_activity = {
        collider.key: bool(
            derived_stage.GetPrimAtPath(collider.mesh_path).IsValid()
            and derived_stage.GetPrimAtPath(collider.mesh_path).IsActive()
        )
        for collider in colliders
    }
    unexpected_original_activity = {
        key: active
        for key, active in original_activity.items()
        if key in REPLACE_KEYS and active
    }
    if unexpected_original_activity:
        raise RuntimeError(
            "Original collider activity does not match the scoped replacement policy: "
            f"{unexpected_original_activity}"
        )

    active_parts: dict[str, int] = {key: 0 for key in REPLACE_KEYS}
    collider_by_key = {collider.key: collider for collider in colliders}
    decomposition_tokens: list[str] = []
    invalid_prebaked_parts: list[str] = []
    part_attachment_records: list[dict[str, Any]] = []
    for prim in derived_stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if str(approximation) == "convexDecomposition":
                decomposition_tokens.append(str(prim.GetPath()))
        path_text = str(prim.GetPath())
        for key in REPLACE_KEYS:
            marker = f"{COLLISION_SCOPE_PREFIX}{key}/part_"
            if marker not in path_text:
                continue
            source = collider_by_key[key]
            active_parts[key] += 1
            if str(approximation) != "convexHull":
                raise RuntimeError(f"Pre-baked part is not authored as convexHull: {prim.GetPath()}")
            if "NewtonMeshCollisionAPI" not in prim.GetAppliedSchemas():
                raise RuntimeError(f"Pre-baked part lacks NewtonMeshCollisionAPI: {prim.GetPath()}")
            max_vertices_attribute = prim.GetAttribute("newton:maxHullVertices")
            if not max_vertices_attribute or max_vertices_attribute.IsCustom():
                raise RuntimeError(
                    f"Pre-baked part does not use the typed newton:maxHullVertices attribute: {prim.GetPath()}"
                )
            max_vertices = max_vertices_attribute.Get()
            if max_vertices != MAX_HULL_VERTICES:
                raise RuntimeError(
                    f"Pre-baked part has newton:maxHullVertices={max_vertices}, "
                    f"expected {MAX_HULL_VERTICES}: {prim.GetPath()}"
                )
            topology = trimesh.Trimesh(
                vertices=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()),
                faces=_triangles(UsdGeom.Mesh(prim)),
                process=True,
                validate=True,
            )
            if key in DECOMPOSE_KEYS and (
                not topology.is_watertight
                or not topology.is_winding_consistent
                or len(topology.vertices) > MAX_HULL_VERTICES
            ):
                invalid_prebaked_parts.append(str(prim.GetPath()))
            nearest_body = _nearest_rigid_body_path(prim)
            if nearest_body != source.body_path:
                raise RuntimeError(
                    f"Pre-baked part {prim.GetPath()} attaches to {nearest_body}, "
                    f"expected source body {source.body_path}"
                )
            if UsdGeom.Xformable(prim).GetOrderedXformOps():
                raise RuntimeError(f"Pre-baked mesh unexpectedly has a local transform: {prim.GetPath()}")
            source_groups = source.source_collision_groups
            replacement_groups = _collision_group_memberships(derived_stage, prim.GetPath())
            if replacement_groups != source_groups:
                raise RuntimeError(
                    f"Collision-group membership changed for {prim.GetPath()}: "
                    f"source={source_groups}, replacement={replacement_groups}"
                )
            replacement_filtered_pairs = _filtered_pair_targets(prim)
            if replacement_filtered_pairs != source.source_filtered_pairs:
                raise RuntimeError(
                    f"Filtered-pair membership changed for {prim.GetPath()}: "
                    f"source={source.source_filtered_pairs}, replacement={replacement_filtered_pairs}"
                )
            replacement_material = _physics_material_semantics(derived_stage, prim)
            part_attachment_records.append(
                {
                    "path": str(prim.GetPath()),
                    "body_path": str(nearest_body),
                    "identity_body_local_transform": True,
                    "collision_groups": replacement_groups,
                    "filtered_pairs": replacement_filtered_pairs,
                    "uses_default_physics_material": replacement_material[
                        "new_parts_use_newton_default_physics_material"
                    ],
                }
            )
    if decomposition_tokens:
        raise RuntimeError(
            "Active convexDecomposition tokens could invoke an unavailable runtime dependency: "
            f"{decomposition_tokens}"
        )
    if invalid_prebaked_parts:
        raise RuntimeError(f"Invalid pre-baked convex parts: {invalid_prebaked_parts}")
    if active_parts != expected_part_counts:
        raise RuntimeError(f"Composed part counts differ: expected {expected_part_counts}, found {active_parts}")
    return {
        "joint_count": len(_physics_signature(derived_stage)["joints"]),
        "mass_property_prim_count": len(_physics_signature(derived_stage)["masses"]),
        "original_collider_activity": original_activity,
        "active_convex_decomposition_tokens": 0,
        "active_prebaked_part_counts": active_parts,
        "max_hull_vertices": MAX_HULL_VERTICES,
        "part_attachment_and_filter_checks": part_attachment_records,
    }


def _verify_newton_model_import(
    asset: Path,
    expected_body_paths: dict[str, str],
    expected_part_counts: dict[str, int],
) -> dict[str, Any]:
    """Prove the runtime importer sees the authored parts and hull limit."""

    import newton

    builder = newton.ModelBuilder()
    import_result = builder.add_usd(str(asset))
    path_shape_map = import_result["path_shape_map"]
    path_body_map = {
        str(path): int(index)
        for path, index in import_result["path_body_map"].items()
    }
    imported_stage = Usd.Stage.Open(str(asset), Usd.Stage.LoadAll)
    if imported_stage is None:
        raise RuntimeError(f"Could not reopen asset for Newton-import verification: {asset}")
    imported_counts = {key: 0 for key in expected_part_counts}
    part_details: list[dict[str, Any]] = []
    for path, shape_index in path_shape_map.items():
        path_text = str(path)
        matching_key = next(
            (
                key
                for key in expected_part_counts
                if f"{COLLISION_SCOPE_PREFIX}{key}/part_" in path_text
            ),
            None,
        )
        if matching_key is None:
            continue
        imported_counts[matching_key] += 1
        shape_type = builder.shape_type[shape_index]
        source = builder.shape_source[shape_index]
        imported_vertex_count = int(len(source.vertices))
        imported_max_hull_vertices = int(source.maxhullvert)
        if shape_type != newton.GeoType.CONVEX_MESH:
            raise RuntimeError(f"Newton did not import pre-baked part as CONVEX_MESH: {path_text}")
        if imported_max_hull_vertices != MAX_HULL_VERTICES:
            raise RuntimeError(
                f"Newton ignored newton:maxHullVertices on {path_text}: "
                f"found {imported_max_hull_vertices}, expected {MAX_HULL_VERTICES}"
            )
        if matching_key in DECOMPOSE_KEYS and imported_vertex_count > MAX_HULL_VERTICES:
            raise RuntimeError(
                f"Newton imported {imported_vertex_count} vertices for {path_text}; "
                f"limit is {MAX_HULL_VERTICES}"
            )
        expected_body_path = expected_body_paths[matching_key]
        expected_body_index = path_body_map.get(expected_body_path)
        if expected_body_index is None:
            raise RuntimeError(f"Newton did not import expected rigid body {expected_body_path}")
        imported_body_index = int(builder.shape_body[shape_index])
        if imported_body_index != expected_body_index:
            raise RuntimeError(
                f"Newton attached {path_text} to body index {imported_body_index}; "
                f"expected {expected_body_index} ({expected_body_path})"
            )
        transform = np.asarray(list(builder.shape_transform[shape_index]), dtype=np.float64)
        if transform.shape != (7,):
            raise RuntimeError(f"Unexpected Newton shape transform for {path_text}: {transform}")
        translation = transform[:3]
        quaternion_xyzw = transform[3:]
        if np.linalg.norm(translation) > 1.0e-8 or not np.isclose(
            abs(quaternion_xyzw[3]), 1.0, rtol=0.0, atol=1.0e-7
        ) or np.linalg.norm(quaternion_xyzw[:3]) > 1.0e-7:
            raise RuntimeError(
                f"Newton imported a non-identity body-local transform for {path_text}: "
                f"{transform.tolist()}"
            )
        authored_prim = imported_stage.GetPrimAtPath(path_text)
        authored_vertices = np.asarray(
            UsdGeom.Mesh(authored_prim).GetPointsAttr().Get(),
            dtype=np.float64,
        )
        imported_vertices = np.asarray(source.vertices, dtype=np.float64)
        imported_scale = np.asarray(list(builder.shape_scale[shape_index]), dtype=np.float64)
        imported_vertices = imported_vertices * imported_scale[None, :]
        authored_bounds = np.stack(
            (authored_vertices.min(axis=0), authored_vertices.max(axis=0))
        )
        imported_bounds = np.stack(
            (imported_vertices.min(axis=0), imported_vertices.max(axis=0))
        )
        if not np.allclose(imported_bounds, authored_bounds, rtol=0.0, atol=2.0e-7):
            raise RuntimeError(
                f"Newton-imported AABB changed for {path_text}: "
                f"authored={authored_bounds.tolist()}, imported={imported_bounds.tolist()}"
            )
        material_fields = {
            "ka": float(builder.shape_material_ka[shape_index]),
            "kd": float(builder.shape_material_kd[shape_index]),
            "ke": float(builder.shape_material_ke[shape_index]),
            "kf": float(builder.shape_material_kf[shape_index]),
            "kh": float(builder.shape_material_kh[shape_index]),
            "mu": float(builder.shape_material_mu[shape_index]),
            "mu_rolling": float(builder.shape_material_mu_rolling[shape_index]),
            "mu_torsional": float(builder.shape_material_mu_torsional[shape_index]),
            "restitution": float(builder.shape_material_restitution[shape_index]),
        }
        expected_material_fields = {
            name: float(getattr(builder.default_shape_cfg, name))
            for name in material_fields
        }
        if material_fields != expected_material_fields:
            raise RuntimeError(
                f"Newton replacement material differs from its default semantics on {path_text}: "
                f"actual={material_fields}, expected={expected_material_fields}"
            )
        part_details.append(
            {
                "path": path_text,
                "shape_type": "CONVEX_MESH",
                "body_path": expected_body_path,
                "body_index": imported_body_index,
                "body_local_transform": transform.tolist(),
                "authored_aabb_m": authored_bounds.tolist(),
                "imported_aabb_m": imported_bounds.tolist(),
                "material": material_fields,
                "imported_vertex_count": imported_vertex_count,
                "imported_max_hull_vertices": imported_max_hull_vertices,
            }
        )
    if imported_counts != expected_part_counts:
        raise RuntimeError(
            f"Newton imported wrong pre-baked part counts: expected {expected_part_counts}, "
            f"found {imported_counts}"
        )
    return {
        "schema": NEWTON_IMPORT_VERIFICATION_SCHEMA,
        "runtime_importer": "newton.ModelBuilder.add_usd",
        "total_shape_count": int(builder.shape_count),
        "prebaked_part_counts": imported_counts,
        "all_parts_imported_as_convex_mesh": True,
        "all_parts_honor_max_hull_vertices": True,
        "all_parts_attached_to_expected_body": True,
        "all_parts_have_identity_body_local_transform": True,
        "all_part_aabbs_match_authored_geometry": True,
        "all_parts_use_newton_default_physics_material": True,
        "parts": part_details,
    }


def _verify_newton_model_import_subprocess(
    asset: Path,
    verification_path: Path,
    expected_body_paths: dict[str, str],
    expected_part_counts: dict[str, int],
) -> dict[str, Any]:
    """Run Newton import in a clean process, isolated from CoACD native state."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--verify-newton-asset",
        str(asset),
        "--verify-output",
        str(verification_path),
        "--expected-body-paths-json",
        json.dumps(expected_body_paths, sort_keys=True),
        "--expected-counts-json",
        json.dumps(expected_part_counts, sort_keys=True),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Fresh-process Newton import verification failed "
            f"(exit {completed.returncode}).\nSTDOUT:\n{completed.stdout[-8000:]}"
            f"\nSTDERR:\n{completed.stderr[-8000:]}"
        )
    if not verification_path.is_file():
        raise RuntimeError("Newton verifier exited successfully without writing its result")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("schema") != NEWTON_IMPORT_VERIFICATION_SCHEMA:
        raise RuntimeError(f"Unexpected Newton verification schema: {verification.get('schema')}")
    if verification.get("prebaked_part_counts") != expected_part_counts:
        raise RuntimeError("Newton child verification returned unexpected part counts")
    return verification


def build(
    source: Path,
    output: Path,
    manifest_path: Path,
    *,
    threshold_m: float,
    seed: int,
    max_parts: int,
    newton_schema_path: Path | None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source:
        raise ValueError("Output must be a derived USD; refusing to overwrite the source asset")
    if output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("Output must have a USD extension")
    if threshold_m <= 0.0:
        raise ValueError("--threshold-m must be positive")
    if max_parts < 2:
        raise ValueError("--max-parts must be at least 2")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError(
            "Deterministic CoACD authoring requires OMP_NUM_THREADS=1. "
            "Use Dockerfile.asset-prep or set it explicitly before launch."
        )

    # Fail before mesh loading/decomposition if the typed Newton schema cannot
    # be authored.  A missing plug-in must never degrade to an untyped custom
    # attribute or a runtime approximation fallback.
    registered_newton_schema_path = _register_newton_schema(newton_schema_path)
    print(
        f"[ASSET-PREP] registered NewtonMeshCollisionAPI from {registered_newton_schema_path}",
        flush=True,
    )
    source_stage = Usd.Stage.Open(str(source), Usd.Stage.LoadAll)
    if source_stage is None or not source_stage.GetDefaultPrim():
        raise RuntimeError(f"Could not open source/default prim: {source}")
    stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(source_stage))
    if not np.isclose(stage_meters_per_unit, 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            "This SO-101 authoring profile requires stage metresPerUnit=1 so body-local "
            f"coordinates and real-metric CoACD settings agree; found {stage_meters_per_unit}. "
            "Convert the source asset explicitly before generating an overlay."
        )
    colliders = _collect_source_colliders(source_stage)
    source_hash = _sha256(source)
    colliders_by_key = {collider.key: collider for collider in colliders}
    preserved_servo_verification = _verify_preserved_servo_clearance(
        colliders_by_key["fixed_servo"],
        source_sha256=source_hash,
    )
    settings = {
        "coacd_version": EXPECTED_COACD_VERSION,
        "threshold_m": threshold_m,
        "real_metric": True,
        "seed": seed,
        "preprocess_mode": "auto",
        "preprocess_resolution": 50,
        "resolution": 2000,
        "mcts_nodes": MCTS_NODES,
        "mcts_iterations": MCTS_ITERATIONS,
        "mcts_max_depth": MCTS_MAX_DEPTH,
        "merge": True,
        "max_convex_hull": max_parts,
        "decimate": True,
        "max_ch_vertex": MAX_HULL_VERTICES,
        "extrude": False,
        "stage_meters_per_unit": stage_meters_per_unit,
        "newton_usd_schema": "NewtonMeshCollisionAPI",
        "omp_num_threads": 1,
    }
    replacement_parts: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    for collider in colliders:
        if collider.key in COPY_SOURCE_HULL_KEYS:
            print(
                f"[ASSET-PREP] copying {collider.key}: {collider.mesh_path} "
                "(clearance-qualified source convexHull; shared instance subtree must be replaced)",
                flush=True,
            )
            source_mesh = trimesh.Trimesh(
                vertices=collider.vertices_body_m,
                faces=collider.faces,
                process=True,
                validate=True,
            )
            replacement_parts[collider.key] = [
                (
                    np.asarray(source_mesh.vertices, dtype=np.float64),
                    np.asarray(source_mesh.faces, dtype=np.int64),
                    float(abs(source_mesh.volume)),
                )
            ]
            continue
        print(f"[ASSET-PREP] decomposing {collider.key}: {collider.mesh_path}", flush=True)
        replacement_parts[collider.key] = _decompose(
            collider,
            threshold_m=threshold_m,
            seed=seed,
            max_parts=max_parts,
        )
        print(f"[ASSET-PREP] {collider.key}: {len(replacement_parts[collider.key])} parts", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived asset: {output}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    temporary_output = output.with_name(f".{output.stem}.tmp-{os.getpid()}{output.suffix}")
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.stem}.tmp-{os.getpid()}{manifest_path.suffix}"
    )
    temporary_verification = manifest_path.with_name(
        f".{manifest_path.stem}.newton-import-{os.getpid()}.json"
    )
    relative_source = os.path.relpath(source, output.parent)
    published_output = False
    published_manifest = False
    try:
        layer = Sdf.Layer.CreateNew(str(temporary_output))
        layer.subLayerPaths = [relative_source]
        stage = Usd.Stage.Open(layer, Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"Could not compose derived stage: {temporary_output}")
        stage.SetDefaultPrim(stage.GetPrimAtPath(source_stage.GetDefaultPrim().GetPath()))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))

        for instance_root_path in sorted(
            {collider.instance_root_path for collider in colliders},
            key=str,
        ):
            stage.OverridePrim(instance_root_path).SetActive(False)
        for collider in colliders:
            # The source `/collisions` branch contains instance-constrained
            # composition.  A new direct child of the rigid body is authorable
            # and has the same body-local transform semantics.
            scope_path = collider.body_path.AppendChild(f"{COLLISION_SCOPE_PREFIX}{collider.key}")
            scope = UsdGeom.Scope.Define(stage, scope_path)
            scope.GetPrim().SetCustomDataByKey("newtonCalibration:sourceMesh", str(collider.mesh_path))
            scope.GetPrim().SetCustomDataByKey("newtonCalibration:generatorSchema", GENERATOR_SCHEMA)
            for index, (vertices, faces, _) in enumerate(replacement_parts[collider.key]):
                _write_mesh(stage, scope_path.AppendChild(f"part_{index:03d}"), vertices, faces)

        default_prim = stage.GetDefaultPrim()
        default_prim.SetCustomDataByKey("newtonCalibration:collisionProfile", COLLISION_PROFILE)
        default_prim.SetCustomDataByKey("newtonCalibration:sourceSha256", source_hash)
        default_prim.SetCustomDataByKey("newtonCalibration:generatorSchema", GENERATOR_SCHEMA)
        default_prim.SetCustomDataByKey("newtonCalibration:coacdSettingsJson", json.dumps(settings, sort_keys=True))
        stage.GetRootLayer().Save()

        derived_stage = Usd.Stage.Open(str(temporary_output), Usd.Stage.LoadAll)
        if derived_stage is None:
            raise RuntimeError(f"Generated USD cannot be reopened: {temporary_output}")
        expected_counts = {key: len(parts) for key, parts in replacement_parts.items()}
        aperture_verification = _verify_fixed_follower_aperture(
            replacement_parts["fixed_follower"]
        )
        verification = _verify_composed_asset(source_stage, derived_stage, colliders, expected_counts)
        runtime_import_verification = _verify_newton_model_import_subprocess(
            temporary_output,
            temporary_verification,
            {collider.key: str(collider.body_path) for collider in colliders},
            expected_counts,
        )
        temporary_verification.unlink(missing_ok=True)
        proxy_parts_by_key: dict[str, list[dict[str, Any]]] = {}
        for collider in colliders:
            scope_path = collider.body_path.AppendChild(
                f"{COLLISION_SCOPE_PREFIX}{collider.key}"
            )
            proxy_parts_by_key[collider.key] = [
                {
                    "path": str(scope_path.AppendChild(f"part_{index:03d}")),
                    "geometry_sha256": _proxy_geometry_sha256(vertices, faces),
                    "vertex_count": int(len(vertices)),
                    "triangle_count": int(len(faces)),
                    "volume_m3": float(volume),
                }
                for index, (vertices, faces, volume) in enumerate(
                    replacement_parts[collider.key]
                )
            ]
        proxy_path_fingerprints = {
            part["path"]: part["geometry_sha256"]
            for parts in proxy_parts_by_key.values()
            for part in parts
        }
        runtime_proxy_paths = {
            str(part["path"]) for part in runtime_import_verification["parts"]
        }
        if runtime_proxy_paths != set(proxy_path_fingerprints):
            missing = sorted(set(proxy_path_fingerprints) - runtime_proxy_paths)
            unexpected = sorted(runtime_proxy_paths - set(proxy_path_fingerprints))
            raise RuntimeError(
                "Newton-imported proxy paths do not match the fingerprinted authored set: "
                f"missing={missing}, unexpected={unexpected}"
            )
        proxy_set_payload = json.dumps(
            proxy_path_fingerprints,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        output_hash = _sha256(temporary_output)
        manifest: dict[str, Any] = {
            "schema": GENERATOR_SCHEMA,
            "source_usd": str(source),
            "source_usd_sha256": source_hash,
            "derived_usd": str(output),
            "derived_usd_sha256": output_hash,
            "source_sublayer": relative_source,
            "generator_settings": settings,
            "proxy_path_fingerprints": proxy_path_fingerprints,
            "proxy_set_sha256": hashlib.sha256(proxy_set_payload).hexdigest(),
            "colliders": [
                {
                    "key": collider.key,
                    "body_path": str(collider.body_path),
                    "source_mesh_path": str(collider.mesh_path),
                    "deactivated_instance_root": str(collider.instance_root_path),
                    "source_vertex_count": int(len(collider.vertices_body_m)),
                    "source_triangle_count": int(len(collider.faces)),
                    "source_collision_attributes": collider.collision_attributes,
                    "physics_material_semantics": collider.material_semantics,
                    "part_count": len(replacement_parts.get(collider.key, [])),
                    "part_vertex_counts": [
                        int(len(part[0])) for part in replacement_parts.get(collider.key, [])
                    ],
                    "part_triangle_counts": [
                        int(len(part[1])) for part in replacement_parts.get(collider.key, [])
                    ],
                    "part_volumes_m3": [
                        part[2] for part in replacement_parts.get(collider.key, [])
                    ],
                    "replacement_scope_path": str(
                        collider.body_path.AppendChild(
                            f"{COLLISION_SCOPE_PREFIX}{collider.key}"
                        )
                    ),
                    "parts": proxy_parts_by_key[collider.key],
                    "replacement_policy": (
                        "prebaked_multi_convex"
                        if collider.key in DECOMPOSE_KEYS
                        else "copied_source_convex_hull"
                    ),
                    "preservation_rationale": (
                        "The source-hashed commissioned approach/descent sweep measured a "
                        f"{preserved_servo_verification['minimum_conservative_clearance_m'] * 1000.0:.3f} mm "
                        "minimum fixed-servo hull clearance, above the 2 mm gate; this "
                        "non-contact housing remains outside the grasp envelope."
                        if collider.key in COPY_SOURCE_HULL_KEYS
                        else None
                    ),
                }
                for collider in colliders
            ],
            "aperture_verification": aperture_verification,
            "preserved_servo_clearance_verification": preserved_servo_verification,
            "verification": verification,
            "newton_runtime_import_verification": runtime_import_verification,
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_output, output)
        published_output = True
        os.replace(temporary_manifest, manifest_path)
        published_manifest = True
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        temporary_verification.unlink(missing_ok=True)
        if published_output and not published_manifest:
            output.unlink(missing_ok=True)
        raise
    print(f"RESULT={output}", flush=True)
    print(f"MANIFEST={manifest_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-newton-asset", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--verify-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-body-paths-json", help=argparse.SUPPRESS)
    parser.add_argument("--expected-counts-json", help=argparse.SUPPRESS)
    parser.add_argument(
        "--threshold-m",
        type=float,
        default=0.0005,
        help="CoACD real-metric concavity threshold in metres (default: 0.5 mm)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--newton-schema-path",
        type=Path,
        help=(
            "Path to the Newton USD schema plug-in directory. Defaults to "
            "NEWTON_USD_SCHEMA_PATH and known Isaac Lab container locations."
        ),
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=32,
        help="Maximum CoACD parts per source collider; must be at least 2 (default: 32)",
    )
    args = parser.parse_args()
    if args.verify_newton_asset is not None:
        if not (
            args.verify_output
            and args.expected_body_paths_json
            and args.expected_counts_json
        ):
            parser.error("internal Newton verification requires output, body paths and counts")
        verification = _verify_newton_model_import(
            args.verify_newton_asset,
            json.loads(args.expected_body_paths_json),
            {key: int(value) for key, value in json.loads(args.expected_counts_json).items()},
        )
        # Newton's native USD importer can leave mesh-owned allocations alive
        # until Python teardown.  With many independent convex meshes, normal
        # interpreter finalization has triggered a glibc double-free after the
        # verification result was complete.  Durably write the child-process
        # result, then exit without running unrelated native finalizers.  The
        # parent still requires exit code 0, a readable JSON result, the exact
        # schema, and the expected part counts before publishing either file.
        with args.verify_output.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(verification, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if not (args.source and args.output and args.manifest):
        parser.error("--source, --output and --manifest are required for asset generation")
    build(
        args.source,
        args.output,
        args.manifest,
        threshold_m=args.threshold_m,
        seed=args.seed,
        max_parts=args.max_parts,
        newton_schema_path=args.newton_schema_path,
    )


if __name__ == "__main__":
    main()
