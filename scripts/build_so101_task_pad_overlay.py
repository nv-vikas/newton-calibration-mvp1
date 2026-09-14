#!/usr/bin/env python3
"""Author a task-scoped SO-101 v4 overlay with matched planar grasp pads.

The v3 asset remains the collision representation for the gripper housing and
linkage.  This overlay disables only the v3 convex parts whose volumes overlap
an inscribed task-pad volume, then adds one deterministic convex pad per jaw.
The pad volumes are sampled against the canonical source meshes and publishing
fails if any sample protrudes beyond the source envelope.

The two gap-facing pad planes are parallel at the declared 18 mm contact
reference configuration.  This is deliberately a task-specific collision
overlay, not a replacement SO-101 asset or a universal gripper calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pxr import Gf, Plug, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


SCHEMA = "newton.calibration/so101-task-pad-overlay@1.0"
IMPORT_SCHEMA = "newton.calibration/newton-task-pad-import-verification@1.0"
COLLISION_PROFILE = "so101_task_planar_pads_v4"
MAX_HULL_VERTICES = 64

SOURCE_ROOT = Sdf.Path("/so101_new_calib")
FIXED_BODY = SOURCE_ROOT.AppendChild("gripper_link")
MOVING_BODY = SOURCE_ROOT.AppendChild("moving_jaw_so101_v1_link")
FIXED_PAD_SCOPE = FIXED_BODY.AppendChild("newton_collision_v4_fixed_pad")
MOVING_PAD_SCOPE = MOVING_BODY.AppendChild("newton_collision_v4_moving_pad")
FIXED_PAD_PATH = FIXED_PAD_SCOPE.AppendChild("part_000")
MOVING_PAD_PATH = MOVING_PAD_SCOPE.AppendChild("part_000")
MATERIAL_PATH = SOURCE_ROOT.AppendChild("newton_calibration_materials").AppendChild(
    "task_pad_v4"
)

V3_FIXED_SCOPE = FIXED_BODY.AppendChild("newton_collision_v3_fixed_follower")
V3_MOVING_SCOPE = MOVING_BODY.AppendChild("newton_collision_v3_moving_jaw")
V3_SERVO_SCOPE = FIXED_BODY.AppendChild("newton_collision_v3_fixed_servo")

SOURCE_TARGETS = {
    "fixed": (FIXED_BODY, "wrist_roll_follower_so101_v1"),
    "moving": (MOVING_BODY, "moving_jaw_so101_v1"),
}

# Geometry commissioned from the source USD and recorded in
# output/controller_commission/gripper_aperture_v2.json.  The q reference is
# where the source-mesh aperture model reaches the 18 mm peg diameter.
APERTURE_AT_ZERO_M = 0.015800002818840475
APERTURE_SLOPE_M_PER_RAD = 0.07727098939348247
PEG_DIAMETER_M = 0.018
CONTACT_REFERENCE_Q_RAD = (PEG_DIAMETER_M - APERTURE_AT_ZERO_M) / APERTURE_SLOPE_M_PER_RAD
CONTACT_CENTER_X_M = 0.00110
OPEN_Q_RAD = 0.30
OPEN_PATH_PEG_CENTER_X_M = 0.00155

# The common source-mesh pad envelope is approximately [-104.425, -90.025]
# mm in the gripper frame at q=0.  The moving finger is tapered/rounded, so a
# wider rectangular pad would leave that source envelope.  This common 5 mm
# band is the largest matched planar prism verified inside both source meshes
# while retaining the grasp aperture at the reference pose.
PAD_Z_RANGE_GRIPPER_M = (-0.1010, -0.0960)
PAD_Y_HALF_WIDTH_M = 0.0020
PAD_THICKNESS_M = 0.0020
PAD_SURFACE_INSET_M = 0.00010
SOURCE_ENVELOPE_TOLERANCE_M = 5.0e-6
MIN_COMMON_Z_BAND_M = 0.0049
MAX_PARALLEL_ERROR_DEG = 0.02
APERTURE_TOLERANCE_M = 0.00025
MIDPOINT_TOLERANCE_M = 0.00015
OPEN_PATH_MIN_CLEARANCE_M = 0.00025

# The gripper joint frame is authored in the canonical USD.  At q=0 the
# moving-jaw body transform is translation followed by Rx(+90 deg).  Positive
# q opens through a local Rz(q).  These values are re-read and checked against
# the source joint before any geometry is authored.
EXPECTED_JOINT_TRANSLATION_M = np.asarray((0.0202, 0.0188, -0.0234), dtype=np.float64)

PAD_MATERIAL = {
    "static_friction": 1.0,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
    "torsional_friction": 0.005,
    "rolling_friction": 0.0001,
}


@dataclass(frozen=True)
class SourceMesh:
    key: str
    body_path: Sdf.Path
    mesh_path: Sdf.Path
    vertices_body_m: np.ndarray
    faces: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def _register_newton_schema(explicit: Path | None) -> Path:
    candidates = _newton_schema_candidates(explicit)
    schema_path = next((path.resolve() for path in candidates if path.is_dir()), None)
    if schema_path is None:
        raise RuntimeError(
            "Newton USD schema plug-in not found; checked: "
            + ", ".join(str(path) for path in candidates)
        )
    Plug.Registry().RegisterPlugins(str(schema_path))
    probe_stage = Usd.Stage.CreateInMemory()
    probe = UsdGeom.Mesh.Define(probe_stage, "/Probe").GetPrim()
    if not probe.ApplyAPI("NewtonMeshCollisionAPI"):
        raise RuntimeError(f"NewtonMeshCollisionAPI could not be applied from {schema_path}")
    material = UsdShade.Material.Define(probe_stage, "/Material").GetPrim()
    if not material.ApplyAPI("NewtonMaterialAPI"):
        raise RuntimeError(f"NewtonMaterialAPI could not be applied from {schema_path}")
    return schema_path


def _triangles(mesh: UsdGeom.Mesh) -> np.ndarray:
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    result: list[tuple[int, int, int]] = []
    cursor = 0
    for count_value in counts:
        count = int(count_value)
        polygon = indices[cursor : cursor + count]
        cursor += count
        for offset in range(1, count - 1):
            result.append((int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1])))
    triangles = np.asarray(result, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError(f"Could not triangulate {mesh.GetPath()}")
    return triangles


def _points_in_body(prim: Usd.Prim, body: Usd.Prim) -> np.ndarray:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_to_body, _ = cache.ComputeRelativeTransform(prim, body)
    return np.asarray(
        [mesh_to_body.Transform(Gf.Vec3d(*point)) for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()],
        dtype=np.float64,
    )


def _collect_source_meshes(stage: Usd.Stage) -> dict[str, SourceMesh]:
    result: dict[str, SourceMesh] = {}
    for key, (body_path, token) in SOURCE_TARGETS.items():
        body = stage.GetPrimAtPath(body_path)
        if not body:
            raise RuntimeError(f"Missing source rigid body {body_path}")
        matches: list[Usd.Prim] = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
            if (
                prim.IsA(UsdGeom.Mesh)
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and token in str(prim.GetPath())
            ):
                matches.append(prim)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one source mesh for {key}, found {[str(p.GetPath()) for p in matches]}")
        prim = matches[0]
        vertices = _points_in_body(prim, body)
        topology = trimesh.Trimesh(
            vertices=vertices,
            faces=_triangles(UsdGeom.Mesh(prim)),
            process=True,
            validate=True,
        )
        if not topology.is_watertight or not topology.is_winding_consistent:
            raise RuntimeError(f"Source mesh is not a closed consistently wound envelope: {prim.GetPath()}")
        topology.fix_normals()
        result[key] = SourceMesh(
            key=key,
            body_path=body_path,
            mesh_path=prim.GetPath(),
            vertices_body_m=np.asarray(topology.vertices, dtype=np.float64),
            faces=np.asarray(topology.faces, dtype=np.int64),
        )
    return result


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def _moving_to_gripper_rotation(q_rad: float) -> np.ndarray:
    return _rotation_x(0.5 * math.pi) @ _rotation_z(q_rad)


def _body_to_gripper(points: np.ndarray, *, moving: bool, q_rad: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if not moving:
        return values.copy()
    rotation = _moving_to_gripper_rotation(q_rad)
    return values @ rotation.T + EXPECTED_JOINT_TRANSLATION_M


def _gripper_to_body(points: np.ndarray, *, moving: bool, q_rad: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if not moving:
        return values.copy()
    rotation = _moving_to_gripper_rotation(q_rad)
    return (values - EXPECTED_JOINT_TRANSLATION_M) @ rotation


def _validate_source_joint(stage: Usd.Stage) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(SOURCE_ROOT.AppendChild("joints").AppendChild("gripper"))
    if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError("Canonical source is missing the SO-101 gripper revolute joint")
    joint = UsdPhysics.RevoluteJoint(prim)
    axis = str(joint.GetAxisAttr().Get())
    translation = np.asarray(joint.GetLocalPos0Attr().Get(), dtype=np.float64)
    local_rot0_value = joint.GetLocalRot0Attr().Get()
    local_rot0 = np.asarray(
        (local_rot0_value.GetReal(), *local_rot0_value.GetImaginary()),
        dtype=np.float64,
    )
    body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
    body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
    expected_quaternion_wxyz = np.asarray((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))
    if axis != "Z" or not np.allclose(translation, EXPECTED_JOINT_TRANSLATION_M, atol=2.0e-6):
        raise RuntimeError(f"Gripper joint frame changed: axis={axis}, translation={translation.tolist()}")
    if not np.allclose(np.abs(local_rot0), np.abs(expected_quaternion_wxyz), atol=5.0e-6):
        raise RuntimeError(f"Gripper joint rotation changed: {local_rot0.tolist()}")
    if body0 != [str(FIXED_BODY)] or body1 != [str(MOVING_BODY)]:
        raise RuntimeError(f"Gripper joint bodies changed: body0={body0}, body1={body1}")
    return {
        "path": str(prim.GetPath()),
        "axis": axis,
        "local_pos0_m": translation.tolist(),
        "local_rot0_wxyz": local_rot0.tolist(),
        "body0": body0[0],
        "body1": body1[0],
        "contact_reference_q_rad": CONTACT_REFERENCE_Q_RAD,
    }


def _box_points(
    *,
    inner_x: float,
    outer_x: float,
    y_half_width: float,
    z_range: tuple[float, float],
) -> np.ndarray:
    return np.asarray(
        [
            (x, y, z)
            for x in (inner_x, outer_x)
            for y in (-y_half_width, y_half_width)
            for z in z_range
        ],
        dtype=np.float64,
    )


def _canonical_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    hull = trimesh.convex.convex_hull(np.asarray(points, dtype=np.float64))
    quantized = np.asarray(hull.vertices, dtype=np.float32).astype(np.float64)
    hull = trimesh.convex.convex_hull(quantized)
    vertices = np.asarray(hull.vertices, dtype=np.float64)
    faces = np.asarray(hull.faces, dtype=np.int64)
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order), dtype=np.int64)
    vertices = vertices[order]
    faces = remap[faces]
    normalized: list[tuple[int, int, int]] = []
    for face in faces:
        values = [int(value) for value in face]
        index = values.index(min(values))
        normalized.append(tuple(values[index:] + values[:index]))
    faces = np.asarray(sorted(normalized), dtype=np.int64)
    if len(vertices) > MAX_HULL_VERTICES or not hull.is_watertight or not hull.is_convex:
        raise RuntimeError("Task pad did not canonicalize to a valid bounded convex hull")
    return vertices, faces, float(abs(hull.volume))


def _grid_in_box(
    *,
    inner_x: float,
    outer_x: float,
    y_half_width: float,
    z_range: tuple[float, float],
    count: int = 7,
) -> np.ndarray:
    return np.asarray(
        [
            (x, y, z)
            for x in np.linspace(min(inner_x, outer_x), max(inner_x, outer_x), count)
            for y in np.linspace(-y_half_width, y_half_width, count)
            for z in np.linspace(z_range[0], z_range[1], count)
        ],
        dtype=np.float64,
    )


def _derive_pads(source: dict[str, SourceMesh]) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = source["fixed"]
    moving = source["moving"]

    fixed_candidates = fixed.vertices_body_m[
        (np.abs(fixed.vertices_body_m[:, 1]) <= 0.0055)
        & (fixed.vertices_body_m[:, 2] >= -0.1045)
        & (fixed.vertices_body_m[:, 2] <= -0.0900)
        & (fixed.vertices_body_m[:, 0] < 0.0)
    ]
    moving_candidates = moving.vertices_body_m[
        (moving.vertices_body_m[:, 1] >= -0.0821)
        & (moving.vertices_body_m[:, 1] <= -0.0660)
        & (moving.vertices_body_m[:, 2] >= 0.0130)
        & (moving.vertices_body_m[:, 2] <= 0.0246)
        & (moving.vertices_body_m[:, 0] < 0.0)
    ]
    if len(fixed_candidates) < 20 or len(moving_candidates) < 20:
        raise RuntimeError("Could not recover the commissioned source pad planes")
    fixed_surface_x = float(np.max(fixed_candidates[:, 0]))
    moving_surface_x_body = float(np.min(moving_candidates[:, 0]))
    if not np.isclose(fixed_surface_x, -0.0079, atol=2.0e-5):
        raise RuntimeError(f"Fixed source pad plane changed: {fixed_surface_x}")
    if not np.isclose(moving_surface_x_body, -0.0123, atol=2.0e-5):
        raise RuntimeError(f"Moving source pad plane changed: {moving_surface_x_body}")

    # Solve the moving source plane's gripper-X location at each corner of the
    # desired common Y/Z band.  A constant plane at the maximum X, plus a tiny
    # inset into material, is parallel to the fixed face and never protrudes
    # into the gap through this band.
    rotation = _moving_to_gripper_rotation(CONTACT_REFERENCE_Q_RAD)
    source_boundary_x: list[float] = []
    for y in (-PAD_Y_HALF_WIDTH_M, PAD_Y_HALF_WIDTH_M):
        for z in PAD_Z_RANGE_GRIPPER_M:
            remainder = (
                rotation[1, 0] * (y - EXPECTED_JOINT_TRANSLATION_M[1])
                + rotation[2, 0] * (z - EXPECTED_JOINT_TRANSLATION_M[2])
            )
            x = EXPECTED_JOINT_TRANSLATION_M[0] + (
                moving_surface_x_body - remainder
            ) / rotation[0, 0]
            source_boundary_x.append(float(x))

    fixed_inner_x = fixed_surface_x - PAD_SURFACE_INSET_M
    fixed_outer_x = fixed_inner_x - PAD_THICKNESS_M
    moving_inner_x = max(source_boundary_x) + PAD_SURFACE_INSET_M
    moving_outer_x = moving_inner_x + PAD_THICKNESS_M

    fixed_gripper_corners = _box_points(
        inner_x=fixed_inner_x,
        outer_x=fixed_outer_x,
        y_half_width=PAD_Y_HALF_WIDTH_M,
        z_range=PAD_Z_RANGE_GRIPPER_M,
    )
    moving_gripper_corners = _box_points(
        inner_x=moving_inner_x,
        outer_x=moving_outer_x,
        y_half_width=PAD_Y_HALF_WIDTH_M,
        z_range=PAD_Z_RANGE_GRIPPER_M,
    )
    fixed_body_corners = fixed_gripper_corners
    moving_body_corners = _gripper_to_body(
        moving_gripper_corners,
        moving=True,
        q_rad=CONTACT_REFERENCE_Q_RAD,
    )
    fixed_vertices, fixed_faces, fixed_volume = _canonical_hull(fixed_body_corners)
    moving_vertices, moving_faces, moving_volume = _canonical_hull(moving_body_corners)

    fixed_samples = _grid_in_box(
        inner_x=fixed_inner_x,
        outer_x=fixed_outer_x,
        y_half_width=PAD_Y_HALF_WIDTH_M,
        z_range=PAD_Z_RANGE_GRIPPER_M,
    )
    moving_samples_gripper = _grid_in_box(
        inner_x=moving_inner_x,
        outer_x=moving_outer_x,
        y_half_width=PAD_Y_HALF_WIDTH_M,
        z_range=PAD_Z_RANGE_GRIPPER_M,
    )
    moving_samples = _gripper_to_body(
        moving_samples_gripper,
        moving=True,
        q_rad=CONTACT_REFERENCE_Q_RAD,
    )
    pads = {
        "fixed": {
            "path": str(FIXED_PAD_PATH),
            "body_path": str(FIXED_BODY),
            "vertices": fixed_vertices,
            "faces": fixed_faces,
            "volume_m3": fixed_volume,
            "samples_body": fixed_samples,
            "corners_gripper_reference": fixed_gripper_corners,
            "inner_x_gripper_reference_m": fixed_inner_x,
            "outer_x_gripper_reference_m": fixed_outer_x,
            "source_inner_plane_body_x_m": fixed_surface_x,
        },
        "moving": {
            "path": str(MOVING_PAD_PATH),
            "body_path": str(MOVING_BODY),
            "vertices": moving_vertices,
            "faces": moving_faces,
            "volume_m3": moving_volume,
            "samples_body": moving_samples,
            "corners_gripper_reference": moving_gripper_corners,
            "inner_x_gripper_reference_m": moving_inner_x,
            "outer_x_gripper_reference_m": moving_outer_x,
            "source_inner_plane_body_x_m": moving_surface_x_body,
            "source_boundary_gripper_x_m": source_boundary_x,
        },
    }
    derivation = {
        "contact_reference_q_rad": CONTACT_REFERENCE_Q_RAD,
        "peg_diameter_m": PEG_DIAMETER_M,
        "target_contact_center_x_m": CONTACT_CENTER_X_M,
        "pad_z_range_gripper_m": list(PAD_Z_RANGE_GRIPPER_M),
        "pad_y_half_width_m": PAD_Y_HALF_WIDTH_M,
        "pad_thickness_m": PAD_THICKNESS_M,
        "pad_surface_inset_m": PAD_SURFACE_INSET_M,
    }
    return pads, derivation


def _source_envelope_validation(source: SourceMesh, pad: dict[str, Any]) -> dict[str, Any]:
    mesh = trimesh.Trimesh(
        vertices=source.vertices_body_m,
        faces=source.faces,
        process=True,
        validate=True,
    )
    mesh.fix_normals()
    samples = np.asarray(pad["samples_body"], dtype=np.float64)
    signed = np.asarray(trimesh.proximity.signed_distance(mesh, samples), dtype=np.float64)
    # trimesh uses positive distance inside a watertight mesh.
    maximum_protrusion = float(max(0.0, -float(np.min(signed))))
    outside_count = int(np.count_nonzero(signed < -SOURCE_ENVELOPE_TOLERANCE_M))
    if outside_count or maximum_protrusion > SOURCE_ENVELOPE_TOLERANCE_M:
        index = int(np.argmin(signed))
        raise RuntimeError(
            f"{source.key} pad leaves source envelope by {-signed[index] * 1000.0:.3f} mm "
            f"at {samples[index].tolist()}"
        )
    return {
        "source_mesh_path": str(source.mesh_path),
        "sample_count": int(len(samples)),
        "minimum_signed_interior_margin_m": float(np.min(signed)),
        "maximum_source_envelope_protrusion_m": maximum_protrusion,
        "allowed_protrusion_m": SOURCE_ENVELOPE_TOLERANCE_M,
        "outside_sample_count": outside_count,
        "passed": True,
        "method": "dense pad-volume samples against watertight source mesh signed distance",
    }


def _pad_geometry_validation(pads: dict[str, Any]) -> dict[str, Any]:
    fixed_inner = float(pads["fixed"]["inner_x_gripper_reference_m"])
    moving_inner = float(pads["moving"]["inner_x_gripper_reference_m"])
    aperture = moving_inner - fixed_inner
    midpoint = 0.5 * (moving_inner + fixed_inner)
    common_z = PAD_Z_RANGE_GRIPPER_M[1] - PAD_Z_RANGE_GRIPPER_M[0]
    fixed_normal = np.asarray((1.0, 0.0, 0.0))
    moving_normal = np.asarray((-1.0, 0.0, 0.0))
    parallel_error = math.degrees(math.acos(np.clip(-float(fixed_normal @ moving_normal), -1.0, 1.0)))
    checks = {
        "common_z_band": common_z >= MIN_COMMON_Z_BAND_M,
        "parallel_faces": parallel_error <= MAX_PARALLEL_ERROR_DEG,
        "aperture": abs(aperture - PEG_DIAMETER_M) <= APERTURE_TOLERANCE_M,
        "midpoint": abs(midpoint - CONTACT_CENTER_X_M) <= MIDPOINT_TOLERANCE_M,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Task-pad geometric gates failed: checks={checks}, aperture={aperture}, midpoint={midpoint}"
        )
    return {
        "reference_frame": "gripper_link",
        "reference_q_rad": CONTACT_REFERENCE_Q_RAD,
        "fixed_inner_plane_x_m": fixed_inner,
        "moving_inner_plane_x_m": moving_inner,
        "aperture_m": aperture,
        "target_aperture_m": PEG_DIAMETER_M,
        "aperture_error_m": aperture - PEG_DIAMETER_M,
        "aperture_tolerance_m": APERTURE_TOLERANCE_M,
        "midpoint_x_m": midpoint,
        "target_midpoint_x_m": CONTACT_CENTER_X_M,
        "midpoint_error_m": midpoint - CONTACT_CENTER_X_M,
        "midpoint_tolerance_m": MIDPOINT_TOLERANCE_M,
        "parallel_error_deg": parallel_error,
        "maximum_parallel_error_deg": MAX_PARALLEL_ERROR_DEG,
        "common_z_range_m": list(PAD_Z_RANGE_GRIPPER_M),
        "common_z_band_m": common_z,
        "minimum_common_z_band_m": MIN_COMMON_Z_BAND_M,
        "common_y_band_m": 2.0 * PAD_Y_HALF_WIDTH_M,
        "checks": checks,
        "passed": True,
    }


def _convex_signed_plane_distance(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True).convex_hull
    origins = np.asarray(hull.vertices)[np.asarray(hull.faces)[:, 0]]
    normals = np.asarray(hull.face_normals)
    offsets = np.einsum("ij,ij->i", normals, origins)
    # <=0 means inside every half-space.  The maximum is a conservative
    # outside distance when positive.
    return np.max(np.asarray(points) @ normals.T - offsets[None, :], axis=1)


def _find_competing_v3_parts(
    base_stage: Usd.Stage,
    pads: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    scopes = {"fixed": V3_FIXED_SCOPE, "moving": V3_MOVING_SCOPE}
    deactivated: dict[str, list[str]] = {"fixed": [], "moving": []}
    active_counts: dict[str, int] = {}
    for key, scope_path in scopes.items():
        scope = base_stage.GetPrimAtPath(scope_path)
        if not scope:
            raise RuntimeError(f"Base v3 asset is missing {scope_path}")
        pad_samples = np.asarray(pads[key]["samples_body"], dtype=np.float64)
        pad_mesh = trimesh.Trimesh(
            vertices=pads[key]["vertices"],
            faces=pads[key]["faces"],
            process=True,
            validate=True,
        ).convex_hull
        total = 0
        for prim in Usd.PrimRange(scope):
            if not prim.IsA(UsdGeom.Mesh) or not prim.IsActive():
                continue
            total += 1
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            faces = _triangles(UsdGeom.Mesh(prim))
            pad_in_part = _convex_signed_plane_distance(pad_samples, vertices, faces)
            part_in_pad = _convex_signed_plane_distance(
                vertices,
                np.asarray(pad_mesh.vertices),
                np.asarray(pad_mesh.faces),
            )
            # The dense volume samples catch pad volume owned by a v3 part;
            # the reciprocal vertex test catches a small v3 part swallowed by
            # the pad.  Boundary-only contact at <2 um is treated as overlap
            # to prevent two colliders from competing for the same contact.
            if np.any(pad_in_part <= 2.0e-6) or np.any(part_in_pad <= 2.0e-6):
                deactivated[key].append(str(prim.GetPath()))
        if total != 32:
            raise RuntimeError(f"Expected 32 active v3 {key} parts, found {total}")
        if not deactivated[key]:
            raise RuntimeError(f"No v3 {key} part overlaps the task pad; pad would be shadowed")
        active_counts[key] = total - len(deactivated[key])
    return deactivated, active_counts


def _cylinder_surface_samples(
    *,
    center_x: float,
    center_z_values: np.ndarray,
    radius: float,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    points: list[tuple[float, float, float]] = []
    for center_z in center_z_values:
        for z_offset in np.linspace(-0.035, 0.035, 15):
            for angle in angles:
                points.append(
                    (
                        center_x + radius * math.cos(float(angle)),
                        radius * math.sin(float(angle)),
                        float(center_z + z_offset),
                    )
                )
    return np.asarray(points, dtype=np.float64)


def _open_path_validation(pads: dict[str, Any]) -> dict[str, Any]:
    # The new geometry-centered grasp puts the settled peg center near pad
    # midpoint Z=-98.5 mm; the approach sweeps it farther down in the gripper
    # frame.  Validate both pad hulls across that full relative path at q_open.
    pad_center_z = 0.5 * sum(PAD_Z_RANGE_GRIPPER_M)
    center_z_values = np.linspace(pad_center_z - 0.043, pad_center_z, 19)
    samples = _cylinder_surface_samples(
        center_x=OPEN_PATH_PEG_CENTER_X_M,
        center_z_values=center_z_values,
        radius=0.5 * PEG_DIAMETER_M,
    )
    results: dict[str, Any] = {}
    minimum = math.inf
    for key, moving in (("fixed", False), ("moving", True)):
        vertices_gripper = _body_to_gripper(
            pads[key]["vertices"],
            moving=moving,
            q_rad=OPEN_Q_RAD,
        )
        clearance = _convex_signed_plane_distance(
            samples,
            vertices_gripper,
            pads[key]["faces"],
        )
        index = int(np.argmin(clearance))
        value = float(clearance[index])
        minimum = min(minimum, value)
        results[key] = {
            "minimum_conservative_clearance_m": value,
            "sample_gripper_m": samples[index].tolist(),
        }
    if minimum < OPEN_PATH_MIN_CLEARANCE_M:
        raise RuntimeError(
            f"Open-path pad clearance is {minimum * 1000.0:.3f} mm; "
            f"required >= {OPEN_PATH_MIN_CLEARANCE_M * 1000.0:.3f} mm"
        )
    return {
        "reference_frame": "gripper_link",
        "open_q_rad": OPEN_Q_RAD,
        "peg_center_x_m": OPEN_PATH_PEG_CENTER_X_M,
        "peg_diameter_m": PEG_DIAMETER_M,
        "peg_center_z_range_m": [float(center_z_values[0]), float(center_z_values[-1])],
        "sample_count": int(len(samples)),
        "minimum_required_clearance_m": OPEN_PATH_MIN_CLEARANCE_M,
        "minimum_clearance_m": minimum,
        "per_pad": results,
        "passed": True,
        "scope": "task-pad shapes only; the authoritative dynamic run must still qualify the full gripper",
    }


def _write_material(stage: Usd.Stage) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, MATERIAL_PATH)
    prim = material.GetPrim()
    standard = UsdPhysics.MaterialAPI.Apply(prim)
    standard.CreateStaticFrictionAttr().Set(PAD_MATERIAL["static_friction"])
    standard.CreateDynamicFrictionAttr().Set(PAD_MATERIAL["dynamic_friction"])
    standard.CreateRestitutionAttr().Set(PAD_MATERIAL["restitution"])
    if not prim.ApplyAPI("NewtonMaterialAPI"):
        raise RuntimeError("Could not apply NewtonMaterialAPI to task pad material")
    prim.GetAttribute("newton:torsionalFriction").Set(PAD_MATERIAL["torsional_friction"])
    prim.GetAttribute("newton:rollingFriction").Set(PAD_MATERIAL["rolling_friction"])
    return material


def _write_pad(
    stage: Usd.Stage,
    *,
    scope_path: Sdf.Path,
    path: Sdf.Path,
    pad: dict[str, Any],
    material: UsdShade.Material,
) -> None:
    scope = UsdGeom.Scope.Define(stage, scope_path)
    scope.GetPrim().SetCustomDataByKey("newtonCalibration:generatorSchema", SCHEMA)
    scope.GetPrim().SetCustomDataByKey("newtonCalibration:role", "taskGraspPad")
    mesh = UsdGeom.Mesh.Define(stage, path)
    vertices = np.asarray(pad["vertices"], dtype=np.float64)
    faces = np.asarray(pad["faces"], dtype=np.int64)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*point) for point in vertices])
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(faces.reshape(-1).astype(int).tolist())
    mesh.CreateDoubleSidedAttr().Set(False)
    mesh.CreateExtentAttr().Set(
        [Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))]
    )
    mesh.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
        UsdPhysics.Tokens.convexHull
    )
    if not mesh.GetPrim().ApplyAPI("NewtonMeshCollisionAPI"):
        raise RuntimeError(f"Could not apply NewtonMeshCollisionAPI to {path}")
    mesh.GetPrim().GetAttribute("newton:maxHullVertices").Set(MAX_HULL_VERTICES)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )


def _physics_signature(stage: Usd.Stage) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName().endswith("Joint") or prim.HasAPI(UsdPhysics.MassAPI):
            result[str(prim.GetPath())] = {
                attribute.GetName(): str(attribute.Get())
                for attribute in prim.GetAttributes()
                if attribute.GetName().startswith(("physics:", "drive:", "limit:"))
            }
    return result


def _verify_composed_asset(
    base_stage: Usd.Stage,
    derived_stage: Usd.Stage,
    deactivated: dict[str, list[str]],
    active_v3_counts: dict[str, int],
) -> dict[str, Any]:
    if _physics_signature(base_stage) != _physics_signature(derived_stage):
        raise RuntimeError("V4 overlay changed authored joint or mass properties")
    for paths in deactivated.values():
        for path in paths:
            prim = derived_stage.GetPrimAtPath(path)
            if prim and prim.IsActive():
                raise RuntimeError(f"Competing v3 part remained active: {path}")
    actual_active: dict[str, int] = {"fixed": 0, "moving": 0, "servo": 0}
    for key, scope_path in {
        "fixed": V3_FIXED_SCOPE,
        "moving": V3_MOVING_SCOPE,
        "servo": V3_SERVO_SCOPE,
    }.items():
        scope = derived_stage.GetPrimAtPath(scope_path)
        if scope:
            actual_active[key] = sum(
                1 for prim in Usd.PrimRange(scope) if prim.IsA(UsdGeom.Mesh) and prim.IsActive()
            )
    expected = {**active_v3_counts, "servo": 1}
    if actual_active != expected:
        raise RuntimeError(f"Unexpected active v3 part counts: actual={actual_active}, expected={expected}")
    for path, body_path in ((FIXED_PAD_PATH, FIXED_BODY), (MOVING_PAD_PATH, MOVING_BODY)):
        prim = derived_stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Missing active task pad {path}")
        candidate = prim.GetParent()
        while candidate and not candidate.HasAPI(UsdPhysics.RigidBodyAPI):
            candidate = candidate.GetParent()
        if not candidate or candidate.GetPath() != body_path:
            raise RuntimeError(f"Task pad {path} is not attached to {body_path}")
        material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if not material or material.GetPath() != MATERIAL_PATH:
            raise RuntimeError(f"Task pad {path} does not resolve explicit material {MATERIAL_PATH}")
        if not relationship or not relationship.IsValid():
            raise RuntimeError(f"Task pad {path} has no valid physics material binding")
    return {
        "active_v3_part_counts": actual_active,
        "deactivated_competing_v3_parts": deactivated,
        "pad_part_counts": {"fixed": 1, "moving": 1},
        "joint_and_mass_properties_unchanged": True,
        "explicit_pad_material_path": str(MATERIAL_PATH),
        "passed": True,
    }


def _verify_newton_import(asset: Path, expected: dict[str, Any]) -> dict[str, Any]:
    import newton

    builder = newton.ModelBuilder()
    imported = builder.add_usd(str(asset))
    path_shape_map = {str(path): int(index) for path, index in imported["path_shape_map"].items()}
    path_body_map = {str(path): int(index) for path, index in imported["path_body_map"].items()}
    for path in expected["deactivated_paths"]:
        if path in path_shape_map:
            raise RuntimeError(f"Newton imported deactivated competing v3 part {path}")
    pad_records: list[dict[str, Any]] = []
    for path, body_path in ((str(FIXED_PAD_PATH), str(FIXED_BODY)), (str(MOVING_PAD_PATH), str(MOVING_BODY))):
        if path not in path_shape_map:
            raise RuntimeError(f"Newton did not import task pad {path}")
        shape_index = path_shape_map[path]
        if builder.shape_type[shape_index] != newton.GeoType.CONVEX_MESH:
            raise RuntimeError(f"Newton did not import {path} as CONVEX_MESH")
        if int(builder.shape_body[shape_index]) != path_body_map[body_path]:
            raise RuntimeError(f"Newton attached {path} to the wrong rigid body")
        source = builder.shape_source[shape_index]
        material = {
            "mu": float(builder.shape_material_mu[shape_index]),
            "mu_rolling": float(builder.shape_material_mu_rolling[shape_index]),
            "mu_torsional": float(builder.shape_material_mu_torsional[shape_index]),
            "restitution": float(builder.shape_material_restitution[shape_index]),
        }
        expected_material = {
            "mu": PAD_MATERIAL["static_friction"],
            "mu_rolling": PAD_MATERIAL["rolling_friction"],
            "mu_torsional": PAD_MATERIAL["torsional_friction"],
            "restitution": PAD_MATERIAL["restitution"],
        }
        if any(not math.isclose(material[key], value, abs_tol=1.0e-8) for key, value in expected_material.items()):
            raise RuntimeError(
                f"Newton material mismatch on {path}: actual={material}, expected={expected_material}"
            )
        pad_records.append(
            {
                "path": path,
                "body_path": body_path,
                "shape_type": "CONVEX_MESH",
                "vertex_count": int(len(source.vertices)),
                "max_hull_vertices": int(source.maxhullvert),
                "material": material,
            }
        )
    active_v3_counts = {"fixed": 0, "moving": 0, "servo": 0}
    for path in path_shape_map:
        if f"{V3_FIXED_SCOPE}/part_" in path:
            active_v3_counts["fixed"] += 1
        elif f"{V3_MOVING_SCOPE}/part_" in path:
            active_v3_counts["moving"] += 1
        elif f"{V3_SERVO_SCOPE}/part_" in path:
            active_v3_counts["servo"] += 1
    if active_v3_counts != expected["active_v3_counts"]:
        raise RuntimeError(
            f"Newton active v3 counts differ: {active_v3_counts} != {expected['active_v3_counts']}"
        )
    return {
        "schema": IMPORT_SCHEMA,
        "runtime_importer": "newton.ModelBuilder.add_usd",
        "total_shape_count": int(builder.shape_count),
        "active_v3_part_counts": active_v3_counts,
        "pad_part_counts": {"fixed": 1, "moving": 1},
        "deactivated_parts_absent": True,
        "explicit_material_matches": True,
        "pads": pad_records,
        "passed": True,
    }


def _verify_newton_import_subprocess(
    asset: Path,
    output: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--verify-newton-asset",
            str(asset),
            "--verify-output",
            str(output),
            "--expected-json",
            json.dumps(expected, sort_keys=True),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Fresh-process Newton import verification failed ({completed.returncode})\n"
            f"STDOUT:\n{completed.stdout[-8000:]}\nSTDERR:\n{completed.stderr[-8000:]}"
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("schema") != IMPORT_SCHEMA or not result.get("passed"):
        raise RuntimeError(f"Invalid Newton verification result: {result}")
    return result


def _pad_manifest_record(pad: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(pad["vertices"], dtype=np.float32)
    faces = np.asarray(pad["faces"], dtype=np.int32)
    digest = hashlib.sha256()
    digest.update(vertices.tobytes())
    digest.update(faces.tobytes())
    return {
        "path": pad["path"],
        "body_path": pad["body_path"],
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "volume_m3": float(pad["volume_m3"]),
        "geometry_sha256": digest.hexdigest(),
        "inner_x_gripper_reference_m": float(pad["inner_x_gripper_reference_m"]),
        "outer_x_gripper_reference_m": float(pad["outer_x_gripper_reference_m"]),
        "source_inner_plane_body_x_m": float(pad["source_inner_plane_body_x_m"]),
        "source_envelope_validation": envelope,
    }


def build(
    *,
    source: Path,
    base_v3: Path,
    base_manifest: Path,
    output: Path,
    manifest_path: Path,
    newton_schema_path: Path | None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    base_v3 = base_v3.expanduser().resolve()
    base_manifest = base_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    for path in (source, base_v3, base_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing v4 asset or manifest")
    schema_path = _register_newton_schema(newton_schema_path)
    source_hash = _sha256(source)
    base_hash = _sha256(base_v3)
    base_record = json.loads(base_manifest.read_text(encoding="utf-8"))
    if base_record.get("source_usd_sha256") != source_hash:
        raise RuntimeError("Canonical source hash does not match the v3 manifest")
    if base_record.get("derived_usd_sha256") != base_hash:
        raise RuntimeError("Base v3 USD hash does not match its manifest")

    source_stage = Usd.Stage.Open(str(source), Usd.Stage.LoadAll)
    base_stage = Usd.Stage.Open(str(base_v3), Usd.Stage.LoadAll)
    if source_stage is None or base_stage is None:
        raise RuntimeError("Could not open source or base-v3 USD")
    joint_validation = _validate_source_joint(source_stage)
    source_meshes = _collect_source_meshes(source_stage)
    pads, derivation = _derive_pads(source_meshes)
    envelope = {
        key: _source_envelope_validation(source_meshes[key], pads[key])
        for key in ("fixed", "moving")
    }
    geometry = _pad_geometry_validation(pads)
    open_path = _open_path_validation(pads)
    deactivated, active_v3_counts = _find_competing_v3_parts(base_stage, pads)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.stem}.tmp-{os.getpid()}{output.suffix}")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.stem}.tmp-{os.getpid()}.json")
    temporary_import = manifest_path.with_name(f".{manifest_path.stem}.import-{os.getpid()}.json")
    published_output = False
    try:
        layer = Sdf.Layer.CreateNew(str(temporary_output))
        layer.subLayerPaths = [os.path.relpath(base_v3, output.parent)]
        stage = Usd.Stage.Open(layer, Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError("Could not compose v4 overlay")
        stage.SetDefaultPrim(stage.GetPrimAtPath(base_stage.GetDefaultPrim().GetPath()))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(base_stage))
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(base_stage))
        for paths in deactivated.values():
            for path in paths:
                stage.OverridePrim(path).SetActive(False)
        material = _write_material(stage)
        _write_pad(
            stage,
            scope_path=FIXED_PAD_SCOPE,
            path=FIXED_PAD_PATH,
            pad=pads["fixed"],
            material=material,
        )
        _write_pad(
            stage,
            scope_path=MOVING_PAD_SCOPE,
            path=MOVING_PAD_PATH,
            pad=pads["moving"],
            material=material,
        )
        default_prim = stage.GetDefaultPrim()
        default_prim.SetCustomDataByKey("newtonCalibration:collisionProfile", COLLISION_PROFILE)
        default_prim.SetCustomDataByKey("newtonCalibration:generatorSchema", SCHEMA)
        default_prim.SetCustomDataByKey("newtonCalibration:sourceSha256", source_hash)
        default_prim.SetCustomDataByKey("newtonCalibration:baseV3Sha256", base_hash)
        stage.GetRootLayer().Save()

        derived_stage = Usd.Stage.Open(str(temporary_output), Usd.Stage.LoadAll)
        if derived_stage is None:
            raise RuntimeError("Generated v4 USD could not be reopened")
        composed = _verify_composed_asset(
            base_stage,
            derived_stage,
            deactivated,
            active_v3_counts,
        )
        expected_import = {
            "deactivated_paths": sorted(path for paths in deactivated.values() for path in paths),
            "active_v3_counts": {**active_v3_counts, "servo": 1},
        }
        imported = _verify_newton_import_subprocess(
            temporary_output,
            temporary_import,
            expected_import,
        )
        temporary_import.unlink(missing_ok=True)
        output_hash = _sha256(temporary_output)
        manifest = {
            "schema": SCHEMA,
            "collision_profile": COLLISION_PROFILE,
            "source_usd": str(source),
            "source_usd_sha256": source_hash,
            "base_v3_usd": str(base_v3),
            "base_v3_usd_sha256": base_hash,
            "base_v3_manifest": str(base_manifest),
            "derived_usd": str(output),
            "derived_usd_sha256": output_hash,
            "registered_newton_schema_path": str(schema_path),
            "task_scope": "SO-101 18 mm cylindrical peg grasp for insertion commissioning",
            "joint_frame_validation": joint_validation,
            "pad_derivation": derivation,
            "explicit_pad_material": {"path": str(MATERIAL_PATH), **PAD_MATERIAL},
            "pads": {
                key: _pad_manifest_record(pads[key], envelope[key])
                for key in ("fixed", "moving")
            },
            "geometric_validation": geometry,
            "open_path_clearance_validation": open_path,
            "v3_housing_preservation": {
                "deactivated_competing_parts": deactivated,
                "active_v3_part_counts": {**active_v3_counts, "servo": 1},
                "selection_method": (
                    "deactivate only v3 convex parts intersecting dense inscribed pad-volume samples "
                    "or having vertices inside the pad hull"
                ),
            },
            "composed_asset_verification": composed,
            "newton_runtime_import_verification": imported,
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_output, output)
        published_output = True
        os.replace(temporary_manifest, manifest_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        temporary_import.unlink(missing_ok=True)
        if published_output and not manifest_path.exists():
            output.unlink(missing_ok=True)
        raise
    print(f"RESULT={output}")
    print(f"MANIFEST={manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--base-v3", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--newton-schema-path", type=Path)
    parser.add_argument("--verify-newton-asset", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--verify-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_newton_asset is not None:
        if args.verify_output is None or args.expected_json is None:
            parser.error("internal Newton verification requires --verify-output and --expected-json")
        result = _verify_newton_import(args.verify_newton_asset, json.loads(args.expected_json))
        with args.verify_output.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    required = (args.source, args.base_v3, args.base_manifest, args.output, args.manifest)
    if any(value is None for value in required):
        parser.error("--source, --base-v3, --base-manifest, --output and --manifest are required")
    build(
        source=args.source,
        base_v3=args.base_v3,
        base_manifest=args.base_manifest,
        output=args.output,
        manifest_path=args.manifest,
        newton_schema_path=args.newton_schema_path,
    )


if __name__ == "__main__":
    main()
