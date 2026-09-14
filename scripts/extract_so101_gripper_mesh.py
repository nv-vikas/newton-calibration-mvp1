#!/usr/bin/env python3
"""Extract SO-101 fixed and moving jaw mesh vertices in their body frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom


def _find_named_prim(stage: Usd.Stage, name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one USD prim named {name!r}; found {len(matches)}")
    return matches[0]


def _mesh_points_in_body(
    stage: Usd.Stage,
    *,
    body_name: str,
    mesh_path_token: str,
    branch: str,
) -> np.ndarray:
    body = _find_named_prim(stage, body_name)
    matches = [
        prim
        for prim in Usd.PrimRange(
            body,
            Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate),
        )
        if (
            prim.IsA(UsdGeom.Mesh)
            and f"/{branch}/" in str(prim.GetPath())
            and mesh_path_token in str(prim.GetPath())
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one mesh containing {mesh_path_token!r}; found {len(matches)}"
        )
    mesh_prim = matches[0]
    points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_to_body, _ = cache.ComputeRelativeTransform(mesh_prim, body)
    result = np.asarray(
        [mesh_to_body.Transform(Gf.Vec3d(*point)) for point in points],
        dtype=np.float64,
    )
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid mesh points for {mesh_prim.GetPath()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stage = Usd.Stage.Open(args.asset)
    if stage is None:
        raise RuntimeError(f"Could not open {args.asset}")
    fixed = _mesh_points_in_body(
        stage,
        body_name="gripper_link",
        mesh_path_token="wrist_roll_follower_so101_v1",
        branch="visuals",
    )
    moving = _mesh_points_in_body(
        stage,
        body_name="moving_jaw_so101_v1_link",
        mesh_path_token="moving_jaw_so101_v1",
        branch="visuals",
    )
    fixed_collision = _mesh_points_in_body(
        stage,
        body_name="gripper_link",
        mesh_path_token="wrist_roll_follower_so101_v1",
        branch="collisions",
    )
    fixed_servo_collision = _mesh_points_in_body(
        stage,
        body_name="gripper_link",
        mesh_path_token="sts3215_03a_v1",
        branch="collisions",
    )
    moving_collision = _mesh_points_in_body(
        stage,
        body_name="moving_jaw_so101_v1_link",
        mesh_path_token="moving_jaw_so101_v1",
        branch="collisions",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fixed_body_points=fixed,
        moving_body_points=moving,
        fixed_collision_body_points=fixed_collision,
        fixed_follower_collision_body_points=fixed_collision,
        fixed_servo_collision_body_points=fixed_servo_collision,
        moving_collision_body_points=moving_collision,
        moving_jaw_collision_body_points=moving_collision,
    )
    print(
        f"RESULT={output} visual_vertices={fixed.shape[0]}/{moving.shape[0]} "
        "collision_vertices="
        f"{fixed_collision.shape[0]}/{fixed_servo_collision.shape[0]}/"
        f"{moving_collision.shape[0]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
