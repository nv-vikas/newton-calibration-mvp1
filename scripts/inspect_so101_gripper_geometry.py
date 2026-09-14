#!/usr/bin/env python3
"""Print geometry bounds under the fixed and moving SO-101 gripper bodies."""

from __future__ import annotations

import argparse
import json

from pxr import Usd, UsdGeom


def _vector(value) -> list[float]:
    return [float(component) for component in value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.asset)
    if stage is None:
        raise RuntimeError(f"Could not open {args.asset}")
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    body_names = {"gripper_link", "moving_jaw_so101_v1_link"}
    result: dict[str, list[dict[str, object]]] = {}
    for body in stage.Traverse():
        if body.GetName() not in body_names:
            continue
        entries: list[dict[str, object]] = []
        body_range = cache.ComputeWorldBound(body).ComputeAlignedRange()
        entries.append(
            {
                "path": str(body.GetPath()),
                "type": body.GetTypeName(),
                "purpose": "aggregate",
                "min_w_m": _vector(body_range.GetMin()),
                "max_w_m": _vector(body_range.GetMax()),
                "center_w_m": _vector(body_range.GetMidpoint()),
                "size_m": _vector(body_range.GetSize()),
            }
        )
        for prim in Usd.PrimRange(
            body,
            Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate),
        ):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            relative = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
            entries.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "purpose": str(UsdGeom.Imageable(prim).GetPurposeAttr().Get()),
                    "min_w_m": _vector(aligned.GetMin()),
                    "max_w_m": _vector(aligned.GetMax()),
                    "center_w_m": _vector(aligned.GetMidpoint()),
                    "size_m": _vector(aligned.GetSize()),
                    "min_body_m": _vector(relative.GetMin()),
                    "max_body_m": _vector(relative.GetMax()),
                    "center_body_m": _vector(relative.GetMidpoint()),
                    "size_body_m": _vector(relative.GetSize()),
                    "physics_attributes": {
                        attribute.GetName(): str(attribute.Get())
                        for attribute in prim.GetAttributes()
                        if any(
                            token in attribute.GetName().lower()
                            for token in ("approximation", "collision", "contact")
                        )
                    },
                }
            )
        result[str(body.GetPath())] = entries
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
