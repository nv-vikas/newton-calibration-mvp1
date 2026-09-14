"""Export a compact SO-101 joint/link description from the source USD."""

from __future__ import annotations

import argparse
import json

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _vec(value) -> list[float]:
    return [float(component) for component in value]


def _quat(value) -> list[float]:
    real = float(value.GetReal())
    imag = value.GetImaginary()
    return [real, float(imag[0]), float(imag[1]), float(imag[2])]


def _matrix(value: Gf.Matrix4d) -> list[list[float]]:
    return [[float(value[row][column]) for column in range(4)] for row in range(4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.asset)
    if stage is None:
        raise RuntimeError(f"Could not open {args.asset}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    joints = []
    links = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            item = {
                "name": prim.GetName(),
                "path": str(prim.GetPath()),
                "body0": str(body0[0]) if body0 else None,
                "body1": str(body1[0]) if body1 else None,
                "axis": str(joint.GetAxisAttr().Get()),
                "local_pos0": _vec(joint.GetLocalPos0Attr().Get()),
                "local_pos1": _vec(joint.GetLocalPos1Attr().Get()),
                "local_rot0": _quat(joint.GetLocalRot0Attr().Get()),
                "local_rot1": _quat(joint.GetLocalRot1Attr().Get()),
                "lower": float(joint.GetLowerLimitAttr().Get()),
                "upper": float(joint.GetUpperLimitAttr().Get()),
            }
            joints.append(item)
            for body in body0 + body1:
                body_prim = stage.GetPrimAtPath(body)
                if body_prim:
                    links[str(body)] = _matrix(cache.GetLocalToWorldTransform(body_prim))
    print(json.dumps({"up_axis": UsdGeom.GetStageUpAxis(stage), "joints": joints, "links": links}, indent=2))


if __name__ == "__main__":
    main()
