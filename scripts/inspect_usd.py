"""Print physics-relevant joint attributes from a USD asset."""

from __future__ import annotations

import argparse

from pxr import Usd, UsdPhysics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset")
    args = parser.parse_args()
    stage = Usd.Stage.Open(args.asset)
    if stage is None:
        raise RuntimeError(f"Could not open {args.asset}")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        print(f"\n{prim.GetPath()} ({prim.GetTypeName()})")
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if any(
                token in name.lower()
                for token in (
                    "armature",
                    "damping",
                    "friction",
                    "stiffness",
                    "maxforce",
                    "lowerlimit",
                    "upperlimit",
                    "axis",
                )
            ):
                print(f"  {name}: {attribute.Get()}")


if __name__ == "__main__":
    main()
