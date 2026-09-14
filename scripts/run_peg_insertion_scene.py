#!/usr/bin/env python3
"""Run and verify the SO-101 peg-insertion scene in Isaac Lab + Newton."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if os.environ.get("ACCEPT_EULA") != "Y":
    raise RuntimeError("Review the NVIDIA Isaac Sim license and export ACCEPT_EULA=Y before running.")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", required=True, help="SO-101 USD path")
parser.add_argument("--output", default="/workspace/output/peg_insertion_scene")
parser.add_argument("--steps", type=int, default=480, help="Physics steps per contact probe")
parser.add_argument("--probe", choices=("layout", "centered", "offset", "both"), default="both")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import importlib.metadata

from newton_calibration.isaaclab.tasks.so101_peg_insertion import PegInsertionMode
from newton_calibration.isaaclab.tasks.so101_peg_insertion.scene import SO101PegInsertionScene


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_probe(scene: SO101PegInsertionScene, name: str, steps: int) -> dict[str, object]:
    scene.reset(probe=name)
    initial = scene.snapshot()
    for _ in range(steps):
        scene.step(render=False, hold_robot=True)
    final_metrics = scene.task_metrics(settled=True)
    final = scene.snapshot()
    final["metrics"]["jammed"] = final_metrics.jammed
    final["metrics"]["seated"] = final_metrics.seated
    if name == "centered":
        expectation = "peg enters to the target depth through the declared aperture"
        passed = (
            final_metrics.insertion_depth_m >= scene.spec.target_insertion_depth_m
            and final_metrics.lateral_offset_m <= scene.spec.radial_clearance_m
        )
    else:
        expectation = "peg outside the aperture is stopped at the socket rim"
        passed = final_metrics.jammed
    return {
        "name": name,
        "steps": steps,
        "expectation": expectation,
        "passed": passed,
        "initial": initial,
        "final": final,
    }


def main() -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scene = SO101PegInsertionScene(
        usd_path=args.asset,
        device=args.device,
        mode=PegInsertionMode.INSERTION,
    )
    try:
        result: dict[str, object] = {
            "runtime": {
                "isaaclab_release": os.environ.get("ISAACLAB_RELEASE", "3.0.0-beta2"),
                "isaaclab_core_extension": package_version("isaaclab"),
                "isaaclab_newton_extension": package_version("isaaclab_newton"),
                "newton": package_version("newton"),
                "device": args.device,
                "physics": "Newton MJWarp",
            },
            "scene": scene.snapshot(),
            "probes": [],
        }
        if args.probe in {"centered", "both"}:
            result["probes"].append(run_probe(scene, "centered", args.steps))
        if args.probe in {"offset", "both"}:
            result["probes"].append(run_probe(scene, "offset", args.steps))
        result["conformance_passed"] = all(probe["passed"] for probe in result["probes"])
        result_path = output / "scene_result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"RESULT={result_path}", flush=True)
        if not result["conformance_passed"]:
            raise RuntimeError("One or more peg/socket conformance probes failed")
    finally:
        scene.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
