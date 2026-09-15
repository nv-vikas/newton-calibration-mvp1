from __future__ import annotations

import argparse
import json
from pathlib import Path

from newton_calibration.adapters.evidence import fetch_anchor_lab_so101
from newton_calibration.adapters.surface import SO101EnvCfg
from newton_calibration.collection import CalibrationRequest, MotionSpec
from newton_calibration.core.models import EnvironmentSpec, jsonable
from newton_calibration.isaaclab import tuning
from newton_calibration.optimizers import list_optimizers


def main() -> None:
    parser = argparse.ArgumentParser(prog="newton-calibration", description="Newton calibration MVP1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch", help="download SO-101 Anchor-Lab evidence and USD")
    fetch_parser.add_argument("--output", default="data/anchor-lab")
    fetch_parser.add_argument("--revision", default="647edd5787cd764cdc041103ad282dc59214d919")
    subparsers.add_parser("optimizers", help="list installed optimizer plug-ins and versions")
    assist_parser = subparsers.add_parser(
        "assist", help="asset-only analyze → collection plan, with video enabled by default"
    )
    assist_parser.add_argument(
        "--config", required=True, help="JSON containing environment and collection (MotionSpec) objects"
    )
    assist_parser.add_argument(
        "--preview-factory", help="Trusted local module:factory returning a bound Isaac Lab/Newton preview adapter"
    )
    assist_parser.add_argument(
        "--design-probe-factory",
        help="Trusted local module:factory returning a Newton sensitivity probe bound to the same scene",
    )
    assist_parser.add_argument("--workdir", default="runs")
    assist_parser.add_argument("--no-preview", action="store_true", help="Explicitly skip the default preview request")

    inspect_parser = subparsers.add_parser("inspect", help="execute analyze + plan without starting physics")
    _add_job_arguments(inspect_parser, include_fit=False)

    run_parser = subparsers.add_parser("run", help="execute analyze → plan → fit → validate → write")
    _add_job_arguments(run_parser, include_fit=True)

    args = parser.parse_args()
    if args.command == "fetch":
        print(json.dumps(fetch_anchor_lab_so101(args.output, args.revision), indent=2))
        return
    if args.command == "optimizers":
        print(json.dumps(list_optimizers(), indent=2, sort_keys=True))
        return
    if args.command == "assist":
        import importlib

        config = json.loads(Path(args.config).read_text())
        preview = None
        if args.preview_factory:
            module, separator, name = args.preview_factory.partition(":")
            if not separator:
                parser.error("--preview-factory must be a trusted local module:factory")
            preview = getattr(importlib.import_module(module), name)()
        design_probe = getattr(preview, "design_probe", None)
        if args.design_probe_factory:
            module, separator, name = args.design_probe_factory.partition(":")
            if not separator:
                parser.error("--design-probe-factory must be a trusted local module:factory")
            design_probe = getattr(importlib.import_module(module), name)()
        result = tuning.assist(
            env=EnvironmentSpec(**config["environment"]),
            request=CalibrationRequest(**config.get("request", {})),
            collection=MotionSpec(**config["collection"]) if config.get("collection") else None,
            preview=preview,
            design_probe=design_probe,
            video=not args.no_preview,
            workdir=args.workdir,
        )
        print(json.dumps(jsonable(result), indent=2))
        if (
            result.status
            in {
                "preview_failed",
                "preview_pending",
                "needs_scene_setup",
                "generation_failed",
                "evidence_action_required",
                "design_failed",
            }
            or result.design.get("adaptive_search", {}).get("status") == "needs_dynamics_probe"
        ):
            raise SystemExit(2)  # files may exist, but requested design/preview is not complete
        return
    env = SO101EnvCfg(
        usd_path=str(Path(args.asset).expanduser().resolve()),
        runtime=args.runtime,
        device=args.device,
        residual_model_path=args.residual_model,
    )
    analysis = tuning.analyze(
        env=env,
        evidence=args.evidence,
        evidence_revision=args.revision,
        workdir=args.workdir,
    )
    calibration_plan = tuning.plan(
        analysis,
        optimizer=args.optimizer,
        optimizer_options=args.optimizer_options_json,
    )
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "analysis": jsonable(analysis),
                    "plan": jsonable(calibration_plan),
                },
                indent=2,
            )
        )
        return
    fit_run = tuning.fit(
        calibration_plan,
        generations=args.generations,
        population=args.population,
        resume=not args.no_resume,
    )
    validation = tuning.validate(fit_run)
    package = tuning.write(validation, output=args.output)
    print(json.dumps(jsonable(package), indent=2))


def _add_job_arguments(parser: argparse.ArgumentParser, *, include_fit: bool) -> None:
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--workdir", default="runs")
    parser.add_argument("--revision", default="local")
    parser.add_argument("--runtime", choices=["isaaclab_newton", "analytic"], default="isaaclab_newton")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--residual-model")
    parser.add_argument(
        "--optimizer",
        help="registered optimizer name; defaults to the recipe optimizer",
    )
    parser.add_argument(
        "--optimizer-options-json",
        type=_json_object,
        metavar="JSON",
        help="optimizer-specific JSON object locked into the calibration plan",
    )
    if include_fit:
        parser.add_argument("--output", default="packages/so101")
        parser.add_argument("--generations", type=int)
        parser.add_argument("--population", type=int)
        parser.add_argument("--no-resume", action="store_true")


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid optimizer JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("optimizer options must be a JSON object")
    return parsed


if __name__ == "__main__":
    main()
