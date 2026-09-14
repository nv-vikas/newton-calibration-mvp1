from __future__ import annotations

import argparse
import json
from pathlib import Path

from newton_calibration.adapters.evidence import fetch_anchor_lab_so101
from newton_calibration.adapters.surface import SO101EnvCfg
from newton_calibration.core.models import jsonable
from newton_calibration.isaaclab import tuning


def main() -> None:
    parser = argparse.ArgumentParser(prog="newton-calibration", description="Newton calibration MVP1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch", help="download SO-101 Anchor-Lab evidence and USD")
    fetch_parser.add_argument("--output", default="data/anchor-lab")
    fetch_parser.add_argument("--revision", default="647edd5787cd764cdc041103ad282dc59214d919")

    inspect_parser = subparsers.add_parser("inspect", help="execute analyze + plan without starting physics")
    _add_job_arguments(inspect_parser, include_fit=False)

    run_parser = subparsers.add_parser("run", help="execute analyze → plan → fit → validate → write")
    _add_job_arguments(run_parser, include_fit=True)

    args = parser.parse_args()
    if args.command == "fetch":
        print(json.dumps(fetch_anchor_lab_so101(args.output, args.revision), indent=2))
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
    calibration_plan = tuning.plan(analysis)
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
    if include_fit:
        parser.add_argument("--output", default="packages/so101")
        parser.add_argument("--generations", type=int)
        parser.add_argument("--population", type=int)
        parser.add_argument("--no-resume", action="store_true")


if __name__ == "__main__":
    main()
