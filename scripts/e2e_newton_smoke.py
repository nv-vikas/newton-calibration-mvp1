"""Exercise all five product calls with real Newton on a bounded smoke slice."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from newton_calibration.adapters.surface import SO101EnvCfg
from newton_calibration.core.io import write_json
from newton_calibration.core.models import jsonable
from newton_calibration.isaaclab import tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all five calibration calls on one short train/holdout slice")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--workdir", default="runs/e2e-smoke")
    parser.add_argument("--output", default="packages/e2e-smoke")
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    env = SO101EnvCfg(
        usd_path=str(Path(args.asset).expanduser().resolve()),
        runtime="isaaclab_newton",
        device=args.device,
    )
    analysis = tuning.analyze(env=env, evidence=args.evidence, workdir=args.workdir)
    full_plan = tuning.plan(analysis)
    smoke_plan = replace(
        full_plan,
        train_episodes=full_plan.train_episodes[:1],
        heldout_episodes=full_plan.heldout_episodes[:1],
        optimizer={
            **full_plan.optimizer,
            "max_episode_duration_s": args.duration,
            "generations": 1,
            "population": 4,
        },
    )
    write_json(Path(smoke_plan.workdir) / "plan.json", smoke_plan)
    fit_run = tuning.fit(smoke_plan, resume=False)
    validation = tuning.validate(fit_run)
    smoke_metrics_passed = validation.passed
    # A truncated integration test must never be mistaken for a releaseable
    # calibration. Preserve the measured gates, then add the coverage gate.
    validation = replace(
        validation,
        passed=False,
        gates={**validation.gates, "full_recipe_coverage": False},
    )
    write_json(Path(smoke_plan.workdir) / "validation.json", validation)
    package = tuning.write(validation, output=args.output)
    print(
        json.dumps(
            jsonable(
                {
                    "run_id": package.run_id,
                    "backend": fit_run.backend,
                    "baseline_score": fit_run.baseline.score,
                    "best_train_score": fit_run.best.score,
                    "heldout_improvement_pct": validation.improvement_pct,
                    "smoke_metrics_passed": smoke_metrics_passed,
                    "activation_allowed": validation.passed,
                    "gates": validation.gates,
                    "package": package,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
