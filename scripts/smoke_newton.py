"""One-episode Newton replay smoke test for container and GPU verification.

This is intentionally a verification harness, not a sixth product API call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newton_calibration.adapters.evidence import AnchorLabSO101Evidence
from newton_calibration.adapters.runtime import create_runtime
from newton_calibration.adapters.surface import SO101EnvCfg
from newton_calibration.core.models import jsonable
from newton_calibration.isaaclab import tuning
from newton_calibration.validation.metrics import compare_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one short Anchor-Lab episode in Newton")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--workdir", default="runs/smoke")
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--base-effort-limit", type=float, default=10.0)
    parser.add_argument("--base-stiffness", type=float, default=1.7453293)
    parser.add_argument("--base-damping", type=float, default=0.017453292)
    args = parser.parse_args()

    env = SO101EnvCfg(
        usd_path=str(Path(args.asset).expanduser().resolve()),
        runtime="isaaclab_newton",
        device=args.device,
        num_substeps=args.substeps,
        base_effort_limit=args.base_effort_limit,
        base_stiffness=args.base_stiffness,
        base_damping=args.base_damping,
    )
    analysis = tuning.analyze(env=env, evidence=args.evidence, workdir=args.workdir)
    calibration_plan = tuning.plan(analysis)
    evidence = AnchorLabSO101Evidence(calibration_plan.evidence_uri)
    episode = evidence.load_episode(
        calibration_plan.train_episodes[0],
        dt=calibration_plan.environment.dt,
        max_duration_s=args.duration,
    )
    candidate = {parameter.name: parameter.initial for parameter in calibration_plan.parameters}

    runtime = create_runtime(calibration_plan.environment)
    try:
        simulated_q, simulated_dq, stable = runtime._rollout(candidate, episode)
        score, metrics = compare_trajectories(
            simulated_q,
            simulated_dq,
            episode.actual_q,
            episode.actual_dq,
            episode.command_q,
            calibration_plan.environment.dt,
            calibration_plan.objective_weights,
        )
    finally:
        runtime.close()

    print(
        json.dumps(
            jsonable(
                {
                    "backend": calibration_plan.environment.adapter,
                    "episode": episode.name,
                    "duration_s": float(episode.time_s[-1]) if len(episode.time_s) else 0.0,
                    "steps": len(episode.time_s),
                    "score": score,
                    "metrics": metrics,
                    "stable": stable,
                    "actual_q_start": episode.actual_q[0].tolist(),
                    "simulated_q_first_five": simulated_q[:5].tolist(),
                    "simulated_q_last": simulated_q[-1].tolist(),
                    "max_abs_simulated_q": float(abs(simulated_q).max()),
                    "max_abs_simulated_dq": float(abs(simulated_dq).max()),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
