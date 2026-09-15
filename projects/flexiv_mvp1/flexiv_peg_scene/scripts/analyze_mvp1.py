"""Run actual toolkit analyze/plan without inventing real Flexiv evidence."""
import json
from pathlib import Path

from newton_calibration.core.models import jsonable
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning

ROOT = Path(__file__).resolve().parents[1]


def main():
    joints = tuple(f"joint{i}" for i in range(1, 8))
    stiffness = [4000., 4000., 3000., 3000., 500., 500., 200.]
    damping = [80., 80., 60., 40., 10., 10., 5.]
    env = ArticulationEnvCfg(
        usd_path=str(ROOT / "assets/Flexiv_Rizon4s_Grav.usd"),
        robot_id="flexiv-rizon4s-grav-mvp1", joint_groups={"arm": joints},
        joint_map={j: j for j in joints}, joint_order=joints,
        profile_confirmed=False, controller_profile_confirmed=False,
        controller_profile_source="Newton scene initialization priors; not OEM-identified RDK controller",
        runtime="isaaclab_newton", dt=1 / 960, num_substeps=1,
        base_stiffness_by_joint=dict(zip(joints, stiffness)),
        base_damping_by_joint=dict(zip(joints, damping)),
        base_effort_limit_by_joint={j: 123. for j in joints},
        base_armature_by_joint={j: .05 for j in joints},
        parameter_bounds={})
    analysis = tuning.analyze(env=env, evidence=None, workdir=ROOT / "output/mvp1-analysis")
    output = ROOT / "mvp1_collection"
    output.mkdir(exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(jsonable(analysis), indent=2) + "\n")
    (output / "proposed_environment.json").write_text(json.dumps(jsonable(env.describe()), indent=2) + "\n")
    try:
        tuning.plan(analysis)
    except ValueError as exc:
        result = dict(call="tuning.plan", status="blocked_expected", run_id=analysis.run_id,
                      reason=str(exc), next_action="Review collection_plan.json, approve the real setup and collect train/held-out trials",
                      fit_executed=False, validation_executed=False, real_samples=0,
                      note="The data-acquisition plan is separate from a locked fitting plan; no gate was bypassed.")
        (output / "fit_plan_status.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(readiness=analysis.readiness, plan=result), indent=2))
    else:
        raise AssertionError("No-evidence fit plan must never be accepted")


if __name__ == "__main__":
    main()
