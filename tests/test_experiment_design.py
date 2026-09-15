"""MVP1 experiment selection and generic extension boundaries (CPU contracts)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from newton_calibration.collection import CalibrationRequest, CollectionPlan, ExperimentSpec, MotionSpec
from newton_calibration.collection.generators import Waveform, generate_waveform, register_generator
from newton_calibration.collection.mvp1 import assess_requirements, select_experiments
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning
from newton_calibration.recipes import get_recipe


@pytest.fixture
def setup(tmp_path):
    asset = tmp_path / "robot.usda"
    asset.write_text(
        '#usda 1.0\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {\n'
        'def PhysicsRevoluteJoint "a" {}\ndef PhysicsRevoluteJoint "b" {}\n}\n'
    )
    env = ArticulationEnvCfg(
        usd_path=str(asset), joint_groups={"arm": ("alpha", "beta")}, joint_map={"alpha": "a", "beta": "b"}
    )
    motion = MotionSpec(
        ("a", "b"),
        (0.0, 0.2),
        (-2.0, -2.0),
        (2.0, 2.0),
        (0.3, 0.2),
        (0.3, 0.2),
        (0.6, 0.5),
        source="Synthetic CPU test envelope, not hardware limits",
        scene_id="test-scene",
    )
    return env, motion


def test_targets_select_distinct_experiments_and_deduplicate_shared_gain_tests(setup, tmp_path):
    env, motion = setup

    def collect(*names):
        return tuning.assist(
            env=env,
            collection=motion,
            request=CalibrationRequest(target_parameters=names),
            video=False,
            workdir=tmp_path,
        )

    gains = collect("arm_stiffness_scale", "arm_damping_scale")
    friction = collect("arm_friction_nm")
    armature = collect("arm_armature")
    assert len(gains.episodes) == 6  # Two families x two joints, plus two holdouts.
    assert {e["recipe_id"] for e in gains.episodes if e["split"] == "train"} == {"servo_sweep@1", "settling@1"}
    assert {e["recipe_id"] for e in friction.episodes if e["split"] == "train"} == {"slow_reversal@1"}
    assert {e["recipe_id"] for e in armature.episodes if e["split"] == "train"} == {"acceleration_sweep@1"}
    assert gains.episodes[0]["sha256"] != friction.episodes[0]["sha256"] != armature.episodes[0]["sha256"]
    assert all(e["target_parameters"] and e["required_signals"] and e["reason"] for e in gains.episodes)
    assert not gains.design["numerical_identifiability_proven"]


def test_per_joint_targets_do_not_excite_unselected_axis(setup, tmp_path):
    env, motion = setup
    env = replace(env, joint_groups={"axis_a": ("alpha",), "axis_b": ("beta",)})
    result = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=("axis_b_friction_nm",)),
        video=False,
        workdir=tmp_path,
    )
    assert len(result.episodes) == 3
    for episode in result.episodes:
        assert episode["excited_usd_joints"] == ["b"]
        data = np.genfromtxt(Path(result.workdir) / episode["command_file"], names=True, delimiter=",")
        np.testing.assert_allclose(data["q1_rad"], motion.center_rad[0])
        assert np.ptp(data["q2_rad"]) > 0.01


def test_effort_only_never_generates_saturation_motion(setup, tmp_path):
    env, motion = setup
    result = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=("arm_effort_scale",)),
        workdir=tmp_path,
    )
    assert isinstance(result, CollectionPlan)
    assert result.status == "evidence_action_required" and not result.episodes
    assert result.design["non_motion_requirements"][0]["disposition"] == "external_evidence_required"
    assert not result.real_execution_approved


def test_missing_clocks_and_sensors_are_explicit_not_invented(setup, tmp_path):
    env, motion = setup
    for request in (
        CalibrationRequest(
            target_parameters=("command_delay_s",), clock_synchronized=False, capability_source="driver manual"
        ),
        CalibrationRequest(
            target_parameters=("arm_stiffness_scale",), available_signals=("command_q",), capability_source="operator"
        ),
    ):
        result = tuning.assist(env=env, collection=motion, request=request, workdir=tmp_path)
        assert result.status == "evidence_action_required"
        assert result.evidence_needs["parameters"][0]["disposition"] == "instrumentation_blocked"
    unknown = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=("command_delay_s",)),
        video=False,
        workdir=tmp_path,
    )
    assert unknown.episodes and not unknown.evidence_needs["parameters"][0]["capabilities_confirmed"]
    assert "synchronized_command_feedback_timestamps" in unknown.episodes[0]["required_signals"]


def test_partial_training_evidence_only_recollects_missing_joint(setup):
    env, motion = setup
    inventory = {"train_episodes": ["known"], "heldout_episodes": ["reserved"], "dynamic_excitation_joints": ["alpha"]}
    request = CalibrationRequest(target_parameters=("arm_stiffness_scale",))
    needs = assess_requirements(env.describe(), request.target_parameters, inventory, {}, request)
    experiments, deferred = select_experiments(needs, request, motion.joint_names)
    assert not deferred
    assert all(e.usd_joints == ("b",) for e in experiments if e.split == "train")
    assert needs["parameters"][0]["missing_usd_joints"] == ["b"]


def test_existing_eligible_evidence_is_reused_and_clock_gap_is_not_fixed_by_more_motion(setup):
    env, motion = setup
    request = CalibrationRequest(target_parameters=("arm_damping_scale", "command_delay_s"))
    inventory = {
        "train_episodes": ["known"],
        "heldout_episodes": ["reserved"],
        "dynamic_excitation_joints": ["alpha", "beta"],
    }
    needs = assess_requirements(
        env.describe(), request.target_parameters, inventory, {"arm_damping_scale": "observed dynamic motion"}, request
    )
    assert [r["disposition"] for r in needs["parameters"]] == ["existing_evidence_eligible", "clock_evidence_required"]
    assert select_experiments(needs, request, motion.joint_names) == ([], [])


def test_budget_preserves_uncovered_experiments_and_locked_plan(setup, tmp_path):
    env, motion = setup
    analysis = tuning.analyze(env=env, request=CalibrationRequest(max_training_experiments=1), workdir=tmp_path)
    result = tuning.plan(analysis, collection=motion, video=False)
    assert len(result.episodes) == 3 and len(result.design["deferred_by_budget"]) == 7
    original = (Path(result.workdir) / "command_plan.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        tuning.plan(analysis, collection=motion, video=False)
    assert (Path(result.workdir) / "command_plan.json").read_bytes() == original


def test_unknown_or_future_parameters_fail_before_generating_commands(setup, tmp_path):
    env, motion = setup
    for name in ("object_friction", "peg_compliance", "solver_magic", "unknown_gain"):
        with pytest.raises(ValueError, match="Unsupported MVP1"):
            tuning.assist(
                env=env, collection=motion, request=CalibrationRequest(target_parameters=(name,)), workdir=tmp_path
            )
    assert not list(tmp_path.rglob("command_plan.json"))


def test_changed_evidence_need_is_not_silently_replanned(setup, tmp_path):
    env, motion = setup
    analysis = tuning.analyze(env=env, workdir=tmp_path)
    analysis.evidence_needs["parameters"][0]["experiment_recipes"] = ["unknown@1"]
    with pytest.raises(ValueError, match="Analysis changed"):
        tuning.plan(analysis, collection=motion, video=False)


def test_recipe_versions_preserve_old_duration_and_new_scope(setup):
    env, _ = setup
    assert get_recipe("articulation.position_pd.free_space.v1", env.describe()).max_episode_duration_s == 12
    assert get_recipe("so101_actuator_dynamics.v1").max_episode_duration_s == 12
    scoped = replace(env.describe(), tuning_targets=("arm_damping_scale",))
    recipe = get_recipe("articulation.position_pd.free_space@3", scoped)
    assert recipe.required_parameter_names == ("arm_damping_scale",)
    assert recipe.max_episode_duration_s is None
    with pytest.raises(ValueError, match="selection requires"):
        get_recipe("articulation.position_pd.free_space.v1", scoped)


def test_extensible_generator_contract_fails_closed():
    t = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="No installed"):
        generate_waveform("unknown@1", t, 1, 0, 0)
    register_generator("contract-test.invalid@1", lambda *args: Waveform(np.ones(3), ()))
    with pytest.raises(ValueError, match="malformed"):
        generate_waveform("contract-test.invalid@1", t, 1, 0, 0)
    with pytest.raises(ValueError, match="unique"):
        register_generator("servo_sweep@1", lambda *args: None)
    register_generator("contract-test.oversized@1", lambda *args: Waveform(np.ones_like(t) * 2., ()))
    with pytest.raises(ValueError, match="amplitude contract"):
        generate_waveform("contract-test.oversized@1", t, 1, 0, 0)
    with pytest.raises(ValueError, match="safe filename"):
        ExperimentSpec("../outside", "servo_sweep@1", "train", ("a",), ("gain",), (), "bad name")


def test_agent_request_provenance_and_durable_human_report(setup, tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        CalibrationRequest(available_signals=("command_q",))
    env, motion = setup
    result = tuning.assist(env=env, collection=motion, video=False, workdir=tmp_path)
    root = Path(result.workdir)
    design = json.loads((root / "experiment_design.json").read_text())
    assert design["catalog"] == "mvp1.free-motion-design@1"
    assert (root / "evidence_needs.json").is_file()
    assert "experiment_design.json" in (root / "COLLECT_NEXT.md").read_text()


def test_scoped_five_calls_keep_full_generated_trials_and_missing_untuned_bounds_do_not_block(
    setup, tmp_path, monkeypatch
):
    """Synthetic CPU pipeline test, NOT real evidence or Newton validation."""
    import pandas as pd

    from newton_calibration.adapters.evidence import TabularJointEvidence
    from newton_calibration.core import JointBinding, LongFormSchema, SignalBinding, bind_evidence_files

    env, motion = setup
    env = replace(
        env,
        runtime="analytic",
        device="cpu",
        dt=0.04,
        profile_confirmed=True,
        controller_profile_confirmed=True,
        controller_profile_source="CPU fixture",
        base_stiffness_by_joint={"alpha": 2.0, "beta": 2.0},
        base_damping_by_joint={"alpha": 1.0, "beta": 1.0},
        base_effort_limit_by_joint={"alpha": 5.0, "beta": 5.0},
        analytic_inertia_by_joint={"alpha": 1.0, "beta": 1.0},
        parameter_bounds={"arm_friction_nm": (0.0, 0.1, 0.01)},
    )
    request = CalibrationRequest(target_parameters=("arm_friction_nm",))
    proposal = tuning.assist(env=env, collection=motion, request=request, video=False, workdir=tmp_path / "collection")
    evidence_root = tmp_path / "synthetic-evidence"
    evidence_root.mkdir()
    episodes = []
    for episode in proposal.episodes:
        data = np.genfromtxt(Path(proposal.workdir) / episode["command_file"], delimiter=",", names=True)
        rows = []
        for index, joint in enumerate(("alpha", "beta"), 1):
            for channel, values in (
                ("target", data[f"q{index}_rad"]),
                ("position", 0.8 * data[f"q{index}_rad"]),
                ("velocity", 0.8 * data[f"dq{index}_rad_s"]),
            ):
                rows.extend(
                    {"t": t, "joint": joint, "signal": channel, "value": value}
                    for t, value in zip(data["time_s"], values)
                )
        filename = episode["name"] + ".csv"
        pd.DataFrame(rows).to_csv(evidence_root / filename, index=False)
        episodes.append({"name": episode["name"], "path": filename, "split": episode["split"], "trial_id": filename})
    spec = bind_evidence_files(
        root=evidence_root,
        episodes=episodes,
        schema=LongFormSchema(
            time_column="t",
            time_unit="s",
            value_column="value",
            joint_column="joint",
            signal_column="signal",
            field_column=None,
        ),
        joint_bindings=tuple(
            JointBinding(s, t, source_unit="rad", usd_unit="rad", transform_confirmed=True)
            for s, t in env.joint_map.items()
        ),
        signal_bindings=(
            SignalBinding("target", "command_q"),
            SignalBinding("position", "actual_q"),
            SignalBinding("velocity", "actual_dq"),
        ),
        revision="synthetic-contract-test",
    )
    evidence = TabularJointEvidence(spec)
    analysis = tuning.analyze(env=env, evidence=evidence, request=request, workdir=tmp_path / "fit")
    assert all(analysis.readiness.values()), analysis.warnings
    plan = tuning.plan(analysis)
    assert [p.name for p in plan.parameters] == ["arm_friction_nm"]
    assert plan.optimizer["max_episode_duration_s"] is None
    durations = []
    original = TabularJointEvidence.load_episode

    def record_load(self, name, **kwargs):
        result = original(self, name, **kwargs)
        durations.append((name, kwargs["max_duration_s"], result.time_s[-1]))
        return result

    monkeypatch.setattr(TabularJointEvidence, "load_episode", record_load)
    fit = tuning.fit(plan, generations=1, population=4)
    validation = tuning.validate(fit)
    package = tuning.write(validation, output=tmp_path / "package")
    assert len(durations) == len(proposal.episodes)
    assert all(cap is None and duration >= 23.9 for _, cap, duration in durations)
    manifest = json.loads(Path(package.manifest_path).read_text())
    assert set(manifest["parameters"]) == {"arm_friction_nm"}
    assert manifest["runtime"]["tuning_targets"] == ["arm_friction_nm"]
    assert not manifest["activation_allowed"]  # A CPU test is never a validated Newton package.
