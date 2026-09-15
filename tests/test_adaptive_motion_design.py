"""CPU contract tests; these never masquerade as Newton or real evidence."""

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from newton_calibration.collection import CalibrationRequest, MotionSpec
from newton_calibration.collection.adaptive import candidate_pool, coverage
from newton_calibration.collection.planning import _generate
from newton_calibration.collection.sensitivity import FiniteDifferenceProbe, SensitivityResult
from newton_calibration.collection.verify_design import verify_design
from newton_calibration.core.io import sha256_file, write_json
from newton_calibration.core.models import ParameterSpec, jsonable
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning


@pytest.fixture
def scene(tmp_path):
    usd = tmp_path / "arm.usda"
    usd.write_text(
        '#usda 1.0\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {\n'
        'def PhysicsRevoluteJoint "a" {}\ndef PhysicsRevoluteJoint "b" {}\n}\n'
    )
    env = ArticulationEnvCfg(usd_path=str(usd), joint_groups={"a": ("a",), "b": ("b",)}, joint_map={"a": "a", "b": "b"})
    motion = MotionSpec(
        ("a", "b"),
        (0.0, 0.2),
        (-2.0, -2.0),
        (2.0, 2.0),
        (0.3, 0.3),
        (0.3, 0.3),
        (0.6, 0.6),
        source="Test simulation envelope",
        scene_id="test-scene",
    )
    return env, motion


class MatrixProbe:
    def __init__(self, env, motion, names, *, confounded=False, failing=False):
        self.env, self.motion, self.names = env, motion, names
        self.confounded, self.failing, self.calls = confounded, failing, []

    def describe(self):
        return {
            "physics": "analytic-contract-test",
            "asset_sha256": sha256_file(self.env.usd_path),
            "scene_id": self.motion.scene_id,
            "ranges": [{"name": n} for n in self.names],
        }

    def __call__(self, path, experiment, names):
        assert experiment.split == "train", "Holdouts must never inform design"
        self.calls.append(experiment)
        if self.failing:
            raise RuntimeError("Deliberate probe failure")
        matrix = np.ones((len(names), len(names))) if self.confounded else np.eye(len(names)) * 10
        return SensitivityResult(names, [matrix.tolist()] * 2, self.describe(), 1)


def test_search_reaches_each_joint_then_stops_and_reserves_holdouts(scene, tmp_path):
    env, motion = scene
    names = ("a_stiffness_scale", "b_stiffness_scale")
    probe = MatrixProbe(env, motion, names)
    result = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=names),
        design_probe=probe,
        workdir=tmp_path / "runs",
        video=False,
    )
    search = result.design["adaptive_search"]
    assert search["status"] == "predicted_coverage_reached"
    assert search["all_requested_parameters_covered"]
    assert {e.usd_joints for e in probe.calls} == {("a",), ("b",)}
    assert len(result.episodes) == 4  # two selected training + two untouched holdouts
    assert search["physics"] == "analytic-contract-test"
    assert not result.real_data and not result.real_execution_approved and not result.fit_allowed
    assert json.loads((Path(result.workdir) / "design_search.json").read_text())["selected"] == search["selected"]


def test_probe_budget_preserves_unexplored_work(scene, tmp_path):
    env, motion = scene
    names = ("a_stiffness_scale", "b_stiffness_scale")
    result = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=names, max_candidate_probes=1),
        design_probe=MatrixProbe(env, motion, names),
        workdir=tmp_path / "runs",
        video=False,
    )
    search = result.design["adaptive_search"]
    assert search["status"] == "probe_budget_reached" and not search["exhausted"]
    assert search["remaining_candidates"]
    assert not search["all_requested_parameters_covered"]
    assert any(not row["predicted_covered"] for row in search["coverage"])


def test_coverage_met_on_budget_boundary_is_not_reported_incomplete(scene, tmp_path):
    env, motion = scene
    names = ("a_stiffness_scale",)
    result = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=names, max_candidate_probes=1),
        design_probe=MatrixProbe(env, motion, names),
        workdir=tmp_path / "runs",
        video=False,
    )
    assert result.design["adaptive_search"]["status"] == "predicted_coverage_reached"


def test_confounded_parameters_force_search_through_variants(scene, tmp_path):
    env, motion = scene
    motion = replace(motion, posture_offsets_rad=((0.1, 0.0),))
    names = ("a_stiffness_scale", "a_damping_scale")
    probe = MatrixProbe(env, motion, names, confounded=True)
    result = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=names),
        design_probe=probe,
        workdir=tmp_path / "runs",
        video=False,
    )
    search = result.design["adaptive_search"]
    assert search["status"] == "candidate_catalog_exhausted" and search["exhausted"]
    assert not search["all_requested_parameters_covered"]
    assert {e.frequency_scale for e in probe.calls} >= {0.45, 1.0, 1.8}
    assert any(e.amplitude_scale < 1 for e in probe.calls)
    assert any(e.posture_offset_rad for e in probe.calls)
    assert any(h["decision"] == "duplicate_commands_skipped" for h in search["history"])
    assert all(not row["predicted_covered"] for row in search["coverage"])


def test_unsupported_ranges_and_probe_failure_cannot_claim_exhaustion(scene, tmp_path):
    env, motion = scene
    names = ("a_stiffness_scale", "b_stiffness_scale")
    result = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=names),
        design_probe=MatrixProbe(env, motion, names[:1], failing=True),
        workdir=tmp_path / "runs",
        video=False,
    )
    search = result.design["adaptive_search"]
    assert search["unsupported_probe_parameters"] == ["b_stiffness_scale"]
    assert search["failed_candidates"] and not search["exhausted"]
    assert not result.episodes and not search["all_requested_parameters_covered"]


def test_binding_mismatch_is_durable_failure(scene, tmp_path):
    env, motion = scene
    names = ("a_stiffness_scale",)
    wrong = replace(motion, scene_id="another-scene")
    result = tuning.assist(
        env=env,
        collection=motion,
        request=CalibrationRequest(target_parameters=names),
        design_probe=MatrixProbe(env, wrong, names),
        workdir=tmp_path / "runs",
        video=False,
    )
    assert result.status == "design_failed" and not result.episodes
    assert "different asset or scene" in result.design["error"]


def test_cli_probe_factory_and_machine_readable_output(scene, tmp_path, monkeypatch, capsys):
    from newton_calibration.cli import main

    env, motion = scene
    names = ("a_stiffness_scale",)
    module = types.ModuleType("test_scene_probe")
    module.factory = lambda: MatrixProbe(env, motion, names)
    monkeypatch.setitem(sys.modules, "test_scene_probe", module)
    config = write_json(
        tmp_path / "config.json",
        {
            "environment": jsonable(env.describe()),
            "collection": jsonable(motion),
            "request": {"target_parameters": names},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "newton-calibration",
            "assist",
            "--config",
            str(config),
            "--design-probe-factory",
            "test_scene_probe:factory",
            "--no-preview",
            "--workdir",
            str(tmp_path / "runs"),
        ],
    )
    main()
    output = capsys.readouterr()
    assert json.loads(output.out)["design"]["adaptive_search"]["status"] == "predicted_coverage_reached"
    assert "MOTION-DESIGN" in output.err


def test_cli_design_failure_exits_nonzero(scene, tmp_path, monkeypatch, capsys):
    from newton_calibration.cli import main

    env, motion = scene
    module = types.ModuleType("test_bad_probe")
    module.factory = lambda: MatrixProbe(env, replace(motion, scene_id="wrong"), ("a_stiffness_scale",))
    monkeypatch.setitem(sys.modules, "test_bad_probe", module)
    config = write_json(
        tmp_path / "config.json",
        {
            "environment": jsonable(env.describe()),
            "collection": jsonable(motion),
            "request": {"target_parameters": ["a_stiffness_scale"]},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "newton-calibration",
            "assist",
            "--config",
            str(config),
            "--design-probe-factory",
            "test_bad_probe:factory",
            "--no-preview",
            "--workdir",
            str(tmp_path / "runs"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "design_failed"


def test_missing_probe_and_explicit_recipe_only_are_truthfully_labeled(scene, tmp_path):
    env, motion = scene
    for mode, expected in (("adaptive", "needs_dynamics_probe"), ("recipe_only", "recipe_only_explicit")):
        result = tuning.assist(
            env=env,
            collection=motion,
            request=CalibrationRequest(design_mode=mode),
            workdir=tmp_path / mode,
            video=False,
        )
        assert result.design["adaptive_search"]["status"] == expected
        assert not result.design["adaptive_search"]["exhausted"]


def test_posture_transition_is_in_exported_commands_and_limits(scene, tmp_path):
    env, motion = scene
    motion = replace(motion, posture_offsets_rad=((0.0, 0.1),))
    analysis = tuning.analyze(
        env=env, request=CalibrationRequest(target_parameters=("a_stiffness_scale",)), workdir=tmp_path / "runs"
    )
    pool, _ = candidate_pool(analysis, motion, CalibrationRequest())
    experiment = next(e for e in pool if e.posture_offset_rad)
    episode = _generate(motion, tmp_path / "commands", [experiment])[0]
    q = np.loadtxt(tmp_path / "commands" / episode["command_file"], delimiter=",", skiprows=1)[:, 1:3]
    np.testing.assert_allclose(q[[0, -1]], [motion.center_rad] * 2)
    assert np.ptp(q[:, 1]) > 0.05
    assert episode["excited_usd_joints"] == ["a", "b"]
    assert max(episode["max_velocity_rad_s_by_joint"]) <= 0.3
    assert max(episode["max_acceleration_rad_s2_by_joint"]) <= 0.6


def test_video_telemetry_is_checked_against_exact_csv_and_simulation_trace(scene, tmp_path):
    from newton_calibration.collection.verify_video import verify_command_telemetry

    env, motion = scene
    result = tuning.assist(env=env, collection=motion, workdir=tmp_path / "runs", video=False)
    plan_path = Path(result.workdir) / "command_plan.json"
    record = {"command_plan_sha256": sha256_file(plan_path), "chapters": [], "telemetry": []}
    screen = {"tests": []}
    for episode in result.episodes:
        table = np.loadtxt(Path(result.workdir) / episode["command_file"], delimiter=",", skiprows=1)
        q = [float(np.interp(12.0, table[:, 0], table[:, j + 1])) for j in range(2)]
        # Contract-test data only; this does not claim to execute Newton.
        record["chapters"].append({"name": episode["name"]})
        record["telemetry"].append({"trial": episode["name"], "time_s": 12.0, "command_q": q, "simulated_q": list(q)})
        screen["tests"].append({"name": episode["name"], "trace": [{"time_s": 12.0, "simulated_q": list(q)}]})
    assert verify_command_telemetry(plan_path, record, screen) == len(result.episodes)
    record["telemetry"][0]["command_q"][0] += 0.1
    with pytest.raises(ValueError, match="exported motion CSV"):
        verify_command_telemetry(plan_path, record, screen)


def test_duplicate_sensitivity_does_not_resolve_confounding():
    request = CalibrationRequest()
    rows = coverage([np.ones((2, 2)) * 1e8] * 2, ["gain", "inertia"], request)
    assert all(not row["predicted_covered"] for row in rows)


@pytest.mark.parametrize("fails", [False, True])
def test_startup_warmup_is_outside_measurements_and_always_resets(fails):
    from newton_calibration.collection.isaaclab_probe import IsaacLabPredictionBackend

    backend = object.__new__(IsaacLabPredictionBackend)
    events = []
    backend.motion = types.SimpleNamespace(center_rad=(0.0,))
    backend.dt, backend.warmup_steps, backend.objects = 0.001, 4, ()
    backend._reset = lambda pose: events.append("reset") or "target"
    backend.robot = types.SimpleNamespace(
        set_joint_position_target_index=lambda **kw: events.append("command"),
        write_data_to_sim=lambda: None,
        update=lambda dt: None,
    )

    def step(**kwargs):
        events.append("step")
        assert kwargs == {"render": False}
        if fails:
            raise RuntimeError("startup failed")

    backend.sim = types.SimpleNamespace(step=step)
    if fails:
        with pytest.raises(RuntimeError, match="startup failed"):
            backend._warmup()
    else:
        backend._warmup()
        assert events.count("step") == 4
    assert events[0] == events[-1] == "reset"


class AnalyticBackend:
    def __init__(self, *, drift=False):
        self.calls, self.drift = 0, drift

    def describe(self):
        return {"physics": "analytic-contract-test"}

    def rollout(self, path, experiment, parameters):
        self.calls += 1
        t = np.linspace(0, 1, 101)
        q = parameters["gain"] * np.sin(t * 6) + parameters["damping"] * np.cos(t * 10)
        if self.drift:
            q += 0.1 * self.calls
        return np.column_stack((q, np.gradient(q, t)))


def test_finite_difference_uses_actual_responses_and_saves_predictions(tmp_path):
    path = tmp_path / "command.csv"
    path.write_text("test command identity")
    ranges = [ParameterSpec(n, 0.5, 2.0, 1.0, "scale", "test", "contract test") for n in ("gain", "damping")]
    backend = AnalyticBackend()
    probe = FiniteDifferenceProbe(backend, ranges, source="analytic test only")
    result = probe(path, None, ["gain", "damping"])
    assert result.rollout_count == backend.calls == 7
    assert all(r["predicted_covered"] for r in coverage(result.information, result.parameters, CalibrationRequest()))
    assert result.metadata["repeatability_error_noise_units"] == 0
    assert result.metadata["simulated_predictions_sha256"] == sha256_file(result.metadata["simulated_predictions_path"])
    assert not result.metadata["range_approved_for_fitting"]
    assert not result.metadata["real_identifiability_proven"]
    with pytest.raises(ValueError, match="repeatability"):
        FiniteDifferenceProbe(AnalyticBackend(drift=True), ranges, source="test")(path, None, ["gain", "damping"])


def test_design_verifier_recomputes_scores_and_detects_changed_commands(tmp_path):
    command = tmp_path / "design_candidates" / "commands" / "candidate.csv"
    command.parent.mkdir(parents=True)
    command.write_text("contract test command")
    ranges = [ParameterSpec(n, 0.5, 2.0, 1.0, "scale", "test", "contract test") for n in ("gain", "damping")]
    result = FiniteDifferenceProbe(AnalyticBackend(), ranges, source="test")(command, None, ["gain", "damping"])
    record = write_json(tmp_path / "design_probes" / "candidate.json", result)
    selected = tmp_path / "commands" / command.name
    selected.parent.mkdir()
    selected.write_bytes(command.read_bytes())
    write_json(
        tmp_path / "design_search.json",
        {
            "physics": "analytic-contract-test",
            "coverage": [{"parameter": n} for n in result.parameters],
            "thresholds": {"minimum_information_gain": 0.02},
            "history": [
                {
                    "candidate": "candidate",
                    "decision": "selected",
                    "information_gain": float(np.linalg.slogdet(np.eye(2) + np.array(result.information)[0])[1]),
                    "probe_record": str(record.relative_to(tmp_path)),
                    "probe_sha256": sha256_file(record),
                    "command_sha256": sha256_file(command),
                }
            ],
        },
    )
    assert verify_design(tmp_path)["verified_probe_count"] == 1
    selected.write_text("changed")
    with pytest.raises(ValueError, match="Selected commands differ"):
        verify_design(tmp_path)
    command.write_text("changed")
    with pytest.raises(ValueError, match="input commands changed"):
        verify_design(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"max_candidate_probes": 0},
        {"max_candidate_probes": True},
        {"design_mode": "invent"},
        {"minimum_information_gain": 0},
        {"sensitivity_floor": float("nan")},
        {"separation_floor": -1},
    ],
)
def test_search_contract_rejects_invalid_limits(change):
    with pytest.raises(ValueError):
        CalibrationRequest(**change)
