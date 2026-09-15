import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from newton_calibration.collection import MotionSpec, verify_commands
from newton_calibration.core.io import sha256_file
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning


@pytest.fixture
def setup(tmp_path):
    asset = tmp_path / "arm.usda"
    asset.write_text(
        '#usda 1.0\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {\n'
        'def PhysicsRevoluteJoint "a" {}\ndef PhysicsRevoluteJoint "b" {}\n}\n'
    )
    env = ArticulationEnvCfg(
        usd_path=str(asset), joint_groups={"arm": ("alpha", "beta")}, joint_map={"alpha": "a", "beta": "b"}
    )
    spec = MotionSpec(
        ("a", "b"),
        (0.0, 0.2),
        (-2.0, -2.0),
        (2.0, 2.0),
        (0.3, 0.2),
        (0.3, 0.2),
        (0.6, 0.5),
        source="test scene simulation envelope",
        scene_id="test-scene",
    )
    return env, spec


def test_auto_no_evidence_generates_reproducible_motion_and_pending_video(setup, tmp_path):
    env, spec = setup
    result = tuning.assist(env=env, collection=spec, workdir=tmp_path / "runs")
    assert result.status == "preview_pending"
    assert len(result.episodes) == 10  # Four evidence-specific experiments per joint + two holdouts.
    assert result.preview["requested"] is True
    assert result.preview["status"] == "blocked"
    assert not result.real_data and not result.real_execution_approved and not result.fit_allowed
    assert not env.profile_confirmed and not env.controller_profile_confirmed and not env.parameter_bounds
    other = tuning.assist(env=env, collection=spec, workdir=tmp_path / "runs")
    assert [e["sha256"] for e in result.episodes] == [e["sha256"] for e in other.episodes]
    assert [e["split"] for e in result.episodes] == ["train"] * 8 + ["heldout"] * 2
    for episode in result.episodes:
        table = np.genfromtxt(Path(result.workdir) / episode["command_file"], delimiter=",", names=True)
        q = np.column_stack([table["q1_rad"], table["q2_rad"]])
        np.testing.assert_allclose(q[0], spec.center_rad)
        np.testing.assert_allclose(q[-1], spec.center_rad)
        assert np.all(q > np.array(spec.lower_rad) + spec.margin_rad)
        assert np.all(q < np.array(spec.upper_rad) - spec.margin_rad)
        dq = np.gradient(q, 1 / spec.command_rate_hz, axis=0, edge_order=2)
        ddq = np.gradient(dq, 1 / spec.command_rate_hz, axis=0, edge_order=2)
        assert np.all(abs(dq).max(axis=0) <= spec.max_velocity_rad_s)
        assert np.all(abs(ddq).max(axis=0) <= spec.max_acceleration_rad_s2)
        # Acceleration-rich trajectories can have smaller amplitude under the
        # same acceleration cap; test the declared excitation threshold.
        assert np.ptp(q, axis=0).max() > 0.01
    with pytest.raises(TypeError, match="CollectionPlan"):
        tuning.fit(result)


def test_default_preview_called_and_screen_failure_is_not_lost(setup, tmp_path):
    env, spec = setup
    calls = []

    def preview(path, output):
        assert verify_commands(path)["asset_sha256"] == sha256_file(env.usd_path)
        calls.append(path)
        output.mkdir()
        paths = {}
        for key in ("video_path", "screen_path", "backend_record_path"):
            dest = output / key
            dest.write_text("test stub, not an actual video")
            paths[key] = str(dest)
        return dict(
            paths, physics="Newton", scene_id=spec.scene_id, command_plan_sha256=sha256_file(path), screen_passed=False
        )

    result = tuning.assist(env=env, collection=spec, preview=preview, workdir=tmp_path)
    assert len(calls) == 1
    assert result.preview["status"] == "complete"  # contract test; actual media is integration-tested
    assert result.preview["screen_passed"] is False
    assert not result.real_execution_approved
    durable = json.loads((Path(result.workdir) / "collection_plan.json").read_text())
    assert durable["preview"]["video_path_sha256"]
    assert (Path(result.workdir) / "COLLECT_NEXT.md").is_file()


def test_no_preview_only_explicitly_and_errors_are_durable(setup, tmp_path):
    env, spec = setup

    def fail(*args):
        raise RuntimeError("renderer unavailable")

    skipped = tuning.assist(env=env, collection=spec, preview=fail, video=False, workdir=tmp_path)
    assert skipped.preview["status"] == "skipped_explicitly"
    failed = tuning.assist(env=env, collection=spec, preview=fail, workdir=tmp_path)
    assert failed.status == "preview_failed"
    assert "renderer unavailable" in failed.preview["error"]
    assert len(failed.episodes) == 10
    assert json.loads((Path(failed.workdir) / "collection_plan.json").read_text())["status"] == "preview_failed"


@pytest.mark.parametrize(
    "changes",
    [
        {"center_rad": (9, 0)},
        {"lower_rad": (0, 0)},
        {"max_velocity_rad_s": (0, 1)},
        {"amplitude_rad": (float("nan"), 1)},
        {"joint_names": ("a", "a")},
        {"duration_s": 3},
        {"command_rate_hz": 0},
        {"source": ""},
        {"scene_id": ""},
    ],
)
def test_invalid_spec_rejected(setup, changes):
    with pytest.raises(ValueError):
        replace(setup[1], **changes)


def test_tampered_commands_and_wrong_joints_rejected(setup, tmp_path):
    env, spec = setup
    with pytest.raises(ValueError, match="exactly match"):
        tuning.assist(env=env, collection=replace(spec, joint_names=("x", "y")), workdir=tmp_path)
    result = tuning.assist(env=env, collection=spec, video=False, workdir=tmp_path)
    root = Path(result.workdir)
    (root / result.episodes[0]["command_file"]).write_text("tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_commands(root / "command_plan.json")


def test_changed_asset_fails_closed(setup, tmp_path):
    env, spec = setup
    a = tuning.analyze(env=env, workdir=tmp_path)
    Path(env.usd_path).write_text("changed")
    with pytest.raises(ValueError, match="changed USD"):
        tuning.plan(a, collection=spec)


def test_surface_supplies_collection_and_preview(setup, tmp_path):
    env, spec = setup

    class Surface:
        def describe(self):
            return env.describe()

        def describe_collection(self):
            return spec

        def preview_collection(self, *args):
            raise RuntimeError("surface preview was invoked")

    result = tuning.assist(env=Surface(), workdir=tmp_path)
    assert "surface preview was invoked" in result.preview["error"]


def test_negligible_excitation_records_generation_failure(setup, tmp_path):
    env, spec = setup
    result = tuning.assist(env=env, collection=replace(spec, max_velocity_rad_s=(1e-8, 1e-8)), workdir=tmp_path)
    assert result.status == "generation_failed"
    assert "too little excitation" in result.preview["reason"]
    assert not result.real_execution_approved


def test_plugin_cannot_rewrite_locked_plan(setup, tmp_path):
    env, spec = setup

    def tamper(path, output):
        content = json.loads(path.read_text())
        content["created_at"] = "rewritten"
        path.write_text(json.dumps(content))
        return {"physics": "Newton", "scene_id": spec.scene_id, "command_plan_sha256": sha256_file(path)}

    result = tuning.assist(env=env, collection=spec, preview=tamper, workdir=tmp_path)
    assert result.status == "preview_failed"
    assert "changed the locked command plan" in result.preview["error"]


def test_cli_reports_pending_preview_as_incomplete(setup, tmp_path, monkeypatch, capsys):
    from newton_calibration.cli import main
    from newton_calibration.core.models import jsonable

    env, spec = setup
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"environment": jsonable(env.describe()), "collection": jsonable(spec)}))
    monkeypatch.setattr(
        "sys.argv", ["newton-calibration", "assist", "--config", str(config), "--workdir", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "preview_pending" and len(result["episodes"]) == 10


def test_runtime_numpy_scalars_are_normalized_before_durable_write(setup, tmp_path):
    env, spec = setup
    center = np.asarray(spec.center_rad, dtype=np.float32)
    native = replace(spec, center_rad=tuple(center), joint_names=list(spec.joint_names))
    result = tuning.assist(env=env, collection=native, video=False, workdir=tmp_path)
    assert result.status == "commands_generated"
    assert all(type(v) is float for v in result.motion_spec.center_rad)
    assert isinstance(result.motion_spec.joint_names, tuple)
    assert json.loads((Path(result.workdir) / "collection_plan.json").read_text())["episodes"]
