"""Progress is observable, but never authoritative or a new calibration gate."""

from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from newton_calibration.collection import MotionSpec
from newton_calibration.core import run_status
from newton_calibration.core.io import sha256_file
from newton_calibration.core.models import CalibrationPlan
from newton_calibration.core.run_status import RunStatus
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning


def read_status(root):
    return json.loads((Path(root) / "status.json").read_text())


@pytest.fixture
def collection(tmp_path):
    asset = tmp_path / "arm.usda"
    asset.write_text(
        '#usda 1.0\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {\n'
        'def PhysicsRevoluteJoint "a" {}\n}\n'
    )
    env = ArticulationEnvCfg(usd_path=str(asset), joint_groups={"arm": ("alpha",)}, joint_map={"alpha": "a"})
    motion = MotionSpec(("a",), (0.0,), (-2.0,), (2.0,), (0.3,), (0.3,), (0.6,), source="test", scene_id="test")
    return env, motion


@pytest.fixture
def fit_plan(tmp_path):
    return CalibrationPlan(
        run_id="test-run",
        created_at="test",
        recipe="test",
        evidence_uri="test",
        evidence_revision="test",
        evidence_fingerprint="test",
        asset_fingerprint="test",
        environment=None,
        parameters=[],
        train_episodes=[],
        heldout_episodes=[],
        objective_weights={},
        optimizer={},
        validation_gates={},
        workdir=str(tmp_path),
    )


@pytest.mark.parametrize("call", ["fit", "validate", "write"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_call_exceptions_report_failure_and_preserve_original(fit_plan, monkeypatch, call, error_type):
    error = error_type("injected failure")
    fit_run = SimpleNamespace(plan=fit_plan, baseline=SimpleNamespace(parameters={}))
    validation = SimpleNamespace(fit=fit_run)

    def crash(*args, **kwargs):
        running = read_status(fit_plan.workdir)
        assert running["current_call"] == call
        assert running["calls"][call]["status"] == "running"
        raise error

    monkeypatch.setattr(tuning, "_evidence_adapter_from_plan", crash)
    monkeypatch.setattr(tuning, "_assert_locked_inputs_unchanged", crash)
    with pytest.raises(error_type) as caught:
        if call == "fit":
            tuning.fit(calibration_plan=fit_plan)
        elif call == "validate":
            tuning.validate(fit_run)
        else:
            tuning.write(validation, output="not-created")
    assert caught.value is error
    state = read_status(fit_plan.workdir)
    assert state["state"] == "FAILED"
    assert state["calls"][call]["status"] == "failed"
    assert "injected failure" in state["calls"][call]["summary"]
    assert state["current_call"] is None and state["progress"] is None
    events = (Path(fit_plan.workdir) / "events.jsonl").read_text()
    assert f"{call} failed" in events


def test_plan_exception_is_recorded(collection, tmp_path):
    env, _ = collection
    analysis = tuning.analyze(env=env, workdir=tmp_path)
    with pytest.raises(ValueError, match="intent"):
        tuning.plan(analysis, intent="not-an-intent")
    assert read_status(analysis.workdir)["calls"]["plan"]["status"] == "failed"


def test_missing_evidence_does_not_preempt_collection(collection, tmp_path):
    env, motion = collection
    analysis = tuning.analyze(env=env, workdir=tmp_path)
    before = read_status(analysis.workdir)
    assert before["state"] == "ANALYZED" and before["calls"]["plan"]["status"] == "waiting"
    assert before["blockers"] == []
    plan = tuning.plan(analysis, collection=motion)
    status = read_status(analysis.workdir)
    assert plan.status == "preview_pending"
    assert status["state"] == "ACTION_REQUIRED"
    assert status["calls"]["plan"]["status"] == "done"
    assert status["calls"]["fit"]["status"] == "waiting"
    assert status["collection"]["command_files"] == len(plan.episodes) > 0
    assert status["collection"]["preview_status"] == "blocked"
    assert status["blockers"] == []
    assert not status["collection"]["fit_allowed"] and not status["collection"]["real_execution_approved"]


def test_explicit_fit_request_stays_blocked_and_retry_can_collect(collection, tmp_path):
    env, motion = collection
    analysis = tuning.analyze(env=env, workdir=tmp_path)
    with pytest.raises(ValueError, match="readiness"):
        tuning.plan(analysis, intent="fit")
    blocked = read_status(analysis.workdir)
    assert blocked["state"] == "BLOCKED" and blocked["calls"]["plan"]["status"] == "blocked"
    assert blocked["blockers"]
    plan = tuning.plan(analysis, intent="collect", collection=motion, video=False)
    state = read_status(plan.workdir)
    assert state["state"] == "ACTION_REQUIRED" and state["blockers"] == []
    assert "readiness checks failed" not in state["calls"]["plan"]["summary"]


def test_missing_scene_setup_explains_real_block(collection, tmp_path):
    env, _ = collection
    plan = tuning.assist(env=env, workdir=tmp_path)
    state = read_status(plan.workdir)
    assert state["state"] == "BLOCKED"
    assert state["collection"]["status"] == "needs_scene_setup"
    assert "starting pose" in state["calls"]["plan"]["summary"]


@pytest.mark.parametrize("kind", ["exception", "generation", "screen_rejected", "screen_ok"])
def test_collection_outcomes_are_not_fitting_success(collection, tmp_path, kind):
    env, motion = collection

    def preview(plan_path, output):
        state = read_status(plan_path.parent)
        assert state["calls"]["plan"]["status"] == "running"
        if kind == "exception":
            raise RuntimeError("renderer unavailable")
        output.mkdir()
        artifacts = {}
        for name in ("video_path", "screen_path", "backend_record_path"):
            path = output / name
            path.write_text("test fixture, NOT a real Newton result")
            artifacts[name] = str(path)
        return dict(
            artifacts,
            physics="Newton",
            scene_id=motion.scene_id,
            command_plan_sha256=sha256_file(plan_path),
            screen_passed=kind == "screen_ok",
        )

    if kind == "generation":
        motion = replace(motion, max_velocity_rad_s=(1e-8,))
    plan = tuning.assist(env=env, collection=motion, preview=preview, workdir=tmp_path)
    state = read_status(plan.workdir)
    assert (
        state["state"]
        == {
            "exception": "FAILED",
            "generation": "FAILED",
            "screen_rejected": "BLOCKED",
            "screen_ok": "ACTION_REQUIRED",
        }[kind]
    )
    assert state["collection"]["status"] == plan.status
    assert state["calls"]["fit"]["status"] == "waiting"
    assert not state["collection"]["fit_allowed"] and not state["collection"]["real_execution_approved"]


def test_display_io_failure_does_not_mask_original_exception(fit_plan, monkeypatch):
    failure = RuntimeError("physics error")
    monkeypatch.setattr(run_status, "atomic_write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    monkeypatch.setattr(tuning, "_evidence_adapter_from_plan", lambda _: (_ for _ in ()).throw(failure))
    with pytest.raises(RuntimeError) as caught:
        tuning.fit(fit_plan)
    assert caught.value is failure


def test_display_io_failure_does_not_break_collection(collection, tmp_path, monkeypatch):
    env, motion = collection
    monkeypatch.setattr(run_status, "atomic_write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    result = tuning.assist(env=env, collection=motion, video=False, workdir=tmp_path)
    assert result.episodes
    assert not (Path(result.workdir) / "status.json").exists()


@pytest.mark.parametrize(
    "corruption", [{"metrics": []}, {"calls": []}, {"calls": {"fit": "bad"}}, {"run_id": "other-run"}]
)
def test_bad_or_wrong_run_status_is_not_trusted(tmp_path, corruption):
    RunStatus(tmp_path, "test").start("fit")
    state = read_status(tmp_path)
    state.update(corruption)
    (tmp_path / "status.json").write_text(json.dumps(state))
    attached = RunStatus.attach(tmp_path, "test")
    attached.finish("fit", "ok", best=1)
    state = read_status(tmp_path)
    assert state["run_id"] == "test" and state["state"] == "FITTED"
    assert state["metrics"]["best"] == 1


def test_disabled_status_keeps_failure_semantics(fit_plan, monkeypatch):
    monkeypatch.setenv("NEWTON_CALIBRATION_STATUS", "0")
    failure = RuntimeError("physics error")
    monkeypatch.setattr(tuning, "_evidence_adapter_from_plan", lambda _: (_ for _ in ()).throw(failure))
    with pytest.raises(RuntimeError) as caught:
        tuning.fit(fit_plan)
    assert caught.value is failure
    assert not (Path(fit_plan.workdir) / "status.json").exists()
    assert not (Path(fit_plan.workdir) / "events.jsonl").exists()


def test_retry_clears_stale_failure_and_progress(tmp_path):
    status = RunStatus(tmp_path, "test")
    status.start("fit")
    status.progress(8, 10, best=3)
    status.fail("fit", "first attempt crashed")
    with RunStatus.attach(tmp_path, "test").observe("fit"):
        state = read_status(tmp_path)
        assert state["state"] == "FITTING" and state["progress"] is None
        assert "summary" not in state["calls"]["fit"] and "ended_at" not in state["calls"]["fit"]
        assert "best" not in state["metrics"]
        RunStatus.attach(tmp_path, "test").finish("fit", "retry completed")
    assert read_status(tmp_path)["state"] == "FITTED"


def test_unknown_collection_outcome_requires_review(tmp_path):
    status = RunStatus(tmp_path, "test")
    status.start("plan")
    status.collection(outcome="unknown_outcome", episodes=0, preview={})
    state = read_status(tmp_path)
    assert state["state"] == "BLOCKED"
    assert "Unrecognized collection outcome" in state["calls"]["plan"]["summary"]
    assert not state["collection"]["fit_allowed"]


def test_decorators_preserve_public_signature():
    assert "calibration_plan" in inspect.signature(tuning.fit).parameters
    assert "optimizer_options" in inspect.signature(tuning.plan).parameters
    assert tuning.fit.__name__ == "fit"


def test_watcher_renders_collection_and_exits_for_required_action(collection, tmp_path):
    env, motion = collection
    result = tuning.assist(env=env, collection=motion, workdir=tmp_path)
    source = Path(__file__).resolve().parents[1] / "scripts/watch_run.py"
    spec = importlib.util.spec_from_file_location("test_watch_run", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.render(read_status(result.workdir), color=False)
    assert "ACTION_REQUIRED" in module.TERMINAL
    assert "evidence collection: preview_pending" in rendered
    assert "Not fit-ready; no real-robot execution approval" in rendered
