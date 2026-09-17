from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any

from .io import atomic_write_text, utc_now
from .models import jsonable

STATUS_SCHEMA = "newton.calibration.run-status/v1"
STATUS_FILENAME = "status.json"
EVENTS_FILENAME = "events.jsonl"

CALLS: tuple[str, ...] = ("analyze", "plan", "fit", "validate", "write")

_DISABLE_ENV = "NEWTON_CALIBRATION_STATUS"

# The run's headline state is derived from the call it is in, so a reader knows
# how far a run got without comparing five call entries.
_RUNNING_STATE = {
    "analyze": "ANALYZING",
    "plan": "PLANNING",
    "fit": "FITTING",
    "validate": "VALIDATING",
    "write": "WRITING",
}
_DONE_STATE = {
    "analyze": "ANALYZED",
    "plan": "PLANNED",
    "fit": "FITTED",
    "validate": "VALIDATED",
    "write": "COMPLETE",
}

_log = logging.getLogger(__name__)


def _best_effort(method):
    """A display failure must neither replace an exception nor fail a healthy job."""

    @wraps(method)
    def wrapped(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception:  # noqa: BLE001 - observation only
            _log.debug("could not update run status", exc_info=True)
            return None

    return wrapped


def observe_call(call: str, argument: str, *path: str):
    """Observe an existing run, without changing call signatures or exceptions.

    ``argument`` identifies the Analysis/Plan/Fit/Validation argument. ``path``
    reaches its locked plan. Invalid arguments remain the function's concern.
    """

    def decorate(function):
        params = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                obj = params.bind(*args, **kwargs).arguments[argument]
                for name in path:
                    obj = getattr(obj, name)
                status = RunStatus.attach(obj.workdir, obj.run_id)
            except Exception:  # noqa: BLE001 - never substitute display errors for API errors
                return function(*args, **kwargs)
            with status.observe(call):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def status_enabled() -> bool:
    """Status publishing is on unless explicitly disabled."""

    return os.environ.get(_DISABLE_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


class RunStatus:
    """Human-facing progress for a single calibration run.

    This is a display surface, never a record.  The fit journal, candidate
    history and per-call result JSON remain the authoritative account of what a
    run did, and nothing here is ever read back to make a decision.  Because it
    is cosmetic, every write is best effort: a failure to publish status must
    not end a job that is otherwise healthy.

    The file is replaced atomically, so a reader polling it never observes a
    partially written object and does not need to coordinate with the writer.
    Every update is written immediately rather than throttled: the update rate
    is bounded by physics rollouts, so one small atomic write per candidate
    costs nothing, and a throttle that drops the newest state is worse than no
    throttle at all.
    """

    def __init__(self, run_dir: str | Path, run_id: str, *, state: dict[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self._enabled = status_enabled()
        if state is None:
            self._state: dict[str, Any] = {
                "schema": STATUS_SCHEMA,
                "run_id": run_id,
                "state": "CREATED",
                "current_call": None,
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "calls": {name: {"status": "waiting"} for name in CALLS},
                "progress": None,
                "metrics": {},
                "last_event": None,
                "blockers": [],
            }
        else:
            self._state = state

    # ------------------------------------------------------------------ open

    @classmethod
    def attach(cls, run_dir: str | Path, run_id: str | None = None) -> RunStatus:
        """Continue an existing run's status, or start one if absent.

        Each of the five calls is a separate process invocation in normal use,
        so the status object is reconstructed from disk rather than passed
        between them.
        """

        directory = Path(run_dir)
        path = directory / STATUS_FILENAME
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(loaded, dict)
                and loaded.get("schema") == STATUS_SCHEMA
                and (run_id is None or loaded.get("run_id") == run_id)
                and isinstance(loaded.get("calls"), dict)
                and all(isinstance(entry, dict) for entry in loaded["calls"].values())
                and isinstance(loaded.get("metrics"), dict)
                and isinstance(loaded.get("blockers"), list)
            ):
                return cls(directory, str(loaded.get("run_id") or run_id or directory.name), state=loaded)
        except (OSError, ValueError):
            pass
        return cls(directory, run_id or directory.name)

    # --------------------------------------------------------------- mutators

    @contextmanager
    def observe(self, call: str):
        """Surface normal exceptions/interruption; always re-raise the original.

        Reload on failure because the call may have published newer progress,
        or explicitly marked an expected readiness block through another object.
        A killed process cannot run this handler; the watcher must detect staleness.
        """
        self.start(call)
        try:
            yield
        except BaseException as error:
            try:
                current = type(self).attach(self.run_dir, self.run_id)
                if current._state.get("calls", {}).get(call, {}).get("status") != "blocked":
                    current.fail(call, f"{type(error).__name__}: {error}")
            except Exception:  # noqa: BLE001 - preserve the original job exception
                _log.debug("could not report failed call", exc_info=True)
            raise

    @_best_effort
    def start(self, call: str) -> None:
        """Mark a call as running and make it the current stage."""

        entry = self._call(call)
        entry.clear()  # Remove old failure summary/end time on an explicit retry.
        entry["status"] = "running"
        entry["started_at"] = utc_now()
        self._state["blockers"] = []
        self._state["progress"] = None
        if call == "plan":
            self._state.pop("collection", None)
        if call == "fit":
            for name in ("baseline", "best"):
                self._state["metrics"].pop(name, None)
        self._state["current_call"] = call
        self._state["state"] = _RUNNING_STATE[call]
        self._transition_event(f"{call} started")
        self._flush()

    @_best_effort
    def finish(self, call: str, summary: str, **metrics: Any) -> None:
        """Mark a call as complete and record the one-line result a human reads."""

        entry = self._call(call)
        entry["status"] = "done"
        entry["summary"] = summary
        entry["ended_at"] = utc_now()
        if metrics:
            entry["metrics"] = jsonable(metrics)
            self._state["metrics"].update(jsonable(metrics))
        self._state["progress"] = None
        if self._state.get("current_call") == call:
            self._state["current_call"] = None
        self._state["state"] = _DONE_STATE[call]
        self._transition_event(f"{call}: {summary}")
        self._flush()

    @_best_effort
    def block(self, call: str, reason: str) -> None:
        """Record why a call cannot proceed.

        A blocked call is an expected outcome, not a crash, and the reason is
        the most useful thing on the screen when one occurs.
        """

        entry = self._call(call)
        entry["status"] = "blocked"
        entry["summary"] = reason
        entry["ended_at"] = utc_now()
        self._state["state"] = "BLOCKED"
        self._state["current_call"] = None
        self._state["progress"] = None
        blockers = self._state.setdefault("blockers", [])
        if reason not in blockers:
            blockers.append(reason)
        self._transition_event(f"{call} blocked: {reason}")
        self._flush()

    @_best_effort
    def fail(self, call: str, error: str) -> None:
        """Record an unexpected failure, distinct from a blocked readiness gate."""

        entry = self._call(call)
        entry["status"] = "failed"
        entry["summary"] = error
        entry["ended_at"] = utc_now()
        self._state["state"] = "FAILED"
        self._state["current_call"] = None
        self._state["progress"] = None
        self._transition_event(f"{call} failed: {error}")
        self._flush()

    @_best_effort
    def progress(self, done: int, total: int, *, unit: str = "candidates", **metrics: Any) -> None:
        """Publish counted progress for the current call.

        The fit loop calls this once per candidate, which is the resolution a
        human watching the run actually wants.
        """

        self._state["progress"] = {
            "call": self._state.get("current_call"),
            "done": int(done),
            "total": int(total),
            "unit": unit,
        }
        if metrics:
            self._state["metrics"].update(jsonable(metrics))
        self._flush()

    @_best_effort
    def event(self, text: str) -> None:
        """Append one line to the event stream and surface it as the latest line."""

        self._transition_event(text)
        self._flush()

    @_best_effort
    def collection(self, *, outcome: str, episodes: int, preview: dict[str, Any]) -> None:
        """Report the collection branch, never fitting readiness or robot approval."""
        self._state["collection"] = {
            "status": outcome,
            "command_files": episodes,
            "preview_status": preview.get("status"),
            "screen_passed": preview.get("screen_passed"),
            "fit_allowed": False,
            "real_execution_approved": False,
        }
        detail = preview.get("error") or preview.get("reason") or outcome.replace("_", " ")
        if outcome in {"design_failed", "generation_failed", "preview_failed"}:
            self.fail("plan", f"Evidence collection {outcome}: {detail}")
        elif outcome in {"needs_scene_setup", "controller_action_required", "evidence_action_required"}:
            self.block("plan", f"Evidence collection needs setup/review: {detail}")
        elif preview.get("screen_passed") is False:
            self.block("plan", f"{episodes} command files retained; simulation screening failed; review required")
        elif outcome not in {"commands_generated", "preview_pending", "preview_complete_review_required"}:
            self.block("plan", f"Unrecognized collection outcome {outcome!r}; inspect the collection record")
        else:
            summary = f"{episodes} collection command files; "
            if preview.get("status") in {"pending", "blocked", "skipped_explicitly"}:
                summary += "preview pending/skipped; screen and review before collecting real evidence"
            else:
                summary += "review collection artifacts, collect real evidence, then re-analyze"
            self.finish("plan", summary)
            self._state["state"] = "ACTION_REQUIRED"
            self._state["blockers"] = []
            self._flush()

    # ---------------------------------------------------------------- private

    def _transition_event(self, text: str) -> None:
        self._state["last_event"] = text
        self._append_event({"at": utc_now(), "run_id": self.run_id, "text": text})

    def _call(self, call: str) -> dict[str, Any]:
        if call not in CALLS:
            raise ValueError(f"unknown call {call!r}; expected one of {CALLS}")
        calls = self._state.setdefault("calls", {})
        return calls.setdefault(call, {"status": "waiting"})

    def _flush(self) -> None:
        if not self._enabled:
            return
        self._state["updated_at"] = utc_now()
        try:
            payload = json.dumps(jsonable(self._state), indent=2, sort_keys=True, allow_nan=False) + "\n"
            atomic_write_text(self.run_dir / STATUS_FILENAME, payload)
        except Exception:  # noqa: BLE001 - status must never end a healthy run
            _log.debug("could not publish run status", exc_info=True)

    def _append_event(self, record: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps(jsonable(record), sort_keys=True, allow_nan=False)
            with (self.run_dir / EVENTS_FILENAME).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # noqa: BLE001 - see _flush
            _log.debug("could not append run event", exc_info=True)
