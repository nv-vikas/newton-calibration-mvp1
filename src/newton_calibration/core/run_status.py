from __future__ import annotations

import json
import logging
import os
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
            if isinstance(loaded, dict) and loaded.get("schema") == STATUS_SCHEMA:
                return cls(directory, str(loaded.get("run_id") or run_id or directory.name), state=loaded)
        except (OSError, ValueError):
            pass
        return cls(directory, run_id or directory.name)

    # --------------------------------------------------------------- mutators

    def start(self, call: str) -> None:
        """Mark a call as running and make it the current stage."""

        entry = self._call(call)
        entry["status"] = "running"
        entry.setdefault("started_at", utc_now())
        self._state["current_call"] = call
        self._state["state"] = _RUNNING_STATE[call]
        self._flush()

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
        self._flush()

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
        blockers = self._state.setdefault("blockers", [])
        if reason not in blockers:
            blockers.append(reason)
        self._flush()

    def fail(self, call: str, error: str) -> None:
        """Record an unexpected failure, distinct from a blocked readiness gate."""

        entry = self._call(call)
        entry["status"] = "failed"
        entry["summary"] = error
        entry["ended_at"] = utc_now()
        self._state["state"] = "FAILED"
        self._flush()

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

    def event(self, text: str) -> None:
        """Append one line to the event stream and surface it as the latest line."""

        self._state["last_event"] = text
        self._append_event({"at": utc_now(), "run_id": self.run_id, "text": text})
        self._flush()

    # ---------------------------------------------------------------- private

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
