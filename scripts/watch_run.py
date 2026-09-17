#!/usr/bin/env python3
"""Live view of a calibration run.

Reads the status file a run publishes and redraws the five calls until the run
reaches a terminal state.  It holds no connection to the job, so it can be
started late, stopped, restarted, or pointed at a directory mounted from
another machine.

    python3 scripts/watch_run.py runs/so101-8f2830f962
    python3 scripts/watch_run.py runs/so101-8f2830f962 --once
    python3 scripts/watch_run.py runs/latest --interval 2

Nothing here is authoritative.  The fit journal and the per-call result JSON
remain the record of what the run did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CALLS = ("analyze", "plan", "fit", "validate", "write")
TERMINAL = {"COMPLETE", "FAILED", "BLOCKED", "ACTION_REQUIRED"}
STALE_AFTER_S = 120.0

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
AMBER = "\033[33m"
RED = "\033[31m"

STATUS_STYLE = {
    "done": (GREEN, "done"),
    "running": (BLUE, "running"),
    "waiting": (DIM, "waiting"),
    "blocked": (AMBER, "blocked"),
    "failed": (RED, "failed"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory, or a status.json path")
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between reads (default 3)")
    parser.add_argument("--once", action="store_true", help="render a single frame and exit")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    return parser.parse_args()


def status_path(run: Path) -> Path:
    return run if run.suffix == ".json" else run / "status.json"


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def age_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds()


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


def bar(done: int, total: int, width: int = 34) -> str:
    if total <= 0:
        return ""
    filled = max(0, min(width, round(width * done / total)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render(state: dict[str, Any], *, color: bool) -> str:
    def paint(text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    lines: list[str] = []
    run_id = state.get("run_id", "unknown")
    run_state = state.get("state", "UNKNOWN")
    elapsed = human_duration(age_seconds(state.get("started_at")))
    since_update = age_seconds(state.get("updated_at"))

    freshness = f"updated {human_duration(since_update)} ago"
    if since_update is not None and since_update > STALE_AFTER_S and run_state not in TERMINAL:
        freshness = paint(f"STALE - no update for {human_duration(since_update)}", AMBER)

    lines.append(paint(run_id, BOLD))
    lines.append(f"{run_state}  ·  {elapsed} elapsed  ·  {freshness}")
    lines.append("")

    calls = state.get("calls", {}) or {}
    progress = state.get("progress") or {}

    for name in CALLS:
        entry = calls.get(name, {}) or {}
        status = str(entry.get("status", "waiting"))
        code, label = STATUS_STYLE.get(status, (DIM, status))
        summary = str(entry.get("summary", "") or "")
        if not summary and status == "waiting":
            summary = "-"
        lines.append(f"  {paint(label.ljust(8), code)} {name.ljust(9)} {summary}")
        if status == "running" and progress.get("call") == name:
            done = int(progress.get("done", 0))
            total = int(progress.get("total", 0))
            unit = str(progress.get("unit", ""))
            metrics = state.get("metrics", {}) or {}
            detail = f"{done} of {total} {unit}".strip()
            best = metrics.get("best")
            baseline = metrics.get("baseline")
            if isinstance(best, (int, float)):
                detail += f"  ·  best {best:.4f}"
            if isinstance(baseline, (int, float)):
                detail += f"  ·  baseline {baseline:.4f}"
            lines.append(f"           {bar(done, total)} {detail}")

    blockers = state.get("blockers") or []
    collection = state.get("collection")
    if isinstance(collection, dict):
        lines.append("")
        lines.append(
            f"  evidence collection: {collection.get('status', 'unknown')} · "
            f"{collection.get('command_files', 0)} command files · "
            f"preview {collection.get('preview_status', 'unknown')}"
        )
        lines.append("  Not fit-ready; no real-robot execution approval.")
    if blockers:
        lines.append("")
        lines.append(paint("  blocked:", AMBER))
        for reason in blockers:
            lines.append(f"    - {reason}")

    # A completed run's last event is mid-fit noise; a blocked or failed one
    # is still the most useful line on the screen.
    last_event = state.get("last_event")
    if last_event and run_state != "COMPLETE":
        lines.append("")
        lines.append(paint(f"  last: {last_event}", DIM))

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    path = status_path(args.run)
    color = not args.no_color and sys.stdout.isatty()

    while True:
        state = read_status(path)
        frame = (
            render(state, color=color)
            if state is not None
            else f"waiting for {path}\n\nNo status yet. The run publishes this file once analyse begins."
        )

        if args.once or not sys.stdout.isatty():
            print(frame)
            return 0

        sys.stdout.write("\033[2J\033[H" + frame + "\n")
        sys.stdout.flush()

        if state is not None and str(state.get("state")) in TERMINAL:
            return 0
        try:
            time.sleep(max(0.5, args.interval))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
