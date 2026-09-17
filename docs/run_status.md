# Run status

A calibration run publishes a small, human-facing status file so its progress can be
watched while it executes. A fit over a real recipe is hours of physics; nobody should
have to tail a log to find out which of the five calls it is on.

**This is a display surface, never a record.** The fit journal, candidate history, and
per-call result JSON remain the authoritative account of what a run did. Nothing reads
the status file back to make a decision, and a failure to publish it never ends a run
that is otherwise healthy.

## Files

Both live in the run directory, alongside `analysis.json`, `plan.json` and the rest.

| File | Contents |
|---|---|
| `status.json` | current state, replaced atomically on every update |
| `events.jsonl` | append-only line per event, for detail after the fact |

`status.json` is replaced with a same-directory temp file and `os.replace`, so a reader
polling it never observes a partial object and does not need to coordinate with the
writer.

## Watching a run

```bash
python3 scripts/watch_run.py runs/so101-8f2830f962
```

```
so101-8f2830f962
FITTING  ·  1h 12m elapsed  ·  updated 4s ago

  done     analyze   16 parameters admitted from 4 train and 4 held-out episodes
  done     plan      locked · 4 fit / 4 sealed · recipe so101_actuator_dynamics.v1
  running  fit
           [#####################.............] 87 of 144 candidates  ·  best 0.1641  ·  baseline 1.9154
  waiting  validate  -
  waiting  write     -

  last: candidate 87 stable, score 0.1664
```

The watcher holds no connection to the job. Start it late, stop it, restart it, or point
it at a directory mounted from another machine — it only reads a file.

```bash
# one frame, for a script or an agent poll
python3 scripts/watch_run.py runs/so101-8f2830f962 --once

# a run on a remote node
ssh newton-tune 'cat /workspace/runs/so101-8f2830f962/status.json'
```

It exits on its own when the run reaches `COMPLETE`, `FAILED`, `BLOCKED` or `ACTION_REQUIRED`, and flags the
status as stale if nothing has updated for two minutes while the run claims to be active.

`plan`, `fit`, `validate` and `write` publish their running state on entry. Normal
exceptions and Python interruptions mark the call failed and then re-raise the
original exception. Status-publishing failures never replace the job's exception.
An explicit fitting-readiness rejection stays `BLOCKED`, not `FAILED`. A retry
clears the previous attempt's failure summary and obsolete progress.

An OS kill, machine loss or power failure cannot execute a Python failure handler;
these still require the watcher's stale warning and authoritative run logs. A long
individual rollout can also be stale without having crashed. Analysis currently
creates its run ID/status after inventory; failures before that point have no run
status file and remain ordinary API exceptions.

## Shape

```json
{
  "schema": "newton.calibration.run-status/v1",
  "run_id": "so101-8f2830f962",
  "state": "FITTING",
  "current_call": "fit",
  "started_at": "2026-09-16T17:41:02+00:00",
  "updated_at": "2026-09-16T18:53:19+00:00",
  "calls": {
    "analyze":  {"status": "done",    "summary": "16 parameters admitted ...", "ended_at": "..."},
    "plan":     {"status": "done",    "summary": "locked · 4 fit / 4 sealed ...", "ended_at": "..."},
    "fit":      {"status": "running", "started_at": "..."},
    "validate": {"status": "waiting"},
    "write":    {"status": "waiting"}
  },
  "progress": {"call": "fit", "done": 87, "total": 144, "unit": "candidates"},
  "metrics": {"baseline": 1.9154, "best": 0.1641},
  "last_event": "candidate 87 stable, score 0.1664",
  "blockers": []
}
```

Call status is one of `waiting`, `running`, `done`, `blocked`, `failed`.

`blocked` and `failed` are different on purpose. A blocked call is an expected outcome —
readiness gates that did not pass, or a gate that rejected the candidate — and its reason
is the most useful thing on the screen when one occurs. A failed call is an exception.

A blocked run looks like this when fitting is explicitly requested before its
readiness requirements are satisfied:

```
flexiv-rizon4s-grav-mvp1-32288a455c
BLOCKED  ·  updated 3s ago

  done     analyze   7 parameters proposed; readiness incomplete
  blocked  plan      readiness checks failed: requested_parameters_identifiable, ...
  waiting  fit       -
  waiting  validate  -
  waiting  write     -
```

## Missing evidence: collection is a valid planning outcome

`analyze` reports incomplete **fitting** readiness but leaves `plan` waiting.
It does not prematurely block a valid evidence-collection route. While `plan`
generates commands or runs a preview, it shows `PLANNING`. On return:

| Collection result | Headline state |
|---|---|
| Commands generated; preview missing or explicitly skipped | `ACTION_REQUIRED`: screen/review before real collection |
| Preview completed, operator review/real evidence still needed | `ACTION_REQUIRED` |
| Scene/controller setup or other evidence action required | `BLOCKED`, with the actual reason |
| Simulation screening rejected the proposed motion | `BLOCKED`; command/video artifacts retained |
| Motion generation, dynamics design or preview failed | `FAILED`; original collection record retained |

The status includes a `collection` object with command count, collection/preview
outcome, screening result when available, and explicit `fit_allowed: false` and
`real_execution_approved: false`. The watcher displays this branch beneath the
five calls. A successful collection plan is **not** a calibrated package or
permission to run hardware. Collect real evidence and re-analyze before fitting.

The separate experimental Cartesian RDK collection scripts are not automatically
instrumented by this five-call status surface.

## For agents

An agent driving a run should not hold it open inside a single tool call. A multi-hour
fit that way reports nothing until it ends, and loses the thread entirely if the agent
dies. Run the job detached, then poll:

```bash
python3 scripts/watch_run.py runs/<run-id> --once
```

That returns one frame and exits, which is the whole summary in a form an agent can
report verbatim. Read `events.jsonl` only when the status shows a failure or has gone
stale.

Because status lives on disk rather than in the agent's memory, a different process — or
a different agent — can pick up and report on a run it did not start.

## Disabling

```bash
NEWTON_CALIBRATION_STATUS=0 python3 ...
```

No status file is written. Nothing else changes.
