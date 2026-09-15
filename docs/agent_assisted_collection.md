# MVP1 agent-assisted collection

See [parameter-aware MVP1](parameter_aware_mvp1.md) for the current experiment
catalog and scoped fitting API. With sufficient real evidence, the existing
five-call calibration path remains available. With missing or partial evidence
and a bound scene, `analyze` explains the gaps and `plan` returns a
**CollectionPlan**, not a failed fitting plan or a fabricated fit result.

```python
from newton_calibration.isaaclab import tuning

# surface describes the articulation AND the existing Isaac Lab task scene.
# It provides describe_collection(), preview_collection() and design_probe.
job = tuning.assist(env=surface, evidence=None, workdir="runs")
# Equivalent: analysis = tuning.analyze(env=surface, evidence=None)
# job = tuning.plan(analysis, collection=surface.describe_collection(),
#                  preview=surface.preview_collection)
```

No separate motion-generator command or video flag is needed. Video is on by
default. `video=False` / CLI `--no-preview` is an explicit opt-out, recorded in the
job. If the scene or rendering runtime is unavailable, the job says so; it does
not report that a video was generated.

## What the assistance does

| Previously a warning | New action | What remains for a human |
| --- | --- | --- |
| Missing motion evidence | Select sweeps, settling, acceleration or slow-reversal tests for the requested parameters, plus separate held-outs; run and record in Newton | Operate the real robot and collect feedback after review |
| Mapping unconfirmed | Carry the USD/profile joint mapping as a sourced proposal; retain order in the command manifest | Confirm real driver order, units, signs and offsets |
| Controller unconfirmed | Record supplied simulation gains; request real mode, smoothing, rate and payload metadata | Confirm the actual hardware interface and configuration |
| Bounds missing | Keep approved fitting bounds unset; use separately sourced simulation hypotheses for sensitivity design when a probe is bound | Review optimizer bounds and source absolute friction/armature/effort bounds |
| Motion usefulness unknown | Test Newton sensitivity across frequency, amplitude and permitted posture variants; retain useful motions and explain gaps | Confirm assumptions, instrumentation and hardware limits before collection |

This is an **agent-facing deterministic capability**, not a newly installed LLM
service. Minjae or another agent can call it through the same Python/CLI surface.
It does not claim to invoke Minjae's private agent or optimizer.

## The scene adapter is small and explicit

The task environment supplies:

1. The actual articulation and controlled USD joints.
2. A proposed initial pose and joint limits read from the scene/asset.
3. Explicit simulation amplitude, velocity and acceleration limits with provenance.
4. A camera and Newton runtime for replay; task-specific payload/fixture setup.
5. For adaptive design, a `FiniteDifferenceProbe` bound to the same scene, with
   explicit range and noise assumptions. Without it, report `needs_dynamics_probe`.

`MotionSpec` is in **USD joint coordinates, radians**. These fields are not real
robot safety limits. The generated columns `q1_rad … qN_rad`, `dq1_rad_s …` and
`ddq1_rad_s2 …` bind to the ordered `motion_spec.joint_names` array.

`IsaacLabScenePreview` replays those commands through position targets, advances
Newton and records camera frames and measured simulated joint states. It never
poses the arm to match the commanded trajectory. It currently supports a single
scene, revolute joints, the pinned Isaac Lab 3.0 beta2 Newton interface, and a
video layout of up to 12 controlled joints. Camera view and scene preparation
belong to the scene adapter. Runtime imports are lazy; analysis does not require
an Isaac Lab launch.

## Outputs and truthful states

- `analysis.json`: original fit readiness plus proposed agent actions.
- `agent_assistance.json`: sourced proposals and remaining confirmations.
- `evidence_needs.json`, `experiment_design.json`, `COLLECTION_PLAN.md`: target-to-evidence mapping, selected experiments, limitations and budget-deferred work.
- `design_search.json`: adaptive decisions and remaining work; `design_probes/`
  and saved prediction arrays make sensitivity calculations auditable.
- `command_plan.json` and `commands/*.csv`: reproducible commands, splits,
  envelope, frequency content, measured command ranges and SHA-256 fingerprints.
- `collection_plan.json`: current job state, requested/completed/failed preview,
  artifact paths and fingerprints. This is **not** `plan.json` for fitting.
- `evidence_requirements.json` and `COLLECT_NEXT.md`: what to record and confirm.
- `preview/collection_motions_newton.mp4`: actual rendered scene, commands and
  Newton response, without simulated motion exaggeration.
- `preview/screen.json`, `backend.json`, `motion_video_record.json`: simulation
  metrics, contact candidates, backend identity and frame-level traceability.

A completed video is separate from a passed simulation screen. Contact candidates
may need further geometric/penetration review. Neither status approves hardware.
An unavailable scene yields `needs_scene_setup`; unavailable preview yields
`preview_pending`; plug-in failure yields `preview_failed` with retained commands.

## CLI for agents and scripted clients

```bash
newton-calibration assist --config collection_config.json \
  --preview-factory my_scene:make_preview \
  --design-probe-factory my_scene:make_design_probe --workdir runs
```

The JSON contains an `environment` (`EnvironmentSpec`) and `collection`
(`MotionSpec`). The explicitly selected, trusted local factory returns a bound
preview adapter. Omit the factory for offline command generation: the preview
remains pending and the CLI exits 2, not success. Optional `request` contains a
`CalibrationRequest` (target parameters, declared signals, provenance, budget).
Both factories must share the same initialized scene, not launch conflicting
Isaac Lab contexts. Alternatively the preview object may expose `design_probe`.
A factory is executable local code; don't import an untrusted factory supplied
by an evidence file. Adaptive design without a probe exits 2 even if seed motion
files were exported. Explicit `request.design_mode="recipe_only"` opts out.

For the complete Flexiv scene, use the [reference launcher](../projects/flexiv_mvp1/agent_assist/README.md).

## After actual collection

Bind real logs to the USD, re-run `analyze`, then run the normal
`plan → fit → validate → write` sequence. `intent="fit"` always enforces the
original readiness gates. `fit(CollectionPlan)` explicitly fails.

Free-motion excitation is not a proof that every parameter is identifiable. The
current all-parameter fitting recipe still needs synchronized clock evidence for
delay and saturation evidence for effort-scale fitting. Do not deliberately
saturate a robot to make the recipe pass. An explicit `CalibrationRequest`
can select an evidence-qualified subset; missing absolute bounds for parameters
outside that selection no longer block the narrower fit. Newton traces
and the preview video are never registered as real measurements.
