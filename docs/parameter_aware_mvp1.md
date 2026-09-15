# Parameter-aware MVP1: analyze → collect → calibrate

## What changed

`analyze` now records **which requested parameters need which evidence**.
`plan` creates free-motion candidates from those gaps. With a bound Newton probe
it now **tests and refines the collection campaign**: frequency, amplitude and
permitted posture variants are scored and selected under a finite budget.
Without a probe it exports seed recipes but reports `needs_dynamics_probe`;
it does not claim motion search or parameter coverage is complete.

The five calls remain `analyze`, `plan`, `fit`, `validate`, `write`. Collection
is one possible output of `plan`, not a sixth scientific call. An optional agent
uses these same deterministic APIs; `assist` is a convenience function, not an
LLM or a claim that Minjae's agent is installed.

| Requested parameter | Selected training experiment | Important qualification |
|---|---|---|
| Stiffness and damping scales | Frequency sweep + smooth transitions with settling holds | Shared experiments are deduplicated. Effective controller response, not necessarily OEM gains. |
| Joint friction | Slow bidirectional reversals | Identifies a proxy unless controller/dynamics or effort is independently anchored. |
| Armature | Acceleration-focused sweep | Encoders at one pose cannot uniquely separate controller gain, link inertia and reflected inertia. |
| Command delay | Frequency sweep with synchronized command/feedback timestamps | Clock offsets and smoothing are separate issues. Existing dynamic data with missing synchronization triggers a clock-evidence action, not more of the same motion. |
| Effort scale | No saturation-seeking experiment | Requires existing approved effort/saturation evidence and an anchored interpretation; never deliberately saturate hardware. |

All generated trajectories contain endpoint holds and obey the **declared
simulation** position, velocity and acceleration envelope. Amplitudes may be
reduced to respect that envelope; actual ranges and scaling are recorded. Two
distinct held-out trajectories are reserved when new motion coverage is needed.
Their outcomes are not used to select or optimize training motions.

## User / agent example

```python
from newton_calibration.collection import CalibrationRequest
from newton_calibration.isaaclab import tuning

# env is a supported ArticulationEnvCfg or a surface with describe().
# The profile declares logical source joints, USD mappings, controller priors
# and one group per joint if independent per-joint parameters are desired.
request = CalibrationRequest(
    target_parameters=("joint1_stiffness_scale", "joint1_damping_scale", "joint1_friction_nm"),
    # Optional declaration, NOT evidence that measurements already exist.
    available_signals=("command_q", "actual_q", "actual_dq"),
    capability_source="Operator-confirmed feedback API",
)
analysis = tuning.analyze(env=env, evidence=existing_logs_or_none, request=request)

# The scene supplies MotionSpec with USD coordinates, initial pose, limits,
# amplitude/speed/acceleration proposals, and a source for those assumptions.
result = tuning.plan(
    analysis, collection=scene_motion_spec, preview=scene_newton_preview,
    design_probe=scene_dynamics_probe,
)  # Video is on by default when this produces a collection plan.
```

With no real evidence, this produces commands and a requested scene preview.
With partial evidence, missing motion coverage is targeted; existing eligible
training measurements are reused. With enough evidence and confirmed mapping,
controller profile and selected parameter bounds, it produces a fitting plan.
If only instrumentation, clock documentation or unsupported evidence is missing,
the result records that action instead of pretending another motion fixes it.
Use `intent="fit"` to require fitting or `intent="collect"` to inspect collection
needs explicitly. Changing the target scope requires a new `analyze` run.

After the operator reviews the proposal and real data is collected, run
`analyze` again with the same target selection and properly bound real logs.
Then `plan → fit → validate → write` produces a scoped result. The manifest
identifies only the parameters actually fitted; other settings use the locked
runtime profile's baselines, not inferred real-robot values. The current runtime
baseline assumes zero joint friction and zero command delay when those terms
are not selected; verify these assumptions before using a narrow recipe.

The generic `articulation.position_pd.free_space@3` recipe is now the default.
It retains **complete episodes**, including holds, in fitting and validation.
Explicit historical v1/v2 recipes and the SO-101 compatibility preset retain
their original duration behavior. Explicit parameter subsets require v3.

## What the user receives before collection

- `analysis.json`: USD/profile checks, real-data eligibility and requested scope.
- `evidence_needs.json`: per-parameter signals, missing joint coverage, limitations.
- `experiment_design.json`: selected recipe IDs, target parameters, reasons,
  required signals, held-outs and any budget-deferred experiments.
- `COLLECTION_PLAN.md`: readable parameter/evidence table.
- `design_search.json`: considered candidates, information gain, decisions,
  range/noise assumptions, predicted coverage, remaining work and stopping reason.
- `design_candidates/commands/`: candidates, including rejected ones. Only the
  top-level `commands/` directory is the selected collection campaign.
- `design_probes/` and `*.predictions.npz`: sensitivity matrices and underlying
  **simulated** q/dq responses with hashes; these are not real evidence.
- `commands/*.csv`: timestamped position, velocity and acceleration proposals.
- `command_plan.json`: links exact command hashes to the analyzed USD and analysis.
- `collection_plan.json`: preview lifecycle and artifact fingerprints.
- Actual Isaac Lab/Newton video + screening report when a working scene adapter
  is bound. No renderer yields `preview_pending`; it never produces a fake video.

An existing collection run is not overwritten: create a new analysis/revision.
The default budgets are 96 candidate probes and 64 selected training experiments.
With five active parameters, a candidate takes 13 Newton rollouts (two anchors,
five perturbations per anchor, and a repeat). Configure limits explicitly in
`CalibrationRequest`. Budget exhaustion is never reported as complete coverage.
Adaptive-search records survive interruption, but automatic mid-search resume
is not implemented in this slice: a new analysis revision reruns the search.
Do not confuse a durable audit log with checkpoint/resume of the GPU experiment.

## How adaptive motion design works

1. Start broad across joints with sweeps, reversals, acceleration sweeps and
   settling. Reserve held-outs first; never use them to score or select motions.
2. Vary frequency, amplitude and permitted posture offsets. Posture transitions
   are smooth out-and-back moves **inside the exported CSV**, not hidden resets.
3. Replay each candidate at two declared range anchors, perturb each relevant
   parameter and measure Newton q/dq changes. Repeat a baseline to check stability.
4. Scale sensitivities by assumed measurement noise. Keep correlated effects in
   the information matrix. Retain useful information gain; reject duplicates and
   low-gain motions. Record all decisions and failures.
5. Stop at predicted coverage for probe-supported targets, finite catalog
   exhaustion, or explicit limits. None of these proves real identifiability.
6. Replay **all selected command files**, including untouched held-outs, in the
   Newton scene. Video labels show family, joints, target parameters, command
   excursion and simulated response at 1× speed.

`predicted_coverage_reached` concerns probe-supported targets only.
`all_requested_parameters_covered` stays false if any requested parameter lacks
a probe range or requires other evidence/instrumentation. Failed probes do not
count as successful exhaustion. No saturation-seeking motion is generated.

`FiniteDifferenceProbe` accepts a prediction backend and sourced `ParameterSpec`
ranges. These are simulation hypotheses, separate from approved fitting bounds.
`IsaacLabPredictionBackend` binds the contract to a running Newton articulation
and restores actuator parameters after each probe. It uses full command files
by default. Optional `probe_window_s` uses a center diagnostic window, resets at
its first command and records that limitation—not full-trajectory sensitivity.

Noise, strength, separation and minimum-gain thresholds are explicit design
assumptions, not measured sensor specifications. Coverage is conditional on other
joint groups and assumed model form. Two range anchors are not an exhaustive
nonlinear sweep or a global multi-joint identifiability certificate.

## Modular boundaries — extensible, MVP1 only

| Module | Responsibility | Extension boundary |
|---|---|---|
| `collection/contracts.py` | Calibration request and experiment specification | Runtime-independent, versionable data contracts |
| `collection/registry.py` | Explicitly installed domain catalogs | A domain provides evidence assessment and experiment selection |
| `collection/mvp1.py` | Free-motion parameter → evidence → experiment mapping | The only implemented domain catalog |
| `collection/generators.py` | Versioned normalized trajectory generators | Trusted code registration; no code executed from evidence/recipe files |
| `collection/planning.py` | Envelope enforcement, CSVs, hashes, durable result | Reuses catalog and generator contracts, independent of robot brand |
| `collection/adaptive.py` | Candidate variants, sensitivity scoring, durable ledger | Uses installed catalog; no hardware driver |
| `collection/sensitivity.py` | Finite differences, repeatability, saved simulated traces | Runtime-neutral prediction backend contract |
| `collection/isaaclab_probe.py` | Apply/read back hypotheses and simulate responses | Adapter for pinned Isaac Lab 3 beta2/Newton scene |
| `collection/isaaclab_preview.py` | Replay + screening in an already running Newton scene | Scene/application owns asset setup, camera and reviewed envelope |
| Existing fitting/runtime/package layers | Optimizer candidates, Newton runs, held-out comparison, package | Same five-call pipeline; target selection persists into the package |

Only catalog and generator registrations for **MVP1 free motion** are shipped.
There is **no new MVP2/MVP3 grasp, contact, object or insertion implementation**.
Existing experimental directories and historical Flexiv bundles are preserved.
Future domains will need their own evidence contracts, parameter bindings,
catalogs, generators and validation gates; they are not automatically supported
just because an extension interface exists.

## Not claimed

This is bounded sensitivity-guided design, **not** mathematical identifiability
certification, completed calibration, exhaustive exploration of every possible
trajectory, or automatic instrumentation discovery. A USD does not tell us
the real robot's driver mapping, payload, controller configuration, accessible
signals or safe workspace. Those remain explicit review requirements.

Simulation replay is not hardware safety certification. Generated commands and
Newton traces are not real evidence. A fit to held-out joint motion does not
prove grasping or insertion transfer. No hardware execution is performed.
