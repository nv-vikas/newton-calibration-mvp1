# Generic articulation MVP1

## Product boundary

MVP1 is a recipe for a supported joint-position-controlled articulation, not a
robot model. SO-101 is a reference profile. A Flexiv, Franka, UR, or another
arm follows the same five calls after its USD, controller baselines, real
signals, joint mapping, and safe bounds pass preflight.

```text
USD inspection ─┐
                ├─> agent proposes profile + mapping ─> user/metadata confirms
real-data scan ─┘                                      │
                                                       v
analyze -> plan -> fit -> validate -> write package
             locked mapping │       held-out only
                            v
                  Newton candidate replays
```

The agent is not inside the numerical loop. It calls the same deterministic
APIs as a developer, explains blockers, records confirmations, starts or
resumes the durable job, and summarizes the validation result.

## What is mapped

For every controlled coordinate, the package locks:

```text
real source name -> USD/runtime DOF
position_usd = sign × scale × unit_convert(position_real) + zero_offset
velocity_usd = sign × scale × unit_convert(velocity_real)
```

The mapping order is also locked; array position is never used to guess joint
identity. Commands use zero-order hold. Measured position and velocity use
interpolation only in their common time range.

## What the agent may automate

- enumerate revolute and prismatic joints from a USD (the current free-space
  recipe accepts mapped revolute joints only);
- inventory real-data columns, joints, signals, rates and explicit splits;
- screen training episodes for per-joint motion excitation and bidirectional
  reversal, without inspecting held-out episodes;
- auto-accept unique exact or punctuation/case-normalized name matches;
- propose a robot profile and evidence manifest;
- select a supported recipe/optimizer and operate all five calls;
- request missing information and resume after it is supplied.

## What the agent must not guess

- joint direction or zero offset;
- radians versus degrees, or linear versus angular coordinates;
- which joints are arm, gripper, passive, coupled, or safety-excluded when the
  USD/profile is ambiguous;
- absolute armature/friction bounds or unsafe effort limits;
- controller stiffness, damping, and effort baselines or their provenance;
- whether command/state clocks were synchronized or an effort limit was
  actually reached;
- Cartesian action transformations that live in an Isaac Lab task/controller.

These become typed readiness blockers. Mapping is preflight configuration, not
an optimizer variable.

## Evidence qualification

The generic actuator recipe does not treat the presence of columns as proof
that every parameter can be fit. `inspect_evidence()` measures per-joint
command/response excursion and velocity reversal on training episodes only.
Delay additionally requires `clock_synchronized=true`. Effort-scale fitting
requires an explicit `effort_saturation_joints` declaration backed by an
independent saturation flag or effort measurement; tracking error alone is not
accepted as saturation evidence. These checks qualify a parameter for the
recipe—they are not a mathematical identifiability proof.

## Output

Generic runs write `newton.calibration.package/v2` with:

- source USD and its fingerprint;
- robot profile and canonical joint order;
- complete bound evidence spec and real-to-USD mapping fingerprint;
- ordered per-joint actuator patch and quantized command delay;
- validation result and complete candidate/job ledger;
- optional actuator residual;
- an explicit claim boundary excluding contact and task transfer.

`VerifiedCalibrationPackage.open()` dispatches legacy SO-101 v1 or generic v2.
`ArticulationEnvCfg.from_calibration()` returns the verified replay surface used
by the Newton adapter.

MVP1 currently packages only a root layer whose complete USD dependency closure
is proven self-contained. Composed production assets fail in `analyze()`—before
an optimization job starts—until dependency vendoring is implemented.

## Runtime qualification still required

Lightweight USDA/OpenUSD inspection helps the agent prepare the job. Before a
generic package should be activated, the Isaac Lab/Newton runtime must import
the articulation, resolve every mapped DOF in canonical order, apply the
actuator values, and verify explicit-controller plus Newton model-backed
armature/friction readback. Fit and held-out baseline/calibrated executions are
bound to distinct episode identities and result fingerprints. The CPU analytic
backend proves contract wiring only and always writes a non-activatable result.
