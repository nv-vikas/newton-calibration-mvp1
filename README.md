# `newton.calibration` MVP1 — articulated robots

This repository implements the first calibration product slice for a supported
position-controlled robot articulation. It is **not tied to SO-101**. The
Anchor-Lab SO-101 data and USD remain the first real-data reference workflow.

The generic product path provides:

- a versioned real-evidence schema for arbitrary joint names and CSV/Parquet layouts;
- deterministic real-coordinate → USD-DOF mapping, including units, sign, scale and zero offset;
- user surface: Isaac Lab configuration plus five Python calls;
- physics execution: kit-less Isaac Lab 3.0 with Newton/MuJoCo-Warp;
- fitting: a replaceable, bounded diagonal CMA-ES plug-in;
- validation: motions excluded from the fit objective, with explicit pass/fail gates;
- output: a setup-scoped package with source USD, per-joint actuator patch, mapping, manifest, report, and complete job history.

MVP1 calibrates free-space articulation dynamics, command timing, and unloaded
end-effector motion. It does **not** claim contact fidelity, object dynamics,
absolute gripping force, pick-and-place, or insertion transfer.

“Any USD” means any supported Newton-compatible, position-controlled,
self-contained articulation for which the agent/user can confirm a one-to-one
joint profile, controller baselines, and safe parameter bounds. MVP1 blocks a
composed USD before fitting until dependency-closure vendoring is implemented.
Multi-axis joints, closed loops, tendons/mimic mechanisms, Cartesian
action pipelines, or unknown units/sign/zero conventions fail with actionable
blockers instead of being guessed.

## Generic five-call path

The agent performs a deterministic preflight before the five scientific calls:

```python
from newton_calibration import (
    JointBinding, LongFormSchema, SignalBinding, TabularJointEvidence,
    bind_evidence_files, inspect_tabular_evidence, propose_joint_mapping,
)
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning

# 1. Inspect the user's explicitly split real logs and the USD.
schema = LongFormSchema(
    time_column="timestamp_ns", time_unit="ns", value_column="value",
    joint_column="joint", signal_column="signal", field_column=None,
)
episodes = [
    {"name": "chirp-a", "path": "chirp-a.parquet", "split": "train", "trial_id": "capture-001"},
    {"name": "chirp-b", "path": "chirp-b.parquet", "split": "heldout", "trial_id": "capture-002"},
]
real = inspect_tabular_evidence(root="real-logs", episodes=episodes, schema=schema)
usd = tuning.inspect_usd("robot.usd")
proposal = propose_joint_mapping(real.source_joints, [joint.name for joint in usd.joints])

# 2. The agent records confirmed units/sign/offset. It never optimizes this map.
confirmed = (
    JointBinding(
        "driver_j1", "usd_joint_1", "rad", "rad",
        sign=1, offset=0.0, transform_confirmed=True,
    ),
    JointBinding(
        "driver_j2", "usd_joint_2", "rad", "rad",
        sign=-1, offset=0.12, transform_confirmed=True,
    ),
)
evidence_spec = bind_evidence_files(
    root="real-logs", episodes=episodes, schema=schema,
    joint_bindings=confirmed,
    signal_bindings=(
        SignalBinding("command", "command_q"),
        SignalBinding("position", "actual_q"),
        SignalBinding("velocity", "actual_dq"),
    ),
    # Set these only when collection metadata establishes the claim. Analyze
    # blocks delay/effort fitting rather than inferring either from tracking error.
    clock_synchronized=True,
    effort_saturation_joints=("driver_j1", "driver_j2"),
)
env = ArticulationEnvCfg(
    usd_path="robot.usd",
    robot_id="customer-arm-cell-a",
    joint_groups={"manipulator": ("driver_j1",), "end_effector": ("driver_j2",)},
    joint_map={item.source_joint: item.usd_joint for item in confirmed},
    profile_confirmed=True,
    controller_profile_confirmed=True,
    controller_profile_source="Isaac Lab task config commit abc123",
    # Per-joint Isaac Lab baselines and robot-specific armature/friction bounds
    # come from the USD, task config, OEM data, or explicit user confirmation.
    base_stiffness_by_joint={"driver_j1": 80.0, "driver_j2": 25.0},
    base_damping_by_joint={"driver_j1": 4.0, "driver_j2": 1.0},
    base_effort_limit_by_joint={"driver_j1": 15.0, "driver_j2": 4.0},
    parameter_bounds={
        "manipulator_effort_scale": (0.25, 1.0, 0.75),
        "manipulator_armature": (0.0, 0.5, 0.02),
        "manipulator_friction_nm": (0.0, 1.0, 0.05),
        "end_effector_effort_scale": (0.25, 1.0, 0.75),
        "end_effector_armature": (0.0, 0.1, 0.005),
        "end_effector_friction_nm": (0.0, 0.3, 0.01),
    },
)

# The stable product lifecycle remains five calls.
analysis = tuning.analyze(env=env, evidence=TabularJointEvidence(evidence_spec))
plan = tuning.plan(analysis)
fit = tuning.fit(plan)
validation = tuning.validate(fit)
package = tuning.write(validation, output="packages/customer-arm-cell-a")
```

`analysis.mapping_report` and `analysis.readiness` tell the calibration agent
exactly what it must resolve. The locked plan persists the evidence schema,
file hashes, split, joint order, affine coordinate transforms, USD mapping,
clock/saturation qualifications, recipe, bounds, optimizer and runtime
settings. The binding schema is `bound-evidence@2`; older bindings must be
recreated so independent trials and coordinate transforms are explicitly
confirmed. Resume refuses any drift. `tuning.inspect_evidence(...)` reports
measured dynamic-excitation and reversal coverage per source joint so the
agent can request a safer trajectory instead of tuning an unsupported term.

## SO-101 compatibility preset

```python
from newton_calibration.isaaclab import SO101EnvCfg, tuning

env = SO101EnvCfg(
    usd_path="data/anchor-lab/robot_assets/so101_no_camera_new_calib.usd",
    runtime="isaaclab_newton",
)

analysis = tuning.analyze(
    env=env,
    evidence="data/anchor-lab/data/so101_arm_50motion",
    evidence_revision="647edd5787cd764cdc041103ad282dc59214d919",
)
calibration_plan = tuning.plan(analysis, recipe="so101_actuator_dynamics.v1")
fit_run = tuning.fit(calibration_plan)
validation = tuning.validate(fit_run)
package = tuning.write(validation, output="packages/so101")
```

The five functions are deterministic product APIs. A calibration agent can
inspect, guide, execute, resume, and explain these same calls without becoming
a dependency of the core; Minjae's agent can use this surface and supply its
optimizer through the plug-in contract described below.

## Load a validated package

The package loader verifies activation status, both held-out baseline and
calibrated executions, the packaged USD hash, controller provenance, parameter
bounds and ownership, YAML/manifest consistency, runtime/solver settings, the
hashed job ledger, and command-delay quantization before returning a configuration.
It deliberately uses the relocatable USD copied into the package rather than
the original absolute asset path.

```python
from newton_calibration.isaaclab import SO101EnvCfg, VerifiedSO101Package

verified = VerifiedSO101Package.open(
    "packages/so101",
    # Recommended when the package crosses a trust boundary:
    expected_manifest_sha256="<digest from the producer>",
)
env_cfg = verified.to_env_cfg(device="cuda:0")

# Equivalent convenience form:
env_cfg = SO101EnvCfg.from_calibration(
    "packages/so101",
    device="cuda:0",
    expected_manifest_sha256="<digest from the producer>",
)
```

For robot-neutral v2 packages, use `VerifiedCalibrationPackage.open(...)` or
`ArticulationEnvCfg.from_calibration(...)`. A separately supplied manifest
digest is the trust anchor when a package crosses machines; the internal
runtime attestations establish cross-file and execution consistency, not a
cryptographic issuer identity.

All 11 values remain attached to `env_cfg.calibration_parameters`; this is a
serializable toolkit replay configuration, not a Gym environment. The Newton
replay adapter applies explicit-PD values, Newton armature/friction, and the
locked physics-step delay. See [`docs/package_loader.md`](docs/package_loader.md)
for the boundary and current task-environment limitation.

An actuator-level residual is available in MVP1 by setting `residual_model_path` on `SO101EnvCfg`. The current plug-in computes a bounded learned `Δtorque(error, velocity, command change, history)` and adds it to the Newton PD torque. It is off by default, fitted only after physical parameters, and must pass the same held-out validation before the package is activatable.

## What each boundary owns

| Component | Responsibility |
|---|---|
| Evidence adapter | Reads long-form Parquet, verifies joints/signals, aligns timestamps, resamples commands with zero-order hold and states by interpolation. |
| Recipe | Declares supported evidence, tunable parameters, bounds, objective weights, and validation gates. |
| Optimizer plug-in | Proposes parameter candidates and consumes scalar scores. It never calls Newton directly. |
| Experiment runner | Applies candidates, replays commands in Newton, captures traces, and calculates metrics. |
| Isaac Lab adapter | Keeps the core independent of backend imports and exposes a serializable environment spec. |
| Package writer | Records scope, provenance, candidate history, calibrated values, and held-out proof. |

## Optimizer plug-ins for Minjae's agent

`fit()` resolves its optimizer through a versioned registry. The built-in
`diagonal-cma-es` remains the default, while an installed package can expose
Minjae's optimizer through the `newton_calibration.optimizers` Python entry-point
group. The agent selects a registered optimizer when it creates the locked plan;
the optimizer proposes candidates, but the toolkit alone validates bounds, runs
Newton, computes scores, checkpoints state, and performs held-out validation.
Each completed generation is committed as an atomic, hashed record, so an
agent can safely resume after interruption without mixing plans or duplicating
candidate history.

```python
calibration_plan = tuning.plan(
    analysis,
    optimizer="minjae-nvopt.v1",
    optimizer_options={"strategy": "adaptive-search"},
)
fit_run = tuning.fit(calibration_plan)
```

The repository provides the integration contract and conformance tests, not
Minjae's proprietary implementation. Selecting an optimizer whose provider is
not installed fails explicitly and lists the available plug-ins. See
[`docs/optimizer_plugins.md`](docs/optimizer_plugins.md) for the provider-side
contract and packaging example.

## Dataset and asset policy

The adapter excludes `present_load_raw` and `tau_abs` from the primary objective because Anchor-Lab does not provide a complete load conversion and `tau_abs` discards sign. Joint position and velocity are treated as radians and radians/s, matching the released SO-101 files.

The released `so101_no_camera_new_calib.usd` is already calibrated. It is appropriate for integration bring-up, but not an unbiased “before” baseline. A product claim must compare against a declared untuned/canonical SO-101 asset or label the work as Newton-specific adaptation of the released asset.

The USD also authors a `10.0` drive `maxForce` but no reflected motor armature. Newton 1.2's native PD importer does not currently honor that `maxForce`. The adapter therefore uses Isaac Lab's explicit torque-clipped PD actuator on top of Newton physics and exposes armature plus arm/gripper saturation as tunable parameters. This avoids silently relying on an unsupported runtime field.

## Docker on Horde

`Dockerfile.kitless` is the preferred MVP1 path: it pins the open-source Isaac Lab `v3.0.0-beta2.patch1` commit and installs only Isaac Lab plus Newton on CUDA 12.8—no Isaac Sim runtime is required. `Dockerfile` remains as a parity image based on NVIDIA's official Isaac Lab container; that image requires separate acceptance of the Isaac Sim Additional Software and Materials License, and the repository deliberately does not bake acceptance into it.

```bash
docker build -f Dockerfile.kitless -t newton-calibration-mvp1:kitless .

# Download the pinned public evidence and asset.
docker run --rm \
  -v "$PWD/data:/workspace/data" \
  newton-calibration-mvp1:kitless fetch --output /workspace/data/anchor-lab

# Run the full Newton job on a GPU node.
./scripts/run_horde_mvp1_kitless.sh

# Exercise all five calls with real Newton on a non-activatable smoke slice.
docker run --rm --gpus all \
  --entrypoint /opt/venv/bin/python \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/runs:/workspace/runs" \
  -v "$PWD/packages:/workspace/packages" \
  newton-calibration-mvp1:kitless \
  /opt/newton-calibration/scripts/e2e_newton_smoke.py \
  --evidence /workspace/data/anchor-lab \
  --asset /workspace/data/anchor-lab/robot_assets/so101_no_camera_new_calib.usd \
  --output /workspace/packages/newton-e2e-smoke
```

The reference image can also exercise the robot-neutral runtime boundary
without pretending that Anchor-Lab proves every generic recipe parameter:

```bash
docker run --rm --gpus all --network none \
  --entrypoint /opt/venv/bin/python \
  -v "$PWD/data:/workspace/data:ro" \
  newton-calibration-mvp1:kitless \
  /opt/newton-calibration/scripts/e2e_generic_newton_boundary.py \
  --evidence /workspace/data/anchor-lab \
  --asset /workspace/data/anchor-lab/robot_assets/so101_no_camera_new_calib.usd
```

That check maps four real evidence coordinates onto non-contiguous runtime
DOFs, leaves two USD DOFs passive, runs episodes in both orders, and verifies
full-articulation reset plus runtime parameter readback. It is a runtime and
mapping qualification—not a second robot calibration and not an activatable
generic package. A generic five-call job still requires the evidence declared
by its recipe, including independent saturation evidence when effort scale is
tuned.

For a fast contract smoke test, pass `--runtime analytic --device cpu --generations 2 --population 4`. That backend tests the product wiring only; validation evidence is publishable only when `runtime=isaaclab_newton`.

## Validation gates

- simulation remains finite and stable;
- weighted held-out trajectory error improves by at least 30%;
- no held-out motion regresses by more than 10%;
- the validation list contains only held-out files;
- the package stores asset/evidence provenance and the complete search history.

## Current actuator surface

The recipe tunes arm and gripper stiffness/damping, reflected armature, Newton joint friction loss, separate arm/gripper effort saturation, and command delay. Geometry, link mass, COM, inertia, time step, substeps, and solver settings are fixed for MVP1 so the optimizer cannot hide asset or numerical problems inside an actuator fit.

## Paused experimental extensions

Earlier grasp, contact, and peg-insertion experiments remain in the repository
for provenance. They are paused and are not part of this MVP1 deliverable. No
result under `output/peg_insertion_*`, `output/so101_controller_mvp2_*`, or the
peg-insertion task modules is used by the calibration package or its claims.

## Verified Horde result

On a Horde GPU node, image
`sha256:5609e3bf9e3164bfe5459c3255d6b3dde93a9a7c68ac7fbe8a9c6cbd6fb29a8b`
executed all five calls with the real `isaaclab_newton` backend against the
pinned Anchor-Lab evidence fingerprint
`a193a700aca2511b59f6a604df1490635e242c8de53b3846afe73e72680e1fd0`.

Run `so101-8f2830f962` used the unmodified production recipe: four 12-second
training recordings, 144 CMA-ES candidates, and four different 12-second
held-out recordings. All 144 candidates were stable. Aggregate held-out
weighted error fell from `1.9154` to `0.1513` (`92.1%` lower); every held-out
episode improved, so all declared gates passed. The package is activatable only
for its recorded scope: **SO-101 free-space arm and unloaded-gripper trajectory
tracking**. It does not validate grasp force, contact, insertion, policy transfer,
or use on a different physical robot.

In the verified working copy, the full run is under
`runs/mvp1-full-20260913/so101-8f2830f962`, the package is under
`packages/so101-mvp1-full-20260913`, and the audit-facing classification is
under `deliverables/mvp1_20260913/truth_report.md`. Those evidence, run,
package, and video artifacts are intentionally excluded from the source
repository; the scripts recreate them from the pinned public dataset and USD.
