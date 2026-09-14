# Newton Calibration MVP1 — SO-101

This repository implements the first calibration product slice for a robot arm and gripper:

- real evidence: NVIDIA Anchor-Lab SO-101 50-motion Parquet logs;
- user surface: Isaac Lab configuration plus five Python calls;
- physics execution: kit-less Isaac Lab 3.0 with Newton/MuJoCo-Warp;
- fitting: a replaceable, bounded diagonal CMA-ES plug-in;
- validation: motions excluded from the fit objective, with explicit pass/fail gates;
- output: a portable setup-scoped package with source USD, provenance layer, Isaac Lab actuator config, manifest, report, and complete job history.

MVP1 calibrates free-space arm dynamics, command timing, and unloaded gripper motion. It does **not** claim contact fidelity, object dynamics, absolute gripping force, pick-and-place, or insertion transfer; those are outside this deliverable.

## Five-call Isaac Lab surface

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

The five functions are deterministic product APIs. A future calibration agent can guide or monitor these same calls without becoming a dependency of the core.

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
