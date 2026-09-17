# Flexiv MVP1 — Cartesian free-motion collection

**15 proposed motions, 390 seconds (6 min 30 s) per reviewed arm posture.**
Direct Flexiv RDK Python, as requested. No ROS installation is needed for this
collector. This is a separate, experimental collection adapter; it does not
change the five-call toolkit's existing joint-PD fitting support.

## Current status — read this first

- Motion files and offline tests: generated and checked locally.
- Real starting pose, controller settings and motion limits: **not provided**.
- Isaac Lab/Newton screening of these Cartesian files: **not performed**.
- Real robot/installed RDK integration: **not tested**. No robot was connected.
- Real execution: **blocked** until setup, simulation review and on-site approval.
- These are proposed commands, not real measurements or calibrated parameters.

The files are not safe-to-run simply because they pass software tests. A small
Cartesian displacement can cause large joint movement near a singularity.
The gripper, other arm links, cables, table and nearby people all matter.

## The motions

| Files | Motion | Purpose | Time |
|---|---|---|---|
| `00_stationary` | Hold the taught pose | Noise, drift and timestamp baseline | 6 s |
| `01`–`03` | X/Y/Z smooth reversals, proposed maximum ±20 mm | Translational tracking, direction reversal, settling | 3 × 24 s |
| `04`–`06` | Three rotational reversals, proposed maximum ±3° | Rotational response | 3 × 24 s |
| `07`–`12` | One axis at a time, 0.05–0.7 Hz chirps; ±6 mm / ±1° maximum | Frequency-dependent response, effective delay | 6 × 30 s |
| `13`–`14` | Different combined translation/rotation motions | Held-out validation; never use for fitting or selecting parameters | 2 × 30 s |

Actual amplitudes are tapered; each file begins and ends with a two-second
stationary segment. Files return to the same taught anchor as part of the
planned trajectory. On abort there is **no automatic return or retreat**.

Repeat at additional independently taught, screened and approved postures.
Do not assume one local Cartesian suite identifies every physical parameter of
all seven joints. Independent friction, inertia, actuator parameters and
redundant-joint dynamics may remain correlated or unexcited. Use real signal
quality plus simulation sensitivity to decide what is identifiable and which
additional experiments are needed. This suite does not exercise gripper closing,
contact, peg grasping or insertion. Keep the gripper fixed and remove the peg.

## First action on the robot workstation: capture setup without moving

Use a qualified operator and a compatible workstation with Python 3.10+ and
**Flexiv RDK Python 1.9.3**. Do not change robot firmware just to match this
script; a different installed SDK needs a source/API and integration review.
The RDK constructor connects to the robot; coordinate that connection with the
operator and do not run concurrently with another controller.

After the operator manually teaches an obstacle-free starting pose, run from
this folder, replacing the serial with the actual robot serial:

```sh
python collect_rdk.py snapshot --robot-serial YOUR_ROBOT_SERIAL --output setup_01
```

This uses read-only accessors. It does **not** enable the robot, change its mode,
move/home it, clear faults, change tools or zero sensors. It produces:

- `setup_01/snapshot.json`: measured pose/joints, timestamps, limits, active tool,
  payload settings and SDK/robot version information.
- `setup_01/profile.to_review.json`: a **draft**, never an approved setup.

It cannot read current Cartesian gains through these accessors. An operator or
controller engineer must supply the intended stiffness/damping, nullspace
weights, command rate, joint-name mapping, workspace and monitoring limits.
Nominal gains reported by `robot.info()` are NOT current controller gains.
The snapshot's joint mapping and clearance are not established automatically.

**Send the snapshot and draft profile back for review before moving the robot.**

## Offline files and binding

`proposal/commands/*.csv` contains **offsets from a fixed anchor**, not cumulative
deltas. Translation and rotation vectors use RDK's robot WORLD axes. The
controlled reference in these files is the flange. They are not Euler-angle
commands and must not be streamed directly to an RDK method.

```sh
python motion_suite.py inspect --bundle proposal
```

The supplied proposal uses **15 Hz as an unconfirmed starting point**. Regenerate
if the chosen direct-RDK deployment rate differs; do not relabel timestamps:

```sh
python motion_suite.py generate --rate 30 --output proposal_30hz
```

After human review of the captured setup, bind to a new output directory:

```sh
python motion_suite.py bind --proposal proposal \
  --profile setup_01/profile.to_review.json --output bound_posture_01
```

Binding requires explicit confirmations and complete numeric settings. It checks
the fixed anchor, active TCP transform, command derivatives and reviewed
workspace box. It is **not** inverse kinematics, collision or singularity
qualification. The bound files contain absolute flange poses with **xyzw**
quaternions; the runner converts the full flange-to-tool transform to RDK TCP
targets with **wxyz** quaternions. Commands are already metric and are not
multiplied by the policy's `action_blend_ratio` again.

## Before any execution

1. Qualify the exact bound bundle in the actual Isaac Lab/Newton scene, including
   self-collision, joint limits, singularities and complete swept volume. An
   old joint-PD screen cannot qualify these Cartesian motions.
2. Review the screen **and real workspace** with the robot operator. Populate
   the expiring `approval.template.json` without changing hashes. Software
   hashes detect changes; they do not authenticate the reviewer or prove safety.
3. Integration-test this new collector and stop behavior under site procedures.
   The approval has an explicit `collector_integration_tested` gate. Unit-test
   mocks are not acceptable evidence for that gate.

Expected screen record fields (actual screening software must produce these,
not someone fabricating a pass): `physics: "Newton"`, `surface: "Isaac Lab"`,
`passed`, `self_collision_enabled`, `joint_limits_checked`, `singularity_checked`,
`swept_volume_checked`, `manifest_sha256`, `profile_sha256`, `screened_episodes`.
These must cover the exact bound configuration. A Cartesian screening adapter
is not included or claimed by this delivery; consequently no execution approval
is included. Do not bypass this gate.

Offline inspection, with **no SDK import or robot connection**:

```sh
python collect_rdk.py run --bundle bound_posture_01 --episode 01_x_reversals
```

Only after the above review, the same command can accept `--execute`,
`--approval PATH_TO_COMPLETED_APPROVAL`, and `--output real_trials`.
It asks the operator to type the robot serial and episode name. It runs **one
episode only**; there is no unattended run-all flag.

The robot must already be operational, stopped, idle, at the taught Cartesian
pose **and joint posture**, with the exact active tool/payload. The runner then
switches to `NRT_CARTESIAN_MOTION_FORCE`, disables force-controlled axes, applies
only the explicitly approved impedance/nullspace settings, and records them.
It never enables, clears faults, homes, changes tool/payload or relaxes robot
safety settings. It requests `Stop()` after success or failure once it owns the
mode. No automatic gain restoration is claimed. Physical E-stop and configured
robot safety remain necessary, especially on process/network failure.

## What is recorded

- `commands.jsonl`: scheduled time, actual host send times, absolute flange
  target, actual RDK TCP target, quaternion convention, terminal velocity and
  wrench arguments. Zero terminal velocity is part of this NRT experiment.
- `feedback.jsonl`: native robot timestamps, host receipt clocks, joint position,
  velocity, torque, motor-side signals when available, flange/TCP poses,
  TCP velocity and external wrench. Samples are polled at best-effort ~200 Hz;
  actual times and duplicate suppression are recorded, not an assumed rate.
- `joint_feedback.csv`: convenient measured joint table. It is **not** a
  joint-position command dataset for the old joint-PD fitter.
- `record.json`: setup/command hashes, split, versions, sent count, status and
  stop result. An API return is not proof that every target was achieved.

Host and robot clocks are deliberately separate. Clock alignment and timing
quality must be evaluated before fitting latency. NRT internal smoothing is
part of the measured system; replaying bare OSC in Newton does not reproduce it
automatically. Cartesian replay/fitting still needs an appropriate simulator
adapter; the existing controller-discovery guard remains intact.

Direct RDK **bypasses** the Isaac ROS deployment bridge, policy inference and its
action decoder. This evidence can calibrate the direct-RDK command-to-motion
path, not by itself validate the full ROS policy deployment pipeline. If final
deployment uses ROS, test that path too with its actual frames, scales and timing.

## Reproduce offline checks

```sh
python -m pytest test_cartesian_collection.py
```

Tests use synthetic fixtures: reproducibility, frames/quaternions, bounded
commands, splits, tampering, setup/approval gates, read-only snapshot, stale
feedback, abort/stop and exact recorded command semantics. They do not run
Newton, RDK bindings or hardware.

API references: [pinned Flexiv RDK 1.9.3 robot interface](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.3/include/flexiv/rdk/robot.hpp),
[robot state fields](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.3/include/flexiv/rdk/data.hpp),
[tool interface](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9.3/include/flexiv/rdk/tool.hpp).
The vendor's contact-search example is not copied: this collector is free-motion
only and contains no force-control/contact-search sequence.
