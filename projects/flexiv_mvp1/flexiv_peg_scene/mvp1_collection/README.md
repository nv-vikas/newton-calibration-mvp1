# Flexiv MVP1 — collect real arm free-motion evidence

This package contains **reference commands**, not real measurements or tuned parameters.
The current evidence count is zero. The Isaac Lab environment now selects **Newton**
by default; the earlier PhysX video is historical, not proof of Newton behavior.

**Newton screening result:** all nine trajectories passed at a 1/960 s physics and
controller update interval. Maximum tracking error was 0.01318 rad and maximum joint
speed was 0.05519 rad/s, with no detected moving-robot contact with table/fixture.
This qualifies the simulated arm replay only. Robot self-collision/cable clearance and
the real setup still require operator review. The separate peg-hold test has residual
slip and remains unqualified; these commands are for an **unloaded** arm.

## What you receive

| File | Purpose |
|---|---|
| `analysis.json` | Actual toolkit `analyze` result on the Flexiv USD, with missing-data gates |
| `fit_plan_status.json` | Actual `plan` attempt: correctly blocked pending evidence and profile confirmation |
| `collection_plan.json` | Data-collection plan, parameter scope, units, proposed mapping and command hashes |
| `commands/train_joint1_reversals.csv` … `train_joint7_reversals.csv` | Seven separate 16-second, individual-joint reference runs, at most ±2° |
| `commands/heldout_multijoint_1.csv`, `heldout_multijoint_2.csv` | Two independent 20-second combined-motion reference trials, at most ±1° per joint |
| `../scripts/collect_flexiv_rdk.py` | Offline inspection plus explicitly gated, one-trial RDK execution and telemetry logging |
| `../scripts/screen_free_motion.py` | Replays these exact references in the Newton tabletop scene |

The nine files total 152 seconds, excluding operator positioning and pauses. They
start/end at the same proposed center and include 2-second holds. This is a **small,
local first collection**, not excitation of the entire robot workspace or a promise
that every parameter will be identifiable. Each repeated real execution gets a new
trial ID. Keep the two held-out trials out of optimization and recipe selection.

The RDK recorder runs independently of the toolkit. To reproduce `analyze_mvp1.py`,
use the existing `newton-calibration-mvp1` checkout with this bundle's
`toolkit_no_evidence.patch` (already applied in this workspace). This small change
allows absent-evidence analysis while leaving fitting blocked. Its three regression
tests are included in `test_missing_evidence.py`. These local changes are not yet
published to GitHub. Isaac Lab execution requires the included Dockerfile or the
pinned Isaac Lab 3.0 beta2 installation; the Mac runs only preparation/inspection.

## Before operating the robot

Use a qualified operator. Remove the peg. Keep the same gripper, a fixed documented
gripper opening, known tool/payload configuration and the real cell clear of contact.
Review the complete swept volume, including cables, robot self-collision and table.
The proposed joint center is a simulation pose, **not a safe automatic homing command**.
The runner refuses to move to it; teach and verify it with the manufacturer's controls.

Confirm each RDK joint index, sign and zero against `joint1` through `joint7` in the USD.
RDK joint-position commands and encoder readings use radians; Elements `JPos` uses
degrees and is **not** this interface. No external axes are supported by this runner.

The Python adapter targets the verified **RDK 1.9 / 1.9.0** four-argument
`SendJointPosition` API. It refuses other versions. Do not downgrade your robot to
match this file: check your installed SDK/robot compatibility and adapt/test the
adapter if necessary. The current 1.9.2/2.x APIs may differ.

## 1. Inspect offline (does not connect)

From the scene directory:

```bash
python scripts/collect_flexiv_rdk.py \
  --plan mvp1_collection/collection_plan.json \
  --episode train_joint1_reversals
```

This validates hashes, timing, finite values and bounded commands. It does not
import RDK, connect, enable the robot, or move anything.

## 2. Review Newton screening and approve the exact real setup

Read `free_motion_screen.json` when supplied alongside this package. A passing
simulation screen does **not** certify real-robot safety. Create an operator approval
JSON using `approval.template.json`; record the exact plan and screening report hashes,
robot serial, payload, checked mapping and on-site safety checks. Do not set a check
true without performing it. Keep the E-stop and manufacturer safety protections active.

## 3. Run ONE approved trial and collect feedback

The robot must already be enabled, stopped, in IDLE and within 0.005 rad of the
approved center. The runner neither clears faults nor enables/auto-homes the robot.
Execute only from the approved local robot-control computer, not from Horde:

```bash
python scripts/collect_flexiv_rdk.py \
  --plan mvp1_collection/collection_plan.json \
  --episode train_joint1_reversals \
  --approval mvp1_collection/approval.json \
  --robot-serial YOUR_ROBOT_SERIAL \
  --output real_trials --execute
```

A typed serial confirmation is required. The runner sends NRT position targets at
100 Hz, with 0.12 rad/s and 0.5 rad/s² trajectory-generator constraints. It checks
feedback freshness, faults, mode, velocity and tracking error; it requests `Stop()`
on completion or error. These checks are **not a safety-rated controller**. An E-stop
remains necessary if the application, network or stop request fails.

Log outputs:

- `raw.jsonl`: requested joint positions, actual send-time bracket, link-side `q/dq`,
  motor-side `theta/dtheta`, measured/desired torque, estimated external torque,
  temperatures and TCP state, with **separate robot and host timestamps**.
- `record.json`: trial ID, train/held-out designation, serial, SDK version, approval,
  file hashes, completed/aborted status and command count. Aborted trials must not
  silently enter fitting.

RDK NRT mode has an internal trajectory generator. The CSV path derivative is
**not** the internal servo target. The runner sends zero terminal velocity, following
the vendor NRT sine-sweep example. Any fitted gains describe the effective closed-loop
command interface; they do not recover Flexiv's proprietary controller coefficients.

## 4. Return the logs; rerun analysis before fitting

We will check sample timing/dropouts, motion coverage, direction reversals, coordinate
mapping and independent trial splits, then replay actual sent commands in Newton.
Clock offset and network timing must be characterized before claiming actuator delay.
The recorder deliberately leaves `clock_synchronized=false`; host receipt is not the
encoder's measurement timestamp. No effort-saturation evidence is manufactured.

**Remaining fitting work:** the default generic recipe currently requires synchronized
clocks and saturation evidence. A reviewed parameter-subset recipe must hold effort
limits fixed and defer delay when these are unavailable. The arm replay adapter must
also retain the fixed gripper/payload and the chosen controller interface. Do not flip
readiness flags to force this evidence through the old all-parameter recipe.

Potential first fit: effective stiffness/damping and, when sufficiently excited,
joint friction and armature. More independent evidence may be needed to separate
these effects. Fix unidentifiable parameters or request better trajectories. These
files do not calibrate gripper actuation/force, peg friction, insertion or task transfer.

## Sources

- [Official Flexiv RDK 1.9 NRT Python example](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9/example_py/intermediate1_non_realtime_joint_position_control.py)
- [Official robot state definitions and units](https://github.com/flexivrobotics/flexiv_rdk/blob/v1.9/include/flexiv/rdk/data.hpp)
- [Official RDK release compatibility changes](https://github.com/flexivrobotics/flexiv_rdk/releases)
- [Isaac Lab 3.0 beta2 multi-backend architecture](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/core-concepts/multi_backend_architecture.html)
