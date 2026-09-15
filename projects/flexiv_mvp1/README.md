# Flexiv MVP 1 — free-motion evidence collection

Separate project for the **Flexiv Rizon 4s + Grav** Isaac Lab / Newton scene and
the nine motion files prepared for real-arm data collection. This is a reference
project, not a replacement for the robot-neutral toolkit at the repository root.

**Status:** all nine reference motions passed the recorded Newton simulation
screen. No real Flexiv measurements have been collected; no Flexiv parameters
have been fitted and no real-data held-out validation has passed.

## Start here

| Deliverable | Location |
|---|---|
| Nine CSV command files | [Motion files](flexiv_peg_scene/mvp1_collection/commands/) |
| Collection instructions and hardware prerequisites | [Collection README](flexiv_peg_scene/mvp1_collection/README.md) |
| Trial definitions, joint mapping, units and hashes | [Collection plan](flexiv_peg_scene/mvp1_collection/collection_plan.json) |
| Offline inspection / gated real-data recorder | [Flexiv RDK Python runner](flexiv_peg_scene/scripts/collect_flexiv_rdk.py) |
| Recorded Newton screening, including joint traces | [Screening report](flexiv_peg_scene/mvp1_collection/free_motion_screen.json) |
| Actual no-evidence analysis and blocked fitting plan | [Analysis](flexiv_peg_scene/mvp1_collection/analysis.json) / [Plan status](flexiv_peg_scene/mvp1_collection/fit_plan_status.json) |
| Portable Newton scene | [Scene USD](flexiv_peg_scene/flexiv_tabletop_newton.usda) |
| Robot, peg and block USDs; source meshes and license | [Assets](flexiv_peg_scene/assets/) |
| Isaac Lab launcher and pinned container definition | [Launcher](flexiv_peg_scene/scripts/run_scene.py) / [Dockerfile](flexiv_peg_scene/Dockerfile) |
| Original file checksums and qualification record | [Bundle record](flexiv_peg_scene/mvp1_collection/bundle_record.json) |

## What the motions do

| Files | Commanded motion | Duration | Intended use after real collection |
|---|---|---:|---|
| `train_joint1_reversals.csv` through `train_joint7_reversals.csv` | One joint at a time, up to ±2° around the proposed center | 7 × 16 s | Fit effective arm dynamics |
| `heldout_multijoint_1.csv` and `heldout_multijoint_2.csv` | Two different combined-joint motions, up to ±1° per joint | 2 × 20 s | Test the fit on trials excluded from optimization and recipe selection |

Total: **152 seconds**, excluding operator positioning and pauses. Each CSV
contains time, seven joint-position targets in radians and reference velocities
at 100 Hz. These are commands, **not measured robot trajectories**.

The intended sequence is: approved real collection → analyze the recorded data
→ fit against Newton replays → validate on the two held-out trials → write a
scoped calibration package. These small, local motions may not identify every
parameter; additional evidence or a narrower parameter recipe may be required.

## Verify and inspect without a robot

From the repository root, using Python 3.9 or newer:

```bash
python3 projects/flexiv_mvp1/verify_bundle.py
python3 -m unittest discover -s projects/flexiv_mvp1/flexiv_peg_scene/scripts -p test_mvp1_collection.py -v
python3 projects/flexiv_mvp1/flexiv_peg_scene/scripts/collect_flexiv_rdk.py \
  --plan projects/flexiv_mvp1/flexiv_peg_scene/mvp1_collection/collection_plan.json \
  --episode train_joint1_reversals
```

These checks do not connect to hardware, execute Newton, or authorize movement.
The verifier checks the archived file hashes and links the nine command files to
the recorded screening report. The unit tests check bounded commands and rejection
of tampered files and missing operator approval. Without `--execute`, the recorder
only inspects a file; it does not load the robot SDK or move the robot.

## Scope and safety boundaries

- The reference runtime is **Isaac Lab 3.0.0-beta2, Newton / MuJoCo Warp**, not PhysX.
- The screen recorded finite motion and no detected moving-robot contact with the
  table or fixture. It is **not a hardware safety certification**; self-collision
  and cable clearance still require on-site review.
- Collection is for the **unloaded arm**, with the peg removed and the gripper
  held at a documented fixed opening. A qualified operator must approve the
  joint mapping, starting pose, tool configuration and real swept volume.
- The separate Newton peg-hold test failed due to residual slip. Preserved peg,
  block and contact-probe artifacts are experimental context, not MVP 2/3 success.
- Flexiv RDK NRT smoothing is not emulated by the existing Newton replay. Matching
  the command interface remains necessary before claiming calibrated dynamics.
- No effort-saturation evidence or synchronized real clocks exists yet. The
  default generic fitting recipe requires a reviewed parameter-subset adaptation
  before this collection can support a fit; do not bypass readiness gates.
- This upload adds no real execution approval and does not change any robot limits.

## Reproduction notes

`flexiv_peg_scene/` is an unchanged snapshot of the delivered collection bundle.
Original report paths refer to the machines that produced them; they are
provenance, not paths a new user must reproduce. Use the relative paths above.
The original scene README mentions `flexiv_tabletop_scene.usda`; the portable
scene included in this Newton bundle is **`flexiv_tabletop_newton.usda`**.

The snapshot's no-evidence `analyze` run used toolkit baseline
`19d9187eaf8014c7b874fdfcf9076cb5c9e9a330` plus the included
[`toolkit_no_evidence.patch`](flexiv_peg_scene/mvp1_collection/toolkit_no_evidence.patch).
The patch and its regression tests are supplied as reference artifacts; this
project upload does **not** merge them into the toolkit core. Inspect and check
the patch against a separate compatible checkout before reproducing that run.
The offline collection checks above do not depend on the patch or toolkit install.

The archived packaging scripts assume the original sibling-directory layout and
generated `output/` records. They are preserved for provenance, not advertised as
one-command rebuilds from this relocated snapshot. The scene launcher and motion
recorder resolve the project assets and commands relative to their own files.
The container definition is included; it was not rebuilt during this upload.

Old PhysX videos, unrelated experiments, Python environments, credentials,
completed approvals and real robot logs are not included. Vendor attribution
and the original Flexiv license are retained with the assets. The two source
STLs were supplied by the user; this private upload makes no new licensing claim.
