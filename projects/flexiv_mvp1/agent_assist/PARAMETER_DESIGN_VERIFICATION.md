# Parameter-aware MVP1 verification — September 15, 2026

## Code verification

- 224 tests passed (Python 3.12); changed-code Ruff checks and `git diff --check` passed.
- Parameter-specific motion selection, shared-test deduplication, independent
  joint targeting, missing-signal/clock actions, budget deferrals, immutable
  collection revisions and invalid generator contracts are tested.
- Synthetic CPU five-call test fits a single selected friction parameter,
  preserves complete 24-second train/held-out episodes and writes a
  nonactivatable package. This is **not a real-data or Newton physics test**.
- Synthetic package-contract test round-trips a gains-only package, preserves
  unselected baseline settings and rejects an injected unvalidated parameter.
- Original Flexiv bundle: all 46 file hashes and nine historical motion files
  verified unchanged. No paused grasp/contact/insertion source files changed.

## Actual Isaac Lab / Newton regression

Existing Horde node; separate preserved directory:
`/home/horde/workspace/mvp1_parameter_design_20260915_v1`.
Container: `mvp1-parameter-design-20260915-v1`, exited 0.
Image: existing `flexiv-peg-scene:agent-assist`, based on Isaac Lab 3.0.0-beta2.
Newton/MuJoCo Warp physics, explicit Isaac Lab PD actuator, native self-collision requested.

Run: `flexiv-agent-assist-mvp1-8e0f5bb901`.
Targets: joint1 stiffness scale, damping scale, armature and joint friction.
Four selected training families plus two held-out trajectories; **6 × 12 seconds**.
Other arm joints and the gripper retain constant commands. Peg remains parked
away from the arm. No grasp, contact-calibration or insertion routine runs.

| Check | Observed result |
|---|---|
| Default video | Completed; 1920×1080, 30 fps, 2,160 frames, 72 seconds |
| Media verification | Full decode passed; all six chapters present; fresh contact sheet visually inspected |
| Kinematic screen | All six episodes passed |
| Largest tracking error | Approximately 0.705 degrees |
| Highest observed joint speed | Approximately 16.88 degrees/second |
| Minimum observed joint-limit margin | Approximately 50 degrees |
| Moving robot / fixture contact candidates | None reported |
| Self-contact candidates | Same eight pairs reported in each trial; unresolved |
| Overall simulation screen | **Not passed — review required** |
| Real evidence / hardware execution / calibration | None |

The eight pairs include several nonadjacent link pairs and gripper-base/link7.
They are raw contact-buffer candidates, not verified active-contact forces.
They have not been suppressed to produce a pass. Neither the kinematic result
nor the completed video is hardware safety certification.

Local artifacts:
`output/mvp1-parameter-design-regression/runs/flexiv-agent-assist-mvp1-8e0f5bb901/`.
Video: `preview/collection_motions_newton.mp4`.
Screen: `preview/screen.json`.
Media checks: `preview/motion_video_validation.json`.

Video SHA-256:
`5928b42a53cfb9f9c7ef1c25729765e2dd1ce6b72a23c31719506b72619817d3`.
Executed command-plan SHA-256:
`7e4dd7c8751db936ad76cb5e1bc722410f5aaff196a8dbf8cd1ffedf55f265a2`.

The remote source snapshot is preserved as executed. Subsequent local changes
add analysis-mutation checks, evidence-quality metadata, stricter generator
contracts and scoped-package checks. Re-generating all six commands with the
final local generators matches the recorded numeric values within **5.6e-13**
(checked with `atol=rtol=1e-11`). The macOS and Linux CSV bytes differ at floating-
point precision; they are **not** claimed to have identical hashes. The video
is linked to the actual executed Linux command hashes, not the regenerated CSVs.

## Scope of this verification

This verifies one-joint use of every implemented collection family in the
actual scene, not the full seven-joint parameter campaign. The new full default
proposal is 28 training experiments + two held-outs, 720 seconds at 24 seconds
per episode; that full campaign has **not** been replayed in this revision.

No new real Flexiv data was collected, no Flexiv tuning was performed, and no
sim-to-real transfer is claimed. The prior SO-101 real-data reference and all
experimental MVP2/MVP3 artifacts remain preserved.
