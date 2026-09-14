# MVP1 presentation video

The renderer is `scripts/render_calibration_videos.py`. It is pinned to the
validated full run `so101-8f2830f962` and refuses stale smoke, rejected, short,
non-finite/unbounded, or differently scoped inputs.

## Physics data that must exist first

Record these two bundles on the Isaac Lab + Newton runtime. The renderer does
not synthesize trajectories.

| Purpose | Expected NPZ | Required episode | Split | Minimum recorded duration |
|---|---|---|---|---:|
| Arm comparison | `deliverables/mvp1_20260913/video_data/heldout_frequency_sweep.npz` | `so101-sysid-50motion-heldout-frequency-sweep` | heldout | 11.5 s |
| Jaw comparison | `deliverables/mvp1_20260913/video_data/heldout_frequency_sweep.npz` | jaw channel of `so101-sysid-50motion-heldout-frequency-sweep` | heldout | 11.5 s |

Each NPZ must contain synchronized arrays:

```text
time_s             [T]
command_q          [T, 6]
measured_q         [T, 6]
measured_dq        [T, 6]
baseline_q         [T, 6]
baseline_dq        [T, 6]
tuned_q            [T, 6]
tuned_dq           [T, 6]
joint_names        [6]
```

Each NPZ needs a same-name JSON sidecar emitted by `record_video_data.py`.
At minimum it must carry the episode, split, duration, runtime,
finite/bounded-rollout flags,
run ID, baseline/tuned errors, scores, objectives, and the exact baseline and
tuned parameter dictionaries. Its `manifest_run_id` must be
`so101-8f2830f962`.

The renderer also reads these durable records directly:

```text
runs/mvp1-full-20260913/so101-8f2830f962/
  analysis.json
  plan.json
  fit.json
  validation.json

packages/so101-mvp1-full-20260913/
  manifest.json
  package.json
```

By default the overview also requires an actual-USD viewport capture at
`deliverables/mvp1_20260913/viewport/`:

```text
live_result.json
live_trajectories.npz
frames/measured/*.png
frames/baseline/*.png
frames/tuned/*.png
```

All three frame lanes must have the declared frame count. The capture must be
an Isaac Sim RTX visualization tied to the same package run, USD hash,
evidence fingerprint, canonical joint order, and exact baseline/tuned
parameter maps. It must also identify itself as
`kinematic_playback_of_verified_trajectories`, carry hashes for the score-locked
source NPZ and sidecar, and prove that every displayed array is an exact window
of that source bundle.

The measured lane is a kinematic playback of real telemetry on the USD; it is
not physical camera footage. The baseline and tuned lanes are kinematic
playbacks of Newton states that were already computed and recorded through the
validated Isaac Lab + Newton workflow. RTX does not run a second physics
experiment and is visualization-only. Use `--no-viewport` only when
intentionally producing a diagram-only draft.

## Render

From the repository root:

```bash
python scripts/render_calibration_videos.py \
  --data-dir deliverables/mvp1_20260913/video_data
```

The default output directory is `deliverables/mvp1_20260913/videos/` and
contains:

```text
so101_mvp1_five_call_overview.mp4
so101_mvp1_five_call_overview_poster.png
so101_mvp1_arm_synchronized.mp4
so101_mvp1_arm_synchronized_poster.png
so101_mvp1_gripper_synchronized.mp4
so101_mvp1_gripper_synchronized_poster.png
video_manifest.json
```

The overview is 53 seconds at 1920×1080 and covers all five calls in detail:
evidence analysis, locked plan, optimizer/runtime loop, held-out validation,
scoped package write, and a three-lane actual-USD RTX visualization of a
verified held-out slice. The arm and jaw clips are synchronized joint-state
reconstructions and plots.

## Required interpretation

- **Measured** means real Anchor-Lab joint telemetry visualized through a
  kinematic reconstruction. It is not physical camera footage.
- **Baseline** means recipe-initial actuator settings applied to the released
  `so101_no_camera_new_calib.usd`, which was already calibrated. It is not an
  untouched or uncalibrated USD default.
- **Tuned** means the selected parameters were executed in Newton through the
  Isaac Lab surface.
- The selected set has three owners: six grouped Isaac Lab explicit-PD
  controller settings (arm/gripper stiffness, damping, and effort scales), four
  Newton dynamics settings (arm/gripper armature and friction), and one toolkit
  timing setting (command delay).
- The optimizer selected a 28.079 ms command delay. At the locked 120 Hz step,
  the runtime applies that setting as three whole steps, or 25.000 ms. The
  package records both the continuous selected value and this effective replay
  behavior.
- `isaaclab_actuator.yaml` and `manifest.json` carry the selected values.
  `calibration.usda` relatively sublayers the packaged copy of the released
  source USD and records package provenance; it does not author those values
  into the USD. `job/candidate-history.jsonl` and the other durable run records
  are included in the portable package.
- **Validated** means 92.099% lower aggregate weighted trajectory score across
  four locked, 12-second held-out motions. Metrics shown beside the animated
  arm and jaw are the displayed frequency-sweep episode only, not the aggregate.
  The plan still uses the dedicated gripper-cycle episode as fit evidence.
- The package scope is free-space arm and unloaded-jaw actuation. It does not
  validate grip force, contact, grasp, insertion, a trained policy, or transfer
  to a physical robot.
