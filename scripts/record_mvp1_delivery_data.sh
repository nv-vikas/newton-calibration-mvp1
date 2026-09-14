#!/usr/bin/env bash
set -euo pipefail

evidence="data/anchor-lab/data/so101_arm_50motion"
asset="data/anchor-lab/robot_assets/so101_no_camera_new_calib.usd"
manifest="packages/so101-mvp1-full-20260913/manifest.json"
output_dir="deliverables/mvp1_20260913/video_data"

record() {
  local episode="$1"
  local output_name="$2"
  /opt/venv/bin/python scripts/record_video_data.py \
    --evidence "$evidence" \
    --asset "$asset" \
    --manifest "$manifest" \
    --output "$output_dir/$output_name.npz" \
    --episode "$episode" \
    --duration 12 \
    --device cuda:0
}

record heldout-frequency-sweep heldout_frequency_sweep
record heldout-friction-gravity heldout_friction_gravity
record heldout-hold-under-gravity heldout_hold_under_gravity
record heldout-backlash-detection heldout_backlash_detection
record train-gripper-cycles train_gripper_cycles
