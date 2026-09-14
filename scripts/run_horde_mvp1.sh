#!/usr/bin/env bash
set -euo pipefail

: "${ACCEPT_EULA:?Review the NVIDIA Isaac Sim license and export ACCEPT_EULA=Y}"

docker build -t newton-calibration-mvp1:latest .
docker run --rm --gpus all --ipc=host \
  -e ACCEPT_EULA="${ACCEPT_EULA}" \
  -v "${PWD}/data:/workspace/data" \
  -v "${PWD}/runs:/workspace/runs" \
  -v "${PWD}/packages:/workspace/packages" \
  newton-calibration-mvp1:latest run \
  --evidence /workspace/data/anchor-lab/data/so101_arm_50motion \
  --asset /workspace/data/anchor-lab/robot_assets/so101_no_camera_new_calib.usd \
  --revision "${ANCHOR_LAB_REVISION:-main}" \
  --workdir /workspace/runs \
  --output /workspace/packages/so101
