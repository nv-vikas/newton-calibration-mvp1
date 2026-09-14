#!/usr/bin/env bash
set -euo pipefail

docker build -f Dockerfile.kitless -t newton-calibration-mvp1:kitless .
docker run --rm --gpus all --ipc=host \
  -v "${PWD}/data:/workspace/data" \
  -v "${PWD}/runs:/workspace/runs" \
  -v "${PWD}/packages:/workspace/packages" \
  newton-calibration-mvp1:kitless run \
  --evidence /workspace/data/anchor-lab/data/so101_arm_50motion \
  --asset /workspace/data/anchor-lab/robot_assets/so101_no_camera_new_calib.usd \
  --revision "${ANCHOR_LAB_REVISION:-647edd5787cd764cdc041103ad282dc59214d919}" \
  --workdir /workspace/runs \
  --output /workspace/packages/so101
