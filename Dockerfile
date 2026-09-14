FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2@sha256:970621075d00059c309f847fa835709de3fa634537407a7009c586c97893d218

USER root
WORKDIR /opt/newton-calibration

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir .

WORKDIR /workspace
ENTRYPOINT ["/workspace/isaaclab/isaaclab.sh", "-p", "-m", "newton_calibration.cli"]
CMD ["--help"]
