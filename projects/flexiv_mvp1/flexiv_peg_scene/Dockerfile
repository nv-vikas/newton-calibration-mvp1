# The Horde reference run used this Isaac Lab 3.0 beta2 base image.
FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2@sha256:970621075d00059c309f847fa835709de3fa634537407a7009c586c97893d218
WORKDIR /workspace/flexiv_peg_scene
COPY assets ./assets
COPY scripts ./scripts
COPY mvp1_collection ./mvp1_collection
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/workspace/isaaclab/isaaclab.sh", "-p", "/workspace/flexiv_peg_scene/scripts/run_scene.py"]
CMD ["--headless", "--enable_cameras", "--capture", "--run-contact-probes", "--output", "/workspace/flexiv_peg_scene/output"]
