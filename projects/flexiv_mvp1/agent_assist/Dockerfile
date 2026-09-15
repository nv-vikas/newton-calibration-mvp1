# Extend the already-built Isaac Lab 3.0.0-beta2 reference scene image.
# No physics/renderer reinstall; only the toolkit's evidence/preview libraries.
FROM flexiv-peg-scene:isaaclab3
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir \
    pandas==2.2.3 pyarrow==22.0.0 huggingface-hub==0.36.2 \
    PyYAML==6.0.3 imageio==2.37.2 imageio-ffmpeg==0.6.0 Pillow==12.1.1
ENTRYPOINT ["/workspace/isaaclab/isaaclab.sh"]
