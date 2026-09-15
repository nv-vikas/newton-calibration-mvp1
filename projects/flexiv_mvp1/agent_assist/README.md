# Agent-assisted Flexiv collection reference

This active adapter leaves `../flexiv_peg_scene/` and its original hashed
delivery bundle unchanged. It reuses that scene's USD assets. No grasp or
insertion implementation is continued here: the peg is parked away, and the
gripper receives a constant command while the unloaded arm moves.

The launch itself runs `tuning.assist(evidence=None)`. That runs `analyze`,
generates collection motions through `plan`, and records their actual Newton
execution by default. There is no separate call to the old motion generator.

From the repository root, with the previously built `flexiv-peg-scene:isaaclab3`
image (Isaac Lab `3.0.0-beta2`, not a nonexistent Isaac Lab 6.x version):

```bash
docker build -t flexiv-peg-scene:agent-assist \
  projects/flexiv_mvp1/agent_assist

docker run --rm --gpus all --shm-size=8g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONPATH=/workspace/repo/src \
  -v "$PWD:/workspace/repo" \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  flexiv-peg-scene:agent-assist \
  -p /workspace/repo/projects/flexiv_mvp1/agent_assist/run_scene.py \
  --headless --output /workspace/repo/output/flexiv-agent-assist
```

The base image's dependency/build recipe is preserved under
`../flexiv_peg_scene/Dockerfile`. The small derived image adds the toolkit's
evidence and preview libraries. Optional `--no-preview` disables video explicitly.

The adapter reads the seven arm joint limits from the active USD. It proposes
the scene's initial pose with joint 1 offset by +0.6 rad, away from the fixture.
These are **simulation-only exploration proposals**, not hardware-approved
coordinates. Motions use a 20° amplitude cap, 0.3 rad/s speed cap and 0.6 rad/s²
acceleration cap; the generator scales them to stay inside that envelope.
Seven 24-second single-joint multi-frequency episodes plus two distinct combined
held-outs produce **216 seconds** of motion, with 2-second endpoint holds.

The exact achieved ranges are recorded; the 20° cap is not a claim that every
joint reaches ±20°. Training frequencies are 0.12 and 0.43 Hz; held-outs use
different joint-dependent frequencies. These motions are a starting collection
proposal, not proof of sufficient excitation for all unknown robot dynamics.

Look at `assist_result.json` for the run directory, video and screening status.
The video overlays commands vs **simulated** response. There is no real Flexiv
data and no new calibration result. Self-collision is requested using Newton's
native setting; reported self-contact candidates remain review items rather
than being filtered to force a pass.

**Do not run the new CSVs through the old ±2° Flexiv RDK runner.** It deliberately
rejects larger motions. A robot operator must confirm the coordinate transform,
real controller/smoothing, payload, joint limits, start pose and swept volume
before an approved hardware adapter can use these proposals. This workflow has
no hardware execution call.
