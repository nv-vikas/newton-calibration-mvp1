# Agent-assisted Flexiv collection reference

[Historical end-to-end run and limitations](VERIFICATION.md): the earlier fixed
nine-motion / 216-second workflow. The current parameter-aware workflow is
described in [the toolkit guide](../../../docs/parameter_aware_mvp1.md).
Hardware execution remains unapproved.

[Adaptive search verification](ADAPTIVE_DESIGN_VERIFICATION.md): actual Newton
sensitivity tests and a 15-motion / 360-second seven-joint collection video.
The bounded test does not establish complete parameter coverage. One reversal
has fingertip/table contact candidates; none of these files is hardware-approved.

[Previous parameter-aware regression](PARAMETER_DESIGN_VERIFICATION.md): 224
tests and a 72-second actual Newton video covering all four collection families
on joint 1. That regression was not the full seven-joint campaign; its artifacts
remain preserved separately.

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
The explicit `--recipe-only` full-scope proposal selects four training families per joint
(servo sweep, settling, acceleration sweep, slow reversal), deduplicates shared
gain/delay tests and adds two distinct combined held-outs: **30 episodes / 720
seconds** at the default 24 seconds each. Effort evidence remains a separate
requirement; no saturation-seeking motion is generated. Joint groups are now
per-axis instead of one shared arm group.

The exact achieved ranges are recorded; the 20° cap is not a claim that every
joint reaches ±20°. The selected recipe's spectrum, target parameters, required
signals and rationale are recorded for every episode. These motions are a
starting collection proposal, not proof that all dynamics are identifiable.

Use repeated `--target-parameter` flags to choose a smaller parameter scope,
for example `--target-parameter joint1_friction_nm`. `--motion-duration 12`
creates shorter episodes for a simulation regression; the default remains 24.
The full proposal is not equivalent to a reviewed real-robot collection campaign.

### Adaptive search (default)

The running-scene Newton probe tests sensitivity of gain, damping, armature,
friction and delay. The candidate catalog varies frequency, amplitude and
declared ±0.12 rad joint-2 posture offsets, all inside the same exported CSVs.
Those offsets and parameter ranges are simulation hypotheses, not OEM limits
or operator-approved configurations. Approved fitting bounds remain unset.

The default uses full command files, up to 96 candidate probes and 64 selected
training motions. Use `--max-candidate-probes` / `--max-training-experiments`
for explicit budgets. Optional `--probe-window 4` is a cheaper center-window
diagnostic, labeled as such; videos still replay the complete selected CSVs.
Budget-limited work, weak sensitivities and external evidence needs remain
visible in `design_search.json`. The number of selected motions is data-driven,
not a fixed claim of 30 episodes. `--recipe-only` explicitly skips this search.

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
