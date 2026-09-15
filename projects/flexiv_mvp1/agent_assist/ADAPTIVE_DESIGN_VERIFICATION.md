# Adaptive MVP1 verification — 2026-09-15

## Implemented scope

`analyze → plan` can now test candidate motions in an existing Newton scene,
retain informative candidates, export the selected campaign, and request its
complete labeled viewport video. Source-level tests cover frequency/amplitude/
posture variants, joint coverage, confounding, budgets, duplicates, failures,
held-out isolation, CLI behavior, numerical audit and startup-state restoration.
Final local suite: **246 tests passed**; changed-code Ruff and diff checks passed.

This remains MVP1 free-space actuation. No new grasp, contact or insertion
implementation; no real robot execution, Flexiv measurements or fitted package.
The original 46-file Flexiv delivery bundle is unchanged and verifies.

## Seven-joint actual-Newton integration run

- Snapshot: `mvp1_adaptive_20260915_v2` on the existing Horde node.
- Container: `mvp1-adaptive-seven-joint-v2`.
- Run: `flexiv-agent-assist-mvp1-6b6694f39c`.
- Local outputs: `output/mvp1-adaptive-seven-joint/`.
- Runtime: actual Isaac Lab 3.0.0-beta2 / Newton / MuJoCo Warp, dt 1/960 s;
  scene robot has 13 DOFs, of which the seven arm joints are controlled.
  The gripper command is constant; the peg is parked away from the arm.
- Explicit verification budget: **14 candidate probes**, with a **4-second
  center diagnostic window** per prediction. This is not full-command sensitivity.
  The product default uses full command files unless a window is requested.
- 13 successful probe records, each containing 13 rollouts: **169 recorded
  successful-probe rollouts**. One startup repeatability rejection is retained.
- Independently recomputed all 13 information matrices and candidate selection
  gains from saved Newton q/dq arrays; command and record hashes also verified.
- Selected **13 training motions + 2 untouched held-outs**, 24 seconds each:
  **15 files / 360 seconds**. All seven joints have selected motion.
- Sweeps have approximately **18.29°** peak-to-peak command excursion; slow
  reversals have **30.50°**. They obey the declared simulation envelope, not
  approved OEM limits. This is not the previous one-joint 72-second regression.

### What this run does not establish

The search stopped at `probe_budget_reached`, with **210 unvisited candidates**.
Only the initial sweep/reversal candidates were tested on this bounded GPU run;
later acceleration, settling, frequency/amplitude and posture variants remain.
CPU contract tests exercise those variant-selection paths, not their real
identifiability. It would be incorrect to call this actual run exhaustive.

Under the stated range and noise assumptions, **14/29 motion-testable targets**
met the predicted-coverage thresholds; **15 remain weak or confounded**. Seven
effort-scale targets additionally require separate evidence. None is calibrated.
Noise and probe ranges are hypotheses, not Flexiv specifications or approved
fitting bounds. Cross-joint/global identifiability is not certified.

The weak targets are all seven armatures; friction on joints 1–4; and stiffness
and damping on joints 1 and 7. More candidates, independently anchored dynamics,
calibrated effort or better real evidence may be needed. No saturation-seeking
motion is generated. Clock synchronization still needs real confirmation.

### Startup rejection and follow-up

The rendered runtime performs deferred CUDA-graph capture. Its implementation
runs an eager integration during capture, contaminating the first baseline.
The 4-second run rejected the first joint-1 sweep when repeatability exceeded
the existing threshold; that record is preserved, not rewritten as a pass.

The adapter now consumes four unmeasured warm-up steps and restores the scene
before any prediction. Unit tests require reset both on success and failure.
The broader run predates this startup-isolation change; its rejection is retained.

The separate `mvp1-startup-full-command-v3` GPU check completed successfully:
run `flexiv-agent-assist-mvp1-fcd377c541`, under
`output/mvp1-startup-full-command/`. It used **full 12-second command files**,
not a diagnostic window, and five rollouts for one stiffness parameter. There
were no rejected probes. Baseline-repeat difference was **0.003449 noise units**,
below the unchanged **0.1** threshold. The saved traces independently reproduce
the information matrix and selection gain. This narrow test is not a replacement
for the seven-joint campaign and did not pass the parameter-coverage threshold.
It also produced a complete 36-second, three-motion native Newton video.

## Video verification

The complete 360-second video contains 10,800 live RTX frames at 1920×1080,
30 fps, 1x playback, and all 15 selected command files. The completed Horde
recording decoded successfully and its chapter/frame checks passed. Initial
inspection confirms visible joints, families, parameter targets and motion.
The downloaded copy also passed a complete decode and 15-chapter check. All
**360 telemetry samples** matched the exact exported command CSVs and the
recorded Newton screening traces. Peak-excursion frames were visually inspected.

Main video SHA-256:
`462bbbef957d348a49643b7f109a0f83dfadef71683fa3fa1edd29feafefa5cc`.
Command-plan SHA-256:
`2e60618c0d565e3ecd1ebe4950b04d5ac16d5293181b57cb3188fe66d1f751c9`.

All 15 kinematic screens passed; maximum tracking error was 1.126°, maximum
joint speed 16.956°/s and minimum joint-limit margin 34.877°. Overall screening
**failed/requires review**, rather than passing, due to the contact flags below.

Video completion is separate from screening approval. Existing self-contact
candidates remain visible. A simulation screen is not hardware certification.
The joint-2 slow reversal additionally reports two fingertip/table contact
candidates. It is **not cleared for real collection**; its pose/amplitude and
full-path clearance need revision and screening. Sensitivity estimates do not
override that failure or certify a free-space experiment.
The old ±2° RDK runner deliberately rejects these larger motions; an operator
must review coordinates, controller mode, swept volume and limits first.

## Rechecking saved artifacts

```bash
python -m newton_calibration.collection.verify_design \
  output/mvp1-adaptive-seven-joint/runs/flexiv-agent-assist-mvp1-6b6694f39c
python -m newton_calibration.collection.verify_video \
  output/mvp1-adaptive-seven-joint/runs/flexiv-agent-assist-mvp1-6b6694f39c/preview
python projects/flexiv_mvp1/verify_bundle.py
```

The first checks saved simulation calculations, the second checks media and
screen-record integrity, and the third checks the historical bundle. None
substitutes for real-data calibration or held-out sim-to-real validation.
