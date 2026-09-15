# Agent-assisted collection verification — 2026-09-15

Implementation commit: `1bb35eca0521f8642749756c1d687c59015784ae`.
Local branch: `codex/mvp1-agent-assist`. This change has not been pushed to GitHub.

## What actually ran

The reference launcher called `tuning.assist(evidence=None)` in the existing
Flexiv tabletop Isaac Lab/Newton scene. `analyze` inspected the asset and
reported zero real evidence. `plan` generated the commands and invoked the
bound scene preview with video enabled by default. No standalone/manual
motion-generator command or video flag was used.

Run: `flexiv-agent-assist-mvp1-cf627de20c`.

| Check | Observed result |
| --- | --- |
| Core/regression tests | 210 passed |
| Changed core-file lint / diff checks | Passed |
| Original snapshot integrity | All 46 hashed files and original nine commands unchanged |
| New motion set | Seven single-joint multi-frequency training episodes + two combined held-outs |
| Command duration | 9 × 24 s = 216 s, excluding simulation resets and settling |
| Command rate / physics timestep | 100 Hz / 1/960 s |
| Actual training-joint command span | 0.476648 rad = approximately 27.31° per trial |
| Actual rendering | 1920 × 1080, 30 fps, 6,480 frames, 1x playback |
| Automatic video verification | Full decode, nine contiguous chapters, correct timing and nonblank viewports passed |
| Visual inspection | All nine chapter thumbnails inspected; actual scene and plots readable |
| Tracking screen | 9/9 passed; largest command-vs-Newton error 0.9832° |
| Maximum simulated joint speed | 16.9905°/s |
| Minimum simulated joint-limit margin | 32.4809° |
| Moving-arm / fixture candidates | None reported in all nine episodes |
| Self-contact candidates | Same eight reported pairs in all episodes; require further review |
| Overall job | `preview_complete_review_required` |
| Overall collision screen / hardware approval | **Not passed / not approved** |

The eight pairs come from the Newton contact-candidate buffer. They are not
automatically equivalent to eight active penetrating collisions or measured
contact forces. No filtering was added to force a pass. The geometry, signed
separation and solver participation remain a review item.

## What is and is not evidence

- **Generated:** USD-coordinate joint commands, collection requirements and
  sourced setup/bounds proposals. The proposed 20° cap is not an OEM limit.
- **Simulated:** Newton arm response in the actual Isaac Lab scene. The peg was
  parked away; the unloaded gripper received a constant command.
- **Recorded:** Renderer video, simulation traces, command fingerprints and
  screening results.
- **Not performed:** Real Flexiv execution, real data collection, fitting,
  real-data held-out validation, calibrated package generation or task transfer.

Held-out here describes the intended split for **future real collection**, not
a held-out real-data calibration result. Real mapping, controller/smoothing,
payload, workspace and operating limits still require confirmation. Do not use
these larger proposals with the archived ±2° RDK runner, which rejects them.

The agent assist is a deterministic capability that Minjae/other agents can
call; no claim is made that Minjae's private agent was installed or invoked.

## Runtime and traceability

Isaac Lab `3.0.0-beta2` reference image; Newton / MuJoCo Warp physics;
native Newton self-collision requested and checked. The derived image adds the
missing pandas library without replacing the base image's compatible packages.

Docker image ID:
`sha256:dbe07ee1f97710410af9232cfb23cf1a86ef2aba823f90c140e593a6917138a8`.

Horde source directory:
`/home/horde/workspace/mvp1_agent_assist_20260915_v3`.
All five collection-module and both scene-adapter file hashes matched the local
implementation. Two earlier failed attempts (missing pandas, then NumPy scalar
serialization) remain in the preceding Horde directories; both issues were
fixed before this successful run, with a serialization regression test added.

Artifact SHA-256 values:

- Command plan: `a173d06e4691fb91d43a4c62a1bd95e174e64632e46ddf18628c92e538af7965`
- Video: `cae40a5bac9e6c01e525c9f335276412ea5141b2f64ced0ca9a430ab685249cc`
- Screen: `38a275db640650cff349b10be384af6934aac825e33abf82914111174bb724d3`

The retrieved files match these fingerprints. Local artifacts are under
`output/flexiv-agent-assist/runs/flexiv-agent-assist-mvp1-cf627de20c/`; generated
media are not checked into Git. Original JSON paths retain the container paths
as provenance. The source guide gives the reproducible Docker command.

## Remaining limitations

This preview adapter is pinned to the present Isaac Lab/Newton interface,
supports one scene, revolute coordinates, and up to 12 joints in its video
layout. Other scenes need their own pose/envelope/camera binding. A missing
scene produces an actionable pending collection job, not invented motions.

The default full-parameter fitting recipe still requires clock and effort
evidence; an evidence-qualified parameter-subset recipe remains separate work.
Do not deliberately saturate the robot to satisfy that recipe. Existing lint
findings in the paused MVP2/3 task code were left untouched.
