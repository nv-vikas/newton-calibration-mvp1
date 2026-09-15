# Flexiv Rizon 4s tabletop peg scene

An actual Isaac Lab **3.0.0-beta2** scene now defaulting to **Newton / MuJoCo Warp**
with Newton collision detection. `--backend physx` retains the earlier implementation.
The original PhysX results and video are preserved, not relabelled as Newton output.
The Flexiv Rizon 4s with Grav gripper is mounted on a 750 mm-high table. The supplied
25 mm peg starts between the gripper fingers, above the central opening of the supplied block.

## Current Newton qualification

- Newton initialized all 13 arm/gripper joints. Instanced vendor arm collision meshes
  are expanded explicitly during spawning so Newton imports the collision geometry.
- Centered peg/fixture and offset-blocking gravity probes passed on Newton.
- The five-second Newton peg-hold check **failed**: residual slow slip remains.
  Do not claim the earlier PhysX hold/video as proof of Newton grasping.
- MVP1 free-motion collection removes the peg from the arm, holds the unloaded
  gripper at a constant command and tests arm trajectories independently.
  See `mvp1_collection/README.md` and the Newton screening report supplied there.
- **All nine arm reference trajectories passed** in `output/newton-v8` at 960 Hz:
  finite motion, maximum tracking error 0.01318 rad, maximum velocity 0.05519 rad/s,
  and no detected moving-arm/gripper contact with the table or fixture. This is
  simulation screening, not a hardware safety or real-data calibration result.
- No real Flexiv data, real robot execution, tuning or transfer validation has occurred.

The Newton arm initialization uses explicit, effort-clipped Isaac Lab PD, with
per-joint stiffness `[4000,4000,3000,3000,500,500,200]` and damping
`[80,80,60,40,10,10,5]`, armature 0.05 kg·m², and 123 Nm effort caps.
The provisional independent gripper drives use stiffness 800, damping 2, armature
0.01 kg·m² and 5 Nm effort caps. These are **simulation priors**, not Flexiv control
parameters or calibrated values. Newton uses 1/960 s with the explicit controller
updated every step, one physics substep,
100 solver iterations, tolerance 1e-6 and 0.05 mm contact margin/gap.

## Open and run

- Open `flexiv_tabletop_scene.usda` in Isaac Sim to inspect the complete portable scene. Keep `assets/` beside it.
- Run `scripts/run_scene.py` through Isaac Lab 3.0 for the tested articulation, dynamic peg, position-hold controls, and capture.
- `assets/Peg_25.usd` and `assets/HoleBlock.usd` can be referenced independently in another scene.

From an Isaac Lab 3.0 installation, use absolute paths to this folder:

```bash
./isaaclab.sh -p /path/to/flexiv_peg_scene/scripts/run_scene.py \
  --headless --backend newton --enable_cameras --capture --steps 4800 --run-contact-probes
```

For an interactive viewport, use the Isaac Lab 3.0 `--viz kit --interactive` options instead of `--headless`.
Without `--interactive`, the script runs bounded validation and exits. This is a scene launcher, not a registered Gym training task.

The prior 24-second PhysX camera tour can be reproduced using `--backend physx --video`.
Newton video recording intentionally refuses a lost peg hold; its grasp setup is not yet qualified.

```bash
./isaaclab.sh -p /path/to/flexiv_peg_scene/scripts/run_scene.py \
  --headless --backend physx --enable_cameras --video --steps 600
```

The camera moves while the simulated arm keeps holding the peg. The video uses
actual RTX frames with short chapter labels. Its `video_capture_record.json`
records camera positions, peg poses, asset hashes, frame count and timing.

Docker (requires NVIDIA GPU support and the Isaac Lab image):

```bash
docker build -t flexiv-peg-scene:isaaclab3 .
docker run --rm --gpus all --shm-size=8g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /absolute/path/to/results:/workspace/flexiv_peg_scene/output \
  flexiv-peg-scene:isaaclab3
```

The reference runtime source reports `VERSION=3.0.0`, and its container release is
`3.0.0-beta2`. Python distribution metadata for the **core extension** reports `6.1.11`;
that is not the Isaac Lab product release. The validation record keeps these fields separate.

## Scene contents

| Item | Implementation |
|---|---|
| Robot | Exact linked Flexiv Rizon4s/Grav source geometry, masses and joint limits, with documented scene adaptations below |
| Table | Fixed 1.6 × 0.9 m top, four legs and a 15 mm robot mounting plate |
| Peg | Supplied STL, interpreted as mm and converted to metres; 25 mm diameter × 75 mm length |
| Block | Supplied 127 × 127 × 40 mm STL, fixed to the tabletop, including all five holes and diameter markings |
| Target opening | Central 25.5 mm bore; nominal radial clearance 0.25 mm |
| Holding | Dynamic peg, no weld: passed on the archived PhysX run; NOT yet qualified on Newton |
| Rendering | Actual RTX frames from Isaac Lab after simulation |

## Physics readiness and assumptions

Both parts have a default prim, metre units, Z-up coordinates, local materials,
collision geometry, positive mass and inertia, and source hashes. They resolve
without an internet connection. This is local simulation readiness, not a formal
SimReady catalog certification or measured physical calibration.

- **Peg:** dynamic rigid body, convex hull collision. Hull volume differs from the STL by about 0.149%.
  Steel density is an assumed 7,850 kg/m³, giving approximately 0.2885 kg.
- **Block:** fixed fixture with exact triangle-mesh collision. It preserves the actual bores;
  a single convex hull would close them. Aluminium density is an assumed 2,700 kg/m³,
  giving 0.8501 kg. Because it is fixed, this mass does not govern its motion.
- Friction priors are 0.6 static / 0.45 dynamic; restitution is zero.
- Contact offset is 0.05 mm, below the 0.25 mm nominal radial clearance.
- The table closes the bottoms of the through-holes in this mounted configuration.

## Robot adaptations

The original vendor file is preserved in `assets/source/`. The prepared robot uses
portable USD PreviewSurface materials. Several gripper links have vendor placeholder
masses of zero or 0.1 g; the scene uses explicitly labelled 20 g / positive-inertia
priors for these links. All six gripper joints are driven together from the vendor
joint frames; this does not reproduce the Flexiv proprietary controller. Arm gains
and gripper gains are scene hold settings, not identified real-robot parameters.
Robot self-collision is disabled for this initial scene, while robot/object/table
collision is enabled.

## What the checks establish

`output/asset_validation.json` records structural checks and ray tests for the open bore.
`output/verified-v2/runtime_validation.json` records the actual hold simulation,
settled peg drift, and independent centered/offset gravity contact probes. These
probes park the arm away from the fixture; they do not exercise an insertion policy.

This deliverable establishes an initialized Newton tabletop scene and scoped checks,
not whole-scene parity with PhysX. It does not claim a trained controller, robot
calibration, reliable Newton grasping, successful insertion, or sim-to-real transfer.

## Rebuild assets

Use a separate asset-preparation Python environment and the included requirements:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-asset-prep.txt
.venv/bin/python scripts/prepare_assets.py
.venv/bin/python scripts/verify_assets.py
```

## Source

[Flexiv robot USD](https://github.com/flexivrobotics/isaac_sim_ws/blob/main/exts/isaacsim.robot.manipulators.examples/data/flexiv/Rizon4s_with_Grav.usd).
The upstream robot license is included as `assets/source/FLEXIV_LICENSE`. The two STL
files were supplied by the user and are preserved unchanged.
