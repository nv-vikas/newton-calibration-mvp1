#!/usr/bin/env python3
"""Capture presentation-ready RTX stills of the SO-101 peg-insertion scene.

The centered and offset images are gravity/contact conformance probes.  This
script deliberately does not pose or animate the arm as though it executed a
grasp or insertion policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if os.environ.get("ACCEPT_EULA") != "Y":
    raise RuntimeError("Review the NVIDIA Isaac Sim license and export ACCEPT_EULA=Y before running.")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", required=True, help="SO-101 USD path")
parser.add_argument("--output", default="/workspace/output/peg_insertion_capture")
parser.add_argument("--probe-steps", type=int, default=480, help="Physics steps per contact probe")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from newton_calibration.isaaclab.tasks.so101_peg_insertion import PegInsertionMode
from newton_calibration.isaaclab.tasks.so101_peg_insertion.scene import SO101PegInsertionScene


BACKGROUND = (8, 19, 30)
WHITE = (245, 248, 250)
MUTED = (168, 181, 194)
NVIDIA_GREEN = (118, 185, 0)
AMBER = (231, 164, 45)


def checkpoint(message: str) -> None:
    print(f"[PEG-RTX] {message}", flush=True)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / family,
        Path("/usr/share/fonts") / family,
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def save_camera_frame(camera: Camera, path: Path) -> None:
    pixels = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, quality=95)
    checkpoint(f"saved {path.name}: {pixels.shape[1]}x{pixels.shape[0]}")


def configure_camera(scene: SO101PegInsertionScene) -> Camera:
    """Attach a 16:9 RTX camera after the reusable physics scene is built."""

    scene.sim.set_setting("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
    sim_utils.create_prim("/World/PegInsertionCamera", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/PegInsertionCamera/CameraSensor",
        update_period=0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=800.0,
            horizontal_aperture=36.0,
            clipping_range=(0.01, 20.0),
        ),
    )
    camera = Camera(cfg=camera_cfg)
    scene.sim.reset()

    eye = Gf.Vec3d(0.72, -0.72, 0.48)
    target = Gf.Vec3d(0.20, 0.0, 0.09)
    camera_prim = scene.sim.stage.GetPrimAtPath(camera_cfg.prim_path)
    if not camera_prim.IsValid():
        raise RuntimeError("Isaac Lab RTX camera prim was not created")
    camera_xform = UsdGeom.Xformable(camera_prim)
    camera_xform.ClearXformOpOrder()
    camera_to_world = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    camera_xform.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(camera_to_world)
    checkpoint(f"camera ready at {tuple(camera_to_world.ExtractTranslation())}")
    return camera


def quaternion_order(scene: SO101PegInsertionScene) -> str:
    """Return the documented convention of the pinned Newton interface."""

    del scene
    return "xyzw"


def create_peg_transform_op(scene: SO101PegInsertionScene) -> UsdGeom.XformOp:
    """Create the one USD transform operation used by every captured state."""

    prim = scene.sim.stage.GetPrimAtPath("/World/Env_0/Peg")
    if not prim.IsValid():
        raise RuntimeError("Peg prim was not created")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.SetResetXformStack(True)
    return xformable.AddTransformOp(
        precision=UsdGeom.XformOp.PrecisionDouble,
        opSuffix="newtonCapture",
    )


def create_robot_transform_ops(
    scene: SO101PegInsertionScene,
) -> list[tuple[int, UsdGeom.XformOp]]:
    """Map Newton articulation bodies to USD links for the beta RTX bridge."""

    operations: list[tuple[int, UsdGeom.XformOp]] = []
    for body_index, body_name in enumerate(scene.robot.body_names):
        prim = scene.sim.stage.GetPrimAtPath(f"/World/Env_0/Robot/{body_name}")
        if not prim.IsValid():
            matches = [candidate for candidate in scene.sim.stage.Traverse() if candidate.GetName() == body_name]
            if len(matches) != 1:
                raise RuntimeError(f"Could not uniquely map Newton body {body_name!r} into the USD stage")
            prim = matches[0]
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.SetResetXformStack(True)
        operation = xformable.AddTransformOp(
            precision=UsdGeom.XformOp.PrecisionDouble,
            opSuffix="newtonCapture",
        )
        operations.append((body_index, operation))
    return operations


def publish_peg_transform(
    scene: SO101PegInsertionScene,
    order: str,
    transform_op: UsdGeom.XformOp,
) -> None:
    """Publish Newton's peg pose to USD for the pinned beta RTX bridge."""

    pose = scene.peg.data.root_pose_w.torch[0].detach().cpu().numpy()
    position = pose[:3]
    quaternion = pose[3:7]
    if order != "xyzw":
        raise ValueError(f"Unsupported rigid-object quaternion order: {order}")
    x, y, z, real = (float(value) for value in quaternion)

    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Quatd(real, Gf.Vec3d(x, y, z)))
    matrix.SetTranslateOnly(Gf.Vec3d(*(float(value) for value in position)))
    transform_op.Set(matrix)


def publish_robot_transforms(
    transform_ops: list[tuple[int, UsdGeom.XformOp]],
    parked_pose: tuple[np.ndarray, np.ndarray],
) -> None:
    """Publish the authoritative parked Newton link poses as USD transforms."""

    positions, orientations = parked_pose
    for body_index, transform_op in transform_ops:
        position = positions[body_index]
        # Isaac Lab's articulation link quaternion array is xyzw in this pinned build.
        x, y, z, real = (float(value) for value in orientations[body_index])
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Quatd(real, Gf.Vec3d(x, y, z)))
        matrix.SetTranslateOnly(Gf.Vec3d(*(float(value) for value in position)))
        transform_op.Set(matrix)


def capture_parked_robot_pose(scene: SO101PegInsertionScene) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot the reset pose before the beta Cubric cache observes any probe resets."""

    positions = scene.robot.data.body_link_pos_w.torch[0].detach().cpu().numpy().copy()
    orientations = scene.robot.data.body_link_quat_w.torch[0].detach().cpu().numpy().copy()
    return positions, orientations


def render_still(
    scene: SO101PegInsertionScene,
    camera: Camera,
    output: Path,
    *,
    order: str,
    transform_op: UsdGeom.XformOp,
    robot_transform_ops: list[tuple[int, UsdGeom.XformOp]],
    parked_robot_pose: tuple[np.ndarray, np.ndarray],
    warmup_frames: int = 8,
) -> None:
    for _ in range(warmup_frames):
        # The pinned beta Cubric adapter overwrites USD transforms on every
        # render. Republish both dynamic objects immediately before each frame.
        publish_peg_transform(scene, order, transform_op)
        publish_robot_transforms(robot_transform_ops, parked_robot_pose)
        scene.sim.render()
        camera.update(dt=scene.sim.get_physics_dt(), force_recompute=True)
    save_camera_frame(camera, output)


def run_probe(
    scene: SO101PegInsertionScene,
    camera: Camera,
    output: Path,
    name: str,
    order: str,
    transform_op: UsdGeom.XformOp,
    robot_transform_ops: list[tuple[int, UsdGeom.XformOp]],
    parked_robot_pose: tuple[np.ndarray, np.ndarray],
) -> dict[str, object]:
    scene.reset(probe=name)
    for _ in range(args.probe_steps):
        scene.step(render=False, hold_robot=True)
    metrics = scene.task_metrics(settled=True)
    render_still(
        scene,
        camera,
        output,
        order=order,
        transform_op=transform_op,
        robot_transform_ops=robot_transform_ops,
        parked_robot_pose=parked_robot_pose,
    )
    entered = (
        metrics.insertion_depth_m >= scene.spec.target_insertion_depth_m
        and metrics.lateral_offset_m <= scene.spec.radial_clearance_m
    )
    passed = entered if name == "centered" else metrics.jammed
    return {
        "probe": name,
        "steps": args.probe_steps,
        "image": output.name,
        "passed": passed,
        "metrics": {
            "insertion_depth_m": metrics.insertion_depth_m,
            "lateral_offset_m": metrics.lateral_offset_m,
            "tilt_deg": metrics.tilt_deg,
            "seated": metrics.seated,
            "jammed": metrics.jammed,
        },
    }


def probe_status(probe: dict[str, object]) -> tuple[str, tuple[int, int, int]]:
    metrics = probe["metrics"]
    if probe["probe"] == "centered" and probe["passed"]:
        return "ENTERED", NVIDIA_GREEN
    if metrics["jammed"]:
        return "JAMMED", AMBER
    return "UNRESOLVED", MUTED


def draw_probe_panel(
    canvas: Image.Image,
    image_path: Path,
    probe: dict[str, object],
    *,
    x: int,
    label: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    image = Image.open(image_path).convert("RGB")
    fitted = ImageOps.fit(image, (858, 483), method=Image.Resampling.LANCZOS)
    status, accent = probe_status(probe)
    draw.rounded_rectangle((x - 4, 196, x + 862, 687), radius=18, fill=accent)
    canvas.paste(fitted, (x, 200))
    draw.text((x, 712), label, font=font(25, bold=True), fill=WHITE)
    draw.text((x + 690, 712), status, font=font(24, bold=True), fill=accent)
    metrics = probe["metrics"]
    detail = (
        f"depth {1000.0 * metrics['insertion_depth_m']:.1f} mm   ·   "
        f"offset {1000.0 * metrics['lateral_offset_m']:.1f} mm   ·   "
        f"tilt {metrics['tilt_deg']:.1f}°"
    )
    draw.text((x, 756), detail, font=font(20), fill=MUTED)


def make_presentation_board(
    output: Path,
    centered_path: Path,
    offset_path: Path,
    centered: dict[str, object],
    offset: dict[str, object],
) -> None:
    canvas = Image.new("RGB", (1920, 1080), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((72, 52), "SO-101 PEG INSERTION · CONTACT CONFORMANCE", font=font(43, bold=True), fill=WHITE)
    draw.text(
        (72, 116),
        "Actual Isaac Lab scene · Newton/MJWarp · same robot, table, metal peg and socket",
        font=font(24),
        fill=MUTED,
    )
    draw_probe_panel(canvas, centered_path, centered, x=72, label="CENTERED DROP")
    draw_probe_panel(canvas, offset_path, offset, x=990, label="OFFSET DROP")
    draw.rounded_rectangle((72, 860, 1848, 1000), radius=18, fill=(17, 34, 48), outline=(52, 74, 91), width=2)
    draw.text((106, 894), "What this proves", font=font(24, bold=True), fill=NVIDIA_GREEN)
    draw.text(
        (106, 936),
        "The scene can distinguish a nominal insertion contact from a rim collision using measured task metrics.",
        font=font(24),
        fill=WHITE,
    )
    draw.text(
        (72, 1030),
        "Scope: gravity/contact probe only · no SO-101 grasp or insertion controller was commanded",
        font=font(20),
        fill=MUTED,
    )
    canvas.save(output, quality=95)
    checkpoint(f"saved presentation board: {output.name}")


def main() -> None:
    if args.width * 9 != args.height * 16:
        raise ValueError("RTX stills must use a 16:9 width and height")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scene = SO101PegInsertionScene(
        usd_path=args.asset,
        device=args.device,
        mode=PegInsertionMode.INSERTION,
    )
    try:
        camera = configure_camera(scene)
        order = quaternion_order(scene)
        peg_transform_op = create_peg_transform_op(scene)
        robot_transform_ops = create_robot_transform_ops(scene)
        scene.reset()
        parked_robot_pose = capture_parked_robot_pose(scene)
        checkpoint(f"Newton root quaternion order: {order}")

        layout_path = output / "peg_insertion_layout.png"
        scene.reset()
        render_still(
            scene,
            camera,
            layout_path,
            order=order,
            transform_op=peg_transform_op,
            robot_transform_ops=robot_transform_ops,
            parked_robot_pose=parked_robot_pose,
        )

        centered_path = output / "peg_insertion_centered_final.png"
        centered = run_probe(
            scene,
            camera,
            centered_path,
            "centered",
            order,
            peg_transform_op,
            robot_transform_ops,
            parked_robot_pose,
        )
        offset_path = output / "peg_insertion_offset_final.png"
        offset = run_probe(
            scene,
            camera,
            offset_path,
            "offset",
            order,
            peg_transform_op,
            robot_transform_ops,
            parked_robot_pose,
        )

        board_path = output / "peg_insertion_contact_probes.png"
        make_presentation_board(board_path, centered_path, offset_path, centered, offset)
        manifest = {
            "task_id": scene.spec.task_id,
            "asset": str(Path(args.asset).resolve()),
            "physics": "Newton/MJWarp",
            "capture": {
                "renderer": "Isaac RTX",
                "resolution": [args.width, args.height],
                "quaternion_order": order,
                "robot_pose_source": "parked Newton link poses captured after reset",
            },
            "scope": "gravity/contact conformance; no robot task execution",
            "conformance_passed": centered["passed"] and offset["passed"],
            "images": {
                "layout": layout_path.name,
                "centered": centered_path.name,
                "offset": offset_path.name,
                "presentation": board_path.name,
            },
            "probes": [centered, offset],
        }
        manifest_path = output / "capture_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2), flush=True)
        print(f"RESULT={board_path}", flush=True)
    finally:
        scene.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
