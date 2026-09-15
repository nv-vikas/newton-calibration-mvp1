"""Record a camera tour from live Isaac Lab RTX frames and active physics stepping."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, UsdGeom


FPS = 30
WIDTH, HEIGHT = 1920, 1080
SHOTS = [
    dict(name="SETUP OVERVIEW", seconds=7,
         eye0=(1.95, -2.9, 2.05), eye1=(1.5, -2.9, 2.10),
         target0=(-.12, 0, .81), target1=(-.12, 0, .81), focal=35.),
    dict(name="GRIPPER + PEG", seconds=7,
         eye0=(.62, -.65, 1.18), eye1=(.50, -.70, 1.13),
         target0=(.15, 0, .88), target1=(.15, 0, .88), focal=55.),
    dict(name="HOLE BLOCK", seconds=5,
         eye0=(.40, -.42, 1.13), eye1=(.32, -.38, 1.19),
         target0=(.15, 0, .79), target1=(.15, 0, .79), focal=52.),
    dict(name="TABLETOP ENVIRONMENT", seconds=5,
         eye0=(1.30, -2.75, 1.95), eye1=(1.75, -2.85, 2.08),
         target0=(-.12, 0, .81), target1=(-.12, 0, .81), focal=35.),
]


def as_tensor(value):
    return value.torch if hasattr(value, "torch") else value


def font(size, bold=False):
    base = Path("/usr/share/fonts/truetype/dejavu")
    path = base / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size=size)


def capture_setup_video(sim, robot, peg, camera, hold, output, backend="newton"):
    output = Path(output)
    samples = output / "video_samples"
    samples.mkdir(exist_ok=True)
    executable = shutil.which("ffmpeg")
    if executable is None:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    path = output / "flexiv_setup_environment.mp4"
    command = [executable, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
               "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    log_path = output / "video_encoder.log"
    title_font, subtitle_font, chapter_font = font(31, True), font(20), font(22, True)
    camera_prim = sim.stage.GetPrimAtPath("/World/Camera")
    camera_xform = UsdGeom.Xformable(camera_prim)
    camera_xform.ClearXformOpOrder()
    camera_op = camera_xform.AddTransformOp()
    trajectory = []
    frame = 0
    dt = sim.get_physics_dt()
    steps_per_frame = round(1 / (FPS * dt))
    total = int(sum(s["seconds"] for s in SHOTS) * FPS)
    with log_path.open("wb") as errors:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for index, shot in enumerate(SHOTS):
                count = shot["seconds"] * FPS
                UsdGeom.Camera(camera_prim).GetFocalLengthAttr().Set(shot["focal"])
                print(f"[FLEXIV-VIDEO] shot {index + 1}/{len(SHOTS)}: {shot['name']}", flush=True)
                for local_frame in range(count):
                    t = local_frame / max(count - 1, 1)
                    t = t * t * (3 - 2 * t)
                    eye = (1 - t) * np.asarray(shot["eye0"]) + t * np.asarray(shot["eye1"])
                    target = (1 - t) * np.asarray(shot["target0"]) + t * np.asarray(shot["target1"])
                    camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
                    # Warm the temporal renderer after each camera cut without
                    # introducing hidden simulation advancement or resets.
                    if local_frame == 0:
                        for _ in range(16):
                            sim.render()
                            camera.update(1 / FPS, force_recompute=True)
                    for _ in range(steps_per_frame):
                        robot.set_joint_position_target_index(target=hold)
                        robot.write_data_to_sim()
                        sim.step(render=False)
                        robot.update(dt)
                        peg.update(dt)
                    sim.render()
                    camera.update(1 / FPS, force_recompute=True)
                    pixels = as_tensor(camera.data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
                    if pixels.shape != (HEIGHT, WIDTH, 3):
                        raise RuntimeError(f"Unexpected RTX frame size: {pixels.shape}")
                    peg_pose = as_tensor(peg.data.root_pose_w).cpu().numpy()[0]
                    if not np.isfinite(peg_pose).all() or peg_pose[2] < .87:
                        raise RuntimeError("Peg hold became invalid while recording")
                    scene_frame = Image.fromarray(pixels)
                    if local_frame == count // 2:
                        scene_frame.save(samples / f"shot_{index + 1}_raw.png")
                    # Titles describe the scene; robot and object pixels come
                    # from the actual active simulation, with no pose edits.
                    draw = ImageDraw.Draw(scene_frame, "RGBA")
                    draw.rectangle((0, 0, WIDTH, 86), fill=(14, 24, 34, 240))
                    draw.rectangle((0, 86, WIDTH, 90), fill=(118, 185, 0, 255))
                    draw.text((34, 23), "Flexiv | Peg-in-hole setup", font=title_font, fill=(255, 255, 255, 255))
                    chapter = f"0{index + 1}  {shot['name']}"
                    bounds = draw.textbbox((0, 0), chapter, font=chapter_font)
                    draw.text((WIDTH - 34 - (bounds[2] - bounds[0]), 32), chapter, font=chapter_font,
                              fill=(183, 226, 104, 255))
                    draw.rectangle((0, HEIGHT - 46, WIDTH, HEIGHT), fill=(14, 24, 34, 230))
                    draw.text((34, HEIGHT - 34), f"Isaac Lab 3.0 beta 2  |  {backend.title()}  |  Live simulated hold + camera tour",
                              font=subtitle_font, fill=(224, 231, 238, 255))
                    if local_frame == count // 2:
                        scene_frame.save(samples / f"shot_{index + 1}.jpg", quality=92)
                    encoder.stdin.write(np.asarray(scene_frame).tobytes())
                    if frame % FPS == 0:
                        trajectory.append({"frame": frame, "time_s": frame / FPS, "shot": index + 1,
                                           "peg_pose": peg_pose.tolist(), "camera_eye": eye.tolist(), "camera_target": target.tolist()})
                    frame += 1
                    if frame % 90 == 0:
                        print(f"[FLEXIV-VIDEO] {frame}/{total} frames", flush=True)
            encoder.stdin.close()
            if encoder.wait(timeout=60):
                raise RuntimeError("Video encoding failed; see video_encoder.log")
        except BaseException:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.terminate()
            encoder.wait(timeout=20)
            raise
    positions = np.asarray([p["peg_pose"][:3] for p in trajectory])
    asset_dir = Path(__file__).resolve().parents[1] / "assets"
    result = {"video": path.name, "width": WIDTH, "height": HEIGHT, "fps": FPS,
              "frames": frame, "duration_s": frame / FPS, "source": "actual Isaac Lab RTX camera frames",
              "physics": backend.title(), "physics_steps_per_video_frame": steps_per_frame, "physics_timestep_s": dt,
              "peg_fixed_attachment": False, "robot_command": "constant position hold",
              "camera_motion_only": True, "peg_hold_valid_throughout": True,
              "max_peg_drift_m_during_video": float(np.max(np.linalg.norm(positions - positions[0], axis=1))),
              "asset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in asset_dir.glob("*.usd")},
              "shots": SHOTS, "samples": trajectory, "video_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "video_capture_record.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[FLEXIV-VIDEO] COMPLETE {path} ({frame} frames)", flush=True)
