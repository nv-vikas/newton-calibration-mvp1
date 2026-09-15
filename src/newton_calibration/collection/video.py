"""Capture live Newton/Isaac Lab RTX frames; add trace labels, never pose edits."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS, WIDTH, HEIGHT = 30, 1920, 1080


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size=size)


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


class MotionRecorder:
    def __init__(self, sim, camera, output, plan_path):
        self.sim, self.camera, self.output = sim, camera, Path(output)
        self.plan_path = Path(plan_path)
        self.plan = json.loads(self.plan_path.read_text())
        self.spec = self.plan["motion_spec"]
        self.center = np.asarray(self.spec["center_rad"])
        self.nj = len(self.spec["joint_names"])
        if self.nj > 12:
            raise ValueError("This video layout supports up to 12 controlled joints")
        self.plot_range = max(3.0, float(np.rad2deg(self.spec["amplitude_rad"]).max()) * 1.1)
        self.dt = sim.get_physics_dt()
        self.steps_per_frame = round(1.0 / FPS / self.dt)
        if abs(self.steps_per_frame * self.dt - 1.0 / FPS) > 1e-10:
            raise ValueError("Video rate must divide the physics rate exactly")
        self.samples = self.output / "motion_video_samples"
        self.samples.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "collection_motions_newton.mp4"
        self.frames = 0
        self.chapters = []
        self.telemetry = []
        self.title_font, self.body_font = font(32, True), font(22)
        self.small_font, self.joint_font = font(18), font(21, True)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.error_stream = (self.output / "motion_video_encoder.log").open("wb")
        self.encoder = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{WIDTH}x{HEIGHT}",
                "-framerate",
                str(FPS),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=self.error_stream,
        )

    def begin_episode(self, episode, index):
        self.episode, self.index = episode, index
        self.points = []
        self.start_frame = self.frames
        for _ in range(12):
            self.sim.render()
            self.camera.update(1.0 / FPS, force_recompute=True)
        print(f"[MOTION-VIDEO] Trial {index + 1}/{len(self.plan['episodes'])} {episode['name']}", flush=True)

    def frame(self, t, command, actual, error, velocity, margin, contacts, self_contacts):
        self.sim.render()
        self.camera.update(1.0 / FPS, force_recompute=True)
        pixels = tensor(self.camera.data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
        if pixels.shape != (HEIGHT, WIDTH, 3):
            raise RuntimeError(f"Unexpected live camera frame: {pixels.shape}")
        raw = Image.fromarray(pixels)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "#111d29")
        # Entire actual viewport is letterboxed, not cropped or altered.
        canvas.paste(raw.resize((1320, 743), Image.Resampling.LANCZOS), (0, 130))
        d = ImageDraw.Draw(canvas)
        white, gray, green, blue = "#f5f8fa", "#b5c4ce", "#a4d65e", "#59b7f1"
        d.text((32, 24), "MVP1 | Agent-assisted evidence collection preview", font=self.title_font, fill=white)
        name = (
            f"JOINT {self.index + 1} · MULTI-FREQUENCY"
            if self.index < self.nj
            else f"HELD-OUT {self.index - self.nj + 1} · COMBINED"
        )
        d.text(
            (32, 78),
            f"{self.index + 1:02d} / {len(self.plan['episodes']):02d}   {name}     {t:05.2f} / {self.episode['duration_s']:.0f} s",
            font=self.body_font,
            fill=green,
        )
        d.text((32, 140), "ACTUAL ISAAC LAB / NEWTON VIEWPORT · 1× SPEED", font=self.small_font, fill="#243443")
        d.text((1344, 143), "Joint displacement from start", font=self.joint_font, fill=white)
        d.text((1344, 179), "Command", font=self.small_font, fill=green)
        d.text((1515, 179), "Newton response", font=self.small_font, fill=blue)
        command_deg, actual_deg = np.rad2deg(command - self.center), np.rad2deg(actual - self.center)
        self.points.append((t, command_deg.tolist(), actual_deg.tolist()))
        x0, x1 = 1400, 1850
        for j in range(self.nj):
            y = 240 + j * min(87, 610 // self.nj)
            d.text((1345, y - 13), f"J{j + 1}", font=self.joint_font, fill=white)
            d.line((x0, y, x1, y), fill="#334555", width=1)
            # Fixed range across all trials; the viewport never exaggerates motion.
            for idx, color in ((1, green), (2, blue)):
                xy = [
                    (
                        x0 + p[0] / self.episode["duration_s"] * (x1 - x0),
                        y
                        - np.clip(p[idx][j], -self.plot_range, self.plot_range)
                        / self.plot_range
                        * min(29, 240 // self.nj),
                    )
                    for p in self.points
                ]
                if len(xy) > 1:
                    d.line(xy, fill=color, width=2)
            d.text((1740, y + 24), f"{actual_deg[j]:+.2f}°", font=self.small_font, fill=blue)
        d.text(
            (1344, 878), f"Plot range: ±{self.plot_range:.0f}° · same scale each trial", font=self.small_font, fill=gray
        )
        d.text(
            (32, 910),
            f"Max tracking error: {np.rad2deg(error):.2f}°    Max speed: {np.rad2deg(velocity):.2f}°/s",
            font=self.body_font,
            fill=white,
        )
        d.text(
            (32, 948),
            f"Fixture candidates: {contacts}    Self-contact candidates: {self_contacts}    Joint-limit margin: {np.rad2deg(margin):.1f}°",
            font=self.body_font,
            fill=white,
        )
        d.rectangle((0, 1002, WIDTH, HEIGHT), fill="#3b2a13")
        d.text(
            (32, 1019),
            "SIMULATION SCREEN ONLY · No real data or calibration · Real-robot safety approval still required",
            font=self.body_font,
            fill="#ffdd92",
        )
        if abs(t - self.episode["duration_s"] / 2) < self.dt:
            raw.save(self.samples / f"trial_{self.index + 1:02d}_raw.png")
            canvas.save(self.samples / f"trial_{self.index + 1:02d}.jpg", quality=94)
        self.encoder.stdin.write(np.asarray(canvas).tobytes())
        if self.frames % FPS == 0:
            self.telemetry.append(
                {
                    "frame": self.frames,
                    "trial": self.episode["name"],
                    "time_s": t,
                    "simulated_state_time_s": t + self.dt,
                    "command_q": command.tolist(),
                    "simulated_q": actual.tolist(),
                }
            )
        self.frames += 1
        if self.frames % 300 == 0:
            print(f"[MOTION-VIDEO] {self.frames} frames captured", flush=True)

    def end_episode(self, result):
        self.chapters.append(
            {
                "name": self.episode["name"],
                "split": self.episode["split"],
                "start_frame": self.start_frame,
                "end_frame_exclusive": self.frames,
                "recorded_screen_passed": result["passed"],
            }
        )

    def finish(self, screening):
        self.encoder.stdin.close()
        code = self.encoder.wait(timeout=60)
        self.error_stream.close()
        if code:
            raise RuntimeError("Video encoder failed")
        expected = round(sum(e["duration_s"] for e in self.plan["episodes"]) * FPS)
        if self.frames != expected:
            raise RuntimeError(f"Expected {expected} frames, received {self.frames}")
        record = {
            "source": "live RTX camera frames from active Isaac Lab Newton simulation",
            "physics": "Newton/MuJoCo Warp",
            "fps": FPS,
            "width": WIDTH,
            "height": HEIGHT,
            "frames": self.frames,
            "duration_s": self.frames / FPS,
            "playback_speed": 1.0,
            "physics_dt_s": self.dt,
            "physics_steps_per_video_frame": self.steps_per_frame,
            "command_plan_sha256": hashlib.sha256(self.plan_path.read_bytes()).hexdigest(),
            "video_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "screen_sha256": hashlib.sha256((self.output / "screen.json").read_bytes()).hexdigest(),
            "all_trials_screen_passed": screening["passed"],
            "self_collision_certified": False,
            "real_data": False,
            "hardware_safety_certified": False,
            "resets": "each episode independently reset and settled for 1 s; reset/settle not in video",
            "arm_motion_exaggerated": False,
            "chapters": self.chapters,
            "telemetry": self.telemetry,
        }
        (self.output / "motion_video_record.json").write_text(json.dumps(record, indent=2) + "\n")
        from .verify_video import verify_video

        verify_video(self.output)
        print(f"[MOTION-VIDEO] COMPLETE {self.path}", flush=True)

    def abort(self):
        if self.encoder.poll() is None:
            if self.encoder.stdin and not self.encoder.stdin.closed:
                self.encoder.stdin.close()
            self.encoder.terminate()
            self.encoder.wait(timeout=20)
        if not self.error_stream.closed:
            self.error_stream.close()
