"""Decode the preview and verify every episode is represented; no physics claims."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from newton_calibration.core.io import sha256_file, write_json


def verify_video(output: str | Path) -> dict:
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image

    output = Path(output)
    video = output / "collection_motions_newton.mp4"
    record = json.loads((output / "motion_video_record.json").read_text())
    screen_path = output / "screen.json"
    screen = json.loads(screen_path.read_text())
    if sha256_file(video) != record["video_sha256"] or sha256_file(screen_path) != record["screen_sha256"]:
        raise ValueError("Preview video/screen fingerprint mismatch")
    if record["command_plan_sha256"] != screen["command_plan_sha256"]:
        raise ValueError("Preview video/screen use different commands")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"],
        check=True,
        timeout=300,
    )
    checks, end = [], 0
    with imageio.get_reader(str(video)) as reader:
        metadata = reader.get_meta_data()
        frames = reader.count_frames()
        if frames != record["frames"] or abs(metadata["duration"] - record["duration_s"]) > 0.05:
            raise ValueError("Incomplete video frames/duration")
        if metadata["size"] != (record["width"], record["height"]) or abs(metadata["fps"] - record["fps"]) > 1e-6:
            raise ValueError("Video geometry or playback timing differs from record")
        chapters = record["chapters"]
        if len(chapters) != len(screen["tests"]):
            raise ValueError("Missing preview chapters")
        sheet = Image.new("RGB", (1920, 360 * ((len(chapters) + 2) // 3)))
        for i, (chapter, test) in enumerate(zip(chapters, screen["tests"])):
            first, last = chapter["start_frame"], chapter["end_frame_exclusive"]
            if first != end or last <= first or chapter["name"] != test["name"]:
                raise ValueError("Chapter discontinuity or wrong screen episode")
            end = last
            frame = reader.get_data((first + last) // 2)
            viewport = frame[180:850, :1320]
            if np.std(viewport) < 5:
                raise ValueError("Blank scene viewport")
            checks.append({"name": chapter["name"], "viewport_std": float(np.std(viewport)), "frames": last - first})
            sheet.paste(Image.fromarray(frame).resize((640, 360)), ((i % 3) * 640, (i // 3) * 360))
        if end != frames:
            raise ValueError("Video does not cover all recorded frames")
    sheet.save(output / "motion_contact_sheet.jpg", quality=95)
    result = {
        "decode_passed": True,
        "frames": frames,
        "duration_s": metadata["duration"],
        "fps": metadata["fps"],
        "size": metadata["size"],
        "chapters": checks,
        "all_simulation_screens_passed": screen["passed"],
        "hardware_safety_certified": False,
    }
    write_json(output / "motion_video_validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(verify_video(parser.parse_args().output), indent=2))
