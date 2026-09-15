"""Decode the preview and verify every episode is represented; no physics claims."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from newton_calibration.core.io import sha256_file, write_json


def verify_command_telemetry(plan_path: Path, record: dict, screen: dict) -> int:
    """Portable archive check: exact CSV replay, not local asset availability."""
    if sha256_file(plan_path) != record["command_plan_sha256"]:
        raise ValueError("Video command plan changed")
    plan = json.loads(plan_path.read_text())
    if plan.get("real_data") is not False:
        raise ValueError("Motion preview must not be labeled real data")
    episodes = plan["episodes"]
    if [e["name"] for e in episodes] != [c["name"] for c in record["chapters"]]:
        raise ValueError("Video omits or reorders selected command files")
    by_name = {test["name"]: test for test in screen["tests"]}
    count = 0
    for episode in episodes:
        path = (plan_path.parent / episode["command_file"]).resolve()
        if not path.is_relative_to(plan_path.parent.resolve()) or sha256_file(path) != episode["sha256"]:
            raise ValueError("Video input command fingerprint changed")
        table = np.loadtxt(path, delimiter=",", skiprows=1)
        width = len(plan["motion_spec"]["joint_names"])
        samples = [s for s in record["telemetry"] if s["trial"] == episode["name"]]
        if not samples:
            raise ValueError("Selected motion has no video telemetry")
        traces = by_name[episode["name"]]["trace"]
        for sample in samples:
            expected = [np.interp(sample["time_s"], table[:, 0], table[:, j + 1]) for j in range(width)]
            if not np.allclose(sample["command_q"], expected, rtol=0, atol=1e-9):
                raise ValueError("Video commands differ from the exported motion CSV")
            trace = min(traces, key=lambda t: abs(t["time_s"] - sample["time_s"]))
            if abs(trace["time_s"] - sample["time_s"]) > 1e-8 or not np.allclose(
                sample["simulated_q"], trace["simulated_q"], rtol=0, atol=1e-9
            ):
                raise ValueError("Video Newton response differs from the screening trace")
            count += 1
    return count


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
    telemetry_count = verify_command_telemetry(output.parent / "command_plan.json", record, screen)
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
            # Midpoints of symmetric reversals often equal the starting pose.
            # Inspect recorded peak excursion instead, without editing motion.
            samples = [s for s in record.get("telemetry", []) if first <= s["frame"] < last]
            chosen = (first + last) // 2
            if samples:
                initial = np.asarray(samples[0]["command_q"])
                chosen = max(samples, key=lambda s: np.linalg.norm(np.asarray(s["command_q"]) - initial))["frame"]
            frame = reader.get_data(chosen)
            viewport = frame[180:850, :1320]
            if np.std(viewport) < 5:
                raise ValueError("Blank scene viewport")
            checks.append(
                {
                    "name": chapter["name"],
                    "viewport_std": float(np.std(viewport)),
                    "frames": last - first,
                    "inspection_frame": chosen,
                    "frame_choice": "recorded peak command excursion" if samples else "midpoint",
                }
            )
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
        "command_telemetry_samples_verified": telemetry_count,
        "all_joint_kinematics_passed": all(test["kinematic_screen_passed"] for test in screen["tests"]),
        "motions_with_fixture_candidates": [
            test["name"] for test in screen["tests"] if test["fixture_contact_candidates"]
        ],
        "motions_with_self_candidates": [test["name"] for test in screen["tests"] if test["self_contact_candidates"]],
    }
    write_json(output / "motion_video_validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(verify_video(parser.parse_args().output), indent=2))
