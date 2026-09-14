"""Render the validated MVP1 SO-101 calibration presentation videos.

This renderer never creates or alters physics results.  It consumes trajectory
arrays recorded by :mod:`record_video_data` plus the durable five-call run and
package records.  The visual labels are intentionally precise:

* measured = real Anchor-Lab telemetry visualized/kinematically reconstructed;
* baseline = recipe-initial actuator settings applied to the already-calibrated
  released SO-101 USD;
* tuned = Newton simulation with the selected calibration parameters.

The overview reports the aggregate held-out validation separately from any
single displayed episode.  MVP1 is scoped to free-space arm and unloaded-jaw
actuation.  Nothing rendered here claims grasp/contact/insertion fidelity,
policy transfer, or physical robot transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = "#FFFFFF"
PANEL = "#FFFFFF"
PANEL_2 = "#F7F8F9"
WHITE = "#14171C"
MUTED = "#6B7280"
GRID = "#E6EAEE"
GREEN = "#76B900"
GREEN_DARK = "#4E7A00"
BLUE = "#1E6FBA"
CORAL = "#C77700"
AMBER = "#C77700"

RUN_ID = "so101-8f2830f962"
AGGREGATE_HELDOUT_IMPROVEMENT_PCT = 92.099450221871
VALIDATED_SCOPE = "SO-101 free-space arm and unloaded gripper actuation"
SOURCE_USD_SHA256 = "c6c82840925ace388b0ff0acb7d8538c2b419d92fbe01dc70fe833b974d6d462"
MEASURED_DISCLOSURE = "real Anchor-Lab telemetry visualized/kinematically reconstructed"
BASELINE_DISCLOSURE = "recipe-initial actuator settings applied to already-calibrated released USD"
TUNED_DISCLOSURE = "Newton simulation with selected calibration settings"
CANONICAL_JOINT_ORDER = ("rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw")

FONT_REGULAR = "/System/Library/Fonts/Avenir Next.ttc"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else FONT_REGULAR
    index = 0 if mono else (0 if bold else 7)
    try:
        return ImageFont.truetype(path, size=size, index=index)
    except (OSError, TypeError):
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=size, index=index)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectory(
    path: Path,
    *,
    expected_split: str,
    minimum_duration_s: float,
    required_improvement_metric: str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load and validate one recorded evidence/Newton trajectory bundle."""

    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Trajectory bundle is incomplete: expected {path} and {metadata_path}. "
            "Run record_video_data.py against the validated full package; do not substitute generated curves."
        )
    raw = np.load(path)
    required = {
        "time_s",
        "command_q",
        "measured_q",
        "measured_dq",
        "baseline_q",
        "baseline_dq",
        "tuned_q",
        "tuned_dq",
        "joint_names",
    }
    missing = sorted(required.difference(raw.files))
    if missing:
        raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
    arrays = {key: np.asarray(raw[key]) for key in raw.files}
    metadata = load_json(metadata_path)
    if metadata.get("manifest_run_id") != RUN_ID:
        raise ValueError(
            f"{metadata_path} belongs to {metadata.get('manifest_run_id')!r}; expected validated full run {RUN_ID!r}."
        )
    if metadata.get("split") != expected_split:
        raise ValueError(f"{metadata_path} split is {metadata.get('split')!r}; expected {expected_split!r}.")
    if float(metadata.get("duration_s", 0.0)) < minimum_duration_s:
        raise ValueError(
            f"{metadata_path} is only {metadata.get('duration_s', 0.0):.3f}s; expected at least {minimum_duration_s:.1f}s."
        )
    if metadata.get("runtime") != "isaaclab_newton":
        raise ValueError(f"{metadata_path} did not record the Isaac Lab + Newton runtime.")
    if not metadata.get("baseline_stable") or not metadata.get("tuned_stable"):
        raise ValueError(f"{metadata_path} contains an unstable Newton rollout.")
    if arrays["measured_q"].shape != arrays["baseline_q"].shape or arrays["measured_q"].shape != arrays["tuned_q"].shape:
        raise ValueError(f"{path} measured/baseline/tuned position arrays are not synchronized.")
    if arrays["measured_q"].ndim != 2 or arrays["measured_q"].shape[1] != 6:
        raise ValueError(f"{path} must contain six synchronized SO-101 joint positions.")
    actual_joint_order = tuple(str(value) for value in arrays["joint_names"].tolist())
    if actual_joint_order != CANONICAL_JOINT_ORDER:
        raise ValueError(
            f"{path} joint order {actual_joint_order!r} does not match canonical order {CANONICAL_JOINT_ORDER!r}."
        )
    if float(metadata["tuned_error"][required_improvement_metric]) >= float(metadata["baseline_error"][required_improvement_metric]):
        raise ValueError(
            f"Refusing to present {path}: tuned {required_improvement_metric} does not improve on the displayed episode."
        )
    return arrays, metadata


def load_run_records(run_dir: Path, package_dir: Path) -> dict:
    names = ("analysis", "plan", "fit", "validation")
    records = {name: load_json(run_dir / f"{name}.json") for name in names}
    records["manifest"] = load_json(package_dir / "manifest.json")
    records["package"] = load_json(package_dir / "package.json")
    for name, record in records.items():
        if record.get("run_id") != RUN_ID:
            raise ValueError(f"{name} record belongs to {record.get('run_id')!r}; expected {RUN_ID!r}.")
    validation = records["validation"]
    manifest = records["manifest"]
    if not validation.get("passed"):
        raise ValueError("Held-out validation did not pass; refusing to render a calibrated result.")
    actual = float(validation.get("improvement_pct", float("nan")))
    if not math.isclose(actual, AGGREGATE_HELDOUT_IMPROVEMENT_PCT, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Unexpected aggregate validation result {actual}; expected {AGGREGATE_HELDOUT_IMPROVEMENT_PCT}.")
    if len(records["plan"].get("heldout_episodes", [])) != 4:
        raise ValueError("Presentation contract requires exactly four locked held-out episodes.")
    if not manifest.get("activation_allowed") or manifest.get("status") != "validated":
        raise ValueError("Package is not the validated, activation-allowed full MVP1 package.")
    if manifest.get("scope") != VALIDATED_SCOPE:
        raise ValueError(f"Unexpected package scope: {manifest.get('scope')!r}")
    return records


def require_matching_scores(label: str, metadata: dict, expected_baseline: float, expected_tuned: float) -> None:
    """Fail closed if a replay sidecar drifts from the durable calibration record."""

    actual_baseline = float(metadata.get("baseline_score", float("nan")))
    actual_tuned = float(metadata.get("tuned_score", float("nan")))
    for name, actual, expected in (
        ("baseline", actual_baseline, float(expected_baseline)),
        ("tuned", actual_tuned, float(expected_tuned)),
    ):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(
                f"{label} replay {name} score {actual:.15g} does not match the durable record "
                f"{expected:.15g}. Refusing to render stale, reordered, or otherwise drifted trajectory data."
            )


def load_viewport_capture(directory: Path | None, records: dict, source_trajectory: Path) -> dict | None:
    """Load a verified three-lane actual-USD RTX capture, if requested."""

    if directory is None:
        return None
    result_path = directory / "live_result.json"
    trajectory_path = directory / "live_trajectories.npz"
    if not result_path.exists() or not trajectory_path.exists():
        raise FileNotFoundError(
            f"Viewport capture is incomplete: expected {result_path} and {trajectory_path}. "
            "Use --no-viewport to intentionally render only kinematic diagrams."
        )
    result = load_json(result_path)
    if result.get("capture_mode") != "kinematic_playback_of_verified_trajectories":
        raise ValueError(
            "Viewport capture is not the safe kinematic playback of the score-locked trajectory bundle."
        )
    if result.get("package_run_id") != RUN_ID:
        raise ValueError(f"Viewport capture belongs to {result.get('package_run_id')!r}; expected {RUN_ID!r}.")
    if result.get("package_status") != "validated" or not result.get("activation_allowed"):
        raise ValueError("Viewport capture is not tied to the activation-allowed validated package.")
    if result.get("calibration_scope") != VALIDATED_SCOPE:
        raise ValueError(f"Viewport capture has unexpected scope {result.get('calibration_scope')!r}.")
    if result.get("asset_sha256") != SOURCE_USD_SHA256:
        raise ValueError("Viewport capture does not use the pinned released SO-101 USD.")
    if result.get("evidence_revision") != records["manifest"]["inputs"]["evidence_revision"]:
        raise ValueError("Viewport capture evidence revision does not match the validated package.")
    if result.get("evidence_fingerprint") != records["manifest"]["inputs"]["evidence_fingerprint"]:
        raise ValueError("Viewport capture evidence fingerprint does not match the validated package.")
    if result.get("episode") not in records["validation"]["per_episode"]:
        raise ValueError("Viewport capture episode is not one of the four locked held-out episodes.")
    if not all(result.get(key) for key in ("baseline_stable", "tuned_stable")):
        raise ValueError("Viewport capture contains an unstable baseline or tuned Newton lane.")
    if "RTX" not in str(result.get("runtime", {}).get("renderer", "")):
        raise ValueError("Viewport capture was not rendered by the declared Isaac Sim RTX camera.")
    if "visualization only" not in str(result.get("isaac_sim_role", "")).lower():
        raise ValueError("Viewport capture does not explicitly declare Isaac Sim RTX as visualization-only.")
    if result.get("source_trajectory_sha256") != sha256(source_trajectory):
        raise ValueError("Viewport capture source trajectory hash does not match the verified renderer input.")
    source_sidecar = source_trajectory.with_suffix(".json")
    if result.get("source_sidecar_sha256") != sha256(source_sidecar):
        raise ValueError("Viewport capture source sidecar hash does not match the verified renderer input.")
    if result.get("output_trajectory_sha256") != sha256(trajectory_path):
        raise ValueError("Viewport capture output trajectory fingerprint is inconsistent.")
    exact_match = result.get("source_array_exact_match", {})
    expected_arrays = {
        "time_s",
        "command_q",
        "measured_q",
        "measured_dq",
        "baseline_q",
        "baseline_dq",
        "tuned_q",
        "tuned_dq",
        "joint_names",
    }
    if set(exact_match) != expected_arrays or not all(exact_match.values()):
        raise ValueError("Viewport arrays are not an exact window of the score-locked source trajectory bundle.")
    expected_baseline = {entry["name"]: float(entry["initial"]) for entry in records["plan"]["parameters"]}
    expected_tuned = {name: float(value) for name, value in records["manifest"]["parameters"].items()}
    for label, actual, expected in (
        ("baseline", result.get("baseline_parameters", {}), expected_baseline),
        ("tuned", result.get("tuned_parameters", {}), expected_tuned),
    ):
        if set(actual) != set(expected) or any(
            not math.isclose(float(actual[name]), value, rel_tol=0.0, abs_tol=1e-12)
            for name, value in expected.items()
        ):
            raise ValueError(f"Viewport {label} parameters do not match the locked full-run records.")
    trajectories = np.load(trajectory_path)
    joint_order = tuple(str(value) for value in trajectories["joint_names"].tolist())
    if joint_order != CANONICAL_JOINT_ORDER:
        raise ValueError(f"Viewport trajectory joint order {joint_order!r} is not canonical.")
    for name in expected_arrays - {"joint_names"}:
        array = trajectories[name]
        if not np.isfinite(array).all():
            raise ValueError(f"Viewport trajectory array {name!r} contains non-finite values.")
    expected_shape = (int(result["steps"]), len(CANONICAL_JOINT_ORDER))
    for name in ("command_q", "measured_q", "measured_dq", "baseline_q", "baseline_dq", "tuned_q", "tuned_dq"):
        if trajectories[name].shape != expected_shape:
            raise ValueError(f"Viewport trajectory array {name!r} has shape {trajectories[name].shape}; expected {expected_shape}.")
    frame_paths = {}
    expected_steps = int(result["steps"])
    for lane in ("measured", "baseline", "tuned"):
        paths = sorted((directory / "frames" / lane).glob("*.png"))
        if len(paths) != expected_steps:
            raise ValueError(f"Viewport {lane} lane has {len(paths)} frames; expected {expected_steps}.")
        frame_paths[lane] = paths
    if float(result["tuned_error"]["all_rmse_rad"]) >= float(result["baseline_error"]["all_rmse_rad"]):
        raise ValueError("Viewport displayed slice does not improve aggregate joint RMSE.")
    return {
        "directory": directory,
        "result": result,
        "trajectory_path": trajectory_path,
        "frame_paths": frame_paths,
    }


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4)) + (alpha,)


def text(draw: ImageDraw.ImageDraw, xy, value: str, size: int, color=WHITE, *, bold=False, anchor=None, mono=False):
    draw.text(xy, value, fill=color, font=font(size, bold=bold, mono=mono), anchor=anchor)


def fit_text(draw: ImageDraw.ImageDraw, xy, value: str, max_width: int, size: int, color=WHITE, *, bold=False):
    candidate = size
    while candidate > 13 and draw.textbbox((0, 0), value, font=font(candidate, bold=bold))[2] > max_width:
        candidate -= 1
    text(draw, xy, value, candidate, color, bold=bold)


def round_rect(draw, box, fill, outline=None, width=1, radius=24):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, color=MUTED, width=5, head=14):
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    points = [
        end,
        (
            end[0] - head * math.cos(angle - math.pi / 6),
            end[1] - head * math.sin(angle - math.pi / 6),
        ),
        (
            end[0] - head * math.cos(angle + math.pi / 6),
            end[1] - head * math.sin(angle + math.pi / 6),
        ),
    ]
    draw.polygon(points, fill=color)


def header(draw, kicker: str, title_value: str, subtitle: str | None = None):
    text(draw, (92, 54), kicker.upper(), 22, GREEN, bold=True)
    text(draw, (92, 92), title_value, 46, WHITE, bold=True)
    if subtitle:
        text(draw, (94, 153), subtitle, 23, MUTED)


def footer(draw, value: str):
    draw.line([(92, 1017), (1828, 1017)], fill=GRID, width=2)
    fit_text(draw, (92, 1034), value, 1400, 17, MUTED)
    text(draw, (1828, 1034), "SO-101 · Isaac Lab + Newton", 17, MUTED, anchor="ra")


def icon(draw, center, kind: str, color: str, scale: float = 1.0):
    x, y = center
    w = int(42 * scale)
    if kind == "analyze":
        draw.rounded_rectangle((x - w, y - w, x + w * 0.55, y + w), radius=7, outline=color, width=5)
        draw.line([(x - w * 0.7, y - w * 0.4), (x + w * 0.25, y - w * 0.4)], fill=color, width=4)
        draw.line([(x - w * 0.7, y), (x + w * 0.1, y)], fill=color, width=4)
        draw.ellipse((x + 4, y + 3, x + w, y + w), outline=color, width=5)
        draw.line([(x + w * 0.76, y + w * 0.78), (x + w * 1.12, y + w * 1.12)], fill=color, width=6)
    elif kind == "plan":
        draw.rounded_rectangle((x - w, y - w * 0.8, x + w, y + w * 0.8), radius=12, outline=color, width=5)
        for offset in (-0.42, 0.0, 0.42):
            draw.ellipse((x - w * 0.7, y + offset * w - 5, x - w * 0.7 + 10, y + offset * w + 5), fill=color)
            draw.line([(x - w * 0.42, y + offset * w), (x + w * 0.62, y + offset * w)], fill=color, width=4)
    elif kind == "fit":
        draw.ellipse((x - w, y - w, x + w, y + w), outline=color, width=5)
        draw.ellipse((x - w * 0.48, y - w * 0.48, x + w * 0.48, y + w * 0.48), outline=color, width=5)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
        draw.line([(x + w * 0.55, y - w * 0.55), (x + w * 1.05, y - w * 1.05)], fill=color, width=6)
    elif kind == "validate":
        points = [(x, y - w), (x + w * 0.85, y - w * 0.55), (x + w * 0.66, y + w * 0.6), (x, y + w), (x - w * 0.66, y + w * 0.6), (x - w * 0.85, y - w * 0.55)]
        draw.polygon(points, outline=color)
        draw.line([(x - w * 0.45, y), (x - w * 0.1, y + w * 0.35), (x + w * 0.5, y - w * 0.38)], fill=color, width=7)
    elif kind == "write":
        draw.polygon([(x - w, y - w * 0.6), (x, y - w), (x + w, y - w * 0.6), (x, y - w * 0.2)], fill=rgba(color, 100), outline=color)
        draw.polygon([(x - w, y - w * 0.6), (x, y - w * 0.2), (x, y + w), (x - w, y + w * 0.55)], outline=color)
        draw.polygon([(x + w, y - w * 0.6), (x, y - w * 0.2), (x, y + w), (x + w, y + w * 0.55)], outline=color)


def interpolate(values: np.ndarray, phase: float) -> np.ndarray:
    position = min(len(values) - 1, max(0.0, phase * (len(values) - 1)))
    low = int(math.floor(position))
    high = min(len(values) - 1, low + 1)
    fraction = position - low
    return values[low] * (1.0 - fraction) + values[high] * fraction


def arm_points(q: np.ndarray):
    yaw = float(q[0])
    shoulder_angle = -float(q[1]) + 0.12
    elbow_angle = shoulder_angle - float(q[2])
    wrist_angle = elbow_angle - float(q[3])
    p0 = np.asarray([0.0, 0.0, 0.0])
    p1 = np.asarray([0.0, 0.0, 0.075])

    def segment(length, angle):
        return length * np.asarray([math.cos(angle) * math.cos(yaw), math.cos(angle) * math.sin(yaw), math.sin(angle)])

    p2 = p1 + segment(0.116, shoulder_angle)
    p3 = p2 + segment(0.135, elbow_angle)
    p4 = p3 + segment(0.071, wrist_angle)
    forward = segment(0.06, wrist_angle)
    p5 = p4 + forward
    side = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0])
    roll = float(q[4])
    side = side * math.cos(roll) + np.asarray([0.0, 0.0, 1.0]) * math.sin(roll)
    jaw_fraction = float(np.clip(q[5] / 0.85, 0.0, 1.25))
    gap = 0.012 + 0.036 * jaw_fraction
    finger_start_a = p5 + side * gap / 2
    finger_start_b = p5 - side * gap / 2
    fingertip_a = finger_start_a + forward * 0.56
    fingertip_b = finger_start_b + forward * 0.56
    return [p0, p1, p2, p3, p4, p5], [(finger_start_a, fingertip_a), (finger_start_b, fingertip_b)]


def project(point: np.ndarray, box):
    left, top, right, bottom = box
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2 - 30
    right_basis = np.asarray([0.80, 0.60, 0.0])
    up_basis = np.asarray([-0.22, 0.30, 0.93])
    scale = min(right - left, bottom - top) * 1.25
    shifted = point - np.asarray([0.13, 0.0, 0.12])
    return center_x + float(shifted @ right_basis) * scale, center_y - float(shifted @ up_basis) * scale


def draw_floor(draw, box):
    left, top, right, bottom = box
    for value in np.linspace(-0.20, 0.45, 11):
        p0 = project(np.asarray([value, -0.28, 0.0]), box)
        p1 = project(np.asarray([value, 0.28, 0.0]), box)
        draw.line([p0, p1], fill=GRID, width=2)
    for value in np.linspace(-0.28, 0.28, 9):
        p0 = project(np.asarray([-0.20, value, 0.0]), box)
        p1 = project(np.asarray([0.45, value, 0.0]), box)
        draw.line([p0, p1], fill=GRID, width=2)


def draw_arm(draw, box, q: np.ndarray, color: str, *, ghost_q: np.ndarray | None = None):
    if ghost_q is not None:
        points, fingers = arm_points(ghost_q)
        screen = [project(point, box) for point in points]
        draw.line(screen, fill=rgba(BLUE, 75), width=16, joint="curve")
        for a, b in fingers:
            draw.line([project(a, box), project(b, box)], fill=rgba(BLUE, 75), width=10)
    points, fingers = arm_points(q)
    screen = [project(point, box) for point in points]
    shadow = [(x + 7, y + 10) for x, y in screen]
    draw.line(shadow, fill=(0, 0, 0, 90), width=28, joint="curve")
    draw.line(screen[:2], fill="#63727D", width=38)
    draw.line(screen[1:], fill=color, width=26, joint="curve")
    for index, point in enumerate(screen[1:-1], start=1):
        radius = 18 if index < 4 else 14
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=PANEL_2, outline=color, width=6)
    base = screen[0]
    draw.rounded_rectangle((base[0] - 44, base[1] - 17, base[0] + 44, base[1] + 27), radius=10, fill="#263744", outline="#80909A", width=3)
    for a, b in fingers:
        draw.line([project(a, box), project(b, box)], fill=color, width=12)
    grip = screen[-1]
    draw.ellipse((grip[0] - 11, grip[1] - 11, grip[0] + 11, grip[1] + 11), fill=WHITE, outline=color, width=4)


def draw_arm_panel(draw, box, label: str, sublabel: str, q: np.ndarray, color: str, *, ghost_q=None, badge=None):
    left, top, right, bottom = box
    round_rect(draw, box, PANEL, outline="#1D3444", width=2, radius=28)
    draw_floor(draw, (left + 16, top + 106, right - 16, bottom - 16))
    draw_arm(draw, (left + 16, top + 106, right - 16, bottom - 16), q, color, ghost_q=ghost_q)
    draw.ellipse((left + 28, top + 28, left + 44, top + 44), fill=color)
    text(draw, (left + 56, top + 22), label, 24, WHITE, bold=True)
    text(draw, (left + 56, top + 53), sublabel, 15, MUTED)
    if badge:
        badge_width = 176
        round_rect(draw, (right - badge_width - 20, top + 20, right - 20, top + 54), "#FFF4CC", outline=AMBER, width=2, radius=14)
        text(draw, (right - badge_width / 2 - 20, top + 37), badge, 14, AMBER, bold=True, anchor="mm")


def draw_metric_card(draw, box, title_value: str, before: float, after: float, unit="rad"):
    left, top, right, bottom = box
    round_rect(draw, box, PANEL_2, outline="#294253", width=2, radius=22)
    text(draw, (left + 24, top + 18), title_value, 17, MUTED, bold=True)
    text(draw, (left + 24, top + 52), f"{before:.2f}", 34, CORAL, bold=True)
    arrow(draw, (left + 122, top + 68), (left + 158, top + 68), MUTED, 3, 8)
    text(draw, (left + 176, top + 52), f"{after:.2f} {unit}", 34, GREEN, bold=True)
    reduction = 100.0 * (before - after) / max(abs(before), 1e-9)
    round_rect(draw, (right - 146, top + 40, right - 20, bottom - 18), "#EDF6DC", outline=GREEN_DARK, width=2, radius=15)
    text(draw, (right - 83, (top + bottom) / 2 + 10), f"{reduction:.0f}% lower", 17, GREEN, bold=True, anchor="mm")


def draw_vertical_metric_card(draw, box, title_value: str, before: float, after: float, unit="rad"):
    left, top, right, bottom = box
    round_rect(draw, box, PANEL_2, outline="#294253", width=2, radius=24)
    text(draw, (left + 28, top + 28), title_value, 17, MUTED, bold=True)
    text(draw, (left + 28, top + 92), "BEFORE", 14, CORAL, bold=True)
    text(draw, (left + 28, top + 120), f"{before:.3f} {unit}", 37, WHITE, bold=True)
    draw.line([(left + 28, top + 184), (right - 28, top + 184)], fill=GRID, width=2)
    text(draw, (left + 28, top + 218), "AFTER TUNING", 14, GREEN, bold=True)
    text(draw, (left + 28, top + 248), f"{after:.3f} {unit}", 37, WHITE, bold=True)
    reduction = 100.0 * (before - after) / max(abs(before), 1e-9)
    round_rect(draw, (left + 28, bottom - 106, right - 28, bottom - 34), "#EDF6DC", outline=GREEN_DARK, width=2, radius=18)
    text(draw, ((left + right) / 2, bottom - 70), f"{reduction:.0f}% lower", 24, GREEN, bold=True, anchor="mm")


def draw_plot(draw, box, series, current_phase: float, *, title_value: str, y_label: str):
    left, top, right, bottom = box
    round_rect(draw, box, PANEL, outline="#1D3444", width=2, radius=24)
    text(draw, (left + 24, top + 18), title_value, 18, WHITE, bold=True)
    plot = (left + 68, top + 62, right - 24, bottom - 48)
    values = np.concatenate([entry[1] for entry in series])
    y_min, y_max = float(values.min()), float(values.max())
    pad = max(0.04, (y_max - y_min) * 0.10)
    y_min, y_max = y_min - pad, y_max + pad
    for fraction in np.linspace(0.0, 1.0, 5):
        y = lerp(plot[3], plot[1], fraction)
        draw.line([(plot[0], y), (plot[2], y)], fill=GRID, width=2)
        value = lerp(y_min, y_max, fraction)
        text(draw, (plot[0] - 10, y), f"{value:.1f}", 13, MUTED, anchor="rm")
    current_x = lerp(plot[0], plot[2], current_phase)
    for label, values_array, color in series:
        limit = max(2, int(current_phase * (len(values_array) - 1)) + 1)
        points = []
        for index, value in enumerate(values_array[:limit]):
            x = lerp(plot[0], plot[2], index / max(1, len(values_array) - 1))
            y = lerp(plot[3], plot[1], (float(value) - y_min) / max(1e-9, y_max - y_min))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=5, joint="curve")
        legend_x = plot[0] + [entry[0] for entry in series].index(label) * 178
        draw.line([(legend_x, bottom - 24), (legend_x + 26, bottom - 24)], fill=color, width=5)
        text(draw, (legend_x + 34, bottom - 31), label, 14, MUTED)
    draw.line([(current_x, plot[1]), (current_x, plot[3])], fill=rgba(WHITE, 80), width=2)
    text(draw, (left + 18, (plot[1] + plot[3]) / 2), y_label, 13, MUTED, anchor="mm")


def pipeline_scene(draw, phase: float):
    header(draw, "MVP 1 · FIVE CALLS", "One repeatable calibration job", "Evidence and Newton remain synchronized from intake through validation.")
    calls = [
        ("1", "analyze()", "Evidence readiness", "analyze"),
        ("2", "plan()", "Recipe + bounds", "plan"),
        ("3", "fit()", "Search candidates", "fit"),
        ("4", "validate()", "Held-out proof", "validate"),
        ("5", "write()", "Scoped record", "write"),
    ]
    card_w, gap, y0, y1 = 278, 52, 300, 650
    x0 = 92
    active = min(4, int(phase * 5.0))
    for index, (number, name, sub, kind) in enumerate(calls):
        left = x0 + index * (card_w + gap)
        right = left + card_w
        highlight = index <= active
        fill = "#F3F9E9" if highlight else PANEL
        outline = GREEN if index == active else (GREEN_DARK if highlight else GRID)
        round_rect(draw, (left, y0, right, y1), fill, outline=outline, width=4 if index == active else 2, radius=28)
        round_rect(draw, (left + 22, y0 + 22, left + 66, y0 + 66), GREEN if highlight else GRID, radius=12)
        text(draw, (left + 44, y0 + 44), number, 18, "#14171C" if highlight else MUTED, bold=True, anchor="mm")
        icon(draw, ((left + right) / 2, y0 + 146), kind, GREEN if highlight else MUTED, scale=0.76)
        text(draw, ((left + right) / 2, y0 + 228), name, 27, WHITE, bold=True, anchor="mm", mono=True)
        text(draw, ((left + right) / 2, y0 + 274), sub, 18, MUTED, anchor="mm")
        if index < len(calls) - 1:
            arrow(draw, (right + 9, (y0 + y1) / 2), (right + gap - 9, (y0 + y1) / 2), GREEN if index < active else MUTED, 5, 12)
    fit_left = x0 + 2 * (card_w + gap)
    newton_box = (fit_left + 14, 724, fit_left + card_w - 14, 846)
    round_rect(draw, newton_box, "#F3F9E9", outline=GREEN_DARK, width=3, radius=22)
    text(draw, ((newton_box[0] + newton_box[2]) / 2, 757), "ISAAC LAB + NEWTON", 16, GREEN, bold=True, anchor="mm")
    text(draw, ((newton_box[0] + newton_box[2]) / 2, 799), "candidate / score", 20, WHITE, anchor="mm")
    draw.line([(fit_left + card_w / 2 - 12, y1), (fit_left + card_w / 2 - 12, newton_box[1])], fill=GREEN, width=4)
    arrow(draw, (fit_left + card_w / 2 + 12, newton_box[1]), (fit_left + card_w / 2 + 12, y1), GREEN, 4, 10)
    round_rect(draw, (92, 892, 1828, 970), PANEL_2, outline="#294253", width=2, radius=19)
    text(draw, (122, 921), "DURABLE RECORD", 15, GREEN, bold=True)
    text(draw, (300, 915), "inputs  ·  parameter bounds  ·  every candidate  ·  metrics  ·  held-out result", 21, WHITE)
    footer(draw, "The same five APIs support CLI, CI, or a future calibration agent.")


def draw_gripper(draw, box, q_value: float, color: str, *, ghost_value: float | None = None):
    left, top, right, bottom = box
    cx, cy = (left + right) / 2, (top + bottom) / 2 - 25
    round_rect(draw, (cx - 98, cy - 148, cx + 98, cy + 42), "#263744", outline="#788996", width=3, radius=24)
    draw.ellipse((cx - 40, cy - 104, cx + 40, cy - 24), fill=PANEL, outline=color, width=7)
    jaw_fraction = float(np.clip(q_value / 0.85, 0.0, 1.3))
    gap = 44 + 126 * jaw_fraction
    if ghost_value is not None:
        ghost_gap = 44 + 126 * float(np.clip(ghost_value / 0.85, 0.0, 1.3))
        for sign in (-1, 1):
            x = cx + sign * ghost_gap / 2
            draw.rounded_rectangle((x - 16, cy - 16, x + 16, cy + 180), radius=12, outline=rgba(BLUE, 90), width=7)
    for sign in (-1, 1):
        x = cx + sign * gap / 2
        draw.rounded_rectangle((x - 19, cy - 18, x + 19, cy + 182), radius=13, fill=color, outline=WHITE, width=2)
        draw.rounded_rectangle((x - 15, cy + 92, x + 15, cy + 174), radius=10, fill="#1C2831")
    draw.line([(cx - gap / 2, cy + 210), (cx + gap / 2, cy + 210)], fill=color, width=5)
    arrow(draw, (cx - gap / 2 + 24, cy + 210), (cx - gap / 2, cy + 210), color, 4, 11)
    arrow(draw, (cx + gap / 2 - 24, cy + 210), (cx + gap / 2, cy + 210), color, 4, 11)
    text(draw, (cx, cy + 235), f"jaw {q_value:.2f} rad", 18, color, bold=True, anchor="mm")


def arm_comparison_frame(data, meta, phase: float, *, compact=False):
    image = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(image, "RGBA")
    header(draw, "DISPLAYED HELD-OUT EPISODE", "SO-101 arm actuation: measured vs. Newton", "Same command · same initial state · synchronized 12 s comparison")
    q_measured = interpolate(data["measured_q"], phase)
    q_baseline = interpolate(data["baseline_q"], phase)
    q_tuned = interpolate(data["tuned_q"], phase)
    panels = [(92, 230, 642, 730), (685, 230, 1235, 730), (1278, 230, 1828, 730)]
    draw_arm_panel(draw, panels[0], "Measured", "real Anchor-Lab telemetry\nkinematically reconstructed", q_measured, BLUE)
    draw_arm_panel(draw, panels[1], "Baseline", "recipe-initial actuator settings\nalready-calibrated released USD", q_baseline, CORAL, ghost_q=q_measured)
    draw_arm_panel(draw, panels[2], "Tuned", "Newton simulation\nselected parameters", q_tuned, GREEN, ghost_q=q_measured, badge="HELD-OUT")
    before = float(meta["baseline_error"]["arm_rmse_rad"])
    after = float(meta["tuned_error"]["arm_rmse_rad"])
    draw_metric_card(draw, (92, 770, 705, 892), "ARM JOINT-SPACE RMSE", before, after)
    round_rect(draw, (742, 770, 1828, 892), PANEL_2, outline="#294253", width=2, radius=22)
    text(draw, (770, 792), "WHAT CHANGED", 15, GREEN, bold=True)
    text(draw, (770, 828), "stiffness · damping · armature · friction · effort · command delay", 23, WHITE)
    progress_x = lerp(92, 1828, phase)
    draw.line([(92, 940), (1828, 940)], fill=GRID, width=7)
    draw.line([(92, 940), (progress_x, 940)], fill=GREEN, width=7)
    draw.ellipse((progress_x - 9, 931, progress_x + 9, 949), fill=WHITE)
    episode = str(meta["episode"]).split("heldout-")[-1].replace("-", " ")
    footer(draw, f"Displayed episode only: {episode} · measured geometry is kinematically reconstructed; aggregate proof is reported separately.")
    return image


def gripper_comparison_frame(data, meta, phase: float):
    image = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(image, "RGBA")
    header(draw, "DISPLAYED HELD-OUT EPISODE", "SO-101 unloaded jaw response", "Jaw channel of the held-out frequency sweep · synchronized command and timing")
    measured = interpolate(data["measured_q"], phase)[5]
    baseline = interpolate(data["baseline_q"], phase)[5]
    tuned = interpolate(data["tuned_q"], phase)[5]
    panels = [(92, 220, 512, 715), (540, 220, 960, 715), (988, 220, 1408, 715)]
    definitions = [
        ("Measured", "real telemetry · reconstructed", measured, BLUE, None),
        ("Baseline", "recipe-initial · calibrated USD", baseline, CORAL, measured),
        ("Tuned Newton", "simulation · selected parameters", tuned, GREEN, measured),
    ]
    for box, (label, sublabel, value, color, ghost) in zip(panels, definitions, strict=True):
        round_rect(draw, box, PANEL, outline="#1D3444", width=2, radius=28)
        draw.ellipse((box[0] + 28, box[1] + 28, box[0] + 44, box[1] + 44), fill=color)
        text(draw, (box[0] + 56, box[1] + 21), label, 22, WHITE, bold=True)
        text(draw, (box[0] + 56, box[1] + 52), sublabel, 14, MUTED)
        draw_gripper(draw, (box[0] + 16, box[1] + 82, box[2] - 16, box[3] - 16), value, color, ghost_value=ghost)
    draw_vertical_metric_card(
        draw,
        (1440, 220, 1828, 715),
        "JAW RMSE",
        float(meta["baseline_error"]["jaw_rmse_rad"]),
        float(meta["tuned_error"]["jaw_rmse_rad"]),
    )
    draw_plot(
        draw,
        (92, 752, 1828, 952),
        [
            ("Measured", data["measured_q"][:, 5], BLUE),
            ("Baseline", data["baseline_q"][:, 5], CORAL),
            ("Tuned", data["tuned_q"][:, 5], GREEN),
        ],
        phase,
        title_value="Jaw angle over the synchronized episode",
        y_label="rad",
    )
    footer(draw, "Displayed held-out episode only · unloaded jaw position response, not grip force, object contact, or grasping.")
    return image


def paste_contained(
    canvas: Image.Image,
    source_path: Path,
    box: tuple[int, int, int, int],
    *,
    normalized_crop: tuple[float, float, float, float] | None = None,
) -> None:
    left, top, right, bottom = box
    with Image.open(source_path) as source:
        frame = source.convert("RGB")
        if normalized_crop is not None:
            crop = (
                int(round(frame.width * normalized_crop[0])),
                int(round(frame.height * normalized_crop[1])),
                int(round(frame.width * normalized_crop[2])),
                int(round(frame.height * normalized_crop[3])),
            )
            frame = frame.crop(crop)
        frame.thumbnail((right - left, bottom - top), Image.Resampling.LANCZOS)
        x = left + (right - left - frame.width) // 2
        y = top + (bottom - top - frame.height) // 2
        canvas.paste(frame, (x, y))


def overview_viewport_scene(canvas: Image.Image, draw, capture: dict, phase: float):
    result = capture["result"]
    header(
        draw,
        "VERIFIED TRAJECTORIES · ACTUAL USD · ISAAC SIM RTX",
        "The score-locked motion rendered on the real asset",
        "Visualization only · physics authority remains the recorded Newton trajectory bundle",
    )
    count = len(capture["frame_paths"]["measured"])
    index = min(count - 1, max(0, int(round(phase * (count - 1)))))
    panels = [(92, 230, 642, 760), (685, 230, 1235, 760), (1278, 230, 1828, 760)]
    lanes = [
        ("measured", "Measured", "real telemetry · kinematic USD playback", BLUE),
        ("baseline", "Baseline", "verified Newton array · USD playback", CORAL),
        ("tuned", "Tuned Newton", "verified Newton array · USD playback", GREEN),
    ]
    for panel, (lane, label, sublabel, accent) in zip(panels, lanes, strict=True):
        left, top, right, bottom = panel
        round_rect(draw, panel, PANEL, outline=accent, width=3, radius=24)
        draw.ellipse((left + 26, top + 25, left + 42, top + 41), fill=accent)
        text(draw, (left + 54, top + 17), label, 22, WHITE, bold=True)
        text(draw, (left + 54, top + 49), sublabel, 14, MUTED)
        paste_contained(
            canvas,
            capture["frame_paths"][lane][index],
            (left + 12, top + 82, right - 12, bottom - 12),
            normalized_crop=(0.111, 0.139, 0.889, 0.785),
        )
        draw.rounded_rectangle((left + 12, top + 82, right - 12, bottom - 12), radius=14, outline=GRID, width=2)
    before = float(result["baseline_error"]["all_rmse_rad"])
    after = float(result["tuned_error"]["all_rmse_rad"])
    draw_metric_card(draw, (312, 806, 1182, 932), "DISPLAYED RTX SLICE · ALL-JOINT RMSE", before, after)
    round_rect(draw, (1225, 806, 1828, 932), PANEL_2, outline=GRID, width=2, radius=22)
    text(draw, (1254, 828), "LOCAL VIEWPORT SLICE", 15, BLUE, bold=True)
    episode = str(result["episode"]).split("heldout-")[-1].replace("-", " ")
    text(draw, (1254, 862), episode, 19, WHITE, bold=True)
    trajectory_source = result["runtime"].get("physics_trajectory_source", "recorded Newton rollout")
    text(draw, (1254, 894), f"{result['steps']} frames · {trajectory_source}", 16, MUTED)
    footer(draw, "RTX is visualization-only · measured=real telemetry; baseline/tuned=verified Newton arrays · aggregate is in CALL 4.")


def overview_arm_scene(draw, data, meta, phase: float):
    header(draw, "EVIDENCE TO PHYSICS", "The gap becomes visible—and measurable", "One held-out command replay, aligned across measured and simulated motion.")
    q_measured = interpolate(data["measured_q"], phase)
    q_baseline = interpolate(data["baseline_q"], phase)
    q_tuned = interpolate(data["tuned_q"], phase)
    panels = [(92, 250, 642, 760), (685, 250, 1235, 760), (1278, 250, 1828, 760)]
    draw_arm_panel(draw, panels[0], "Measured", "real Anchor-Lab telemetry\nkinematically reconstructed", q_measured, BLUE)
    draw_arm_panel(draw, panels[1], "Baseline", "recipe-initial actuator settings\nalready-calibrated released USD", q_baseline, CORAL, ghost_q=q_measured)
    draw_arm_panel(draw, panels[2], "Tuned Newton", "simulation · selected parameters", q_tuned, GREEN, ghost_q=q_measured)
    before = float(meta["baseline_error"]["arm_rmse_rad"])
    after = float(meta["tuned_error"]["arm_rmse_rad"])
    draw_metric_card(draw, (390, 806, 1530, 932), "HELD-OUT ARM RMSE", before, after)
    footer(draw, "Displayed episode only · 4-episode aggregate is in CALL 4.")


def overview_gripper_scene(draw, data, meta, phase: float):
    header(draw, "HELD-OUT JAW CHANNEL", "The same unseen motion tests jaw response", "The displayed frequency-sweep episode exercises a 0.30 rad jaw-command range.")
    measured = interpolate(data["measured_q"], phase)[5]
    baseline = interpolate(data["baseline_q"], phase)[5]
    tuned = interpolate(data["tuned_q"], phase)[5]
    boxes = [(92, 250, 610, 770), (701, 250, 1219, 770), (1310, 250, 1828, 770)]
    for box, label, value, color, ghost in [
        (boxes[0], "Measured", measured, BLUE, None),
        (boxes[1], "Baseline", baseline, CORAL, measured),
        (boxes[2], "Tuned Newton", tuned, GREEN, measured),
    ]:
        round_rect(draw, box, PANEL, outline="#1D3444", width=2, radius=28)
        text(draw, ((box[0] + box[2]) / 2, box[1] + 42), label, 25, WHITE, bold=True, anchor="mm")
        draw_gripper(draw, (box[0] + 18, box[1] + 55, box[2] - 18, box[3] - 12), value, color, ghost_value=ghost)
    draw_metric_card(
        draw,
        (390, 810, 1530, 932),
        "DISPLAYED HELD-OUT JAW RMSE",
        float(meta["baseline_error"]["jaw_rmse_rad"]),
        float(meta["tuned_error"]["jaw_rmse_rad"]),
    )
    footer(draw, "Displayed held-out episode metric only · unloaded jaw position, not gripping-force or contact validation.")


def title_scene(draw, phase: float):
    glow = int(45 + 40 * math.sin(phase * math.pi))
    for radius in range(360, 80, -24):
        alpha = int(glow * (1.0 - radius / 390.0))
        draw.ellipse((960 - radius, 510 - radius, 960 + radius, 510 + radius), outline=rgba(GREEN, alpha), width=3)
    round_rect(draw, (140, 108, 1780, 920), "#FFFFFF", outline=GREEN_DARK, width=3, radius=42)
    round_rect(draw, (164, 134, 470, 184), GREEN, radius=20)
    text(draw, (317, 159), "NEWTON CALIBRATION · MVP 1", 18, "#14171C", bold=True, anchor="mm")
    text(draw, (960, 334), "SO-101 physics tuning", 64, WHITE, bold=True, anchor="mm")
    text(draw, (960, 420), "Measured evidence  ·  repeatable Newton calibration  ·  traceable proof", 27, MUTED, anchor="mm")
    draw.line([(410, 550), (1510, 550)], fill=GRID, width=4)
    for x, label, color in [(530, "REAL EVIDENCE", BLUE), (960, "ISAAC LAB + NEWTON", GREEN), (1390, "TRACEABLE RECORD", "#141D42")]:
        draw.ellipse((x - 34, 516, x + 34, 584), fill=PANEL_2, outline=color, width=6)
        text(draw, (x, 636), label, 18, color, bold=True, anchor="mm")
    arrow(draw, (600, 550), (878, 550), MUTED, 5, 14)
    arrow(draw, (1042, 550), (1320, 550), MUTED, 5, 14)
    text(draw, (960, 790), "Five calls. One durable job. No one-off fitting glue.", 31, WHITE, bold=True, anchor="mm")
    text(draw, (960, 854), "Validated scope: free-space arm + unloaded jaw actuation", 19, GREEN_DARK, anchor="mm")


def call_scene_header(draw, call_number: int, call_name: str, title_value: str, subtitle: str):
    header(draw, f"CALL {call_number} OF 5 · {call_name.upper()}()", title_value, subtitle)
    labels = ("analyze", "plan", "fit", "validate", "write")
    x = 1160
    for index, label in enumerate(labels, start=1):
        fill = GREEN if index == call_number else ("#F3F9E9" if index < call_number else PANEL_2)
        outline = GREEN if index <= call_number else GRID
        round_rect(draw, (x, 60, x + 125, 102), fill, outline=outline, width=2, radius=13)
        text(draw, (x + 62, 81), label, 15, "#FFFFFF" if index == call_number else (GREEN_DARK if index < call_number else MUTED), bold=True, anchor="mm")
        x += 138


def detail_card(draw, box, title_value: str, body: str, *, accent=GREEN, value: str | None = None):
    left, top, right, bottom = box
    round_rect(draw, box, PANEL, outline=GRID, width=2, radius=22)
    draw.rectangle((left, top, left + 9, bottom), fill=accent)
    text(draw, (left + 34, top + 26), title_value, 18, accent, bold=True)
    if value:
        text(draw, (left + 34, top + 66), value, 37, WHITE, bold=True)
        body_y = top + 119
    else:
        body_y = top + 70
    for index, line in enumerate(body.split("\n")):
        text(draw, (left + 34, body_y + 31 * index), line, 18, MUTED)


def analyze_scene(draw, records: dict):
    analysis = records["analysis"]
    call_scene_header(draw, 1, "analyze", "Check whether the evidence can support this recipe", "Nothing is tuned yet. The toolkit inspects provenance, signals, splits, and claim limits.")
    rates = analysis["sample_rates_hz"]
    detail_card(draw, (92, 245, 618, 535), "REAL EVIDENCE", "42 train + 8 held-out episodes\n6 joints · q, dq, command_q\nrevision and fingerprint locked", accent=BLUE, value="50 episodes")
    detail_card(draw, (697, 245, 1223, 535), "TIMING QUALITY", f"joint state ≈ {rates['actual_q']:.0f} Hz\ncommand ≈ {rates['command_q']:.0f} Hz\nindependent streams expose delay", accent=GREEN, value="READY")
    detail_card(draw, (1302, 245, 1828, 535), "RECIPE READINESS", "evidence exposes arm + jaw motion\nrecipe permits controller, dynamics\nand timing calibration", accent=GREEN_DARK, value="11 tunables")
    round_rect(draw, (92, 610, 1828, 905), "#FDF4E8", outline=AMBER, width=2, radius=24)
    text(draw, (128, 645), "ANALYZE ALSO LIMITS THE CLAIM", 18, AMBER, bold=True)
    notes = [
        "Raw load and tau_abs are excluded: conversion and torque sign are unavailable.",
        "The released SO-101 USD is already calibrated; the baseline is recipe-initial actuator settings on that USD.",
        "Evidence supports free-space arm and unloaded-jaw motion—not force, contact, grasp, insertion, policy, or real transfer.",
    ]
    for index, line in enumerate(notes):
        draw.ellipse((132, 705 + index * 62, 148, 721 + index * 62), fill=AMBER)
        text(draw, (174, 696 + index * 62), line, 21, WHITE)
    footer(draw, f"Evidence fingerprint · {analysis['evidence_fingerprint'][:16]}… · physical Anchor-Lab telemetry")


def plan_scene(draw, records: dict):
    plan = records["plan"]
    call_scene_header(draw, 2, "plan", "Lock one reproducible calibration job", "The recipe determines evidence use, tunable bounds, objective, optimizer budget, and held-out gates.")
    optimizer = plan["optimizer"]
    cards = [
        ("FIT SET", "4 × 12 s", "step response\nchirp sweep\nstatic holding\ngripper cycles", BLUE),
        ("HELD-OUT SET", "4 × 12 s", "frequency sweep\nfriction + gravity\nhold under gravity\nbacklash detection", GREEN),
        ("SEARCH", "144 candidates", f"{optimizer['name']}\n{optimizer['generations']} generations × {optimizer['population']}\nseed {optimizer['seed']} · resumable", AMBER),
    ]
    for index, (name, value, body, accent) in enumerate(cards):
        left = 92 + index * 579
        detail_card(draw, (left, 246, left + 526, 600), name, body, accent=accent, value=value)
    round_rect(draw, (92, 665, 1828, 910), PANEL_2, outline=GRID, width=2, radius=24)
    text(draw, (128, 700), "PARAMETER OWNERSHIP", 17, GREEN, bold=True)
    text(draw, (128, 744), "Isaac Lab explicit-PD · 6", 24, WHITE, bold=True)
    text(draw, (128, 782), "arm/jaw stiffness · damping · effort", 17, MUTED)
    draw.line((655, 720, 655, 834), fill=GRID, width=2)
    text(draw, (700, 744), "Newton dynamics · 4", 24, WHITE, bold=True)
    text(draw, (700, 782), "arm/jaw armature · joint friction", 17, MUTED)
    draw.line((1217, 720, 1217, 834), fill=GRID, width=2)
    text(draw, (1260, 744), "Toolkit timing · 1", 24, WHITE, bold=True)
    text(draw, (1260, 782), "delay · 0–80 ms; applied in 8.33 ms steps", 16, MUTED)
    text(draw, (128, 853), "Locked gate · ≥30% aggregate improvement · ≤10% episode regression · finite/bounded rollouts", 20, GREEN_DARK, bold=True)
    footer(draw, f"Recipe · {plan['recipe']} · dt {plan['environment']['dt']:.6f} s · solver fixed at {plan['environment']['solver_iterations']} iterations")


def fit_scene(draw, records: dict, phase: float):
    fit = records["fit"]
    baseline = float(fit["baseline"]["score"])
    best = float(fit["best"]["score"])
    candidate_count = int(fit["completed_generations"]) * int(fit["plan"]["optimizer"]["population"])
    call_scene_header(draw, 3, "fit", "Run candidate experiments in Newton", "Optimizer proposes values; the toolkit applies them, replays commands, scores trajectories, and checkpoints.")
    x_positions = (245, 680, 1115, 1550)
    boxes = [
        ("OPTIMIZER", "proposes bounded\nparameter vector", AMBER),
        ("TOOLKIT", "applies candidate\n+ records provenance", GREEN),
        ("ISAAC LAB + NEWTON", "replays four\nfit commands", GREEN_DARK),
        ("METRICS", "returns weighted\ntrajectory score", BLUE),
    ]
    for index, (title_value, body, accent) in enumerate(boxes):
        x = x_positions[index]
        round_rect(draw, (x - 175, 300, x + 175, 545), PANEL, outline=accent, width=3, radius=25)
        text(draw, (x, 348), title_value, 18, accent, bold=True, anchor="mm")
        for line_index, line in enumerate(body.split("\n")):
            text(draw, (x, 415 + line_index * 34), line, 21, WHITE, bold=True, anchor="mm")
        if index < len(boxes) - 1:
            arrow(draw, (x + 190, 423), (x_positions[index + 1] - 190, 423), MUTED, 4, 12)
    arrow(draw, (1550, 580), (245, 580), GREEN, 4, 12)
    text(draw, (895, 606), "candidate score closes the loop", 17, GREEN_DARK, bold=True, anchor="mm")
    round_rect(draw, (92, 690, 1828, 908), PANEL_2, outline=GRID, width=2, radius=24)
    progress = min(1.0, max(0.0, phase))
    text(draw, (128, 724), f"{candidate_count} candidates · {fit['completed_generations']} completed generations · backend {fit['backend']}", 20, MUTED)
    text(draw, (128, 770), f"FIT SCORE   {baseline:.4f}", 32, CORAL, bold=True)
    arrow(draw, (610, 792), (728, 792), MUTED, 4, 13)
    text(draw, (770, 770), f"{best:.4f}", 32, GREEN, bold=True)
    reduction = 100.0 * (baseline - best) / max(abs(baseline), 1e-12)
    text(draw, (1070, 770), f"{reduction:.1f}% lower on fit evidence", 27, GREEN_DARK, bold=True)
    draw.line((128, 856, 1788, 856), fill=GRID, width=14)
    draw.line((128, 856, lerp(128, 1788, progress), 856), fill=GREEN, width=14)
    footer(draw, "Fit improvement selects a candidate; it is not the held-out claim. CALL 4 decides whether the package passes.")


def validate_scene(draw, records: dict):
    validation = records["validation"]
    before = float(validation["baseline_metrics"]["score"])
    after = float(validation["calibrated_metrics"]["score"])
    call_scene_header(draw, 4, "validate", "Validate on evidence excluded from the fit objective", "The aggregate gate and every episode regression check are evaluated on four locked 12 s holdouts.")
    round_rect(draw, (92, 240, 720, 900), "#F3F9E9", outline=GREEN, width=3, radius=28)
    text(draw, (132, 286), "AGGREGATE HELD-OUT SCORE", 18, GREEN_DARK, bold=True)
    text(draw, (132, 348), f"{before:.4f}", 52, CORAL, bold=True)
    arrow(draw, (368, 380), (458, 380), MUTED, 5, 14)
    text(draw, (500, 348), f"{after:.4f}", 52, GREEN, bold=True)
    text(draw, (132, 458), f"{validation['improvement_pct']:.3f}% lower", 45, GREEN_DARK, bold=True)
    text(draw, (132, 530), "4 held-out episodes × 12 s", 24, WHITE, bold=True)
    gates = validation["gates"]
    gate_rows = [
        ("held-out only", gates["heldout_only"]),
        ("≥30% aggregate gain", gates["minimum_improvement"]),
        ("no episode >10% worse", gates["no_large_episode_regression"]),
        ("finite/bounded rollouts", gates["stable"]),
    ]
    for index, (label, passed) in enumerate(gate_rows):
        y = 605 + index * 59
        draw.ellipse((134, y, 162, y + 28), fill=GREEN if passed else CORAL)
        text(draw, (148, y + 14), "P" if passed else "X", 15, "#FFFFFF", bold=True, anchor="mm")
        text(draw, (188, y - 2), label, 21, WHITE)

    round_rect(draw, (780, 240, 1828, 900), PANEL, outline=GRID, width=2, radius=28)
    text(draw, (824, 286), "PER-EPISODE WEIGHTED SCORE", 18, BLUE, bold=True)
    y = 350
    for episode, values in validation["per_episode"].items():
        label = episode.split("heldout-")[-1].replace("-", " ")
        b = float(values["baseline"]["score"])
        a = float(values["calibrated"]["score"])
        gain = 100.0 * (b - a) / max(abs(b), 1e-12)
        text(draw, (824, y), label, 20, WHITE, bold=True)
        text(draw, (1430, y), f"{b:.3f}  to  {a:.3f}", 19, MUTED)
        text(draw, (1775, y), f"{gain:.1f}%", 19, GREEN_DARK, bold=True, anchor="ra")
        draw.line((824, y + 40, 1776, y + 40), fill=GRID, width=2)
        y += 118
    footer(draw, "92.099% is the weighted aggregate validation result—not the metric of one displayed trajectory.")


def package_scene(draw, records: dict):
    manifest = records["manifest"]
    plan = records["plan"]
    selected = manifest["parameters"]
    at_bounds = []
    for parameter in plan["parameters"]:
        value = float(selected[parameter["name"]])
        if math.isclose(value, float(parameter["lower"]), abs_tol=1e-9) or math.isclose(value, float(parameter["upper"]), abs_tol=1e-9):
            at_bounds.append(parameter["name"])
    selected_delay_ms = 1000.0 * float(selected["command_delay_s"])
    dt = float(plan["environment"]["dt"])
    effective_delay_steps = int(round(float(selected["command_delay_s"]) / dt))
    effective_delay_ms = 1000.0 * effective_delay_steps * dt
    call_scene_header(draw, 5, "write", "Write the setup-scoped package and job record", "The portable package carries the source USD, selected values, validation proof, and the complete calibration ledger.")
    round_rect(draw, (92, 238, 1090, 910), PANEL, outline=GREEN, width=3, radius=30)
    round_rect(draw, (130, 278, 1052, 375), "#F3F9E9", outline=GREEN, width=2, radius=18)
    text(draw, (166, 300), "SETUP-SCOPED CALIBRATION PACKAGE", 18, GREEN_DARK, bold=True)
    text(draw, (166, 337), f"run {manifest['run_id']}", 19, WHITE, mono=True)
    artifacts = [
        ("so101_no_camera_new_calib.usd", "portable copy of the pinned source asset"),
        ("calibration.usda", "relative source sublayer + provenance metadata"),
        ("isaaclab_actuator.yaml", "explicit-PD + Newton dynamics values + delay"),
        ("manifest + validation", "fingerprints · scope · four holdouts + gate"),
        ("job/candidate-history.jsonl", "all 144 candidates + complete run records"),
    ]
    for index, (name, detail) in enumerate(artifacts):
        y = 420 + index * 82
        draw.ellipse((154, y + 8, 174, y + 28), fill=GREEN)
        text(draw, (198, y), name, 18, WHITE, bold=True, mono=True)
        text(draw, (545, y + 2), detail, 17, MUTED)
    round_rect(draw, (1135, 238, 1828, 910), PANEL_2, outline=GRID, width=2, radius=30)
    text(draw, (1175, 282), "ACTIVATION DECISION", 17, GREEN, bold=True)
    text(draw, (1175, 326), "PASS", 48, GREEN_DARK, bold=True)
    text(draw, (1175, 400), "Scoped to", 17, MUTED, bold=True)
    text(draw, (1175, 438), "free-space arm +", 27, WHITE, bold=True)
    text(draw, (1175, 477), "unloaded jaw actuation", 27, WHITE, bold=True)
    text(draw, (1175, 552), f"{len(selected)} selected values", 22, WHITE, bold=True)
    text(draw, (1175, 591), f"{len(at_bounds)} at declared bounds", 20, AMBER, bold=True)
    text(draw, (1175, 643), f"Delay selected · {selected_delay_ms:.3f} ms", 18, MUTED)
    text(draw, (1175, 675), f"Runtime applied · {effective_delay_steps} steps = {effective_delay_ms:.3f} ms", 18, MUTED)
    text(draw, (1175, 713), "Setup-scoped—not universal SO-101 defaults", 17, MUTED)
    round_rect(draw, (1175, 755, 1788, 855), "#FDF4E8", outline=AMBER, width=2, radius=18)
    text(draw, (1202, 778), "NOT VALIDATED", 16, AMBER, bold=True)
    text(draw, (1202, 815), "force · contact · grasp · insertion · policy transfer", 18, WHITE)
    footer(draw, "The package leaves CALL 5 with both the values and the evidence-backed limit of their claim.")


def truth_scene(draw, records: dict):
    selected_delay_s = float(records["manifest"]["parameters"]["command_delay_s"])
    dt = float(records["plan"]["environment"]["dt"])
    delay_steps = int(round(selected_delay_s / dt))
    effective_delay_ms = 1000.0 * delay_steps * dt
    header(draw, "WHAT IS REAL, SIMULATED, TUNED, AND VALIDATED", "A defensible MVP1 result", "The presentation ends where the evidence ends.")
    rows = [
        ("REAL", MEASURED_DISCLOSURE, BLUE),
        ("SIMULATED", "SO-101 USD commanded through Isaac Lab + Newton/MuJoCo-Warp", GREEN),
        ("BASELINE", BASELINE_DISCLOSURE, CORAL),
        (
            "TUNED",
            TUNED_DISCLOSURE
            + f" · 6 Isaac Lab explicit-PD + 4 Newton dynamics settings · delay {1000.0 * selected_delay_s:.3f} ms selected / {delay_steps} steps = {effective_delay_ms:.3f} ms applied",
            GREEN_DARK,
        ),
        ("VALIDATED", "92.099% lower aggregate score · 4 held-out 12 s motions · free-space arm + unloaded jaw", BLUE),
    ]
    y = 240
    for label, body, accent in rows:
        round_rect(draw, (92, y, 1828, y + 120), PANEL if label != "VALIDATED" else "#F3F9E9", outline=accent, width=2, radius=20)
        text(draw, (126, y + 24), label, 17, accent, bold=True)
        fit_text(draw, (310, y + 42), body, 1460, 24, WHITE, bold=True)
        y += 142
    round_rect(draw, (92, 954, 1828, 997), "#FDF4E8", outline=AMBER, width=2, radius=13)
    text(draw, (960, 976), "No contact, grasp, insertion, policy, or physical robot transfer claim.", 18, AMBER, bold=True, anchor="mm")
    footer(draw, "MVP1 · evidence-backed actuator calibration only")


def overview_frame(records, heldout_data, heldout_meta, gripper_data, gripper_meta, viewport, time_s: float, duration: float):
    image = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(image, "RGBA")
    if time_s < 2.7:
        title_scene(draw, ease(time_s / 2.7))
    elif time_s < 7.2:
        pipeline_scene(draw, (time_s - 2.7) / 4.5)
    elif time_s < 12.3:
        analyze_scene(draw, records)
    elif time_s < 17.4:
        plan_scene(draw, records)
    elif time_s < 23.5:
        fit_scene(draw, records, (time_s - 17.4) / 6.1)
    elif time_s < 29.7:
        validate_scene(draw, records)
    elif time_s < 35.7:
        if viewport is not None:
            overview_viewport_scene(image, draw, viewport, (time_s - 29.7) / 6.0)
        else:
            overview_arm_scene(draw, heldout_data, heldout_meta, (time_s - 29.7) / 6.0)
    elif time_s < 41.5:
        overview_gripper_scene(draw, gripper_data, gripper_meta, (time_s - 35.7) / 5.8)
    elif time_s < 47.5:
        package_scene(draw, records)
    else:
        truth_scene(draw, records)
    return image


def render_frames(output_dir: Path, name: str, duration: float, fps: int, callback):
    frame_dir = output_dir / f"{name}_frames"
    frame_dir.mkdir(parents=True, exist_ok=False)
    count = int(round(duration * fps))
    for frame_index in range(count):
        time_s = frame_index / fps
        phase = min(1.0, time_s / max(1e-9, duration - 1.0 / fps))
        image = callback(time_s, phase)
        image.save(frame_dir / f"frame_{frame_index:05d}.png", optimize=True)
    return frame_dir, count


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        binaries = sorted((Path(__file__).resolve().parents[1] / ".video-deps" / "imageio_ffmpeg" / "binaries").glob("ffmpeg-*"))
        if binaries:
            return str(binaries[0])
    raise RuntimeError("ffmpeg is required to encode MP4 outputs")


def encode_video(frame_dir: Path, output: Path, fps: int) -> None:
    subprocess.run(
        [
            find_ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def render_product(output_dir: Path, name: str, duration: float, fps: int, callback, *, keep_frames: bool) -> dict:
    if (output_dir / f"{name}.mp4").exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir / f'{name}.mp4'}")
    if keep_frames:
        frame_parent = output_dir
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix=f"{name}-")
        frame_parent = Path(cleanup.name)
    try:
        frame_dir, count = render_frames(frame_parent, name, duration, fps, callback)
        poster = output_dir / f"{name}_poster.png"
        Image.open(frame_dir / "frame_00000.png").save(poster)
        video = output_dir / f"{name}.mp4"
        encode_video(frame_dir, video, fps)
        product = {
            "name": name,
            "video": str(video),
            "video_sha256": sha256(video),
            "poster": str(poster),
            "poster_sha256": sha256(poster),
            "frames": str(frame_dir) if keep_frames else None,
            "frame_count": count,
            "duration_s": duration,
        }
    finally:
        if cleanup is not None:
            cleanup.cleanup()
    return product


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mvp1-full-20260913") / RUN_ID)
    parser.add_argument("--package-dir", type=Path, default=Path("packages/so101-mvp1-full-20260913"))
    parser.add_argument("--data-dir", type=Path, default=Path("deliverables/mvp1_20260913/video_data"))
    parser.add_argument("--arm-data", type=Path, help="Override data-dir/heldout_frequency_sweep.npz")
    parser.add_argument("--gripper-data", type=Path, help="Override the held-out arm bundle for the jaw-channel clip")
    parser.add_argument("--viewport-dir", type=Path, default=Path("deliverables/mvp1_20260913/viewport"))
    parser.add_argument("--no-viewport", action="store_true", help="Intentionally omit the actual-USD RTX segment.")
    parser.add_argument("--output-dir", type=Path, default=Path("deliverables/mvp1_20260913/videos"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--keep-frames", action="store_true", help="Preserve PNG frame directories next to the MP4s.")
    parser.add_argument(
        "--only",
        choices=("all", "overview", "arm", "gripper"),
        default="all",
        help="Render one product or all three.",
    )
    args = parser.parse_args()
    arm_data_path = args.arm_data or args.data_dir / "heldout_frequency_sweep.npz"
    gripper_data_path = args.gripper_data or arm_data_path
    records = load_run_records(args.run_dir, args.package_dir)
    heldout_data, heldout_meta = load_trajectory(
        arm_data_path,
        expected_split="heldout",
        minimum_duration_s=11.5,
        required_improvement_metric="arm_rmse_rad",
    )
    gripper_data, gripper_meta = load_trajectory(
        gripper_data_path,
        expected_split="heldout",
        minimum_duration_s=11.5,
        required_improvement_metric="jaw_rmse_rad",
    )
    viewport = load_viewport_capture(
        None if args.no_viewport else args.viewport_dir,
        records,
        arm_data_path,
    )
    expected_baseline = {entry["name"]: float(entry["initial"]) for entry in records["plan"]["parameters"]}
    expected_tuned = {name: float(value) for name, value in records["manifest"]["parameters"].items()}
    for label, metadata in (("arm", heldout_meta), ("gripper", gripper_meta)):
        if metadata.get("manifest_status") != "validated" or not metadata.get("activation_allowed"):
            raise ValueError(f"{label} trajectory sidecar is not from the activation-allowed validated package.")
        for group_name, actual, expected in (
            ("baseline", metadata.get("baseline_parameters", {}), expected_baseline),
            ("tuned", metadata.get("tuned_parameters", {}), expected_tuned),
        ):
            if set(actual) != set(expected) or any(
                not math.isclose(float(actual[name]), value, rel_tol=0.0, abs_tol=1e-12)
                for name, value in expected.items()
            ):
                raise ValueError(f"{label} trajectory {group_name} parameters do not match the locked full-run records.")
    validation_names = set(records["validation"]["per_episode"])
    if heldout_meta["episode"] not in validation_names:
        raise ValueError(f"Displayed arm episode {heldout_meta['episode']!r} is not one of the four validated holdouts.")
    if gripper_meta["episode"] not in validation_names:
        raise ValueError(f"Displayed jaw episode {gripper_meta['episode']!r} is not one of the four validated holdouts.")
    heldout_record = records["validation"]["per_episode"][heldout_meta["episode"]]
    require_matching_scores(
        "arm held-out",
        heldout_meta,
        heldout_record["baseline"]["score"],
        heldout_record["calibrated"]["score"],
    )
    gripper_episode = gripper_meta["episode"]
    gripper_record = records["validation"]["per_episode"][gripper_episode]
    require_matching_scores(
        "jaw held-out",
        gripper_meta,
        gripper_record["baseline"]["score"],
        gripper_record["calibrated"]["score"],
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    products = []
    if args.only in ("all", "overview"):
        products.append(render_product(
            output_dir,
            "so101_mvp1_five_call_overview",
            53.0,
            args.fps,
            lambda time_s, _phase: overview_frame(
                records, heldout_data, heldout_meta, gripper_data, gripper_meta, viewport, time_s, 53.0
            ),
            keep_frames=args.keep_frames,
        ))

    if args.only in ("all", "arm"):
        products.append(render_product(
            output_dir,
            "so101_mvp1_arm_synchronized",
            12.0,
            args.fps,
            lambda _time_s, phase: arm_comparison_frame(heldout_data, heldout_meta, ease(phase)),
            keep_frames=args.keep_frames,
        ))

    if args.only in ("all", "gripper"):
        products.append(render_product(
            output_dir,
            "so101_mvp1_gripper_synchronized",
            11.0,
            args.fps,
            lambda _time_s, phase: gripper_comparison_frame(gripper_data, gripper_meta, ease(phase)),
            keep_frames=args.keep_frames,
        ))

    manifest = {
        "schema": "newton.calibration.presentation-video/v1",
        "run_id": RUN_ID,
        "run_dir": str(args.run_dir),
        "package_dir": str(args.package_dir),
        "source_trajectories": {
            "arm": {"path": str(arm_data_path), "sha256": sha256(arm_data_path), "episode": heldout_meta["episode"], "split": "heldout"},
            "gripper": {"path": str(gripper_data_path), "sha256": sha256(gripper_data_path), "episode": gripper_meta["episode"], "split": "heldout", "channel": "jaw"},
        },
        "viewport_capture": None
        if viewport is None
        else {
            "path": str(viewport["directory"]),
            "result_sha256": sha256(viewport["directory"] / "live_result.json"),
            "trajectory_sha256": sha256(viewport["trajectory_path"]),
            "episode": viewport["result"]["episode"],
            "steps": viewport["result"]["steps"],
            "renderer": viewport["result"]["runtime"]["renderer"],
            "capture_mode": viewport["result"]["capture_mode"],
            "source_trajectory_sha256": viewport["result"]["source_trajectory_sha256"],
            "source_sidecar_sha256": viewport["result"]["source_sidecar_sha256"],
            "display_start_s": viewport["result"]["display_start_s"],
            "display_end_s": viewport["result"]["display_end_s"],
            "local_score_improvement_pct": viewport["result"]["score_improvement_pct"],
        },
        "truth_labels": {
            "measured": MEASURED_DISCLOSURE,
            "baseline": BASELINE_DISCLOSURE,
            "tuned": TUNED_DISCLOSURE,
        },
        "aggregate_validation": {
            "improvement_pct": records["validation"]["improvement_pct"],
            "heldout_episode_count": len(records["validation"]["per_episode"]),
            "duration_s_each": records["plan"]["optimizer"]["max_episode_duration_s"],
            "scope": records["manifest"]["scope"],
        },
        "excluded_claims": ["grip force", "contact", "grasp", "insertion", "policy transfer", "physical robot transfer"],
        "fps": args.fps,
        "width": 1920,
        "height": 1080,
        "products": products,
    }
    (output_dir / "video_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
