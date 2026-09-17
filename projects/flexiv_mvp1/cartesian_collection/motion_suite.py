"""Deterministic Cartesian collection proposals. No robot or simulator imports.

Offsets are from one fixed, operator-taught flange pose, NOT incremental moves
from the preceding sample. World/base-frame rotation vectors left-multiply the
anchor quaternion. The collector converts flange targets to the active-tool TCP
and calls RDK directly; ROS/policy inference latency is outside this boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

SCHEMA = "flexiv.cartesian_collection/v1"
BOUNDARY = "direct_rdk_cartesian_tcp_targets_derived_from_flange_reference"
FIELDS = ["time_s", "dx_m", "dy_m", "dz_m", "rx_rad", "ry_rad", "rz_rad"]
POSE_FIELDS = ["time_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"]
AXES = ("x", "y", "z", "rx", "ry", "rz")
ENVELOPE = {  # Engineering proposals, not Flexiv ratings or hardware approval.
    "translation_norm_m": 0.025,
    "rotation_norm_rad": math.radians(4),
    "linear_speed_m_s": 0.05,
    "angular_speed_rad_s": 0.16,
    "linear_acceleration_m_s2": 0.20,
    "angular_acceleration_rad_s2": 0.8,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def norm(values):
    return math.sqrt(sum(x * x for x in values))


def numbers(values, length, label):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{label} needs {length} values")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
        raise ValueError(f"{label} needs finite numeric values")
    return values


def positive(value, label):
    numbers([value], 1, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def quat_multiply(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
        w * W - x * X - y * Y - z * Z,
    ]


def rotation_quat(vector):
    angle = norm(vector)
    scale = math.sin(angle / 2) / angle if angle > 1e-12 else 0.5
    return [v * scale for v in vector] + [math.cos(angle / 2)]


def rotation_distance(a, b):
    return 2 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b))) / (norm(a) * norm(b))))


def rotate(q, xyz):
    return quat_multiply(quat_multiply(q, list(xyz) + [0]), [-q[0], -q[1], -q[2], q[3]])[:3]


def target_pose(anchor, offsets):
    """Apply base-frame displacement once to the immutable taught anchor."""
    return [anchor[i] + offsets[i] for i in range(3)] + quat_multiply(rotation_quat(offsets[3:]), anchor[3:])


def tcp_target(flange_pose, tool_pose):
    """Full rigid transform, not just a translation; both poses use xyzw."""
    offset = rotate(flange_pose[3:], tool_pose[:3])
    return [x + y for x, y in zip(flange_pose[:3], offset)] + quat_multiply(flange_pose[3:], tool_pose[3:])


def to_rdk_pose(xyzw):
    return list(xyzw[:3]) + [xyzw[6]] + list(xyzw[3:6])


def from_rdk_pose(wxyz):
    numbers(list(wxyz), 7, "RDK pose")
    return list(wxyz[:3]) + list(wxyz[4:]) + [wxyz[3]]


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def trajectory(kind, axis, duration, rate):
    rows = []
    for index in range(round(duration * rate) + 1):
        t = index / rate
        # Two-second stationary bookends and C2 tapered excitation.
        u, length = t - 2, duration - 4
        window = smoothstep(u / 3) * smoothstep((length - u) / 3)
        values = [0.0] * 6
        if 0 < u < length and kind != "stationary":
            if kind == "reversals":
                amplitude = 0.020 if axis < 3 else math.radians(3)
                values[axis] = amplitude * window * math.sin(2 * math.pi * 2 * u / length)
            elif kind == "chirp":
                amplitude = 0.006 if axis < 3 else math.radians(1)
                phase = 2 * math.pi * (0.05 * u + 0.5 * (0.7 - 0.05) * u * u / length)
                values[axis] = amplitude * window * math.sin(phase)
            elif kind in {"heldout_a", "heldout_b"}:
                for j in range(6):
                    frequency = (0.07 + j * 0.023) if kind == "heldout_a" else (0.11 + j * 0.019)
                    amplitude = 0.009 if j < 3 else math.radians(1.1)
                    values[j] = amplitude * window * math.sin(2 * math.pi * frequency * u + j * 0.31)
            else:
                raise ValueError("Unknown motion family")
        rows.append([t] + values)
    return rows


def metrics(rows, rate):
    values = [r[1:] for r in rows]
    velocity = [[(b - a) * rate for a, b in zip(x, y)] for x, y in itertools.pairwise(values)]
    acceleration = [[(b - a) * rate for a, b in zip(x, y)] for x, y in itertools.pairwise(velocity)]
    return {
        "translation_norm_m": max(norm(r[:3]) for r in values),
        "rotation_norm_rad": max(norm(r[3:]) for r in values),
        "linear_speed_m_s": max(norm(r[:3]) for r in velocity),
        "angular_speed_rad_s": max(norm(r[3:]) for r in velocity),
        "linear_acceleration_m_s2": max(norm(r[:3]) for r in acceleration),
        "angular_acceleration_rad_s2": max(norm(r[3:]) for r in acceleration),
    }


def check_tcp_rates(poses, profile):
    """Bound command derivatives at the real controlled TCP, not only flange."""
    rate = profile["command_rate_hz"]
    targets = [tcp_target(row[1:], profile["active_tool"]["tcp_location_xyzw"]) for row in poses]
    linear, angular = [], []
    for a, b in itertools.pairwise(targets):
        linear.append([(y - x) * rate for x, y in zip(a[:3], b[:3])])
        delta = quat_multiply(b[3:], [-a[3], -a[4], -a[5], a[6]])
        if delta[3] < 0:
            delta = [-x for x in delta]
        sine = norm(delta[:3])
        factor = 2 * math.atan2(sine, delta[3]) / sine if sine > 1e-12 else 2
        angular.append([v * factor * rate for v in delta[:3]])
    for velocities, speed_key, accel_key in (
        (linear, "linear_speed_m_s", "linear_acceleration_m_s2"),
        (angular, "angular_speed_rad_s", "angular_acceleration_rad_s2"),
    ):
        if max(norm(v) for v in velocities) > profile["limits"][speed_key] + 1e-8:
            raise ValueError("Active TCP command speed exceeds profile")
        if (
            max(norm([(y - x) * rate for x, y in zip(a, b)]) for a, b in itertools.pairwise(velocities))
            > profile["limits"][accel_key] + 1e-8
        ):
            raise ValueError("Active TCP command acceleration exceeds profile")


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows([[f"{x:.12g}" for x in row] for row in rows])


def read_csv(path, fields):
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader) != fields:
            raise ValueError("Unexpected columns/units")
        rows = [[float(x) for x in row] for row in reader]
    for row in rows:
        numbers(row, len(fields), "CSV row")
    return rows


def profile_template(rate):
    return {
        "schema": SCHEMA,
        "confirmed": False,
        "robot_serial": None,
        "posture_id": None,
        "base_frame": "RDK_WORLD",
        "command_boundary": BOUNDARY,
        "command_rate_hz": rate,
        "rate_confirmed": False,
        "rdk_control_mode": "NRT_CARTESIAN_MOTION_FORCE",
        "rdk_version": "1.9.3",
        "robot_software_version": None,
        "controller_settings_source": None,
        "anchor_flange_pose_xyzw": None,
        "anchor_joint_positions_rad": None,
        "joint_names": None,
        "joint_lower_rad": None,
        "joint_upper_rad": None,
        "joint_limit_margin_rad": 0.1,
        "max_joint_speed_rad_s": None,
        "tool_payload_description": None,
        "fixed_gripper_description": None,
        "active_tool": None,
        "cartesian_stiffness": None,
        "cartesian_damping_ratio": None,
        "nullspace_objectives": None,
        "snapshot_path": None,
        "snapshot_sha256": None,
        "max_external_wrench_abs": None,
        "reviewed_workspace_min_m": None,
        "reviewed_workspace_max_m": None,
        "limits": dict(ENVELOPE),
        "start_translation_tolerance_m": 0.002,
        "start_rotation_tolerance_rad": math.radians(1),
        "start_joint_tolerance_rad": 0.03,
        "max_tracking_translation_m": 0.015,
        "max_tracking_rotation_rad": math.radians(5),
        "feedback_timeout_s": 0.15,
        "max_schedule_lateness_s": 0.4 / rate,
        "notes": "Unconfirmed proposals only. Teach a free-space anchor with peg removed. "
        "No automatic homing, tool changes, controller changes or real execution approval.",
    }


def validate_profile(profile, root):
    if profile.get("schema") != SCHEMA or profile.get("command_boundary") != BOUNDARY:
        raise ValueError("Unsupported schema or command boundary")
    if profile.get("confirmed") is not True or profile.get("rate_confirmed") is not True:
        raise ValueError("Operator must confirm the deployment profile and rate")
    for key in (
        "robot_serial",
        "posture_id",
        "base_frame",
        "tool_payload_description",
        "fixed_gripper_description",
        "robot_software_version",
        "controller_settings_source",
    ):
        if not isinstance(profile.get(key), str) or not profile[key].strip() or profile[key] in {"TBD", "REQUIRED"}:
            raise ValueError(f"Missing {key}")
    if profile.get("rdk_control_mode") != "NRT_CARTESIAN_MOTION_FORCE" or profile.get("rdk_version") != "1.9.3":
        raise ValueError("This collector targets only the reviewed RDK 1.9.3 NRT Cartesian API")
    if profile["base_frame"] != "RDK_WORLD":
        raise ValueError("RDK motion commands must use RDK_WORLD, not a guessed ROS/world transform")
    rate = positive(profile.get("command_rate_hz"), "command rate")
    if rate < 10 or rate > 100:
        raise ValueError("Requalify this collector for rates outside 10–100 Hz")
    anchor = numbers(profile.get("anchor_flange_pose_xyzw"), 7, "anchor pose")
    if abs(norm(anchor[3:]) - 1) > 1e-6:
        raise ValueError("Anchor quaternion must be normalized, xyzw")
    for key, length in (
        ("anchor_joint_positions_rad", 7),
        ("joint_lower_rad", 7),
        ("joint_upper_rad", 7),
        ("max_joint_speed_rad_s", 7),
        ("cartesian_stiffness", 6),
        ("max_external_wrench_abs", 6),
        ("nullspace_objectives", 3),
        ("cartesian_damping_ratio", 6),
        ("reviewed_workspace_min_m", 3),
        ("reviewed_workspace_max_m", 3),
    ):
        numbers(profile.get(key), length, key)
    names = profile.get("joint_names")
    if (
        not isinstance(names, list)
        or len(names) != 7
        or len(set(names)) != 7
        or not all(isinstance(n, str) and n for n in names)
    ):
        raise ValueError("Seven unique real feedback joint names required")
    for key in (
        "joint_limit_margin_rad",
        "start_translation_tolerance_m",
        "start_rotation_tolerance_rad",
        "start_joint_tolerance_rad",
        "max_tracking_translation_m",
        "max_tracking_rotation_rad",
        "feedback_timeout_s",
        "max_schedule_lateness_s",
    ):
        positive(profile.get(key), key)
    if profile["feedback_timeout_s"] > 0.25 or profile["max_schedule_lateness_s"] >= 1 / rate:
        raise ValueError("Feedback timeout/lateness exceeds collector hard cap")
    for key in ("max_joint_speed_rad_s", "cartesian_stiffness", "cartesian_damping_ratio", "max_external_wrench_abs"):
        for value in profile[key]:
            positive(value, key)
    for key, value in ENVELOPE.items():
        if positive(profile["limits"].get(key), key) > value + 1e-12:
            raise ValueError("Wider envelopes require a reviewed generator change")
    for q, lo, hi in zip(profile["anchor_joint_positions_rad"], profile["joint_lower_rad"], profile["joint_upper_rad"]):
        if not lo + profile["joint_limit_margin_rad"] < q < hi - profile["joint_limit_margin_rad"]:
            raise ValueError("Taught joint anchor violates reviewed limits")
    if any(not 0.3 <= x <= 0.8 for x in profile["cartesian_damping_ratio"]):
        raise ValueError("RDK damping ratio must lie in [0.3, 0.8]")
    objectives = profile["nullspace_objectives"]
    if not (0 <= objectives[0] <= 1 and 0 <= objectives[1] <= 1 and 0.1 <= objectives[2] <= 1):
        raise ValueError("Invalid nullspace objective weights")
    tool = profile.get("active_tool")
    if not isinstance(tool, dict) or not tool.get("name"):
        raise ValueError("Active tool snapshot required")
    location = numbers(tool.get("tcp_location_xyzw"), 7, "tool TCP pose")
    if abs(norm(location[3:]) - 1) > 1e-6:
        raise ValueError("Tool orientation must be a normalized xyzw quaternion")
    numbers([tool.get("mass")], 1, "tool mass")
    if tool["mass"] < 0:
        raise ValueError("Negative tool mass")
    numbers(tool.get("CoM"), 3, "tool CoM")
    numbers(tool.get("inertia"), 6, "tool inertia")
    for key in ("snapshot",):
        path = Path(profile.get(key + "_path") or "")
        path = path if path.is_absolute() else Path(root) / path
        if not path.is_file() or digest(path) != profile.get(key + "_sha256"):
            raise ValueError(f"Missing or changed {key} file")
    snapshot = json.loads(path.read_text())
    if snapshot.get("robot_serial") != profile["robot_serial"] or snapshot.get("active_tool") != tool:
        raise ValueError("Profile and captured robot/tool disagree")
    if (
        snapshot.get("stationary_samples") is not True
        or snapshot.get("fresh_samples") is not True
        or snapshot.get("stopped") is not True
    ):
        raise ValueError("Capture a fresh, stationary, stopped anchor before binding")
    captured = snapshot["samples"][-1]
    if (
        norm([a - b for a, b in zip(from_rdk_pose(captured["flange_pose"])[:3], anchor[:3])]) > 1e-6
        or rotation_distance(from_rdk_pose(captured["flange_pose"])[3:], anchor[3:]) > 1e-6
        or max(abs(a - b) for a, b in zip(captured["q"], profile["anchor_joint_positions_rad"])) > 1e-6
    ):
        raise ValueError("Anchor differs from snapshot; teach and capture a new posture")
    return profile


def check_workspace(pose, profile):
    tcp = tcp_target(pose, profile["active_tool"]["tcp_location_xyzw"])
    for point in (pose[:3], tcp[:3]):
        if not all(
            lo < p < hi
            for p, lo, hi in zip(point, profile["reviewed_workspace_min_m"], profile["reviewed_workspace_max_m"])
        ):
            raise ValueError("Flange or active TCP outside reviewed workspace")
    # This box does not screen other links, self-collision, obstacles or singularities.


def generate(output, rate=15.0):
    if not 10 <= rate <= 100 or not math.isfinite(rate):
        raise ValueError("Proposed command rate must be 10–100 Hz")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "commands").mkdir()
    definitions = [("00_stationary", "stationary", None, 6, "train")]
    definitions += [(f"{i + 1:02d}_{axis}_reversals", "reversals", i, 24, "train") for i, axis in enumerate(AXES)]
    definitions += [(f"{i + 7:02d}_{axis}_chirp", "chirp", i, 30, "train") for i, axis in enumerate(AXES)]
    definitions += [
        ("13_heldout_coupled_a", "heldout_a", None, 30, "heldout"),
        ("14_heldout_coupled_b", "heldout_b", None, 30, "heldout"),
    ]
    episodes = []
    for name, kind, axis, duration, split in definitions:
        path = root / "commands" / (name + ".csv")
        write_csv(path, FIELDS, trajectory(kind, axis, duration, rate))
        rows = read_csv(path, FIELDS)
        measured = metrics(rows, rate)
        if any(measured[k] > ENVELOPE[k] for k in measured):
            raise ValueError("Generated trajectory exceeds proposed envelope")
        episodes.append(
            {
                "id": name,
                "kind": kind,
                "axis": AXES[axis] if axis is not None else "coupled/none",
                "split": split,
                "duration_s": duration,
                "samples": len(rows),
                "path": str(path.relative_to(root)),
                "sha256": digest(path),
                "metrics": measured,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "status": "proposal_unbound_not_robot_approved",
        "boundary": BOUNDARY,
        "command_rate_hz": rate,
        "rate_source": "15 Hz proposal inspired by tutorial; NOT a confirmed deployment rate",
        "offset_semantics": "fixed taught flange anchor; translation/rotation vectors in confirmed base frame",
        "real_execution_approved": False,
        "newton_screened": False,
        "real_data": False,
        "duration_s": sum(e["duration_s"] for e in episodes),
        "limits": ENVELOPE,
        "generator_sha256": digest(__file__),
        "episodes": episodes,
        "identifiability": "Excitation proposals, not proof of identifying all seven-joint parameters. "
        "Use multiple reviewed postures; inspect excitation and sensitivity after collection.",
    }
    save_json(root / "manifest.json", manifest)
    save_json(root / "deployment_profile.template.json", profile_template(rate))
    verify(root)
    return manifest


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("boundary") != BOUNDARY:
        raise ValueError("Unexpected manifest semantics")
    rate = positive(manifest["command_rate_hz"], "rate")
    if manifest.get("generator_sha256") != digest(__file__):
        raise ValueError("Generator changed; regenerate and requalify")
    episodes = manifest["episodes"]
    if len(episodes) != 15 or len({e["id"] for e in episodes}) != 15:
        raise ValueError("Expected complete 15-motion suite")
    for episode in episodes:
        source = (root / episode["path"]).resolve()
        if not source.is_relative_to(root) or not source.is_file() or digest(source) != episode["sha256"]:
            raise ValueError("Command file path/hash invalid")
        rows = read_csv(source, FIELDS)
        if len(rows) != episode["samples"] or len(rows) < 3:
            raise ValueError("Invalid sample count")
        if abs(rows[-1][0] - episode["duration_s"]) > 1e-8:
            raise ValueError("Invalid duration")
        if any(abs(row[0] - i / rate) > 1e-8 for i, row in enumerate(rows)):
            raise ValueError("Invalid time grid")
        if any(abs(v) > 1e-10 for row in (rows[0], rows[-1]) for v in row[1:]):
            raise ValueError("Motion must start/end at taught anchor")
        measured = metrics(rows, rate)
        for key, value in measured.items():
            if value > min(ENVELOPE[key], manifest["limits"][key]) + 1e-9:
                raise ValueError("Command envelope exceeded")
            if abs(value - episode["metrics"][key]) > 1e-9:
                raise ValueError("Metrics do not match commands")
        if episode["split"] != ("heldout" if episode["id"].startswith(("13_", "14_")) else "train"):
            raise ValueError("Held-out split changed")
    return manifest


def bind(proposal, profile_path, output):
    proposal, profile_path = Path(proposal), Path(profile_path)
    manifest = verify(proposal)
    profile = validate_profile(json.loads(profile_path.read_text()), profile_path.parent)
    if profile["command_rate_hz"] != manifest["command_rate_hz"]:
        raise ValueError("Regenerate motions at the confirmed deployment rate")
    prepared = []
    for e in manifest["episodes"]:
        if any(e["metrics"][k] > profile["limits"][k] for k in ENVELOPE):
            raise ValueError("Motion exceeds profile limits; revise proposals before binding")
        rows = read_csv(proposal / e["path"], FIELDS)
        poses = [[row[0]] + target_pose(profile["anchor_flange_pose_xyzw"], row[1:]) for row in rows]
        for row in poses:
            check_workspace(row[1:], profile)
        check_tcp_rates(poses, profile)
        prepared.append((e, poses))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "commands").mkdir()
    # Preserve the captured setup in the portable bound bundle.
    profile = json.loads(json.dumps(profile))
    for key in ("snapshot",):
        source = Path(profile[key + "_path"])
        source = source if source.is_absolute() else profile_path.parent / source
        destination = out / (key + source.suffix)
        destination.write_bytes(source.read_bytes())
        profile[key + "_path"] = destination.name
    save_json(out / "profile.json", profile)
    entries = []
    for e, rows in prepared:
        path = out / "commands" / (e["id"] + ".csv")
        write_csv(path, POSE_FIELDS, rows)
        entries.append(
            {**e, "path": str(path.relative_to(out)), "sha256": digest(path), "proposal_sha256": e["sha256"]}
        )
    bound = {
        **manifest,
        "schema": SCHEMA + "/bound",
        "status": "bound_needs_newton_screen_and_operator_approval",
        "source_manifest_sha256": digest(proposal / "manifest.json"),
        "profile_sha256": digest(out / "profile.json"),
        "episodes": entries,
    }
    save_json(out / "manifest.json", bound)
    save_json(
        out / "approval.template.json",
        {
            "operator_name": None,
            "robot_serial": profile["robot_serial"],
            "expires_utc": None,
            "manifest_sha256": digest(out / "manifest.json"),
            "profile_sha256": digest(out / "profile.json"),
            "collector_sha256": digest(Path(__file__).with_name("collect_rdk.py")),
            "screen_report_path": None,
            "screen_report_sha256": None,
            "approved_episodes": [],
            **{k: False for k in APPROVAL_CHECKS},
        },
    )
    return bound


APPROVAL_CHECKS = (
    "qualified_operator",
    "estop_available",
    "peg_removed",
    "gripper_fixed",
    "real_swept_volume_reviewed",
    "taught_pose_and_joint_branch_verified",
    "joint_mapping_verified",
    "controller_tool_payload_confirmed",
    "robot_limits_and_stop_behavior_reviewed",
    "exclusive_rdk_control_no_other_clients",
    "newton_screen_reviewed",
    "collector_integration_tested",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("generate")
    p.add_argument("--output", required=True)
    p.add_argument("--rate", type=float, default=15.0)
    p = subs.add_parser("inspect")
    p.add_argument("--bundle", required=True)
    p = subs.add_parser("bind")
    p.add_argument("--proposal", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(args.output, args.rate)
    elif args.command == "inspect":
        result = verify(args.bundle)
    else:
        result = bind(args.proposal, args.profile, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "motions": len(result["episodes"]),
                "duration_s": result["duration_s"],
                "newton_screened": result["newton_screened"],
                "real_execution_approved": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
