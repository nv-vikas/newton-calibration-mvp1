"""Direct Flexiv RDK collection. Offline by default; ONE approved trial at a time.

snapshot connects and READS only. run needs --execute, an expiring approval and
a passed, hash-bound Newton screen. Never enables, homes, clears faults, changes
tools, zeros sensors or relaxes robot safety limits. A software stop is not an
E-stop; process/network failure remains dependent on the robot's configured stop
behavior. Hardware/SDK integration has not been performed by local unit tests.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import json
import time
import uuid
from pathlib import Path

from motion_suite import (
    APPROVAL_CHECKS,
    BOUNDARY,
    POSE_FIELDS,
    SCHEMA,
    check_tcp_rates,
    check_workspace,
    digest,
    from_rdk_pose,
    norm,
    numbers,
    profile_template,
    read_csv,
    rotation_distance,
    save_json,
    tcp_target,
    to_rdk_pose,
    validate_profile,
)

SDK_VERSION = "1.9.3"


def sdk():
    version = importlib.metadata.version("flexivrdk")
    if version != SDK_VERSION:
        raise RuntimeError(f"Reviewed API is flexivrdk=={SDK_VERSION}; found {version}. Version review required.")
    import flexivrdk

    return flexivrdk


def serialize_state(state):
    native_stamp = list(state.timestamp)
    if len(native_stamp) != 2 or any(not isinstance(x, int) for x in native_stamp):
        raise ValueError("Invalid native robot timestamp")
    if native_stamp[0] <= 0 or not 0 <= native_stamp[1] < 1_000_000_000:
        raise ValueError("Invalid native robot timestamp")
    result = {
        "host_receive_monotonic_ns": time.monotonic_ns(),
        "host_receive_unix_ns": time.time_ns(),
        "robot_timestamp": native_stamp,
    }
    for name, size in (
        ("q", 7),
        ("dq", 7),
        ("tcp_pose", 7),
        ("flange_pose", 7),
        ("tcp_vel", 6),
        ("tau", 7),
        ("ext_wrench_in_world", 6),
    ):
        result[name] = numbers(list(getattr(state, name)), size, name)
    for name in ("theta", "dtheta", "tau_des", "tau_ext", "ext_wrench_in_tcp", "temperature"):
        value = getattr(state, name, None)
        result[name] = list(value) if value is not None else None
    json.dumps(result, allow_nan=False)
    for key in ("tcp_pose", "flange_pose"):
        if abs(norm(result[key][3:]) - 1) > 1e-5:
            raise ValueError("Invalid feedback quaternion")
    return result


def tool_snapshot(tool):
    p = tool.params()
    return {
        "name": tool.name(),
        "mass": float(p.mass),
        "CoM": list(p.CoM),
        "inertia": list(p.inertia),
        "tcp_location_xyzw": from_rdk_pose(p.tcp_location),
    }


def info_snapshot(info):
    if info.DoF != 7 or info.DoF_e != 0:
        raise ValueError("Only seven-joint arms without external axes are supported")
    return {
        "serial_num": info.serial_num,
        "software_ver": info.software_ver,
        "model_name": info.model_name,
        "DoF": info.DoF,
        "DoF_e": info.DoF_e,
        **{k: list(getattr(info, k)) for k in ("q_min", "q_max", "dq_max", "tau_max", "K_x_nom")},
    }


def snapshot(serial, output, rate):
    """Read-only accessor calls. No control/settings methods, including Stop()."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rdk = sdk()
    robot = rdk.Robot(serial)
    info = info_snapshot(robot.info())
    tool = tool_snapshot(rdk.Tool(robot))
    samples = []
    for _ in range(10):
        samples.append(serialize_state(robot.states()))
        time.sleep(0.02)
    stable = all(max(abs(v) for v in s["dq"]) < 0.01 for s in samples)
    stable &= all(max(abs(a - b) for a, b in zip(s["q"], samples[0]["q"])) < 0.001 for s in samples)
    fresh = tuple(samples[-1]["robot_timestamp"]) > tuple(samples[0]["robot_timestamp"])
    captured = {
        "schema": SCHEMA + "/snapshot",
        "robot_serial": info["serial_num"],
        "rdk_version": SDK_VERSION,
        "robot_info": info,
        "active_tool": tool,
        "mode": str(robot.mode()),
        "operational": robot.operational(),
        "fault": robot.fault(),
        "stopped": robot.stopped(),
        "stationary_samples": stable,
        "fresh_samples": fresh,
        "samples": samples,
        "no_motion_commands_sent": True,
        "hardware_safe": False,
        "controller_gains_read": False,
    }
    save_json(out / "snapshot.json", captured)
    profile = profile_template(rate)
    profile.update(
        robot_serial=info["serial_num"],
        robot_software_version=info["software_ver"],
        active_tool=tool,
        joint_lower_rad=info["q_min"],
        joint_upper_rad=info["q_max"],
        snapshot_path="snapshot.json",
        snapshot_sha256=digest(out / "snapshot.json"),
    )
    if stable and fresh and captured["stopped"]:
        profile.update(
            anchor_flange_pose_xyzw=from_rdk_pose(samples[-1]["flange_pose"]),
            anchor_joint_positions_rad=samples[-1]["q"],
        )
    save_json(out / "profile.to_review.json", profile)
    return {
        "status": "read_only_snapshot_saved_needs_operator_review",
        "output": str(out),
        "motion_commands_sent": 0,
        "stable": stable,
        "fresh": fresh,
    }


def load_bound(bundle, episode_id):
    root = Path(bundle).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA + "/bound" or manifest.get("boundary") != BOUNDARY:
        raise ValueError("Bind proposals to a reviewed profile first; never run relative CSV directly")
    if digest(root / "profile.json") != manifest["profile_sha256"]:
        raise ValueError("Profile hash mismatch")
    if manifest["generator_sha256"] != digest(Path(__file__).with_name("motion_suite.py")):
        raise ValueError("Generator changed; regenerate and requalify")
    profile = validate_profile(json.loads((root / "profile.json").read_text()), root)
    if profile["command_rate_hz"] != manifest["command_rate_hz"]:
        raise ValueError("Rate mismatch")
    episode = next((e for e in manifest["episodes"] if e["id"] == episode_id), None)
    if episode is None:
        raise ValueError("Unknown episode")
    path = (root / episode["path"]).resolve()
    if not path.is_relative_to(root) or digest(path) != episode["sha256"]:
        raise ValueError("Command file path/hash mismatch")
    rows = read_csv(path, POSE_FIELDS)
    if len(rows) != episode["samples"] or len(rows) < 3:
        raise ValueError("Command sample count mismatch")
    rate = profile["command_rate_hz"]
    anchor = profile["anchor_flange_pose_xyzw"]
    for i, row in enumerate(rows):
        if abs(row[0] - i / rate) > 1e-8:
            raise ValueError("Invalid command timing")
        pose = row[1:]
        if abs(norm(pose[3:]) - 1) > 1e-6:
            raise ValueError("Invalid target quaternion")
        if norm([a - b for a, b in zip(pose[:3], anchor[:3])]) > profile["limits"]["translation_norm_m"] + 1e-9:
            raise ValueError("Translation envelope exceeded")
        if rotation_distance(pose[3:], anchor[3:]) > profile["limits"]["rotation_norm_rad"] + 1e-9:
            raise ValueError("Rotation envelope exceeded")
        check_workspace(pose, profile)
    if abs(rows[-1][0] - episode["duration_s"]) > 1e-8:
        raise ValueError("Duration mismatch")
    for row in (rows[0], rows[-1]):
        if norm([a - b for a, b in zip(row[1:4], anchor[:3])]) > 1e-9 or rotation_distance(row[4:], anchor[3:]) > 1e-6:
            raise ValueError("Trial must begin and end at reviewed anchor")
    check_tcp_rates(rows, profile)
    return manifest, profile, episode, rows


def validate_approval(path, bundle, profile, episode_id, now=None):
    path, root = Path(path), Path(bundle)
    approval = json.loads(path.read_text())
    for key in APPROVAL_CHECKS:
        if approval.get(key) is not True:
            raise ValueError(f"On-site approval missing: {key}")
    if not isinstance(approval.get("operator_name"), str) or not approval["operator_name"].strip():
        raise ValueError("Named qualified operator required")
    if approval.get("robot_serial") != profile["robot_serial"] or episode_id not in approval.get(
        "approved_episodes", []
    ):
        raise ValueError("Robot/trial not approved")
    expiry = dt.datetime.fromisoformat(approval.get("expires_utc", "").replace("Z", "+00:00"))
    if expiry.tzinfo is None or expiry <= (now or dt.datetime.now(dt.timezone.utc)):
        raise ValueError("Expired or timezone-ambiguous approval")
    for key, source in (
        ("manifest_sha256", root / "manifest.json"),
        ("profile_sha256", root / "profile.json"),
        ("collector_sha256", Path(__file__)),
    ):
        if approval.get(key) != digest(source):
            raise ValueError(f"Approval hash mismatch: {key}")
    screen = path.parent / (approval.get("screen_report_path") or "MISSING")
    if not screen.is_file() or digest(screen) != approval.get("screen_report_sha256"):
        raise ValueError("Missing/changed Newton screening report")
    report = json.loads(screen.read_text())
    for key in (
        "passed",
        "self_collision_enabled",
        "joint_limits_checked",
        "singularity_checked",
        "swept_volume_checked",
    ):
        if report.get(key) is not True:
            raise ValueError(f"Newton screening incomplete: {key}")
    if (
        report.get("physics") != "Newton"
        or report.get("surface") != "Isaac Lab"
        or report.get("manifest_sha256") != digest(root / "manifest.json")
        or report.get("profile_sha256") != digest(root / "profile.json")
        or episode_id not in report.get("screened_episodes", [])
    ):
        raise ValueError("Newton screen does not cover this exact bundle/profile/trial")
    return approval


class FeedbackGuard:
    def __init__(self, profile):
        self.p = profile
        self.last_stamp = None
        self.last_advance = None

    def check(self, state, now, target, *, at_start=False):
        p = self.p
        stamp = tuple(state["robot_timestamp"])
        if self.last_stamp is not None and stamp < self.last_stamp:
            raise RuntimeError("Robot timestamp reversed")
        if stamp != self.last_stamp:
            self.last_stamp, self.last_advance = stamp, now
        if now - self.last_advance > p["feedback_timeout_s"]:
            raise RuntimeError("Robot feedback stale")
        for q, v, lo, hi, vmax in zip(
            state["q"], state["dq"], p["joint_lower_rad"], p["joint_upper_rad"], p["max_joint_speed_rad_s"]
        ):
            if not lo + p["joint_limit_margin_rad"] < q < hi - p["joint_limit_margin_rad"] or abs(v) > vmax:
                raise RuntimeError("Joint limit/speed guard")
        flange = from_rdk_pose(state["flange_pose"])
        check_workspace(flange, p)
        actual_tcp = from_rdk_pose(state["tcp_pose"])
        expected_tcp = tcp_target(flange, p["active_tool"]["tcp_location_xyzw"])
        if (
            norm([a - b for a, b in zip(actual_tcp[:3], expected_tcp[:3])]) > 0.002
            or rotation_distance(actual_tcp[3:], expected_tcp[3:]) > 0.01
        ):
            raise RuntimeError("Tool/TCP transform differs from confirmed profile")
        for x, limit in zip(state["ext_wrench_in_world"], p["max_external_wrench_abs"]):
            if abs(x) > limit:
                raise RuntimeError("External-wrench guard; no contact trial allowed")
        tlim = p["start_translation_tolerance_m"] if at_start else p["max_tracking_translation_m"]
        rlim = p["start_rotation_tolerance_rad"] if at_start else p["max_tracking_rotation_rad"]
        if (
            norm([a - b for a, b in zip(flange[:3], target[:3])]) > tlim
            or rotation_distance(flange[3:], target[3:]) > rlim
        ):
            raise RuntimeError("Start-pose/Cartesian tracking guard")
        if at_start:
            if max(abs(v) for v in state["dq"]) > 0.01:
                raise RuntimeError("Robot is moving")
            if (
                max(abs(a - b) for a, b in zip(state["q"], p["anchor_joint_positions_rad"]))
                > p["start_joint_tolerance_rad"]
            ):
                raise RuntimeError("Wrong joint branch/posture; no automatic homing")


def check_hardware(robot, tool, rdk, profile, *, idle=False):
    if not robot.connected() or robot.fault() or not robot.operational():
        raise RuntimeError("Robot not connected, operational and fault-free")
    wanted = rdk.Mode.IDLE if idle else rdk.Mode.NRT_CARTESIAN_MOTION_FORCE
    if robot.mode() != wanted or (idle and (not robot.stopped() or robot.busy())):
        raise RuntimeError("Unexpected robot mode/activity; refusing takeover")
    if tool_snapshot(tool) != profile["active_tool"]:
        raise RuntimeError("Tool/payload configuration changed")


def execute(bundle, episode_id, approval_path, output):
    manifest, profile, episode, rows = load_bound(bundle, episode_id)
    approval = validate_approval(approval_path, bundle, profile, episode_id)
    rdk = sdk()
    destination = Path(output) / f"{episode_id}-{uuid.uuid4().hex[:10]}"
    destination.mkdir(parents=True, exist_ok=False)
    robot, owns_mode = None, False
    record = {
        "status": "preflight",
        "trial_id": destination.name,
        "split": episode["split"],
        "profile": profile,
        "approval": approval,
        "manifest_sha256": digest(Path(bundle) / "manifest.json"),
        "command_sha256": episode["sha256"],
        "rdk_version": SDK_VERSION,
        "boundary": BOUNDARY,
        "real_data": False,
        "hardware_validated": False,
        "clock_synchronized": False,
        "commands_sent": 0,
        "note": "Direct RDK includes NRT interpolation; excludes ROS bridge and policy inference. "
        "Host and robot clocks are preserved separately. Not a fitted/validated calibration.",
    }
    try:
        robot = rdk.Robot(profile["robot_serial"])
        tool = rdk.Tool(robot)
        info = info_snapshot(robot.info())
        record["robot_info"] = info
        if info["serial_num"] != profile["robot_serial"] or info["software_ver"] != profile["robot_software_version"]:
            raise RuntimeError("Robot identity/software changed")
        if (
            any(lo < actual for lo, actual in zip(profile["joint_lower_rad"], info["q_min"]))
            or any(hi > actual for hi, actual in zip(profile["joint_upper_rad"], info["q_max"]))
            or any(v > actual for v, actual in zip(profile["max_joint_speed_rad_s"], info["dq_max"]))
            or any(k > nominal for k, nominal in zip(profile["cartesian_stiffness"], info["K_x_nom"]))
        ):
            raise RuntimeError("Profile exceeds actual robot capabilities")
        check_hardware(robot, tool, rdk, profile, idle=True)
        guard = FeedbackGuard(profile)
        for _ in range(10):
            guard.check(serialize_state(robot.states()), time.monotonic(), rows[0][1:], at_start=True)
            time.sleep(0.02)
        print(f"ONE trial: {episode_id}, {episode['duration_s']} s. Peg removed. Physical E-stop available.")
        token = f"RUN {profile['robot_serial']} {episode_id}"
        if input(f"Type '{token}' to proceed: ").strip() != token:
            raise RuntimeError("Operator cancelled")
        # Recheck the approval, source bytes, robot state and pose AFTER the human pause.
        validate_approval(approval_path, bundle, profile, episode_id)
        fresh_manifest, fresh_profile, _, fresh_rows = load_bound(bundle, episode_id)
        if fresh_manifest != manifest or fresh_profile != profile or fresh_rows != rows:
            raise RuntimeError("Configuration changed during confirmation")
        check_hardware(robot, tool, rdk, profile, idle=True)
        guard = FeedbackGuard(profile)
        guard.check(serialize_state(robot.states()), time.monotonic(), rows[0][1:], at_start=True)
        owns_mode = True  # Ensure Stop is attempted even if a transition partially fails.
        robot.SwitchMode(rdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        # Apply only values explicitly recorded in the operator-confirmed profile.
        robot.SetForceControlAxis([False] * 6)
        robot.SetCartesianImpedance(profile["cartesian_stiffness"], profile["cartesian_damping_ratio"])
        robot.SetNullSpacePosture(profile["anchor_joint_positions_rad"])
        robot.SetNullSpaceObjectives(*profile["nullspace_objectives"])
        guard.check(serialize_state(robot.states()), time.monotonic(), rows[0][1:], at_start=True)
        record["real_data"] = True
        record["status"] = "recording"
        with (
            (destination / "feedback.jsonl").open("x", buffering=1) as telemetry,
            (destination / "commands.jsonl").open("x", buffering=1) as commands,
            (destination / "joint_feedback.csv").open("x", newline="", buffering=1) as joints,
        ):
            writer = csv.writer(joints)
            writer.writerow(
                ["host_receive_monotonic_ns", "robot_seconds", "robot_nanoseconds"]
                + [f"q{i + 1}_rad" for i in range(7)]
                + [f"dq{i + 1}_rad_s" for i in range(7)]
            )
            start = time.monotonic()
            index, last_stamp = 0, None
            target = rows[0][1:]
            next_tool_check = start
            while index < len(rows) or time.monotonic() < start + rows[-1][0] + 1:
                now = time.monotonic()
                if robot.fault() or not robot.operational() or robot.mode() != rdk.Mode.NRT_CARTESIAN_MOTION_FORCE:
                    raise RuntimeError("Robot readiness/mode changed")
                state = serialize_state(robot.states())
                guard.check(state, now, target)
                stamp = tuple(state["robot_timestamp"])
                if stamp != last_stamp:
                    telemetry.write(json.dumps(state, allow_nan=False) + "\n")
                    writer.writerow(
                        [state["host_receive_monotonic_ns"]] + state["robot_timestamp"] + state["q"] + state["dq"]
                    )
                    last_stamp = stamp
                if now >= next_tool_check:
                    if tool_snapshot(tool) != profile["active_tool"]:
                        raise RuntimeError("Tool/payload changed during trial")
                    next_tool_check = now + 1
                if index < len(rows) and time.monotonic() >= start + rows[index][0]:
                    lateness = time.monotonic() - (start + rows[index][0])
                    if lateness > profile["max_schedule_lateness_s"]:
                        raise RuntimeError("Scheduling deadline missed; abort rather than catch up")
                    target = rows[index][1:]
                    check_workspace(target, profile)
                    guard.check(state, time.monotonic(), target)
                    command = to_rdk_pose(tcp_target(target, profile["active_tool"]["tcp_location_xyzw"]))
                    limits = profile["limits"]
                    before = time.monotonic_ns()
                    robot.SendCartesianMotionForce(
                        command,
                        [0.0] * 6,
                        [0.0] * 6,
                        limits["linear_speed_m_s"],
                        limits["angular_speed_rad_s"],
                        limits["linear_acceleration_m_s2"],
                        limits["angular_acceleration_rad_s2"],
                    )
                    after = time.monotonic_ns()
                    commands.write(
                        json.dumps(
                            {
                                "sequence": index,
                                "reference_time_s": rows[index][0],
                                "planned_flange_pose_xyzw": target,
                                "sent_tcp_pose_wxyz": command,
                                "target_wrench": [0.0] * 6,
                                "terminal_velocity": [0.0] * 6,
                                "send_before_monotonic_ns": before,
                                "send_after_monotonic_ns": after,
                                "feedback_before_command_robot_timestamp": state["robot_timestamp"],
                                "schedule_lateness_s": lateness,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    index += 1
                    record["commands_sent"] = index
                time.sleep(0.005)  # Best-effort ~200 Hz feedback; actual clocks preserved.
        record["status"] = "command_sequence_completed_not_calibration_validated"
    except BaseException as exc:
        record.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if owns_mode and robot is not None:
            try:
                robot.Stop()
                record["stop_requested"] = True
                record["stopped_after_request"] = robot.stopped()
                if not record["stopped_after_request"]:
                    record["status"] = "stop_not_confirmed"
                    print("STOP NOT CONFIRMED. Use the physical E-stop.", flush=True)
            except Exception as exc:  # noqa: BLE001 - preserve evidence even if the SDK stop fails.
                record.update(status="stop_failed", stop_error=str(exc))
                print("STOP REQUEST FAILED. Use the physical E-stop.", flush=True)
        record["files"] = {p.name: digest(p) for p in destination.iterdir() if p.is_file()}
        save_json(destination / "record.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("snapshot", help="Connect and read only; never moves or enables robot")
    p.add_argument("--robot-serial", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rate", type=float, default=15.0)
    p = subs.add_parser("run", help="Offline validation unless --execute is explicitly supplied")
    p.add_argument("--bundle", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--approval")
    p.add_argument("--output", default="real_trials")
    p.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "snapshot":
        result = snapshot(args.robot_serial, args.output, args.rate)
    elif not args.execute:
        _, profile, episode, _ = load_bound(args.bundle, args.episode)
        result = {
            "status": "offline_only_no_sdk_import_or_robot_connection",
            "episode": episode,
            "robot_serial": profile["robot_serial"],
            "requires_approval_and_newton_screen": True,
        }
    else:
        if not args.approval:
            parser.error("--execute requires --approval; no bypass")
        result = execute(args.bundle, args.episode, args.approval, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("status") in {"aborted", "stop_failed", "stop_not_confirmed"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
