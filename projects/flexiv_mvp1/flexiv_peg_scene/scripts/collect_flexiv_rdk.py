"""Offline by default. Execute ONE operator-approved Flexiv RDK 1.9 trial.

No auto-homing, auto-enable, fault clearing, torque control or safety-limit edits.
This application is not a safety controller. Keep the physical E-stop available.
NRT smoothing remains part of the measured command-to-motion system.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import uuid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_commands(plan_path, episode_name):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    episode = next(e for e in plan["episodes"] if e["name"] == episode_name)
    source = (plan_path.parent / episode["command_file"]).resolve()
    if not source.is_relative_to(plan_path.parent):
        raise ValueError("Command path escapes collection package")
    if digest(source) != episode["sha256"]:
        raise ValueError("Command file hash changed; regenerate and reapprove")
    with source.open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if len(rows) != episode["samples"] or len(rows) < 2:
        raise ValueError("Command count mismatch")
    center = plan["center_rad"]
    last = None
    last_dq = None
    for row in rows:
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("Non-finite commands")
        t = row["time_s"]
        if last is not None and abs(t - last - .01) > 1e-7:
            raise ValueError("Expected uniform 100 Hz reference commands")
        for i in range(7):
            q, dq = row[f"q{i+1}_rad"], row[f"dq{i+1}_rad_s"]
            if not plan["lower_rad"][i] + .15 < q < plan["upper_rad"][i] - .15:
                raise ValueError("Joint limit margin violated")
            if abs(q - center[i]) > math.radians(2.01) or abs(dq) > .12:
                raise ValueError("Approved small-motion envelope exceeded")
            if last_dq and abs(dq - last_dq[i]) / .01 > .5:
                raise ValueError("Acceleration envelope exceeded")
        last, last_dq = t, [row[f"dq{i+1}_rad_s"] for i in range(7)]
    if rows[0]["time_s"] != 0 or abs(rows[-1]["time_s"] - episode["duration_s"]) > 1e-7:
        raise ValueError("Reference duration mismatch")
    for row in (rows[0], rows[-1]):
        if any(abs(row[f"q{i+1}_rad"] - center[i]) > 1e-8 for i in range(7)):
            raise ValueError("Trial must start and end at its reviewed center")
    return plan, episode, rows


def validate_approval(approval, plan, plan_path, serial):
    for key in ("qualified_operator", "mapping_confirmed", "payload_confirmed", "peg_removed",
                "gripper_fixed", "real_swept_volume_reviewed", "safety_limits_configured",
                "estop_available", "center_taught_and_verified"):
        if approval.get(key) is not True:
            raise ValueError(f"On-site approval missing: {key}")
    if not approval.get("operator_name") or not approval.get("tool_payload_description"):
        raise ValueError("Named operator and tool/payload description required")
    if approval.get("robot_serial") != serial or not serial:
        raise ValueError("Approval must identify this exact robot")
    if approval.get("plan_sha256") != digest(plan_path):
        raise ValueError("Approval does not match this exact collection plan")
    confirmed_mapping = [{**m, "confirmed": True} for m in plan["proposed_mapping"]]
    if approval.get("confirmed_rdk_to_usd_mapping") != confirmed_mapping:
        raise ValueError("Review the exact proposed mapping; changed mapping requires regenerated commands")
    if approval.get("newton_free_motion_report_sha256") in (None, "", "REQUIRED"):
        raise ValueError("Review and record the Newton free-motion screening report first")


def state_payload(state):
    # Keep native clock and host receipt clock distinct. Do not assign a false
    # shared acquisition timestamp or call telemetry 'commanded' motion.
    payload = {"host_receive_monotonic_ns": time.monotonic_ns(),
               "robot_timestamp": list(state.timestamp)}
    for key in ("q", "dq", "theta", "dtheta", "tau", "tau_des", "tau_ext",
                "temperature", "tcp_pose", "ext_wrench_in_tcp"):
        payload[key] = [float(v) for v in getattr(state, key)]
    if len(payload["q"]) != 7 or len(payload["dq"]) != 7:
        raise ValueError("Only seven-axis Rizon without external axes is supported")
    if not all(math.isfinite(v) for key in ("q", "dq") for v in payload[key]):
        raise ValueError("Invalid encoder feedback")
    return payload


def execute(args, plan, episode, rows):
    approval = json.loads(Path(args.approval).read_text())
    validate_approval(approval, plan, args.plan, args.robot_serial)
    report_path = Path(args.approval).resolve().parent / approval.get("newton_free_motion_report", "MISSING")
    if not report_path.is_file() or digest(report_path) != approval["newton_free_motion_report_sha256"]:
        raise ValueError("Newton screening report is missing or its hash changed")
    report = json.loads(report_path.read_text())
    if report.get("passed") is not True or report.get("physics") != "Newton/MuJoCo Warp":
        raise ValueError("A passed Newton screening report is required")
    if report.get("command_plan_sha256") != digest(args.plan):
        raise ValueError("Newton screening used a different command plan")
    version = importlib.metadata.version("flexivrdk")
    if version not in ("1.9", "1.9.0"):
        raise RuntimeError(f"This runner targets RDK 1.9 API, installed {version}; adapt/version-test before execution")
    import flexivrdk
    destination = Path(args.output) / f"{episode['name']}-{uuid.uuid4().hex[:12]}"
    destination.mkdir(parents=True, exist_ok=False)
    record = dict(trial_id=destination.name, split=episode["split"], robot_serial=args.robot_serial,
                  rdk_version=version, plan_sha256=digest(args.plan), command_sha256=episode["sha256"],
                  approval=approval, status="started", clock_synchronized=False,
                  command_mode="NRT_JOINT_POSITION", real_data=True,
                  note="Host-command-to-received-feedback data; encoder clock alignment must be characterized before latency fitting")
    robot = None
    mode_taken = False
    sent = 0
    try:
        robot = flexivrdk.Robot(args.robot_serial)
        if robot.fault() or not robot.operational() or robot.busy() or not robot.stopped():
            raise RuntimeError("Robot must already be enabled, fault-free, stopped and idle under operator control")
        if robot.mode() != flexivrdk.Mode.IDLE:
            raise RuntimeError("Refusing to take over a robot that is not in IDLE")
        info = robot.info()
        if info.DoF != 7:
            raise RuntimeError("Unexpected DOF layout; external axes are not supported")
        initial = state_payload(robot.states())
        if any(abs(a - b) > .005 for a, b in zip(initial["q"], plan["center_rad"])):
            raise RuntimeError("Not at approved center (0.005 rad tolerance). Teach it manually; no automatic move is made")
        if max(abs(v) for v in initial["dq"]) > .01:
            raise RuntimeError("Robot is moving")
        for row in rows:
            if any(not info.q_min[i] + .15 < row[f"q{i+1}_rad"] < info.q_max[i] - .15 for i in range(7)):
                raise RuntimeError("Trajectory violates actual robot limits")
        record["robot_limits"] = {k: list(getattr(info,k)) for k in ("q_min", "q_max", "dq_max", "tau_max")}
        if any(v < .12 for v in info.dq_max):
            raise RuntimeError("Actual velocity capability is below this reviewed collection profile")
        print("Ready for ONE bounded trial. Keep workspace clear and E-stop available.", flush=True)
        if input(f"Type the robot serial {args.robot_serial} to execute: ").strip() != args.robot_serial:
            raise RuntimeError("Execution cancelled")
        # Recheck after human confirmation, which may take arbitrary time.
        initial = state_payload(robot.states())
        if robot.busy() or robot.mode() != flexivrdk.Mode.IDLE or not robot.operational() or not robot.stopped():
            raise RuntimeError("Robot status changed during confirmation")
        if any(abs(a-b) > .005 for a,b in zip(initial["q"], plan["center_rad"])):
            raise RuntimeError("Center changed during confirmation")
        mode_taken = True
        robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        previous_clock = tuple(initial["robot_timestamp"])
        last_fresh = time.monotonic()
        start = time.monotonic()
        with (destination / "raw.jsonl").open("x", buffering=1) as stream:
            for row in rows:
                deadline = start + row["time_s"]
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                if time.monotonic() - deadline > .05:
                    raise RuntimeError("Command scheduling stalled >50ms; stop, do not catch up")
                if robot.fault() or not robot.operational() or robot.mode() != flexivrdk.Mode.NRT_JOINT_POSITION:
                    raise RuntimeError("Robot fault/readiness/mode changed")
                state = state_payload(robot.states())
                stamp = tuple(state["robot_timestamp"])
                if stamp < previous_clock:
                    raise RuntimeError("Robot timestamp reversed")
                if stamp > previous_clock:
                    last_fresh = time.monotonic()
                if time.monotonic() - last_fresh > .1:
                    raise RuntimeError("Feedback stale >100ms")
                previous_clock = stamp
                q = [row[f"q{i}_rad"] for i in range(1, 8)]
                dq = [row[f"dq{i}_rad_s"] for i in range(1, 8)]
                if any(abs(a-b) > .10 for a,b in zip(q,state["q"])) or max(abs(v) for v in state["dq"]) > .15:
                    raise RuntimeError("Tracking/speed envelope exceeded")
                before = time.monotonic_ns()
                # Zero terminal velocity, matching the vendor NRT sine-sweep
                # example. dq in the CSV is the reference path derivative,
                # not a claim to observe the controller's internal target.
                robot.SendJointPosition(q, [0.] * 7, [.12] * 7, [.5] * 7)
                after = time.monotonic_ns()
                stream.write(json.dumps(dict(sequence=sent, reference_time_s=row["time_s"],
                    command_q=q, command_dq=[0.] * 7, reference_path_dq=dq,
                    command_send_before_ns=before, command_send_after_ns=after,
                    feedback_before_command=state), allow_nan=False) + "\n")
                sent += 1
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "aborted"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if robot is not None and mode_taken:
            try:
                robot.Stop()
                record["stop_requested"] = True
            except Exception as exc:
                record["stop_error"] = str(exc)
                record["status"] = "stop_failed"
                print("STOP REQUEST FAILED. Use the physical E-stop.", flush=True)
        record["commands_sent"] = sent
        record["raw_sha256"] = digest(destination / "raw.jsonl") if (destination / "raw.jsonl").exists() else None
        (destination / "record.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval")
    parser.add_argument("--robot-serial")
    parser.add_argument("--output", default="real_trials")
    args = parser.parse_args()
    plan, episode, rows = load_commands(args.plan, args.episode)
    if not args.execute:
        print(json.dumps(dict(mode="OFFLINE ONLY; no RDK import or connection", episode=episode,
                              center_rad=plan["center_rad"], preconditions=plan["preconditions"]), indent=2))
        return
    if not args.approval or not args.robot_serial:
        parser.error("--execute requires --approval and --robot-serial")
    execute(args, plan, episode, rows)


if __name__ == "__main__":
    main()
