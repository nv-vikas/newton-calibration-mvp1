"""CPU contract tests; synthetic fixtures are not hardware/Newton validation."""

import datetime as dt
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect_rdk as c
import motion_suite as m


@pytest.fixture
def proposal(tmp_path):
    root = tmp_path / "proposal"
    m.generate(root)
    return root


@pytest.fixture
def profile(tmp_path):
    p = m.profile_template(15)
    tool = {
        "name": "synthetic-gripper",
        "mass": 0.5,
        "CoM": [0, 0, 0.05],
        "inertia": [0.01, 0.01, 0.01, 0, 0, 0],
        "tcp_location_xyzw": [0, 0, 0.15, 0, 0, 0, 1],
    }
    p.update(
        confirmed=True,
        rate_confirmed=True,
        robot_serial="TEST-ONLY",
        posture_id="synthetic",
        robot_software_version="test-version",
        controller_settings_source="test fixture, not hardware",
        anchor_flange_pose_xyzw=[0.4, 0, 0.5, 0, 0, 0, 1],
        anchor_joint_positions_rad=[0] * 7,
        joint_names=[f"joint{i + 1}" for i in range(7)],
        joint_lower_rad=[-2] * 7,
        joint_upper_rad=[2] * 7,
        max_joint_speed_rad_s=[0.2] * 7,
        active_tool=tool,
        cartesian_stiffness=[100] * 6,
        cartesian_damping_ratio=[0.7] * 6,
        nullspace_objectives=[0, 0, 0.5],
        max_external_wrench_abs=[10, 10, 10, 2, 2, 2],
        tool_payload_description="synthetic fixture",
        fixed_gripper_description="synthetic fixed opening",
        reviewed_workspace_min_m=[0.1, -0.3, 0.2],
        reviewed_workspace_max_m=[0.7, 0.3, 0.9],
        snapshot_path="snapshot.json",
    )
    m.save_json(
        tmp_path / "snapshot.json",
        {
            "robot_serial": p["robot_serial"],
            "active_tool": tool,
            "stationary_samples": True,
            "fresh_samples": True,
            "stopped": True,
            "samples": [{"flange_pose": m.to_rdk_pose(p["anchor_flange_pose_xyzw"]), "q": [0] * 7}],
        },
    )
    p["snapshot_sha256"] = m.digest(tmp_path / "snapshot.json")
    m.save_json(tmp_path / "profile.json", p)
    return tmp_path / "profile.json"


@pytest.fixture
def bound(proposal, profile, tmp_path):
    root = tmp_path / "bound"
    m.bind(proposal, profile, root)
    return root


def state(profile, stamp=(100, 0)):
    flange = profile["anchor_flange_pose_xyzw"]
    tcp = m.tcp_target(flange, profile["active_tool"]["tcp_location_xyzw"])
    return {
        "robot_timestamp": list(stamp),
        "q": [0] * 7,
        "dq": [0] * 7,
        "flange_pose": m.to_rdk_pose(flange),
        "tcp_pose": m.to_rdk_pose(tcp),
        "ext_wrench_in_world": [0] * 6,
    }


def test_deterministic_complete_suite(proposal, tmp_path):
    a = m.verify(proposal)
    b = m.generate(tmp_path / "another")
    assert len(a["episodes"]) == 15
    assert a["duration_s"] == 390
    assert a["episodes"] == b["episodes"]
    assert [e["split"] for e in a["episodes"]].count("heldout") == 2
    assert not a["real_data"] and not a["newton_screened"] and not a["real_execution_approved"]
    for e in a["episodes"]:
        rows = m.read_csv(proposal / e["path"], m.FIELDS)
        assert rows[0][1:] == [0] * 6 == rows[-1][1:]
        if e["kind"] == "reversals":
            axis = m.AXES.index(e["axis"])
            assert max(r[axis + 1] for r in rows) > (0.014 if axis < 3 else 0.035)
            assert min(r[axis + 1] for r in rows) < (-0.014 if axis < 3 else -0.035)
        if e["split"] == "heldout":
            assert all(max(abs(r[j + 1]) for r in rows) > 0.005 for j in range(6))


@pytest.mark.parametrize("rate", [10, 15, 30, 100 / 3, 100])
def test_rate_regeneration(tmp_path, rate):
    root = tmp_path / "motions"
    m.generate(root, rate)
    assert m.verify(root)["command_rate_hz"] == rate


@pytest.mark.parametrize("rate", [0, 1, 101, float("nan"), float("inf")])
def test_reject_invalid_rate(tmp_path, rate):
    with pytest.raises(ValueError):
        m.generate(tmp_path / "bad", rate)


def test_no_overwrite(proposal):
    with pytest.raises(FileExistsError):
        m.generate(proposal)


def test_detect_tamper(proposal):
    path = proposal / "commands/01_x_reversals.csv"
    path.write_text(path.read_text() + "0,0,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="hash"):
        m.verify(proposal)


def test_path_escape(proposal):
    path = proposal / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"][0]["path"] = "../outside.csv"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        m.verify(proposal)


def test_fixed_anchor_not_incremental():
    anchor = [0.4, 0, 0.5, 0, 0, 0, 1]
    target = m.target_pose(anchor, [0.01, 0, 0, 0, 0, 0])
    for _ in range(100):
        assert m.target_pose(anchor, [0.01, 0, 0, 0, 0, 0]) == target
    assert target[0] == pytest.approx(0.41)


def test_full_tcp_rotation_and_wxyz_conversion():
    q90 = m.rotation_quat([0, 0, 1.5707963267948966])
    flange = [1, 2, 3] + q90
    tool = [0.1, 0, 0] + q90
    target = m.tcp_target(flange, tool)
    assert target[:3] == pytest.approx([1, 2.1, 3])
    assert m.rotation_distance(target[3:], m.rotation_quat([0, 0, 3.141592653589793])) < 1e-7
    assert m.from_rdk_pose(m.to_rdk_pose(target)) == target


def test_bound_replays_exact_offsets(bound, proposal):
    manifest, profile, e, rows = c.load_bound(bound, "01_x_reversals")
    offsets = m.read_csv(proposal / e["path"], m.FIELDS)
    for row, source in zip(rows, offsets):
        assert row[1:] == pytest.approx(m.target_pose(profile["anchor_flange_pose_xyzw"], source[1:]))
    assert not manifest["newton_screened"]
    with pytest.raises(ValueError, match="approval missing"):
        c.validate_approval(bound / "approval.template.json", bound, profile, e["id"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("confirmed", False),
        ("rate_confirmed", False),
        ("base_frame", "world"),
        ("rdk_version", "0.9"),
        ("cartesian_damping_ratio", [1] * 6),
        ("cartesian_stiffness", [0] * 6),
        ("anchor_flange_pose_xyzw", [0] * 7),
        ("max_external_wrench_abs", [float("nan")] * 6),
        ("joint_names", ["same"] * 7),
    ],
)
def test_profile_rejects_missing_or_wrong_configuration(profile, key, value):
    p = json.loads(profile.read_text())
    p[key] = value
    with pytest.raises(ValueError):
        m.validate_profile(p, profile.parent)


def test_too_small_workspace_blocks_bind(proposal, profile, tmp_path):
    p = json.loads(profile.read_text())
    p["reviewed_workspace_max_m"][2] = 0.6  # Flange fits; active TCP does NOT.
    profile.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="workspace"):
        m.bind(proposal, profile, tmp_path / "unsafe")
    assert not (tmp_path / "unsafe").exists()


def test_rate_mismatch_blocks_bind(proposal, profile, tmp_path):
    p = json.loads(profile.read_text())
    p["command_rate_hz"] = 30
    profile.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="rate"):
        m.bind(proposal, profile, tmp_path / "wrong-rate")


def test_stale_and_backwards_feedback(profile):
    p = json.loads(profile.read_text())
    g = c.FeedbackGuard(p)
    g.check(state(p), 0, p["anchor_flange_pose_xyzw"], at_start=True)
    with pytest.raises(RuntimeError, match="stale"):
        g.check(state(p), 0.2, p["anchor_flange_pose_xyzw"])
    with pytest.raises(RuntimeError, match="reversed"):
        g.check(state(p, (99, 0)), 0.2, p["anchor_flange_pose_xyzw"])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("q", [1] * 7, "joint branch"),
        ("q", [1.99] * 7, "Joint limit"),
        ("dq", [0.3] * 7, "speed"),
        ("ext_wrench_in_world", [20] * 6, "wrench"),
        ("tcp_pose", [0, 0, 0, 1, 0, 0, 0], "transform"),
    ],
)
def test_feedback_guards(profile, field, value, match):
    p = json.loads(profile.read_text())
    s = state(p)
    s[field] = value
    with pytest.raises(RuntimeError, match=match):
        c.FeedbackGuard(p).check(s, 0, p["anchor_flange_pose_xyzw"], at_start=True)


def approved_fixture(bound):
    a = json.loads((bound / "approval.template.json").read_text())
    a.update({k: True for k in m.APPROVAL_CHECKS})
    a.update(
        operator_name="SYNTHETIC TEST ONLY",
        approved_episodes=["00_stationary"],
        expires_utc="2099-01-01T00:00:00Z",
        screen_report_path="synthetic_screen.json",
    )
    report = {
        "physics": "Newton",
        "surface": "Isaac Lab",
        "passed": True,
        "self_collision_enabled": True,
        "joint_limits_checked": True,
        "singularity_checked": True,
        "swept_volume_checked": True,
        "manifest_sha256": m.digest(bound / "manifest.json"),
        "profile_sha256": m.digest(bound / "profile.json"),
        "screened_episodes": ["00_stationary"],
        "fixture_only_not_real_validation": True,
    }
    m.save_json(bound / "synthetic_screen.json", report)
    a["screen_report_sha256"] = m.digest(bound / "synthetic_screen.json")
    m.save_json(bound / "test_approval.json", a)
    return bound / "test_approval.json"


def test_approval_expiry_and_binding(bound):
    _, p, _, _ = c.load_bound(bound, "00_stationary")
    path = approved_fixture(bound)
    assert c.validate_approval(path, bound, p, "00_stationary")
    with pytest.raises(ValueError, match="not approved"):
        c.validate_approval(path, bound, p, "01_x_reversals")
    with pytest.raises(ValueError, match="Expired"):
        c.validate_approval(path, bound, p, "00_stationary", dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc))
    a = json.loads(path.read_text())
    a["collector_sha256"] = "changed"
    path.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="hash"):
        c.validate_approval(path, bound, p, "00_stationary")


def test_snapshot_has_no_control_calls(tmp_path, monkeypatch, profile):
    p = json.loads(profile.read_text())

    class ReadOnlyRobot:
        def __init__(self, serial):
            assert serial == "TEST-ONLY"
            self.tick = 0

        def info(self):
            return types.SimpleNamespace(
                DoF=7,
                DoF_e=0,
                serial_num="TEST-ONLY",
                software_ver="test-version",
                model_name="synthetic",
                q_min=[-2] * 7,
                q_max=[2] * 7,
                dq_max=[1] * 7,
                tau_max=[100] * 7,
                K_x_nom=[1000] * 6,
            )

        def states(self):
            self.tick += 1
            s = state(p, (100, self.tick))
            s["timestamp"] = s.pop("robot_timestamp")
            return types.SimpleNamespace(**s, tcp_vel=[0] * 6, tau=[0] * 7)

        def mode(self):
            return "IDLE"

        def operational(self):
            return True

        def fault(self):
            return False

        def stopped(self):
            return True

        # All setters/control methods intentionally absent: any call fails.

    class Tool:
        def __init__(self, robot):
            pass

        def name(self):
            return p["active_tool"]["name"]

        def params(self):
            t = p["active_tool"]
            return types.SimpleNamespace(
                mass=t["mass"], CoM=t["CoM"], inertia=t["inertia"], tcp_location=m.to_rdk_pose(t["tcp_location_xyzw"])
            )

    monkeypatch.setattr(c, "sdk", lambda: types.SimpleNamespace(Robot=ReadOnlyRobot, Tool=Tool))
    monkeypatch.setattr(c.time, "sleep", lambda _: None)
    result = c.snapshot("TEST-ONLY", tmp_path / "capture", 15)
    assert result["motion_commands_sent"] == 0
    capture = json.loads((tmp_path / "capture/profile.to_review.json").read_text())
    assert capture["confirmed"] is False and capture["cartesian_stiffness"] is None
    assert capture["anchor_flange_pose_xyzw"] == p["anchor_flange_pose_xyzw"]


def test_sdk_wrong_version_fails_before_import(monkeypatch):
    monkeypatch.setattr(c.importlib.metadata, "version", lambda _: "0.9")
    with pytest.raises(RuntimeError, match="Version review"):
        c.sdk()


@pytest.mark.parametrize("fail_send", [False, True])
def test_mock_trial_records_exact_commands_and_stops(bound, monkeypatch, tmp_path, fail_send):
    """Synthetic SDK verifies lifecycle, NOT actual RDK/network/physics behavior."""
    approval = approved_fixture(bound)
    _, p, _, rows = c.load_bound(bound, "00_stationary")
    clock = [0.0]
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: int(clock[0] * 1e9))
    monkeypatch.setattr(c.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    calls = []

    class Robot:
        def __init__(self, serial):
            self.current_mode = "idle"
            self.tick = 0
            self.sent = 0

        def info(self):
            return types.SimpleNamespace(
                DoF=7,
                DoF_e=0,
                serial_num="TEST-ONLY",
                software_ver="test-version",
                model_name="synthetic",
                q_min=[-2] * 7,
                q_max=[2] * 7,
                dq_max=[1] * 7,
                tau_max=[100] * 7,
                K_x_nom=[1000] * 6,
            )

        def states(self):
            self.tick += 1
            s = state(p, (100, self.tick))
            s["timestamp"] = s.pop("robot_timestamp")
            return types.SimpleNamespace(**s, tcp_vel=[0] * 6, tau=[0] * 7)

        def mode(self):
            return self.current_mode

        def operational(self):
            return True

        def connected(self):
            return True

        def busy(self):
            return False

        def fault(self):
            return False

        def stopped(self):
            return True

        def SwitchMode(self, mode):
            calls.append(("mode", mode))
            self.current_mode = mode

        def SetForceControlAxis(self, axes):
            calls.append(("force_axes", axes))

        def SetCartesianImpedance(self, stiffness, damping):
            calls.append(("impedance", stiffness, damping))

        def SetNullSpacePosture(self, pose):
            calls.append(("nullspace", pose))

        def SetNullSpaceObjectives(self, *weights):
            calls.append(("objectives", weights))

        def SendCartesianMotionForce(self, *values):
            if fail_send and self.sent == 2:
                raise RuntimeError("synthetic transport failure")
            calls.append(("command", values))
            self.sent += 1

        def Stop(self):
            calls.append(("stop",))
            self.current_mode = "idle"

        # No Enable, ClearFault, homing, safety edits or tool setters exist.

    class Tool:
        def __init__(self, robot):
            pass

        def name(self):
            return p["active_tool"]["name"]

        def params(self):
            t = p["active_tool"]
            return types.SimpleNamespace(
                mass=t["mass"], CoM=t["CoM"], inertia=t["inertia"], tcp_location=m.to_rdk_pose(t["tcp_location_xyzw"])
            )

    monkeypatch.setattr(
        c,
        "sdk",
        lambda: types.SimpleNamespace(
            Robot=Robot, Tool=Tool, Mode=types.SimpleNamespace(IDLE="idle", NRT_CARTESIAN_MOTION_FORCE="cartesian")
        ),
    )
    monkeypatch.setattr("builtins.input", lambda _: "RUN TEST-ONLY 00_stationary")
    output = tmp_path / "runs"
    if fail_send:
        with pytest.raises(RuntimeError, match="transport"):
            c.execute(bound, "00_stationary", approval, output)
    else:
        c.execute(bound, "00_stationary", approval, output)
    record = json.loads(next(output.glob("*/record.json")).read_text())
    assert record["commands_sent"] == (2 if fail_send else len(rows))
    assert record["stop_requested"] and record["stopped_after_request"]
    assert calls[-1] == ("stop",)
    assert not record["clock_synchronized"] and not record["hardware_validated"]
    sent = next(output.glob("*/commands.jsonl")).read_text().splitlines()
    assert len(sent) == record["commands_sent"]
    first = json.loads(sent[0])
    assert first["sent_tcp_pose_wxyz"] == pytest.approx([0.4, 0, 0.65, 1, 0, 0, 0])
    assert first["planned_flange_pose_xyzw"] == p["anchor_flange_pose_xyzw"]
