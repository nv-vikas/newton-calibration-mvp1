"""Generate bounded, reference-only free-motion commands; no hardware access.

Run from the scene folder. These are collection inputs, never real evidence.
The on-site operator must approve mapping, pose, payload and collision clearance.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [f"joint{i}" for i in range(1, 8)]
RATE = 100
HEADERS = ["time_s"] + [f"q{i}_rad" for i in range(1, 8)] + [f"dq{i}_rad_s" for i in range(1, 8)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trajectory(center, amplitudes, frequencies, duration=16.):
    t = np.arange(round(duration * RATE) + 1) / RATE
    x = np.clip(t - 2., 0., duration - 4.)
    span = duration - 4.
    window = np.sin(np.pi * x / span) ** 2
    dw = np.pi / span * np.sin(2 * np.pi * x / span)
    phase = 2 * np.pi * x[:, None] * frequencies
    q = np.asarray(center) + amplitudes * window[:, None] * np.sin(phase)
    dq = amplitudes * (dw[:, None] * np.sin(phase) + window[:, None] * 2 * np.pi * frequencies * np.cos(phase))
    return np.column_stack([t, q, dq])


def main():
    output = ROOT / "mvp1_collection"
    output.mkdir(exist_ok=True)
    (output / "commands").mkdir(exist_ok=True)
    scene = json.loads((ROOT / "assets/asset_manifest.json").read_text())
    center = np.array([scene["scene"]["joint_positions_rad"][j] for j in JOINTS])
    # Rotate the collection pose away from the fixture. No file moves the real
    # robot to this pose: an operator must teach/approve it independently.
    center[0] += .6
    lower = np.deg2rad([-165, -135, -175, -112, -175, -85, -175])
    upper = np.deg2rad([165, 135, 175, 159, 175, 265, 175])
    jobs = []
    for i in range(7):
        amplitudes = np.zeros(7)
        amplitudes[i] = np.deg2rad(2.)
        jobs.append((f"train_joint{i + 1}_reversals", "train", amplitudes, np.full(7, .25), 16.))
    for k in range(2):
        jobs.append((f"heldout_multijoint_{k + 1}", "heldout", np.full(7, np.deg2rad(1.)),
                     np.array([.11, .13, .17, .19, .23, .29, .31]) + k * .02, 20.))
    episodes = []
    for name, split, amplitude, frequency, duration in jobs:
        values = trajectory(center, amplitude, frequency, duration)
        q, dq = values[:, 1:8], values[:, 8:15]
        ddq = np.gradient(dq, 1 / RATE, axis=0)
        assert np.isfinite(values).all()
        assert (q > lower + .15).all() and (q < upper - .15).all()
        assert np.abs(dq).max() < .12 and np.abs(ddq).max() < .5
        assert np.max(np.abs(q[[0, -1]] - center)) < 1e-10
        filename = output / "commands" / f"{name}.csv"
        with filename.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADERS)
            writer.writerows(values.tolist())
        episodes.append(dict(name=name, split=split, command_file=f"commands/{name}.csv", sha256=digest(filename),
                             duration_s=duration, samples=len(values),
                             max_abs_velocity_rad_s=float(np.abs(dq).max()),
                             max_abs_acceleration_rad_s2=float(np.abs(ddq).max()),
                             status="reference_commands_only; not collected evidence"))
    manifest = dict(schema="flexiv.mvp1.collection/v1", robot="Flexiv Rizon 4s with Grav",
        purpose="unloaded arm free-motion system identification; gripper held fixed",
        asset_sha256=digest(ROOT / "assets/Flexiv_Rizon4s_Grav.usd"),
        command_rate_hz=RATE, expected_joint_order=JOINTS, center_rad=center.tolist(),
        lower_rad=lower.tolist(), upper_rad=upper.tolist(), episodes=episodes,
        real_samples=0, calibrated=False, real_motion_authorized=False,
        rdk_api_target="1.9 (four-argument SendJointPosition); incompatible APIs fail closed",
        runtime_backend="isaaclab_newton", command_mode="NRT_JOINT_POSITION",
        timing_note="NRT controller smooths targets. Log actual send times and robot timestamps separately. Host receive time is not encoder capture time.",
        proposed_mapping=[dict(rdk_index=i, usd_joint=j, units="rad", sign=1, offset_rad=0., confirmed=False) for i,j in enumerate(JOINTS)],
        preconditions=["Qualified operator, E-stop and guarded clear workspace", "Remove peg; no contact with fixture/table",
            "Confirm RDK version and exact joint index/sign/zero mapping", "Confirm fixed gripper opening, tool mass/COM/inertia and mounting",
            "Independently teach and check the collection center; runner never auto-homes",
            "Review Newton sweep report AND real swept volume, cables and self-collision clearance",
            "Configure existing hardware safety limits; runner never disables or changes them"],
        fitting_scope=dict(propose=["effective stiffness", "effective damping", "joint friction", "armature if evidence supports it"],
            hold_fixed=["effort limits (no intentional saturation)", "link mass/inertia absent independent measurement"],
            conditional=["command delay only with characterized clocks", "residual only after structured held-out error"],
            exclude=["gripping force", "peg/object friction", "insertion", "real task transfer"]),
        current_toolkit_gap="Default generic recipe still requires effort-saturation evidence and synchronized clocks. A reviewed parameter-subset recipe/controller adapter is needed before fit; never spoof these declarations.")
    (output / "collection_plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(files=len(episodes), duration_s=sum(e["duration_s"] for e in episodes),
                         output=str(output), real_samples=0), indent=2))


if __name__ == "__main__":
    main()
