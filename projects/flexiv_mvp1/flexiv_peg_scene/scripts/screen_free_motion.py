"""Replay reference commands in Newton; never turn simulation into real evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def screen(sim, robot, peg, plan_path, output):
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from collect_flexiv_rdk import load_commands
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    base = tensor(robot.data.default_joint_pos).clone()
    ids = [robot.joint_names.index(j) for j in plan["expected_joint_order"]]
    base[:, ids] = torch.tensor(plan["center_rad"], device=base.device)
    # Unloaded, fixed-command gripper. The peg is parked off the table; it is
    # not attached to the arm and contributes no payload during these trials.
    pose = tensor(peg.data.default_root_pose).clone()
    pose[:, :3] = torch.tensor([2., 0., .2], device=base.device)
    peg.write_root_pose_to_sim_index(root_pose=pose)
    peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6),device=base.device))
    peg.reset()
    result = dict(schema="flexiv.newton.motion-screen/v1", physics="Newton/MuJoCo Warp",
                  command_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  real_data=False, real_execution_approved=False, self_collision_certified=False,
                  gripper="constant commanded opening; no peg", scene="same tabletop and hole fixture",
                  tests=[], limitations=["Not a hardware safety certification",
                    "Real cables, limits, mounting and swept volume require on-site review",
                    "Independent gripper drives are a model approximation, not OEM linkage validation",
                    "Newton directly receives position references; Flexiv NRT smoothing is not emulated"])
    output = Path(output)
    model = NewtonManager._model
    labels = list(model.shape_label)
    dt = sim.get_physics_dt()
    def advance(q):
        robot.set_joint_position_target_index(target=q)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        peg.update(dt)
    for episode in plan["episodes"]:
        _, _, rows = load_commands(plan_path, episode["name"])
        values = np.asarray([[r["time_s"]] + [r[f"q{i}_rad"] for i in range(1,8)] for r in rows])
        robot.write_joint_position_to_sim_index(position=base)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(base))
        robot.reset()
        for _ in range(round(1. / dt)):
            advance(base)
        n = round(episode["duration_s"] / dt) + 1
        error = 0.
        velocity = 0.
        contacts_seen = set()
        trace = []
        for k in range(n):
            t = k * dt
            target = base.clone()
            command = np.array([np.interp(t, values[:,0], values[:,i+1]) for i in range(7)])
            target[:, ids] = torch.tensor(command, device=base.device, dtype=base.dtype)
            advance(target)
            actual = tensor(robot.data.joint_pos)[0,ids].cpu().numpy()
            dq = tensor(robot.data.joint_vel)[0,ids].cpu().numpy()
            if not np.isfinite(actual).all() or not np.isfinite(dq).all():
                raise RuntimeError("Non-finite Newton motion")
            error = max(error, float(np.abs(command-actual).max()))
            velocity = max(velocity, float(np.abs(dq).max()))
            # Query every simulated timestep. Contacts among static fixtures
            # and the fixed robot base are expected, not free-motion failures.
            contacts = NewtonManager.get_contacts()
            count = int(contacts.rigid_contact_count.numpy()[0])
            a = contacts.rigid_contact_shape0.numpy()[:count]
            b = contacts.rigid_contact_shape1.numpy()[:count]
            for ia, ib in zip(a,b):
                if ia < 0 or ib < 0:
                    continue
                la, lb = labels[int(ia)], labels[int(ib)]
                moving = lambda name: "/Robot/link" in name or "/Robot/Grav_gripper/" in name
                fixture = lambda name: "/Table/" in name or "/HoleBlock/" in name or name == "/World/Ground"
                if (moving(la) and fixture(lb)) or (moving(lb) and fixture(la)):
                    contacts_seen.add(tuple(sorted((la, lb))))
            if k % 24 == 0:
                trace.append(dict(time_s=t, command_q=command.tolist(), simulated_q=actual.tolist()))
        item = dict(name=episode["name"], split=episode["split"], simulated_steps=n,
                    finite=True, max_position_error_rad=error, max_joint_velocity_rad_s=velocity,
                    moving_robot_fixture_contact_pairs=sorted(contacts_seen),
                    passed=bool(error < .10 and velocity < .15 and not contacts_seen),
                    trace=trace)
        result["tests"].append(item)
        result["passed"] = len(result["tests"]) == len(plan["episodes"]) and all(x["passed"] for x in result["tests"])
        (output / "free_motion_screen.json").write_text(json.dumps(result, indent=2)+"\n")
        print("[FLEXIV-SCREEN] " + json.dumps({k:v for k,v in item.items() if k != "trace"}), flush=True)
    return result
