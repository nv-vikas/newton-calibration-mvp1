"""Flexiv scene binding. Generic collection logic stays in the toolkit."""
import numpy as np
import torch
from pxr import Gf, UsdGeom

from newton_calibration.collection import MotionSpec
from newton_calibration.collection.isaaclab_preview import IsaacLabScenePreview
from newton_calibration.core.io import write_json
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning


def run_assisted_collection(sim, robot, peg, camera, assets, output, backend, *, video=True):
    joints = tuple(f"joint{i}" for i in range(1, 8))
    tensor = lambda value: value.torch if hasattr(value, "torch") else value
    order = [robot.joint_names.index(j) for j in joints]
    center = tensor(robot.data.default_joint_pos)[0, order].cpu().numpy().copy()
    center[0] += .6  # same explicitly proposed fixture-avoiding center as the prior preview
    limits = {}
    for prim in sim.stage.Traverse():
        if prim.GetName() in joints:
            lower = prim.GetAttribute("physics:lowerLimit").Get()
            upper = prim.GetAttribute("physics:upperLimit").Get()
            if lower is not None and upper is not None:
                if prim.GetName() in limits:
                    raise ValueError("Ambiguous joint limits in scene")
                limits[prim.GetName()] = (np.deg2rad(lower), np.deg2rad(upper))
    if set(limits) != set(joints):
        raise ValueError("Cannot discover complete position limits from the active scene USD")
    spec = MotionSpec(
        joint_names=joints, center_rad=tuple(center), lower_rad=tuple(limits[j][0] for j in joints),
        upper_rad=tuple(limits[j][1] for j in joints), amplitude_rad=(float(np.deg2rad(20)),)*7,
        max_velocity_rad_s=(.3,)*7, max_acceleration_rad_s2=(.6,)*7,
        source="Position limits: active USD joint attributes. Center: scene initial pose +0.6 rad joint1. "
               "20-degree amplitude cap, 0.3 rad/s and 0.6 rad/s² are simulation exploration proposals, NOT OEM operating limits",
        scene_id="flexiv-tabletop-unloaded-newton-v1",
    )
    kp = [4000., 4000., 3000., 3000., 500., 500., 200.]
    kd = [80., 80., 60., 40., 10., 10., 5.]
    env = ArticulationEnvCfg(
        usd_path=str(assets / "Flexiv_Rizon4s_Grav.usd"), robot_id="flexiv-agent-assist-mvp1",
        joint_groups={"arm": joints}, joint_map={j: j for j in joints}, joint_order=joints,
        profile_confirmed=False, controller_profile_confirmed=False,
        controller_profile_source="Active Newton scene explicit IdealPD initialization priors; not OEM/Flexiv NRT gains",
        dt=sim.get_physics_dt(), base_stiffness_by_joint=dict(zip(joints, kp)),
        base_damping_by_joint=dict(zip(joints, kd)), base_effort_limit_by_joint={j: 123. for j in joints},
        base_armature_by_joint={j: .05 for j in joints}, parameter_bounds={},
    )
    # MVP1 unloaded arm only. Preserve fixture; do not run grasp/contact/insertion tasks.
    pose = tensor(peg.data.default_root_pose).clone()
    pose[:, :3] = torch.tensor([2., 0., .2], device=pose.device)
    peg.write_root_pose_to_sim_index(root_pose=pose)
    peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=pose.device))
    peg.reset()
    if camera:
        prim = sim.stage.GetPrimAtPath("/World/Camera")
        UsdGeom.Camera(prim).GetFocalLengthAttr().Set(43.)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.45, -2.45, 1.85), Gf.Vec3d(-.1, 0., 1.02), Gf.Vec3d(0, 0, 1)).GetInverse())
    preview = IsaacLabScenePreview(sim=sim, robot=robot, camera=camera, scene_id=spec.scene_id,
                                  fixture_prefixes=("/World/Table/", "/World/HoleBlock/", "/World/Ground"),
                                  fixed_robot_prefixes=("/World/Robot/base_link/",),
                                  backend_record=backend, update_objects=(peg,))
    # The actual user/agent call: no real evidence and no separate motion generator.
    result = tuning.assist(env=env, evidence=None, collection=spec, preview=preview, video=video,
                           workdir=output / "runs")
    write_json(output / "assist_result.json", result)
    print("[AGENT-ASSIST] " + result.status + " | " + result.workdir, flush=True)
    if result.status == "preview_failed":
        raise RuntimeError(result.preview["error"])
