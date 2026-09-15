"""Flexiv scene binding. Generic collection logic stays in the toolkit."""

import numpy as np
import torch
from pxr import Gf, UsdGeom

from newton_calibration.collection import CalibrationRequest, MotionSpec
from newton_calibration.collection.isaaclab_preview import IsaacLabScenePreview
from newton_calibration.collection.isaaclab_probe import IsaacLabPredictionBackend
from newton_calibration.collection.sensitivity import FiniteDifferenceProbe
from newton_calibration.core.io import write_json
from newton_calibration.core.models import ParameterSpec
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning


def run_assisted_collection(
    sim,
    robot,
    peg,
    camera,
    assets,
    output,
    backend,
    *,
    video=True,
    targets=(),
    duration_s=24.0,
    max_candidate_probes=96,
    max_training_experiments=64,
    probe_window_s=None,
    recipe_only=False,
):
    joints = tuple(f"joint{i}" for i in range(1, 8))
    tensor = lambda value: value.torch if hasattr(value, "torch") else value
    order = [robot.joint_names.index(j) for j in joints]
    center = tensor(robot.data.default_joint_pos)[0, order].cpu().numpy().copy()
    center[0] += 0.6  # same explicitly proposed fixture-avoiding center as the prior preview
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
        joint_names=joints,
        center_rad=tuple(center),
        lower_rad=tuple(limits[j][0] for j in joints),
        upper_rad=tuple(limits[j][1] for j in joints),
        amplitude_rad=(float(np.deg2rad(20)),) * 7,
        max_velocity_rad_s=(0.3,) * 7,
        max_acceleration_rad_s2=(0.6,) * 7,
        source="Position limits: active USD joint attributes. Center: scene initial pose +0.6 rad joint1. "
        "20-degree amplitude cap, 0.3 rad/s and 0.6 rad/s² are simulation exploration proposals, NOT OEM operating limits",
        scene_id="flexiv-tabletop-unloaded-newton-v1",
        duration_s=duration_s,
        posture_offsets_rad=((0.0, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, -0.12, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    kp = [4000.0, 4000.0, 3000.0, 3000.0, 500.0, 500.0, 200.0]
    kd = [80.0, 80.0, 60.0, 40.0, 10.0, 10.0, 5.0]
    env = ArticulationEnvCfg(
        usd_path=str(assets / "Flexiv_Rizon4s_Grav.usd"),
        robot_id="flexiv-agent-assist-mvp1",
        # Independent joint parameters, not one shared gain/friction for seven axes.
        joint_groups={j: (j,) for j in joints},
        joint_map={j: j for j in joints},
        joint_order=joints,
        profile_confirmed=False,
        controller_profile_confirmed=False,
        controller_profile_source="Active Newton scene explicit IdealPD initialization priors; not OEM/Flexiv NRT gains",
        dt=sim.get_physics_dt(),
        base_stiffness_by_joint=dict(zip(joints, kp)),
        base_damping_by_joint=dict(zip(joints, kd)),
        base_effort_limit_by_joint={j: 123.0 for j in joints},
        base_armature_by_joint={j: 0.05 for j in joints},
        parameter_bounds={},
    )
    # MVP1 unloaded arm only. Preserve fixture; do not run grasp/contact/insertion tasks.
    pose = tensor(peg.data.default_root_pose).clone()
    pose[:, :3] = torch.tensor([2.0, 0.0, 0.2], device=pose.device)
    peg.write_root_pose_to_sim_index(root_pose=pose)
    peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=pose.device))
    peg.reset()
    if camera:
        prim = sim.stage.GetPrimAtPath("/World/Camera")
        UsdGeom.Camera(prim).GetFocalLengthAttr().Set(43.0)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(1.45, -2.45, 1.85), Gf.Vec3d(-0.1, 0.0, 1.02), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
    preview = IsaacLabScenePreview(
        sim=sim,
        robot=robot,
        camera=camera,
        scene_id=spec.scene_id,
        fixture_prefixes=("/World/Table/", "/World/HoleBlock/", "/World/Ground"),
        fixed_robot_prefixes=("/World/Robot/base_link/",),
        backend_record=backend,
        update_objects=(peg,),
    )
    # Explicit simulation hypotheses, NOT measured values or approved fitting bounds.
    ranges = []
    for joint in joints:
        for kind, lo, hi, initial, unit in (
            ("stiffness_scale", 0.5, 2.0, 1.0, "scale"),
            ("damping_scale", 0.5, 2.0, 1.0, "scale"),
            ("armature", 0.025, 0.1, 0.05, "kg m^2"),
            ("friction_nm", 0.0, 0.5, 0.0, "N m"),
        ):
            ranges.append(
                ParameterSpec(
                    f"{joint}_{kind}",
                    lo,
                    hi,
                    initial,
                    unit,
                    "simulation-proposal",
                    "Exploratory sensitivity range, not a Flexiv specification",
                )
            )
    ranges.append(ParameterSpec("command_delay_s", 0.0, 0.08, 0.0, "s", "toolkit", "Exploratory delay hypothesis"))
    probe = (
        None
        if recipe_only
        else FiniteDifferenceProbe(
            IsaacLabPredictionBackend(
                sim=sim,
                robot=robot,
                environment=env.describe(),
                motion=spec,
                update_objects=(peg,),
                probe_window_s=probe_window_s,
            ),
            ranges,
            source="Active IdealPD baselines: gain scales 0.5–2, armature 0.025–0.1, exploratory friction 0–0.5 Nm and delay 0–80 ms; operator review required",
        )
    )
    # The actual user/agent call: no real evidence and no separate motion generator.
    result = tuning.assist(
        env=env,
        evidence=None,
        collection=spec,
        preview=preview,
        design_probe=probe,
        video=video,
        request=CalibrationRequest(
            target_parameters=tuple(targets),
            max_candidate_probes=max_candidate_probes,
            max_training_experiments=max_training_experiments,
            design_mode="recipe_only" if recipe_only else "adaptive",
        ),
        workdir=output / "runs",
    )
    write_json(output / "assist_result.json", result)
    print("[AGENT-ASSIST] " + result.status + " | " + result.workdir, flush=True)
    if result.status == "preview_failed":
        raise RuntimeError(result.preview["error"])
    if (
        result.status == "design_failed"
        or result.design.get("adaptive_search", {}).get("status") == "no_informative_motion_selected"
    ):
        raise RuntimeError("Adaptive motion design failed; inspect collection_plan.json/design_search.json")
