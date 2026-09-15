"""Actual Isaac Lab 3.0 Flexiv tabletop scene, pre-grasp setup and contact probes.

This validates a scene, not a trained insertion policy or real robot transfer.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=None)
parser.add_argument("--steps", type=int, default=2400)
parser.add_argument("--backend", choices=["newton", "physx"], default="newton")
parser.add_argument("--collection-plan", help="Screen free-motion command files in this Newton scene; no hardware access")
parser.add_argument("--capture", action="store_true")
parser.add_argument("--video", action="store_true", help="Record a 24-second camera tour of the live simulated setup")
parser.add_argument("--run-contact-probes", action="store_true")
parser.add_argument("--interactive", action="store_true", help="Keep the scene running after checks; use with --viz kit")
parser.add_argument("--probe", choices=["held", "centered", "offset"], default="held")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.capture = True
if args.capture:
    args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import importlib.metadata
import json
import os
import numpy as np
import torch
from PIL import Image
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
if args.backend == "newton":
    from isaaclab_newton.assets import Articulation, RigidObject
    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg
else:
    from isaaclab_physx.assets import Articulation, RigidObject
from isaaclab.sim import SimulationCfg, SimulationContext

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUT = Path(args.output) if args.output else ROOT / "output" / args.probe
OUTPUT.mkdir(parents=True, exist_ok=True)
MANIFEST = json.loads((ASSETS / "asset_manifest.json").read_text())
POSE = MANIFEST["scene"]
DT = 1 / 960 if args.backend == "newton" else 1 / 240


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def log(message):
    print("[FLEXIV] " + message, flush=True)


def box(stage, path, pos, size, color, *, collision=True, metallic=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    mat = UsdShade.Material.Define(stage, path + "/Material")
    shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def main():
    physics_options = {}
    if args.backend == "newton":
        physics_options = dict(
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(iterations=100, tolerance=1e-6,
                    integrator="implicitfast", use_mujoco_contacts=False,
                    njmax=1500, nconmax=512),
                default_shape_cfg=NewtonShapeCfg(margin=.00005, gap=.00005),
                num_substeps=1, use_cuda_graph=True),
            use_newton_actuators=False)
    sim = SimulationContext(SimulationCfg(dt=DT, device=args.device, render_interval=4, **physics_options))
    stage = sim.stage
    sim.set_camera_view((1.9, -2.7, 2.1), (-.12, 0, .87))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    log("Isaac Lab SimulationContext created")
    box(stage, "/World/Ground", (0, 0, -.025), (10., 10., .05), (.17, .20, .23))
    dome = sim_utils.DomeLightCfg(intensity=1600, color=(.86, .91, 1.0))
    dome.func("/World/Dome", dome)
    for name, pos, intensity, size in [("Key", (.3, -1.0, 2.8), 1600, 2.0),
                                       ("Fill", (-.9, .6, 2.4), 1000, 1.5)]:
        light = UsdLux.RectLight.Define(stage, "/World/" + name)
        light.CreateIntensityAttr(intensity)
        light.CreateWidthAttr(size)
        light.CreateHeightAttr(size)
        light.AddTranslateOp().Set(Gf.Vec3d(*pos))
    box(stage, "/World/Table/Top", (-.05, 0, .7225), (1.6, .9, .055), (.42, .46, .49), metallic=.3)
    for x in [-.74, .64]:
        for y in [-.34, .34]:
            box(stage, f"/World/Table/Leg_{str(x).replace('-', 'n').replace('.', '_')}_{str(y).replace('-', 'n').replace('.', '_')}",
                (x, y, .35), (.065, .065, .7), (.09, .12, .15), metallic=.3)
    box(stage, "/World/Table/MountPlate", (-.45, 0, .7575), (.23, .23, .015), (.13, .15, .17), metallic=.6)
    # Bolt heads make the robot mounting explicit; the plate is the physical mount.
    for dx in [-.09, .09]:
        for dy in [-.09, .09]:
            p = f"/World/Table/Bolt_{'p' if dx > 0 else 'n'}{'p' if dy > 0 else 'n'}"
            cyl = UsdGeom.Cylinder.Define(stage, p)
            cyl.CreateRadiusAttr(.006)
            cyl.CreateHeightAttr(.004)
            cyl.AddTranslateOp().Set(Gf.Vec3d(-.45 + dx, dy, .767))
            cyl.CreateDisplayColorAttr([(.16, .17, .18)])

    fixture = stage.DefinePrim("/World/HoleBlock", "Xform")
    fixture.GetReferences().AddReference(str(ASSETS / "HoleBlock.usd"))
    UsdGeom.Xformable(fixture).AddTranslateOp().Set(Gf.Vec3d(*POSE["fixture_position_m"]))
    actuator_cls = IdealPDActuatorCfg if args.backend == "newton" else ImplicitActuatorCfg
    arm_limit = dict(effort_limit=123., effort_limit_sim=1e9) if args.backend == "newton" else dict(effort_limit_sim=123.)
    gripper_limit = dict(effort_limit=5., effort_limit_sim=1e9) if args.backend == "newton" else dict(effort_limit_sim=5.)
    # Explicit sampled PD cannot reuse PhysX's implicit-drive gains on tiny
    # wrist/finger inertias. These are bounded simulation initialization priors,
    # not identified controller values or robot hardware settings.
    arm_pd = dict(stiffness=4000., damping=180.)
    grip_pd = dict(stiffness=80., damping=2.)
    if args.backend == "newton":
        arm_pd = dict(stiffness={"joint[1-2]": 4000., "joint[3-4]": 3000., "joint[5-6]": 500., "joint7": 200.},
                      damping={"joint[1-2]": 80., "joint3": 60., "joint4": 40., "joint[5-6]": 10., "joint7": 5.},
                      armature=.05)
        grip_pd = dict(stiffness=800., damping=2., armature=.01)
    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "Flexiv_Rizon4s_Grav.usd"),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=32,
                solver_velocity_iteration_count=8)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-.45, 0, .765),
            joint_pos=POSE["joint_positions_rad"], joint_vel={".*": 0.0}),
        actuators={
            "arm": actuator_cls(joint_names_expr=["joint[1-7]"], **arm_pd, **arm_limit),
            "gripper": actuator_cls(joint_names_expr=["finger_joint", ".*_knuckle_joint", ".*_finger_joint"],
                                          **grip_pd, **gripper_limit)},
    ))
    root_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/World/Robot/joints/root_joint"))
    root_joint.CreateLocalPos0Attr(Gf.Vec3f(-.45, 0, .765))
    peg_pos = [.15, 0, .925]
    if args.probe != "held":
        # Tip begins 10mm above the block. Offset goes onto solid material.
        peg_pos = [.15 + (.017 if args.probe == "offset" else 0), 0, .875]
    peg = RigidObject(RigidObjectCfg(
        prim_path="/World/Peg",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "Peg_25.usd")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(peg_pos), rot=(0., 1., 0., 0.)),
    ))
    # Tight clearance requires contact offsets well below the 0.25mm radial gap.
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            cfg = PhysxSchema.PhysxCollisionAPI.Apply(p)
            cfg.CreateContactOffsetAttr(.00005)
            cfg.CreateRestOffsetAttr(0.0)
    if args.backend == "newton":
        # Expand vendor instances so Newton does not omit arm collision meshes.
        for p in list(stage.Traverse()):
            if p.IsInstance():
                p.SetInstanceable(False)
        for p in list(stage.Traverse()):
            if str(p.GetPath()).startswith("/World/Robot/") and "/collisions/" in str(p.GetPath()) and p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
    camera = None
    if args.capture:
        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        camera = Camera(CameraCfg(
            prim_path="/World/Camera", height=1080, width=1920,
            data_types=["rgb"], renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(focal_length=30., horizontal_aperture=36., clipping_range=(.01, 20.))))
    log("Spawned robot, table, exact hole mesh and dynamic peg; resetting")
    sim.reset()
    if args.backend == "newton":
        from isaaclab_newton.physics.newton_manager import NewtonManager
        model = NewtonManager._model
        backend_record = {
            "physics": "Newton", "solver": "MuJoCo Warp", "contact_pipeline": "Newton",
            "newton_version": importlib.metadata.version("newton"),
            "body_count": model.body_count, "shape_count": model.shape_count,
            "body_labels": list(model.body_label), "shape_labels": list(model.shape_label),
            "shape_friction": model.shape_material_mu.numpy().tolist(),
            "dt_s": DT, "num_substeps": 1, "solver_iterations": 100, "solver_tolerance": 1e-6,
            "actuator": "Isaac Lab explicit IdealPD with effort clipping",
        }
        (OUTPUT / "newton_backend_attestation.json").write_text(json.dumps(backend_record, indent=2) + "\n")
        log("Verified Newton model: " + str(model.body_count) + " bodies")
    log("Runtime initialized; joints=" + str(robot.joint_names))
    q = tensor(robot.data.default_joint_pos).clone()
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    robot.reset()
    root_pose = tensor(peg.data.default_root_pose).clone()
    log("Peg initial default pose=" + str(root_pose.cpu().tolist()))
    peg.write_root_pose_to_sim_index(root_pose=root_pose)
    peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=args.device))
    peg.reset()
    hold = q.clone()
    if args.probe == "held":
        close_delta = .02 if args.backend == "newton" else .012
        for i, name in enumerate(robot.joint_names):
            if name in {"finger_joint", "right_outer_knuckle_joint", "left_inner_knuckle_joint", "right_inner_knuckle_joint"}:
                hold[:, i] -= close_delta
            elif name in {"left_outer_finger_joint", "right_outer_finger_joint"}:
                hold[:, i] += close_delta
    # Contact probes park the arm away from the falling peg using the authored
    # pre-grasp pose: its fingers are 50mm above the contact-probe start.
    trace = []
    initial = tensor(peg.data.root_pose_w).cpu().numpy().copy()
    for step in range(args.steps):
        robot.set_joint_position_target_index(target=hold)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
        peg.update(DT)
        pos = tensor(peg.data.root_pose_w).cpu().numpy()[0]
        joints = tensor(robot.data.joint_pos).cpu().numpy()[0]
        if not np.isfinite(pos).all() or not np.isfinite(joints).all():
            raise RuntimeError("Non-finite simulation state")
        if step % 24 == 0 or step == args.steps - 1:
            trace.append({"step": step, "peg_pose": pos.tolist(), "joint_positions_rad": joints.tolist(), "max_joint_error_rad": float(np.max(np.abs(joints - q.cpu().numpy()[0])))})
    final = tensor(peg.data.root_pose_w).cpu().numpy()[0]
    displacement = float(np.linalg.norm(final[:3] - initial[0, :3]))
    log(f"Completed {args.steps} steps; peg displacement={displacement:.6f}m; pose={final.tolist()}")
    if camera:
        def capture(name, eye, target, focal):
            p = stage.GetPrimAtPath("/World/Camera")
            UsdGeom.Camera(p).GetFocalLengthAttr().Set(focal)
            x = UsdGeom.Xformable(p)
            x.ClearXformOpOrder()
            x.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
            for _ in range(24):
                sim.render()
                camera.update(1 / 60, force_recompute=True)
            pixels = tensor(camera.data.output["rgb"])[0].cpu().numpy()[..., :3].astype(np.uint8)
            Image.fromarray(pixels).save(OUTPUT / name)
            log("Saved " + name)
        capture("flexiv_scene_overview.png", (1.9, -2.7, 2.1), (-.12, 0, .87), 37.)
        capture("flexiv_gripper_peg_closeup.png", (.62, -.65, 1.18), (.15, 0, .87), 60.)
        capture("hole_block_top.png", (.18, -.16, 1.19), (.15, 0, .77), 55.)
        if args.video:
            from capture_video import capture_setup_video
            capture_setup_video(sim, robot, peg, camera, hold, OUTPUT, backend=args.backend)
    # Export the complete authored reset scene, portable references made relative.
    # The exported scene should also hold the peg when opened and played directly.
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in robot.joint_names:
            index = robot.joint_names.index(prim.GetName())
            UsdPhysics.DriveAPI.Apply(prim, "angular").CreateTargetPositionAttr(float(np.rad2deg(hold[0, index].item())))
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    layer_path = OUTPUT / "flexiv_tabletop_scene.usda"
    stage.GetRootLayer().Export(str(layer_path))
    result = {"runtime": {"isaac_lab_release": os.environ.get("ISAACLAB_RELEASE", "3.0.0-beta2"),
                          "source_tree_version": Path(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"), "VERSION").read_text().strip(),
                          "core_extension_distribution_version": importlib.metadata.version("isaaclab"),
                          "physics": "Newton" if args.backend == "newton" else "PhysX", "device": args.device},
              "probe": args.probe, "steps": args.steps, "seconds": args.steps * DT,
              "peg_initial_pose": initial[0].tolist(), "peg_final_pose": final.tolist(),
              "peg_displacement_m": displacement, "trace": trace,
              "finite_state": True, "robot_joint_names": robot.joint_names,
              "scene_only": True, "trained_policy": False, "real_transfer_validated": False,
              "peg_fixed_attachment": False, "self_collisions": False,
              "peg_held_pass": bool(displacement < .015) if args.probe == "held" else None}
    settled_positions = np.asarray([x["peg_pose"][:3] for x in trace if x["step"] * DT >= 1.0])
    if len(settled_positions):
        drift = float(np.max(np.linalg.norm(settled_positions - settled_positions[-1], axis=1)))
        result["settled_peg_drift_m"] = drift
        if args.probe == "held":
            result["peg_held_pass"] = bool(drift < .001 and final[2] > .87)
    if args.run_contact_probes:
        result["contact_probes"] = []
        # Reset the arm to a parked pose away from the fixture for independent
        # gravity/contact checks. This is not an insertion controller.
        park = q.clone()
        park[:, robot.joint_names.index("joint1")] += .9
        robot.write_joint_position_to_sim_index(position=park)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(park))
        robot.reset()
        for name, offset in [("centered", 0.), ("offset", .017)]:
            probe_pose = root_pose.clone()
            probe_pose[0, :3] = torch.tensor([.15 + offset, 0., .875], device=args.device)
            peg.write_root_pose_to_sim_index(root_pose=probe_pose)
            peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=args.device))
            peg.reset()
            probe_steps = round(2. / DT)
            for _ in range(probe_steps):
                robot.set_joint_position_target_index(target=park)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(DT)
                peg.update(DT)
            actual = tensor(peg.data.root_pose_w).cpu().numpy()[0]
            rotation = Gf.Rotation(Gf.Quatd(float(actual[6]), Gf.Vec3d(*[float(x) for x in actual[3:6]])))
            tip = actual[:3] + np.asarray(rotation.TransformDir(Gf.Vec3d(0, 0, .075)))
            passed = bool(abs(tip[2] - .75) < .003 and np.linalg.norm(actual[:2] - [.15, 0]) < .001) if name == "centered" else bool(tip[2] > .783)
            probe_result = {"name": name, "offset_m": offset, "steps": probe_steps, "tip_position_m": tip.tolist(),
                            "final_pose": actual.tolist(), "passed": passed,
                            "expectation": "enters central bore and rests on tabletop" if name == "centered" else "stopped by fixture material"}
            result["contact_probes"].append(probe_result)
            log("Contact probe: " + json.dumps(probe_result))
    (OUTPUT / "runtime_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.collection_plan:
        if args.backend != "newton":
            raise ValueError("MVP1 collection screening requires Newton")
        from screen_free_motion import screen
        log("Starting Newton free-motion screening: " + args.collection_plan)
        screen(sim, robot, peg, args.collection_plan, OUTPUT)
    log("Artifacts in " + str(OUTPUT))
    if args.interactive:
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        robot.reset()
        peg.write_root_pose_to_sim_index(root_pose=root_pose)
        peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=args.device))
        peg.reset()
        while app.is_running():
            robot.set_joint_position_target_index(target=hold)
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(DT)
            peg.update(DT)


try:
    main()
except BaseException:
    # Kit's close can terminate the interpreter before an unwinding traceback
    # is printed. Persist failure evidence before shutting down the app.
    import traceback
    traceback.print_exc()
    (OUTPUT / "failure.txt").write_text(traceback.format_exc())
    raise
finally:
    app.close()
