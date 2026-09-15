"""Build portable SI-unit USD assets and solve a pre-grasp Flexiv scene pose.

Source files are preserved. Physics values are declared engineering priors.
The fixture stays static, so an exact triangle mesh preserves every bore.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import trimesh
from scipy.optimize import least_squares, brentq
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def transform(position=(0, 0, 0), rotation=None):
    t = np.eye(4)
    t[:3, 3] = position
    if rotation is not None:
        t[:3, :3] = rotation
    return t


def quat_rotation(q):
    return Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()


def set_transform(prim, matrix):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))


def material(stage, path, color, metallic=0.0, roughness=0.4):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def mesh_prim(stage, path, mesh):
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    geom.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    geom.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    geom.CreateSubdivisionSchemeAttr("none")
    geom.CreateExtentAttr([Gf.Vec3f(*x) for x in mesh.bounds])
    return geom.GetPrim()


def make_part(source, name, density, color, dynamic):
    mesh = trimesh.load_mesh(ASSETS / "source" / source)
    assert mesh.is_watertight and mesh.volume > 0, source
    mesh.apply_scale(0.001)
    path = ASSETS / (name + ".usd")
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/" + name).GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("calibration:sourceUnits", "millimetres; inferred from supplied STL dimensions")
    root.SetCustomDataByKey("calibration:qualification", "local structural and runtime checks; not catalog certification")
    root.SetCustomDataByKey("calibration:materialBasis", "unmeasured density and friction priors")
    root.SetCustomDataByKey("calibration:sourceSha256", hashlib.sha256((ASSETS / "source" / source).read_bytes()).hexdigest())
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
    props = UsdPhysics.MassAPI.Apply(root)
    props.CreateMassAttr(float(mesh.volume * density))
    props.CreateDensityAttr(float(density))
    props.CreateCenterOfMassAttr(Gf.Vec3f(*mesh.center_mass))
    values, vectors = np.linalg.eigh(mesh.moment_inertia * density)
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] *= -1
    q = Rotation.from_matrix(vectors).as_quat()
    props.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
    props.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
    looks = material(stage, "/" + name + "/Looks/Metal", color, metallic=0.7, roughness=0.27)
    pm = UsdPhysics.MaterialAPI.Apply(looks.GetPrim())
    pm.CreateStaticFrictionAttr(0.6)
    pm.CreateDynamicFrictionAttr(0.45)
    pm.CreateRestitutionAttr(0.0)
    visual = mesh_prim(stage, "/" + name + "/Visual", mesh)
    UsdShade.MaterialBindingAPI.Apply(visual).Bind(looks)
    collision_mesh = mesh.convex_hull if dynamic else mesh
    collision = mesh_prim(stage, "/" + name + "/Collision", collision_mesh)
    UsdGeom.Imageable(collision).CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(collision).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collision).CreateApproximationAttr("convexHull" if dynamic else "none")
    UsdShade.MaterialBindingAPI.Apply(collision).Bind(looks, materialPurpose="physics")
    stage.GetRootLayer().Save()
    return {"source": source, "usd": path.name, "bounds_m": mesh.bounds.tolist(),
            "mass_kg_prior": float(mesh.volume * density), "density_kg_m3_prior": density,
            "watertight": True, "dynamic": dynamic,
            "collision": "convex hull" if dynamic else "exact static triangle mesh",
            "hull_volume_error_fraction": float(collision_mesh.volume / mesh.volume - 1),
            "positive_inertia": bool(np.all(values > 0))}


def robot_pose():
    stage = Usd.Stage.Open(str(ASSETS / "source/Rizon4s_with_Grav.usd"))
    root_path = str(stage.GetDefaultPrim().GetPath())
    cache = UsdGeom.XformCache()
    rest = {str(p.GetPath()): np.array(cache.GetLocalToWorldTransform(p)).T
            for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        a = transform(joint.GetLocalPos0Attr().Get(), quat_rotation(joint.GetLocalRot0Attr().Get()))
        b = transform(joint.GetLocalPos1Attr().Get(), quat_rotation(joint.GetLocalRot1Attr().Get()))
        axis = prim.GetAttribute("physics:axis").Get() or "Z"
        joints.append((prim, str(b0[0]), str(b1[0]), a, b, "XYZ".index(axis)))

    def fk(angles):
        poses = {root_path + "/base_link": np.eye(4)}
        pending = list(joints)
        while pending:
            before = len(pending)
            for item in list(pending):
                prim, parent, child, a, b, axis = item
                if parent not in poses:
                    continue
                rot = np.zeros(3)
                rot[axis] = angles.get(prim.GetName(), 0)
                poses[child] = poses[parent] @ a @ transform(rotation=Rotation.from_rotvec(rot).as_matrix()) @ np.linalg.inv(b)
                pending.remove(item)
            assert len(pending) < before, "disconnected robot joints"
        return poses

    def gripper_angles(q):
        return {"finger_joint": q, "right_outer_knuckle_joint": q,
                "left_inner_knuckle_joint": q, "right_inner_knuckle_joint": q,
                "left_outer_finger_joint": -q, "right_outer_finger_joint": -q}

    # Exact vendor finger collision vertices, represented in each link frame.
    tips = []
    for side in ("left", "right"):
        body = root_path + "/Grav_gripper/" + side + "_finger_tip"
        p = stage.GetPrimAtPath(body + "/collisions")
        m = np.linalg.inv(rest[body]) @ np.array(cache.GetLocalToWorldTransform(p)).T
        v = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=float)
        tips.append((body, (m @ np.column_stack([v, np.ones(len(v))]).T).T[:, :3]))

    def finger_geometry(q):
        poses = fk(gripper_angles(q))
        base = poses[root_path + "/Grav_gripper/gripper_base"]
        points = []
        for body, v in tips:
            m = np.linalg.inv(base) @ poses[body]
            points.append((m @ np.column_stack([v, np.ones(len(v))]).T).T[:, :3])
        left, right = points
        # At zero, left finger is on negative local y; inner face is max y.
        gap = right[:, 1].min() - left[:, 1].max()
        xyz = np.array([(left[:, 0].mean() + right[:, 0].mean()) / 2,
                        (right[:, 1].min() + left[:, 1].max()) / 2,
                        min(left[:, 2].min(), right[:, 2].min())])
        return gap, xyz

    g0, g1 = finger_geometry(0)[0], finger_geometry(0.5)[0]
    print("FINGER gaps", g0, g1, flush=True)
    grip = brentq(lambda q: finger_geometry(q)[0] - 0.025, -0.15, 0.7)
    gap, finger_near = finger_geometry(grip)
    # Peg starts 3mm behind the proximal edge of the fingertip, so ~40mm protrudes.
    peg_local = transform((finger_near[0], finger_near[1], finger_near[2] - 0.003))
    base_world = transform((-0.45, 0.0, 0.765))
    target_peg = transform((0.15, 0.0, 0.925), Rotation.from_euler("x", np.pi).as_matrix())
    names = ["joint" + str(i) for i in range(1, 8)]
    def objective(q):
        pose = base_world @ fk(dict(zip(names, q)))[root_path + "/Grav_gripper/gripper_base"] @ peg_local
        return np.concatenate(((pose[:3, 3] - target_peg[:3, 3]) * 8,
                               Rotation.from_matrix(target_peg[:3, :3].T @ pose[:3, :3]).as_rotvec()))
    lower, upper = [], []
    for name in names:
        j = stage.GetPrimAtPath(root_path + "/joints/" + name)
        lower.append(np.deg2rad(j.GetAttribute("physics:lowerLimit").Get()) + .02)
        upper.append(np.deg2rad(j.GetAttribute("physics:upperLimit").Get()) - .02)
    rng = np.random.default_rng(7)
    candidates = []
    for seed in [[0, -.7, 0, 1.4, 0, .9, 0]] + [rng.uniform(np.maximum(lower, -1.5), np.minimum(upper, 1.5)) for _ in range(18)]:
        solution = least_squares(objective, np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=400)
        if np.linalg.norm(objective(solution.x)) < 1e-5:
            poses = fk(dict(zip(names, solution.x)))
            elbow = poses[root_path + "/link4"][:3, 3]
            # Prefer elbow-up poses; keep major arm links above the table.
            min_z = min(poses[root_path + "/link" + str(i)][2, 3] for i in range(2, 8))
            score = np.linalg.norm(solution.x) - 4 * elbow[2] + (100 if min_z < .08 else 0)
            candidates.append((score, solution.x))
    assert candidates, "IK failed"
    q = sorted(candidates, key=lambda v: v[0])[0][1]
    angles = {**dict(zip(names, q)), **gripper_angles(grip)}
    poses = fk(angles)
    # Build a portable robot copy; replace runtime MDL references with PreviewSurface.
    prepared = Usd.Stage.CreateNew(str(ASSETS / "Flexiv_Rizon4s_Grav.usd"))
    prepared.GetRootLayer().TransferContent(stage.Flatten())
    colors = []
    for prim in prepared.Traverse():
        if prim.IsA(UsdShade.Material):
            shaders = [p for p in prim.GetChildren() if p.IsA(UsdShade.Shader)]
            color = next((p.GetAttribute("inputs:diffuse_color_constant").Get() for p in shaders
                          if p.GetAttribute("inputs:diffuse_color_constant").Get() is not None), (.6, .6, .6))
            colors.append((str(prim.GetPath()), tuple(color)))
    for path, color in colors:
        prepared.RemovePrim(path)
        material(prepared, path, color, metallic=.1, roughness=.35)
    for path, pose in poses.items():
        prim = prepared.GetPrimAtPath(path)
        parent_world = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path).GetParent())).T
        set_transform(prim, np.linalg.inv(parent_world) @ pose)
        # Vendor fingers include zero-mass and 0.1g placeholder links.
        mass = UsdPhysics.MassAPI(prim)
        if "Grav_gripper" in path and (mass.GetMassAttr().Get() or 0) < .001:
            mass.CreateMassAttr(.02)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5))
            prim.SetCustomDataByKey("calibration:massBasis", "20g scene stability prior; unmeasured")
    for prim, _, _, _, _, _ in joints:
        p = prepared.GetPrimAtPath(prim.GetPath())
        if p.IsA(UsdPhysics.RevoluteJoint):
            value = angles.get(p.GetName(), 0.0)
            p.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(float(np.rad2deg(value)))
            p.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
            # Explicit drives on all six gripper joints; no undeclared mimic runtime dependency.
            drive = UsdPhysics.DriveAPI.Apply(p, "angular")
            drive.CreateTargetPositionAttr(float(np.rad2deg(value)))
            arm = p.GetName() in names
            drive.CreateStiffnessAttr(4000.0 if arm else 80.0)
            drive.CreateDampingAttr(180.0 if arm else 2.0)
            drive.CreateMaxForceAttr(123.0 if p.GetName() in names[:2] else 64.0 if arm else 5.0)
    prepared.GetDefaultPrim().SetCustomDataByKey("calibration:controller", "scene position hold; vendor real controller not reproduced")
    prepared.GetRootLayer().Save()
    return {"base_world": base_world.tolist(), "joint_positions_rad": {k: float(v) for k, v in angles.items()},
            "peg_in_gripper": peg_local.tolist(), "peg_world": target_peg.tolist(),
            "gripper_body": "Grav_gripper/gripper_base", "gripper_gap_m": float(gap),
            "ik_residual": float(np.linalg.norm(objective(q))),
            "fixture_position_m": [.15, 0, .75], "table_top_m": .75,
            "robot_body_rest": {k.removeprefix(root_path + "/"): v.tolist() for k, v in poses.items()}}


def main():
    parts = [make_part("Peg_25.stl", "Peg_25", 7850, (.55, .60, .66), True),
             make_part("hole_block.stl", "HoleBlock", 2700, (.23, .29, .34), False)]
    pose = robot_pose()
    result = {"parts": parts, "scene": pose,
              "central_bore_diameter_m": .0255, "peg_diameter_m": .025,
              "radial_clearance_m": .00025,
              "qualification": "simulation assets; no measured calibration or formal SimReady certification"}
    (ASSETS / "asset_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"parts": parts, "joints": pose["joint_positions_rad"], "ik_residual": pose["ik_residual"]}, indent=2))


if __name__ == "__main__":
    main()
