from __future__ import annotations

from pathlib import Path

from newton_calibration.core.models import EnvironmentSpec

_SUPPORTED_PARAMETERS = {
    "arm_stiffness_scale",
    "arm_damping_scale",
    "arm_armature",
    "arm_friction_nm",
    "arm_effort_scale",
    "gripper_stiffness_scale",
    "gripper_damping_scale",
    "gripper_armature",
    "gripper_friction_nm",
    "gripper_effort_scale",
    "command_delay_s",
}


def supported_parameter_names(environment: EnvironmentSpec) -> set[str]:
    """Return the parameter writers exposed by the selected thin adapter."""
    if environment.adapter not in {"isaaclab_newton", "analytic"}:
        return set()
    if environment.profile_schema == "articulation-profile/v1":
        names = {"command_delay_s"}
        for group in environment.joint_groups:
            names.update(
                {
                    f"{group}_stiffness_scale",
                    f"{group}_damping_scale",
                    f"{group}_armature",
                    f"{group}_friction_nm",
                    f"{group}_effort_scale",
                }
            )
        return names
    return set(_SUPPORTED_PARAMETERS)


def validate_so101_asset(environment: EnvironmentSpec) -> list[str]:
    path = Path(environment.asset_path).expanduser()
    errors: list[str] = []
    if not path.exists():
        errors.append(f"USD asset does not exist: {path}")
    elif path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        errors.append(f"Unsupported asset extension: {path.suffix}")
    missing = [
        joint
        for joint in ("rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw")
        if joint not in environment.joint_map
    ]
    if missing:
        errors.append(f"Dataset-to-USD joint map is missing: {missing}")
    if len(set(environment.joint_map.values())) != len(environment.joint_map):
        errors.append("Dataset-to-USD joint map contains duplicate target joints")
    if errors or environment.adapter != "isaaclab_newton":
        return errors
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        # Lightweight clients can still analyze evidence. The Newton container
        # performs the authoritative structural check before a real fit.
        return errors
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        errors.append(f"OpenUSD could not open asset: {path}")
        return errors
    revolute_joints = {prim.GetName(): prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)}
    missing_joints = sorted(set(environment.joint_map.values()) - set(revolute_joints))
    if missing_joints:
        errors.append(f"USD is missing mapped revolute joints: {missing_joints}")
    roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not roots:
        errors.append("USD has no PhysicsArticulationRootAPI")
    for joint_name in set(environment.joint_map.values()) & set(revolute_joints):
        drive = UsdPhysics.DriveAPI.Get(revolute_joints[joint_name], "angular")
        if not drive:
            errors.append(f"USD joint has no angular drive: {joint_name}")
    return errors
