from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from newton_calibration.core.models import EnvironmentSpec

_USD_EXTENSIONS = {".usd", ".usda", ".usdc"}
_JOINT_DECLARATION = re.compile(r"\b(?:def|over)\s+Physics(?P<kind>Revolute|Prismatic)Joint\s+\"(?P<name>[^\"]+)\"")


@dataclass(frozen=True)
class UsdJoint:
    name: str
    prim_path: str
    kind: str
    driven: bool | None = None


@dataclass(frozen=True)
class UsdInspection:
    asset_path: str
    joints: tuple[UsdJoint, ...] = ()
    articulation_roots: tuple[str, ...] = ()
    inspection_backend: str = "unavailable"
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ready_for_profile(self) -> bool:
        return not self.blockers and bool(self.joints)

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_usd(path: str | Path) -> UsdInspection:
    """Inspect articulation joints without importing Isaac Lab.

    OpenUSD is authoritative when available.  A conservative USDA text parser
    exists so an agent can still build a mapping proposal in a lightweight
    client; the Newton runtime performs the final import/readback check.
    """

    asset = Path(path).expanduser().resolve()
    if not asset.is_file():
        return UsdInspection(str(asset), blockers=(f"USD asset does not exist: {asset}",))
    if asset.suffix.lower() not in _USD_EXTENSIONS:
        return UsdInspection(str(asset), blockers=(f"Unsupported asset extension: {asset.suffix}",))

    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return _inspect_usda_text(asset)

    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        return UsdInspection(str(asset), inspection_backend="openusd", blockers=("OpenUSD could not open asset",))
    joints: list[UsdJoint] = []
    roots: list[str] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        kind = None
        drive_token = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            kind, drive_token = "revolute", "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            kind, drive_token = "prismatic", "linear"
        if kind:
            joints.append(
                UsdJoint(
                    name=prim.GetName(),
                    prim_path=str(prim.GetPath()),
                    kind=kind,
                    driven=bool(UsdPhysics.DriveAPI.Get(prim, drive_token)),
                )
            )
    blockers: list[str] = []
    if not joints:
        blockers.append("USD contains no supported revolute or prismatic joints")
    if not roots:
        blockers.append("USD contains no PhysicsArticulationRootAPI")
    if len(roots) > 1:
        blockers.append("USD contains multiple articulation roots; select one explicitly")
    names = [joint.name for joint in joints]
    if len(names) != len(set(names)):
        blockers.append("USD contains duplicate joint leaf names; full prim-path mappings are not supported in MVP1")
    return UsdInspection(
        asset_path=str(asset),
        joints=tuple(joints),
        articulation_roots=tuple(roots),
        inspection_backend="openusd",
        blockers=tuple(blockers),
    )


def validate_articulation_asset(environment: EnvironmentSpec) -> UsdInspection:
    report = inspect_usd(environment.asset_path)
    blockers = list(report.blockers)
    warnings = list(report.warnings)
    groups = environment.joint_groups
    ordered = [joint for members in groups.values() for joint in members]
    if not groups:
        blockers.append("Robot profile has no actuator groups")
    elif any(not members for members in groups.values()):
        blockers.append("Robot profile contains an empty actuator group")
    if len(ordered) != len(set(ordered)):
        blockers.append("A logical joint occurs in more than one actuator group")
    missing_mappings = sorted(set(ordered) - set(environment.joint_map))
    if missing_mappings:
        blockers.append(f"Real-data-to-USD joint map is missing logical joints: {missing_mappings}")
    extra_mappings = sorted(set(environment.joint_map) - set(ordered))
    if extra_mappings:
        blockers.append(f"Real-data-to-USD joint map contains uncontrolled logical joints: {extra_mappings}")
    mapped_targets = [environment.joint_map[name] for name in ordered if name in environment.joint_map]
    if len(mapped_targets) != len(set(mapped_targets)):
        blockers.append("Real-data-to-USD joint map contains duplicate target DOFs")
    if not environment.profile_confirmed:
        blockers.append("Robot profile and joint mapping have not been explicitly confirmed")
    if not environment.controller_profile_confirmed:
        blockers.append("Controller profile baselines have not been explicitly confirmed")
    if not environment.controller_profile_source.strip():
        blockers.append("Controller profile is missing its source/provenance")
    controlled = set(ordered)
    for field_name, values in (
        ("base_stiffness_by_joint", environment.base_stiffness_by_joint),
        ("base_damping_by_joint", environment.base_damping_by_joint),
        ("base_effort_limit_by_joint", environment.base_effort_limit_by_joint),
    ):
        if set(values) != controlled:
            blockers.append(f"Controller profile {field_name} must explicitly cover every controlled joint")
    for group in groups:
        effort_bounds = f"{group}_effort_scale"
        selected = not environment.tuning_targets or effort_bounds in environment.tuning_targets
        if selected and effort_bounds not in environment.parameter_bounds:
            blockers.append(f"Robot-specific safe bounds are required for {effort_bounds}")
    portable, dependency_reason = _root_layer_is_self_contained(Path(environment.asset_path).expanduser().resolve())
    if not portable:
        blockers.append(
            "MVP1 packaging currently requires a self-contained root USD; " + dependency_reason
        )
    discovered = {joint.name for joint in report.joints}
    if discovered:
        missing_targets = sorted(set(mapped_targets) - discovered)
        if missing_targets:
            blockers.append(f"USD is missing mapped joints: {missing_targets}")
        prismatic_targets = sorted(
            joint.name for joint in report.joints if joint.name in mapped_targets and joint.kind == "prismatic"
        )
        if prismatic_targets:
            blockers.append(
                "The MVP1 position-PD recipe currently supports revolute coordinates only; "
                f"mapped prismatic joints are unsupported: {prismatic_targets}"
            )
    elif report.inspection_backend == "usda-text":
        warnings.append("Static text inspection is provisional; Newton import/readback is still required")
    return UsdInspection(
        asset_path=report.asset_path,
        joints=report.joints,
        articulation_roots=report.articulation_roots,
        inspection_backend=report.inspection_backend,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _root_layer_is_self_contained(path: Path) -> tuple[bool, str]:
    """Fail closed before fitting when MVP1 cannot package the USD closure."""

    if not path.is_file():
        return False, "the root layer is missing"
    if path.suffix.lower() == ".usda":
        try:
            references = re.findall(r"@([^@]+)@", path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return False, "the USDA root layer could not be decoded"
        if references:
            return False, f"external dependencies are not yet vendored: {sorted(set(references))}"
        return True, "root layer has no external asset references"
    try:
        from pxr import UsdUtils

        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - OpenUSD/Newton environment
        return False, f"dependency closure could not be proven: {type(exc).__name__}"
    external_layers = [layer for layer in layers if Path(layer.realPath).resolve() != path]
    if external_layers or assets or unresolved:
        return False, "external USD dependency closure is not yet vendored"
    return True, "OpenUSD dependency closure contains only the root layer"


def _inspect_usda_text(asset: Path) -> UsdInspection:
    if asset.suffix.lower() != ".usda":
        return UsdInspection(
            str(asset),
            inspection_backend="unavailable",
            blockers=("USD joint inventory is unavailable outside an OpenUSD/Newton environment",),
            warnings=("OpenUSD is unavailable; binary USD joint discovery requires the Newton container",),
        )
    try:
        payload = asset.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return UsdInspection(
            str(asset),
            inspection_backend="unavailable",
            blockers=("USD joint inventory is unavailable outside an OpenUSD/Newton environment",),
            warnings=("USD is not readable USDA text; inspect it in the Newton container",),
        )
    joints = tuple(
        UsdJoint(match.group("name"), f"/{match.group('name')}", match.group("kind").lower(), None)
        for match in _JOINT_DECLARATION.finditer(payload)
    )
    root_declarations = re.findall(r"(?<![A-Za-z])(?:Physics)?ArticulationRootAPI(?![A-Za-z])", payload)
    has_root = bool(root_declarations)
    blockers: list[str] = []
    if not joints:
        blockers.append("USD contains no supported revolute or prismatic joints")
    if not has_root:
        blockers.append("USD contains no PhysicsArticulationRootAPI")
    if len(root_declarations) > 1:
        blockers.append("USD contains multiple articulation roots; select one explicitly")
    names = [joint.name for joint in joints]
    if len(names) != len(set(names)):
        blockers.append("USD contains duplicate joint leaf names; full prim-path mappings are not supported in MVP1")
    return UsdInspection(
        str(asset),
        joints=joints,
        articulation_roots=("<declared>",) if has_root else (),
        inspection_backend="usda-text",
        blockers=tuple(blockers),
        warnings=("USDA text inspection is provisional; Newton import/readback remains authoritative",),
    )
