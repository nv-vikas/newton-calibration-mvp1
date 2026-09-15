from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from newton_calibration.adapters.runtime import create_runtime
from newton_calibration.core.models import EnvironmentSpec


@dataclass(frozen=True)
class ArticulationEnvCfg:
    """Robot-agnostic Isaac Lab/Newton calibration surface.

    ``joint_groups`` contains logical coordinates from the real evidence.
    ``joint_map`` binds each of those coordinates to a USD/runtime DOF.  Both
    are explicit because a USD cannot reveal the real robot driver's names,
    units, sign convention, or zero offsets.
    """

    usd_path: str
    joint_groups: dict[str, tuple[str, ...]]
    joint_map: dict[str, str]
    joint_order: tuple[str, ...] = ()
    robot_id: str = "articulation"
    profile_confirmed: bool = False
    controller_profile_confirmed: bool = False
    controller_profile_source: str = ""
    runtime: str = "isaaclab_newton"
    device: str = "cuda:0"
    dt: float = 1.0 / 120.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    base_stiffness: float = 1.0
    base_damping: float = 0.1
    base_effort_limit: float = 1.0
    base_armature: float = 0.0
    base_stiffness_by_joint: dict[str, float] = field(default_factory=dict)
    base_damping_by_joint: dict[str, float] = field(default_factory=dict)
    base_effort_limit_by_joint: dict[str, float] = field(default_factory=dict)
    base_armature_by_joint: dict[str, float] = field(default_factory=dict)
    analytic_inertia_by_joint: dict[str, float] = field(default_factory=dict)
    parameter_bounds: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    tuning_targets: tuple[str, ...] = ()
    num_substeps: int = 1
    solver_iterations: int = 100
    solver_tolerance: float = 1e-6
    residual_model_path: str | None = None
    residual_model_sha256: str | None = None
    calibration_run_id: str | None = None
    calibration_manifest_path: str | None = None
    calibration_manifest_sha256: str | None = None
    calibration_parameters: dict[str, float] = field(default_factory=dict)

    @property
    def logical_joint_order(self) -> tuple[str, ...]:
        return self.joint_order or tuple(joint for members in self.joint_groups.values() for joint in members)

    @classmethod
    def from_calibration(
        cls,
        package: str | Path,
        *,
        device: str | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> ArticulationEnvCfg:
        from .package_loader import VerifiedArticulationPackage

        verified = VerifiedArticulationPackage.open(
            package,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        return verified.to_env_cfg(device=device)

    def describe(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            adapter=self.runtime,
            asset_path=self.usd_path,
            robot_id=self.robot_id,
            profile_schema="articulation-profile/v1",
            device=self.device,
            dt=self.dt,
            gravity=self.gravity,
            joint_map=dict(self.joint_map),
            joint_groups={name: tuple(members) for name, members in self.joint_groups.items()},
            joint_order=self.logical_joint_order,
            profile_confirmed=self.profile_confirmed,
            controller_profile_confirmed=self.controller_profile_confirmed,
            controller_profile_source=self.controller_profile_source,
            base_stiffness=self.base_stiffness,
            base_damping=self.base_damping,
            base_effort_limit=self.base_effort_limit,
            base_armature=self.base_armature,
            base_stiffness_by_joint=dict(self.base_stiffness_by_joint),
            base_damping_by_joint=dict(self.base_damping_by_joint),
            base_effort_limit_by_joint=dict(self.base_effort_limit_by_joint),
            base_armature_by_joint=dict(self.base_armature_by_joint),
            analytic_inertia_by_joint=dict(self.analytic_inertia_by_joint),
            parameter_bounds={name: tuple(bounds) for name, bounds in self.parameter_bounds.items()},
            tuning_targets=tuple(self.tuning_targets),
            num_substeps=self.num_substeps,
            solver_iterations=self.solver_iterations,
            solver_tolerance=self.solver_tolerance,
            residual_model_path=self.residual_model_path,
            residual_model_sha256=self.residual_model_sha256,
            calibration_run_id=self.calibration_run_id,
            calibration_manifest_path=self.calibration_manifest_path,
            calibration_manifest_sha256=self.calibration_manifest_sha256,
            calibration_parameters=dict(self.calibration_parameters),
        )

    def calibrated_candidate(self) -> dict[str, float]:
        if not self.calibration_parameters:
            raise RuntimeError("ArticulationEnvCfg was not created from a calibration package")
        return dict(self.calibration_parameters)


@dataclass(frozen=True)
class SO101EnvCfg:
    """Serializable Isaac Lab replay-environment configuration consumed by the toolkit."""

    usd_path: str
    runtime: str = "isaaclab_newton"
    device: str = "cuda:0"
    dt: float = 1.0 / 120.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    joint_map: dict[str, str] = field(
        default_factory=lambda: {
            "rotation": "shoulder_pan",
            "pitch": "shoulder_lift",
            "elbow": "elbow_flex",
            "wrist_pitch": "wrist_flex",
            "wrist_roll": "wrist_roll",
            "jaw": "gripper",
        }
    )
    base_stiffness: float = 1.7453293
    base_damping: float = 0.017453292
    base_effort_limit: float = 10.0
    base_armature: float = 0.0
    num_substeps: int = 1
    solver_iterations: int = 100
    solver_tolerance: float = 1e-6
    residual_model_path: str | None = None
    residual_model_sha256: str | None = None
    calibration_run_id: str | None = None
    calibration_manifest_path: str | None = None
    calibration_manifest_sha256: str | None = None
    calibration_parameters: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_calibration(
        cls,
        package: str | Path,
        *,
        device: str | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> SO101EnvCfg:
        """Load a validated package without importing Isaac Lab at load time."""

        from .package_loader import VerifiedSO101Package

        verified = VerifiedSO101Package.open(
            package,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        return verified.to_env_cfg(device=device)

    def describe(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            adapter=self.runtime,
            asset_path=self.usd_path,
            robot_id="so101",
            profile_schema="legacy-so101/v1",
            device=self.device,
            dt=self.dt,
            gravity=self.gravity,
            joint_map=dict(self.joint_map),
            base_stiffness=self.base_stiffness,
            base_damping=self.base_damping,
            base_effort_limit=self.base_effort_limit,
            base_armature=self.base_armature,
            num_substeps=self.num_substeps,
            solver_iterations=self.solver_iterations,
            solver_tolerance=self.solver_tolerance,
            residual_model_path=self.residual_model_path,
            residual_model_sha256=self.residual_model_sha256,
            calibration_run_id=self.calibration_run_id,
            calibration_manifest_path=self.calibration_manifest_path,
            calibration_manifest_sha256=self.calibration_manifest_sha256,
            calibration_parameters=dict(self.calibration_parameters),
        )

    def calibrated_candidate(self) -> dict[str, float]:
        """Return all calibrated values or fail rather than run a partial profile."""

        if not self.calibration_parameters:
            raise RuntimeError("SO101EnvCfg was not created from a calibration package")
        return dict(self.calibration_parameters)


class IsaacLabCalibrationAdapter:
    """The narrow environment boundary used by calibration core."""

    def __init__(self, cfg: SO101EnvCfg | ArticulationEnvCfg):
        self.cfg = cfg
        self.runtime = create_runtime(cfg.describe())

    @classmethod
    def from_calibration(
        cls,
        package: str | Path,
        *,
        device: str | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> IsaacLabCalibrationAdapter:
        """Load either a legacy SO-101 or generic articulation package.

        Package schema dispatch belongs at this shared entry point so callers
        do not need robot-specific branching before creating the Isaac Lab
        calibration surface.
        """

        from .package_loader import VerifiedCalibrationPackage

        verified = VerifiedCalibrationPackage.open(
            package,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        return cls(verified.to_env_cfg(device=device))

    def describe(self) -> EnvironmentSpec:
        return self.cfg.describe()

    def evaluate(self, candidate, episodes, objective_weights, *, phase: str = "unscoped", **context):
        return self.runtime.evaluate(candidate, episodes, objective_weights, phase=phase, **context)

    def evaluate_calibrated(self, episodes, objective_weights, *, phase: str = "package-replay", **context):
        """Replay with the complete package profile; useful for package smoke tests."""

        return self.runtime.evaluate(
            self.cfg.calibrated_candidate(), episodes, objective_weights, phase=phase, **context
        )

    def close(self) -> None:
        self.runtime.close()
