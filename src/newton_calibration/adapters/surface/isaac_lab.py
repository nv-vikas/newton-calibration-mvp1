from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from newton_calibration.adapters.runtime import create_runtime
from newton_calibration.core.models import EnvironmentSpec


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

    def __init__(self, cfg: SO101EnvCfg):
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
        return cls(
            SO101EnvCfg.from_calibration(
                package,
                device=device,
                expected_manifest_sha256=expected_manifest_sha256,
            )
        )

    def describe(self) -> EnvironmentSpec:
        return self.cfg.describe()

    def evaluate(self, candidate, episodes, objective_weights):
        return self.runtime.evaluate(candidate, episodes, objective_weights)

    def evaluate_calibrated(self, episodes, objective_weights):
        """Replay with the complete package profile; useful for package smoke tests."""

        return self.runtime.evaluate(self.cfg.calibrated_candidate(), episodes, objective_weights)

    def close(self) -> None:
        self.runtime.close()
