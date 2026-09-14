from __future__ import annotations

from dataclasses import dataclass, field

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
        )


class IsaacLabCalibrationAdapter:
    """The narrow environment boundary used by calibration core."""

    def __init__(self, cfg: SO101EnvCfg):
        self.cfg = cfg
        self.runtime = create_runtime(cfg.describe())

    def describe(self) -> EnvironmentSpec:
        return self.cfg.describe()

    def evaluate(self, candidate, episodes, objective_weights):
        return self.runtime.evaluate(candidate, episodes, objective_weights)

    def close(self) -> None:
        self.runtime.close()
