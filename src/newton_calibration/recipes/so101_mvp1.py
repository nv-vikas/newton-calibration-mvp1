from __future__ import annotations

from dataclasses import dataclass, field

from newton_calibration.core.models import ParameterSpec


@dataclass(frozen=True)
class SO101MVP1Recipe:
    name: str = "so101_actuator_dynamics.v1"
    train_selectors: tuple[str, ...] = (
        "train-step-response",
        "train-chirp-sweep",
        "train-static-holding",
        "train-gripper-cycles",
    )
    heldout_selectors: tuple[str, ...] = (
        "heldout-frequency-sweep",
        "heldout-friction-gravity",
        "heldout-hold-under-gravity",
        "heldout-backlash-detection",
    )
    max_episode_duration_s: float = 12.0
    parameters: tuple[ParameterSpec, ...] = (
        ParameterSpec("arm_stiffness_scale", 0.25, 8.0, 1.0, "scale", "isaaclab_explicit_pd", "arm servo response"),
        ParameterSpec("arm_damping_scale", 0.25, 20.0, 1.0, "scale", "isaaclab_explicit_pd", "arm overshoot and settling"),
        ParameterSpec("arm_armature", 0.001, 0.08, 0.02, "kg*m^2", "newton", "reflected motor inertia"),
        ParameterSpec("arm_friction_nm", 0.0, 0.35, 0.02, "N*m", "newton", "arm holding and low-speed error"),
        ParameterSpec("arm_effort_scale", 0.005, 0.2, 0.05, "scale", "isaaclab_explicit_pd", "arm saturation response"),
        ParameterSpec("gripper_stiffness_scale", 0.25, 5.0, 1.0, "scale", "isaaclab_explicit_pd", "gripper servo response"),
        ParameterSpec("gripper_damping_scale", 0.25, 20.0, 1.0, "scale", "isaaclab_explicit_pd", "gripper overshoot and settling"),
        ParameterSpec("gripper_armature", 0.0001, 0.03, 0.005, "kg*m^2", "newton", "gripper reflected inertia"),
        ParameterSpec("gripper_friction_nm", 0.0, 0.25, 0.01, "N*m", "newton", "gripper hysteresis proxy"),
        ParameterSpec("gripper_effort_scale", 0.001, 0.05, 0.01, "scale", "isaaclab_explicit_pd", "gripper saturation response"),
        ParameterSpec("command_delay_s", 0.0, 0.08, 0.02, "s", "toolkit", "command-to-motion latency"),
    )
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {
            "position_nrmse": 0.60,
            "velocity_nrmse": 0.15,
            "phase_error_s": 0.15,
            "hold_nrmse": 0.10,
        }
    )
    optimizer: dict[str, float | int] = field(
        default_factory=lambda: {"name": "diagonal-cma-es", "population": 12, "generations": 12, "seed": 7}
    )
    validation_gates: dict[str, float] = field(
        default_factory=lambda: {"minimum_improvement_pct": 30.0, "maximum_episode_regression_pct": 10.0}
    )


def get_recipe(name: str) -> SO101MVP1Recipe:
    if name != "so101_actuator_dynamics.v1":
        raise KeyError(f"Unknown recipe: {name}")
    return SO101MVP1Recipe()
