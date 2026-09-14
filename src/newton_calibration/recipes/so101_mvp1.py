from __future__ import annotations

from dataclasses import dataclass, field

from newton_calibration.core.models import EnvironmentSpec, ParameterSpec


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
        ParameterSpec(
            "arm_damping_scale", 0.25, 20.0, 1.0, "scale", "isaaclab_explicit_pd", "arm overshoot and settling"
        ),
        ParameterSpec("arm_armature", 0.001, 0.08, 0.02, "kg*m^2", "newton", "reflected motor inertia"),
        ParameterSpec("arm_friction_nm", 0.0, 0.35, 0.02, "N*m", "newton", "arm holding and low-speed error"),
        ParameterSpec("arm_effort_scale", 0.005, 0.2, 0.05, "scale", "isaaclab_explicit_pd", "arm saturation response"),
        ParameterSpec(
            "gripper_stiffness_scale", 0.25, 5.0, 1.0, "scale", "isaaclab_explicit_pd", "gripper servo response"
        ),
        ParameterSpec(
            "gripper_damping_scale", 0.25, 20.0, 1.0, "scale", "isaaclab_explicit_pd", "gripper overshoot and settling"
        ),
        ParameterSpec("gripper_armature", 0.0001, 0.03, 0.005, "kg*m^2", "newton", "gripper reflected inertia"),
        ParameterSpec("gripper_friction_nm", 0.0, 0.25, 0.01, "N*m", "newton", "gripper hysteresis proxy"),
        ParameterSpec(
            "gripper_effort_scale", 0.001, 0.05, 0.01, "scale", "isaaclab_explicit_pd", "gripper saturation response"
        ),
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

    @property
    def required_parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)


@dataclass(frozen=True)
class ArticulationMVP1Recipe:
    """Free-space position-PD recipe instantiated from a robot profile."""

    name: str
    parameters: tuple[ParameterSpec, ...]
    required_parameter_names: tuple[str, ...]
    train_selectors: tuple[str, ...] = ()
    heldout_selectors: tuple[str, ...] = ()
    max_episode_duration_s: float = 12.0
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


def get_recipe(name: str, environment: EnvironmentSpec | None = None) -> SO101MVP1Recipe | ArticulationMVP1Recipe:
    if name == "so101_actuator_dynamics.v1":
        return SO101MVP1Recipe()
    if name not in {"articulation.position_pd.free_space.v1", "articulation.position_pd.free_space@2"}:
        raise KeyError(f"Unknown recipe: {name}")
    if environment is None:
        raise ValueError(f"Recipe {name!r} must be instantiated with a robot environment/profile")
    if not environment.joint_groups:
        return ArticulationMVP1Recipe(name=name, parameters=(), required_parameter_names=())

    parameters: list[ParameterSpec] = []
    required: list[str] = []
    scale_templates = {
        "stiffness_scale": (0.25, 8.0, 1.0, "servo response", "isaaclab_explicit_pd"),
        "damping_scale": (0.25, 20.0, 1.0, "overshoot and settling", "isaaclab_explicit_pd"),
        "effort_scale": (0.1, 2.0, 1.0, "effort saturation", "isaaclab_explicit_pd"),
    }
    absolute_templates = {
        "armature": ("kg*m^2", "reflected inertia", "newton"),
        "friction_nm": ("N*m", "low-speed and holding error", "newton"),
    }
    joint_order = list(environment.joint_order) or [
        joint for members in environment.joint_groups.values() for joint in members
    ]
    group_by_joint = {
        joint: group for group, members in environment.joint_groups.items() for joint in members
    }
    group_order = list(dict.fromkeys(group_by_joint[joint] for joint in joint_order if joint in group_by_joint))
    for group in environment.joint_groups:
        if group not in group_order:
            group_order.append(group)
    for group in group_order:
        for suffix, (lower, upper, initial, rationale, owner) in scale_templates.items():
            parameter_name = f"{group}_{suffix}"
            required.append(parameter_name)
            declared = environment.parameter_bounds.get(parameter_name)
            if declared is None and suffix != "effort_scale":
                declared = (lower, upper, initial)
            if declared is None:
                continue
            parameters.append(ParameterSpec(parameter_name, *declared, "scale", owner, f"{group} {rationale}"))
        for suffix, (unit, rationale, owner) in absolute_templates.items():
            parameter_name = f"{group}_{suffix}"
            required.append(parameter_name)
            declared = environment.parameter_bounds.get(parameter_name)
            if declared is not None:
                parameters.append(ParameterSpec(parameter_name, *declared, unit, owner, f"{group} {rationale}"))
    required.append("command_delay_s")
    delay = environment.parameter_bounds.get("command_delay_s", (0.0, 0.08, 0.02))
    parameters.append(
        ParameterSpec(
            "command_delay_s",
            *delay,
            "s",
            "toolkit",
            "command-to-motion latency",
        )
    )
    return ArticulationMVP1Recipe(
        name=name,
        parameters=tuple(parameters),
        required_parameter_names=tuple(required),
    )
