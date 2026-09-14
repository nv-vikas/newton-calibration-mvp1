from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from newton_calibration.actuators import load_residual
from newton_calibration.core.attestation import episode_inputs, evaluation_result_fingerprint
from newton_calibration.core.io import sha256_file
from newton_calibration.core.models import EnvironmentSpec
from newton_calibration.validation.metrics import compare_trajectories

_SO101_JOINT_ORDER = ("rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw")
_SO101_GROUPS = {
    "arm": _SO101_JOINT_ORDER[:5],
    "gripper": _SO101_JOINT_ORDER[5:],
}
_SO101_ANALYTIC_STIFFNESS = (35.0, 35.0, 35.0, 35.0, 35.0, 28.0)
_SO101_ANALYTIC_DAMPING = (2.0, 2.0, 2.0, 2.0, 2.0, 1.5)
_SO101_ANALYTIC_INERTIA = (1.8, 1.6, 1.2, 0.7, 0.5, 0.35)
_GROUP_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class _JointLayout:
    """Validated evidence-to-runtime joint ordering shared by both backends."""

    groups: tuple[tuple[str, tuple[str, ...]], ...]
    logical_names: tuple[str, ...]
    runtime_names: tuple[str, ...]
    group_by_joint: tuple[str, ...]
    legacy_so101: bool


def _resolve_joint_layout(environment: EnvironmentSpec) -> _JointLayout:
    """Resolve configured groups without relying on robot-specific joint counts.

    ``joint_groups`` contains logical evidence coordinates. Flattening its
    insertion order defines the vector order used by evidence, residuals and
    both runtime adapters. Empty groups retain the original SO-101 contract.
    """

    configured_groups = getattr(environment, "joint_groups", None) or {}
    profile_schema = getattr(environment, "profile_schema", "legacy-so101/v1")
    legacy_so101 = profile_schema == "legacy-so101/v1"
    if profile_schema not in {"legacy-so101/v1", "articulation-profile/v1"}:
        raise ValueError(f"Unsupported robot profile schema: {profile_schema!r}")
    if not legacy_so101 and not configured_groups:
        raise ValueError("Generic articulation profile must declare at least one joint group")
    groups = _SO101_GROUPS if legacy_so101 else configured_groups
    normalized: list[tuple[str, tuple[str, ...]]] = []
    configured_members: list[str] = []
    joint_to_group: dict[str, str] = {}
    for raw_group_name, raw_members in groups.items():
        group_name = str(raw_group_name)
        if not _GROUP_NAME.fullmatch(group_name):
            raise ValueError(f"Joint group name {group_name!r} is not a valid calibration parameter prefix")
        if isinstance(raw_members, (str, bytes)):
            raise TypeError(
                f"Joint group {group_name!r} members must be a sequence of logical joint names, not a string"
            )
        members = tuple(str(member) for member in raw_members)
        if not members:
            raise ValueError(f"Joint group {group_name!r} must contain at least one logical joint")
        normalized.append((group_name, members))
        configured_members.extend(members)
        for member in members:
            joint_to_group[member] = group_name

    duplicates = sorted({name for name in configured_members if configured_members.count(name) > 1})
    if duplicates:
        raise ValueError(f"Logical joints may belong to only one joint group: {duplicates}")
    declared_order = tuple(getattr(environment, "joint_order", ()) or ())
    logical_names = list(declared_order) if declared_order else configured_members
    if declared_order and (
        len(declared_order) != len(set(declared_order)) or set(declared_order) != set(configured_members)
    ):
        raise ValueError("joint_order must contain every grouped logical joint exactly once")
    group_by_joint = [joint_to_group[name] for name in logical_names]
    missing = [name for name in logical_names if name not in environment.joint_map]
    if missing:
        raise ValueError(f"Joint groups reference logical joints absent from joint_map: {missing}")
    runtime_names = tuple(environment.joint_map[name] for name in logical_names)
    duplicate_runtime_names = sorted({name for name in runtime_names if runtime_names.count(name) > 1})
    if duplicate_runtime_names:
        raise ValueError(f"Each calibration coordinate must map to a distinct runtime joint: {duplicate_runtime_names}")
    return _JointLayout(
        groups=tuple(normalized),
        logical_names=tuple(logical_names),
        runtime_names=runtime_names,
        group_by_joint=tuple(group_by_joint),
        legacy_so101=legacy_so101,
    )


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not np.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}; found {value!r}")
    return number


def _base_vector(
    environment: EnvironmentSpec,
    layout: _JointLayout,
    *,
    map_field: str,
    scalar_field: str,
    positive: bool = False,
    legacy_values: tuple[float, ...] | None = None,
) -> np.ndarray:
    configured = dict(getattr(environment, map_field, None) or {})
    unknown = sorted(set(configured) - set(layout.logical_names))
    if unknown:
        raise ValueError(f"{map_field} contains joints outside joint_groups: {unknown}")
    if layout.legacy_so101 and not configured and legacy_values is not None:
        values = legacy_values
    else:
        scalar = getattr(environment, scalar_field)
        values = tuple(configured.get(name, scalar) for name in layout.logical_names)
    return np.asarray(
        [
            _finite_number(value, f"{map_field}[{name!r}]", positive=positive)
            for name, value in zip(layout.logical_names, values)
        ],
        dtype=np.float64,
    )


def _analytic_inertia_vector(environment: EnvironmentSpec, layout: _JointLayout) -> np.ndarray:
    configured = dict(getattr(environment, "analytic_inertia_by_joint", None) or {})
    unknown = sorted(set(configured) - set(layout.logical_names))
    if unknown:
        raise ValueError(f"analytic_inertia_by_joint contains joints outside joint_groups: {unknown}")
    if layout.legacy_so101 and not configured:
        values = _SO101_ANALYTIC_INERTIA
    else:
        missing = [name for name in layout.logical_names if name not in configured]
        if missing:
            raise ValueError(
                "The analytic reference runtime requires inertia for every configured joint; "
                f"missing analytic_inertia_by_joint values for {missing}"
            )
        values = tuple(configured[name] for name in layout.logical_names)
    return np.asarray(
        [
            _finite_number(value, f"analytic_inertia_by_joint[{name!r}]", positive=True)
            for name, value in zip(layout.logical_names, values)
        ],
        dtype=np.float64,
    )


def _group_candidate_vector(
    candidate: dict[str, float],
    layout: _JointLayout,
    suffix: str,
    *,
    default: float | np.ndarray,
    positive: bool = False,
) -> np.ndarray:
    defaults = np.broadcast_to(np.asarray(default, dtype=np.float64), (len(layout.logical_names),))
    values: list[float] = []
    for index, (logical_name, group_name) in enumerate(zip(layout.logical_names, layout.group_by_joint)):
        key = f"{group_name}_{suffix}"
        value = candidate.get(key, float(defaults[index]))
        values.append(_finite_number(value, key, positive=positive))
    return np.asarray(values, dtype=np.float64)


def _joint_properties(
    environment: EnvironmentSpec,
    layout: _JointLayout,
    candidate: dict[str, float],
    *,
    analytic: bool,
) -> dict[str, np.ndarray]:
    stiffness = _base_vector(
        environment,
        layout,
        map_field="base_stiffness_by_joint",
        scalar_field="base_stiffness",
        legacy_values=_SO101_ANALYTIC_STIFFNESS if analytic else None,
    )
    damping = _base_vector(
        environment,
        layout,
        map_field="base_damping_by_joint",
        scalar_field="base_damping",
        legacy_values=_SO101_ANALYTIC_DAMPING if analytic else None,
    )
    effort = _base_vector(
        environment,
        layout,
        map_field="base_effort_limit_by_joint",
        scalar_field="base_effort_limit",
        positive=True,
    )
    base_armature = _base_vector(
        environment,
        layout,
        map_field="base_armature_by_joint",
        scalar_field="base_armature",
    )
    return {
        "stiffness": stiffness * _group_candidate_vector(candidate, layout, "stiffness_scale", default=1.0),
        "damping": damping * _group_candidate_vector(candidate, layout, "damping_scale", default=1.0),
        "effort": effort * _group_candidate_vector(candidate, layout, "effort_scale", default=1.0, positive=True),
        "friction": _group_candidate_vector(candidate, layout, "friction_nm", default=0.0),
        "armature": _group_candidate_vector(candidate, layout, "armature", default=base_armature),
    }


def _validate_episode_width(episode: object, expected_joint_count: int) -> None:
    for field_name in ("command_q", "actual_q", "actual_dq"):
        values = np.asarray(getattr(episode, field_name))
        if values.ndim != 2 or values.shape[1] != expected_joint_count:
            raise ValueError(
                f"Episode {getattr(episode, 'name', '<unnamed>')!r} {field_name} must have "
                f"shape [samples, {expected_joint_count}]; found {values.shape}"
            )


class AnalyticPDReplayRuntime:
    """CPU reference backend for contract tests; it is not the product physics backend."""

    def __init__(self, environment: EnvironmentSpec):
        self.environment = environment
        self.joint_layout = _resolve_joint_layout(environment)
        # Resolve immutable physical defaults at construction so malformed
        # profiles fail before an optimizer spends evaluations on them.
        self.base_inertia = _analytic_inertia_vector(environment, self.joint_layout)
        _joint_properties(environment, self.joint_layout, {}, analytic=True)
        residual_path = environment.residual_model_path
        residual_before = sha256_file(residual_path) if residual_path else None
        self.residual = load_residual(residual_path, list(self.joint_layout.logical_names))
        residual_after = sha256_file(residual_path) if residual_path else None
        if residual_before != residual_after:
            raise RuntimeError("Actuator residual changed while the analytic runtime was loading it")
        if environment.residual_model_sha256 not in (None, residual_after):
            raise RuntimeError("Actuator residual does not match the locked environment fingerprint")
        self._loaded_residual_sha256 = residual_after

    def describe(self) -> EnvironmentSpec:
        return self.environment

    def attestation(self) -> dict[str, object]:
        if not hasattr(self, "_last_evaluation"):
            raise RuntimeError("Analytic runtime cannot attest before a complete evidence evaluation succeeds")
        return {
            "schema": "newton.calibration.runtime-attestation/v2",
            "backend": "analytic",
            "authoritative": False,
            "robot_id": self.environment.robot_id,
            "logical_joints": list(self.joint_layout.logical_names),
            "runtime_joints": list(self.joint_layout.runtime_names),
            "runtime_dt_s": self.environment.dt,
            "gravity": list(self.environment.gravity),
            "num_substeps": self.environment.num_substeps,
            "solver_iterations": self.environment.solver_iterations,
            "solver_tolerance": self.environment.solver_tolerance,
            "selected_joint_scoped": True,
            "readback_parameters": [],
            "residual_sha256": self._loaded_residual_sha256,
            **self._last_evaluation,
        }

    def evaluate(
        self,
        candidate,
        episodes: Sequence,
        objective_weights,
        *,
        phase: str = "unscoped",
        run_id: str = "",
        plan_sha256: str = "",
        evidence_fingerprint: str = "",
        mapping_fingerprint: str = "",
    ):
        candidate = self._resolve_candidate(candidate)
        aggregate: list[dict[str, float]] = []
        per_episode: dict[str, dict[str, float]] = {}
        stable = True
        for episode in episodes:
            simulated_q, simulated_dq = self._rollout(candidate, episode)
            score, metrics = compare_trajectories(
                simulated_q,
                simulated_dq,
                episode.actual_q,
                episode.actual_dq,
                episode.command_q,
                self.environment.dt,
                objective_weights,
            )
            metrics["score"] = score
            per_episode[episode.name] = metrics
            aggregate.append(metrics)
            stable = stable and np.isfinite(simulated_q).all() and float(np.max(np.abs(simulated_q))) < 100.0
        names = aggregate[0].keys()
        means = {name: float(np.mean([item[name] for item in aggregate])) for name in names}
        self._last_evaluation = {
            "evaluation_phase": phase,
            "run_id": run_id,
            "plan_sha256": plan_sha256,
            "evidence_fingerprint": evidence_fingerprint,
            "mapping_fingerprint": mapping_fingerprint,
            "evidence_episodes": episode_inputs(episodes),
            "result_sha256": evaluation_result_fingerprint(
                score=means["score"], metrics=means, episodes=per_episode, stable=stable
            ),
        }
        return means["score"], means, per_episode, stable

    def _resolve_candidate(self, candidate: dict[str, float]) -> dict[str, float]:
        resolved = dict(self.environment.calibration_parameters)
        resolved.update(candidate)
        return resolved

    def _rollout(self, candidate, episode):
        _validate_episode_width(episode, len(self.joint_layout.logical_names))
        dt = self.environment.dt
        q = episode.actual_q[0].copy()
        dq = episode.actual_dq[0].copy()
        q_history = np.empty_like(episode.actual_q)
        dq_history = np.empty_like(episode.actual_dq)
        delay_steps = max(0, round(candidate.get("command_delay_s", 0.0) / dt))
        properties = _joint_properties(
            self.environment,
            self.joint_layout,
            candidate,
            analytic=True,
        )
        kp = properties["stiffness"]
        kd = properties["damping"]
        friction = properties["friction"]
        effort = properties["effort"]
        inertia = self.base_inertia + properties["armature"]
        if self.residual is not None:
            self.residual.reset(episode.command_q[0])
        for step in range(len(episode.time_s)):
            command = episode.command_q[max(0, step - delay_steps)]
            torque = kp * (command - q) - kd * dq - friction * np.tanh(dq / 0.01)
            if self.residual is not None:
                torque += self.residual.compute(command, q, dq)
            torque = np.clip(torque, -effort, effort)
            ddq = torque / inertia
            dq += dt * ddq
            q += dt * dq
            q_history[step] = q
            dq_history[step] = dq
        return q_history, dq_history

    def close(self) -> None:
        return None
