from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LinearHistoryResidual:
    """Small actuator-level residual: Δtorque(error, velocity, command change, bias)."""

    joint_names: list[str]
    weights: np.ndarray
    clip_nm: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.clip_nm = np.asarray(self.clip_nm, dtype=np.float64)
        expected = (len(self.joint_names), 4)
        if self.weights.shape != expected:
            raise ValueError(f"Residual weights must have shape {expected}, got {self.weights.shape}")
        if self.clip_nm.shape != (len(self.joint_names),):
            raise ValueError("Residual clip_nm must contain one value per joint")
        self._previous_command = np.zeros(len(self.joint_names), dtype=np.float64)

    def reset(self, command: np.ndarray) -> None:
        self._previous_command = np.asarray(command, dtype=np.float64).copy()

    def compute(self, command: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float64)
        features = np.stack(
            [command - q, dq, command - self._previous_command, np.ones_like(command)],
            axis=1,
        )
        torque = np.sum(self.weights * features, axis=1)
        self._previous_command = command.copy()
        return np.clip(torque, -self.clip_nm, self.clip_nm)


def load_residual(path: str | Path | None, expected_joints: list[str]) -> LinearHistoryResidual | None:
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "newton.calibration.actuator-residual/v1":
        raise ValueError(f"Unsupported actuator residual schema in {source}")
    joints = payload["joint_names"]
    if joints != expected_joints:
        raise ValueError(f"Residual joints {joints} do not match environment joints {expected_joints}")
    return LinearHistoryResidual(joints, np.asarray(payload["weights"]), np.asarray(payload["clip_nm"]))
