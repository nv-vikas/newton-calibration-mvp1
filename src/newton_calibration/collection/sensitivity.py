"""Finite-difference dynamics experiments; no evidence or hardware is synthesized."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from newton_calibration.core.io import sha256_file
from newton_calibration.core.models import ParameterSpec, jsonable


class PredictionBackend(Protocol):
    def describe(self) -> dict: ...
    def rollout(self, command_path, experiment, parameters: dict[str, float]) -> np.ndarray:
        """Return [time, q followed by dq] in the declared controlled-joint order."""
        ...


@dataclass
class SensitivityResult:
    parameters: list[str]
    information: list[list[list[float]]]
    metadata: dict
    rollout_count: int


class FiniteDifferenceProbe:
    """Conditional sensitivity at two points in explicitly supplied probe ranges.

    Probe ranges are simulation hypotheses, NOT approved fitting bounds. The
    signal noise floors are assumptions until measured, not sensor discoveries.
    Mean Gram matrices prevent treating every simulation frame as an independent
    real trial. Correlated parameters are retained, not scored independently.
    """

    def __init__(
        self,
        backend: PredictionBackend,
        ranges: list[ParameterSpec],
        *,
        source: str,
        position_noise_rad: float = 0.001,
        velocity_noise_rad_s: float = 0.01,
        anchors=(0.25, 0.75),
        perturbation_fraction=0.1,
    ):
        self.backend = backend
        self.ranges = {p.name: p for p in ranges}
        self.source = source
        self.anchors = tuple(anchors)
        self.step = perturbation_fraction
        self.noise = (position_noise_rad, velocity_noise_rad_s)
        if not source.strip() or not ranges or len(self.ranges) != len(ranges):
            raise ValueError("Sensitivity ranges need unique parameters and provenance")
        if len(self.anchors) != 2 or not all(0 < a < 1 for a in self.anchors):
            raise ValueError("Two interior range anchors are required")
        if not 0 < self.step <= 0.2 or any(a + self.step > 1 for a in self.anchors):
            raise ValueError("Perturbations must stay inside probe ranges")
        if any(not np.isfinite(v) or v <= 0 for v in self.noise):
            raise ValueError("Sensitivity noise assumptions must be finite and positive")

    def describe(self):
        return {
            **self.backend.describe(),
            "method": "one-sided finite difference at two range anchors",
            "ranges": [jsonable(p) for p in self.ranges.values()],
            "range_source": self.source,
            "anchors": self.anchors,
            "perturbation_fraction": self.step,
            "assumed_position_noise_rad": self.noise[0],
            "assumed_velocity_noise_rad_s": self.noise[1],
            "range_approved_for_fitting": False,
            "real_identifiability_proven": False,
            "scope": "Conditional on other joint groups and model form; not a global identifiability certificate",
        }

    def __call__(self, command_path, experiment, parameters: list[str]) -> SensitivityResult:
        if not parameters or any(name not in self.ranges for name in parameters):
            raise ValueError("Requested sensitivity parameter has no declared probe range")
        matrices, runs, repeat_error = [], 0, 0.0
        traces = {}
        for anchor_index, anchor in enumerate(self.anchors):
            nominal = {
                name: self.ranges[name].lower + anchor * (self.ranges[name].upper - self.ranges[name].lower)
                for name in parameters
            }
            baseline = self._trace(command_path, experiment, nominal)
            traces[f"anchor_{anchor_index}_baseline"] = baseline
            runs += 1
            widths = baseline.shape[1] // 2
            noise = np.array([self.noise[0]] * widths + [self.noise[1]] * widths)
            if anchor_index == 0:
                repeated = self._trace(command_path, experiment, nominal)
                traces[f"anchor_{anchor_index}_repeat"] = repeated
                runs += 1
                if repeated.shape != baseline.shape:
                    raise ValueError("Repeated prediction changed output shape")
                repeat_error = float(np.max(abs(repeated - baseline) / noise))
                if repeat_error > 0.1:
                    raise ValueError(
                        f"Newton repeatability error exceeds 10% of assumed measurement noise: {repeat_error}"
                    )
            columns = []
            for name in parameters:
                changed = dict(nominal)
                changed[name] += self.step * (self.ranges[name].upper - self.ranges[name].lower)
                response = self._trace(command_path, experiment, changed)
                traces[f"anchor_{anchor_index}_{name}"] = response
                runs += 1
                if response.shape != baseline.shape:
                    raise ValueError("Perturbed prediction changed output shape")
                columns.append(((response - baseline) / noise / self.step).reshape(-1))
            jacobian = np.column_stack(columns)
            matrices.append((jacobian.T @ jacobian / len(baseline)).tolist())
        trace_path = Path(command_path).with_suffix(".predictions.npz")
        np.savez_compressed(trace_path, **traces)
        return SensitivityResult(
            parameters,
            matrices,
            {
                **self.describe(),
                "repeatability_error_noise_units": repeat_error,
                "command_sha256": sha256_file(command_path),
                "simulated_predictions_path": str(trace_path),
                "simulated_predictions_sha256": sha256_file(trace_path),
            },
            runs,
        )

    def _trace(self, path, experiment, parameters):
        value = np.asarray(self.backend.rollout(path, experiment, parameters), dtype=float)
        if value.ndim != 2 or len(value) < 3 or value.shape[1] % 2 or not np.isfinite(value).all():
            raise ValueError("Dynamics probe returned invalid q/dq predictions")
        return value
