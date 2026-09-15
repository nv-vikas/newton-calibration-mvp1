"""Runtime-independent experiment contracts. No scene, optimizer or robot driver."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationRequest:
    """Requested scope and *declared* collection capabilities, never measured facts.

    Empty targets means the selected fitting recipe's complete parameter surface.
    None capabilities means unknown, not unavailable. A motion proposal can be
    produced conditionally while hardware instrumentation is still unconfirmed.
    """

    target_parameters: tuple[str, ...] = ()
    available_signals: tuple[str, ...] | None = None
    clock_synchronized: bool | None = None
    capability_source: str = ""
    max_training_experiments: int = 64
    design_mode: str = "adaptive"
    max_candidate_probes: int = 96
    minimum_information_gain: float = 0.02
    sensitivity_floor: float = 1.0
    separation_floor: float = 0.01

    def __post_init__(self):
        for field in ("target_parameters", "available_signals"):
            values = getattr(self, field)
            if values is None and field == "available_signals":
                continue
            if isinstance(values, str) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{field} must be a sequence of nonempty names")
            if len(values) != len(set(values)):
                raise ValueError(f"{field} contains duplicate names")
            object.__setattr__(self, field, tuple(values))
        if self.clock_synchronized is not None and type(self.clock_synchronized) is not bool:
            raise ValueError("clock_synchronized must be true, false or unknown")
        if (
            self.available_signals is not None or self.clock_synchronized is not None
        ) and not self.capability_source.strip():
            raise ValueError("Declared measurement capabilities require provenance")
        if type(self.max_training_experiments) is not int or not 1 <= self.max_training_experiments <= 256:
            raise ValueError("max_training_experiments must be an integer in [1, 256]")
        if self.design_mode not in {"adaptive", "recipe_only"}:
            raise ValueError("design_mode must be adaptive or recipe_only")
        if type(self.max_candidate_probes) is not int or not 1 <= self.max_candidate_probes <= 4096:
            raise ValueError("max_candidate_probes must be an integer in [1, 4096]")
        import math

        for name in ("minimum_information_gain", "sensitivity_floor", "separation_floor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ExperimentSpec:
    """One independently replayable experiment selected by a domain recipe."""

    name: str
    recipe_id: str
    split: str
    usd_joints: tuple[str, ...]
    target_parameters: tuple[str, ...]
    required_signals: tuple[str, ...]
    reason: str
    variant: int = 0
    frequency_scale: float = 1.0
    amplitude_scale: float = 1.0
    posture_offset_rad: tuple[float, ...] = ()

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("Experiment name must be a safe filename component")
        if self.split not in {"train", "heldout"} or not self.usd_joints:
            raise ValueError("An experiment needs train/heldout role and controlled coordinates")
        if len(self.usd_joints) != len(set(self.usd_joints)):
            raise ValueError("An experiment cannot repeat a controlled coordinate")
        import math

        if not math.isfinite(self.frequency_scale) or not 0.1 <= self.frequency_scale <= 4:
            raise ValueError("frequency_scale must be in [0.1, 4]")
        if not math.isfinite(self.amplitude_scale) or not 0 < self.amplitude_scale <= 1:
            raise ValueError("amplitude_scale must be in (0, 1]")
        if any(not math.isfinite(float(v)) for v in self.posture_offset_rad):
            raise ValueError("Posture offset must be finite")
        object.__setattr__(self, "posture_offset_rad", tuple(float(v) for v in self.posture_offset_rad))
