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

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("Experiment name must be a safe filename component")
        if self.split not in {"train", "heldout"} or not self.usd_joints:
            raise ValueError("An experiment needs train/heldout role and controlled coordinates")
        if len(self.usd_joints) != len(set(self.usd_joints)):
            raise ValueError("An experiment cannot repeat a controlled coordinate")
