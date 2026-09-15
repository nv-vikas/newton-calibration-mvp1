"""Versioned normalized trajectory generators, independent of robot and physics.

The common writer enforces the scene envelope after generation. A new domain
must register its own generator explicitly; no arbitrary agent-authored code is
evaluated from a recipe or evidence file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Waveform:
    values: np.ndarray
    frequencies_hz: tuple[float, ...]


Generator = Callable[[np.ndarray, float, int, int], Waveform]
_GENERATORS: dict[str, Generator] = {}


def register_generator(recipe_id: str, generator: Generator):
    if "@" not in recipe_id or recipe_id in _GENERATORS or not callable(generator):
        raise ValueError("Generator needs a unique versioned ID and callable implementation")
    _GENERATORS[recipe_id] = generator


def generate_waveform(recipe_id: str, active_time: np.ndarray, duration: float, joint: int, variant: int) -> Waveform:
    if recipe_id not in _GENERATORS:
        raise ValueError(f"No installed motion generator for {recipe_id!r}")
    result = _GENERATORS[recipe_id](active_time, duration, joint, variant)
    if not isinstance(result, Waveform) or not isinstance(result.values, np.ndarray):
        raise TypeError("Motion generator must return a Waveform with an ndarray")
    if result.values.shape != active_time.shape or not np.isfinite(result.values).all():
        raise ValueError("Motion generator returned malformed or nonfinite values")
    if np.max(np.abs(result.values)) > 1.0 + 1e-12:
        raise ValueError("Normalized motion generator exceeded its amplitude contract")
    if any(not np.isfinite(frequency) or frequency <= 0 for frequency in result.frequencies_hz):
        raise ValueError("Motion generator returned invalid frequencies")
    return result


def _sweep(t, duration, joint, variant, *, fast=False):
    f0, f1 = (0.20, 0.80) if fast else (0.08, 0.45)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    return Waveform(np.sin(phase), (f0, f1))


def _acceleration(t, duration, joint, variant):
    return _sweep(t, duration, joint, variant, fast=True)


def _settling(t, duration, joint, variant):
    # Quintic transitions + plateaus, never instantaneous position steps.
    knots = np.array([0, 0.12, 0.32, 0.46, 0.66, 0.80, 1.0]) * duration
    values = (0.0, 0.7, 0.7, -0.7, -0.7, 0.0, 0.0)
    q = np.zeros_like(t)
    for a, b, qa, qb in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
        mask = (t >= a) & (t < b)
        u = np.clip((t[mask] - a) / (b - a), 0, 1)
        q[mask] = qa + (qb - qa) * (10 * u**3 - 15 * u**4 + 6 * u**5)
    return Waveform(q, ())


def _slow_reversal(t, duration, joint, variant):
    frequency = 2.0 / duration
    return Waveform(np.sin(2 * np.pi * frequency * t), (frequency,))


def _heldout(t, duration, joint, variant):
    f0, f1 = 0.085 + 0.009 * joint + 0.025 * variant, 0.31 + 0.013 * joint + 0.04 * variant
    return Waveform(0.65 * (0.85 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(2 * np.pi * f1 * t)), (f0, f1))


for _name, _generator in (
    ("servo_sweep@1", _sweep),
    ("settling@1", _settling),
    ("acceleration_sweep@1", _acceleration),
    ("slow_reversal@1", _slow_reversal),
    ("heldout_multisine@1", _heldout),
):
    register_generator(_name, _generator)
