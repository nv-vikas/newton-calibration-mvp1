from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any


def parameter_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Fingerprint one complete parameter candidate deterministically."""

    normalized: dict[str, float] = {}
    for name, value in parameters.items():
        if not isinstance(name, str) or not name:
            raise ValueError("calibration parameter names must be non-empty strings")
        if isinstance(value, bool):
            raise TypeError(f"calibration parameter {name!r} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"calibration parameter {name!r} must be a finite number")
        normalized[name] = number
    return _canonical_sha256(normalized)


def record_fingerprint(value: Any) -> str:
    """Fingerprint a finite JSON-compatible locked record."""

    return _canonical_sha256(value)


def numeric_surface_fingerprint(values: Mapping[str, Sequence[Any]]) -> str:
    """Fingerprint ordered numeric values read back from a runtime surface."""

    normalized: dict[str, list[float]] = {}
    for name, items in values.items():
        # Isaac Lab actuator tensors are float32. Canonicalize both ideal
        # Python/NumPy values and observed tensors to the same representation
        # before hashing; readback equality itself is checked with tolerance.
        converted = [_float32(float(value)) for value in items]
        if any(not math.isfinite(value) for value in converted):
            raise ValueError(f"runtime surface {name!r} contains a non-finite value")
        normalized[str(name)] = converted
    return _canonical_sha256(normalized)


def evaluation_result_fingerprint(
    *, score: Any, metrics: Mapping[str, Any], episodes: Mapping[str, Mapping[str, Any]], stable: Any
) -> str:
    """Bind a runtime attestation to the exact comparison result it produced."""

    if not isinstance(stable, bool):
        raise TypeError("evaluation stability must be a boolean")
    return _canonical_sha256(
        {
            "score": _finite_number(score, "evaluation score"),
            "metrics": _finite_mapping(metrics, "evaluation metrics"),
            "episodes": {
                str(name): _finite_mapping(values, f"evaluation episode {name!r}")
                for name, values in episodes.items()
            },
            "stable": stable,
        }
    )


def episode_inputs(episodes: Sequence[Any]) -> list[dict[str, str]]:
    """Return ordered, immutable identities for the real episodes actually replayed."""

    result: list[dict[str, str]] = []
    for episode in episodes:
        item = {
            "name": _nonempty_text(getattr(episode, "name", None), "episode name"),
            "split": _nonempty_text(getattr(episode, "split", None), "episode split"),
            "trial_id": _nonempty_text(getattr(episode, "trial_id", None), "episode trial_id"),
            "source_sha256": _nonempty_text(
                getattr(episode, "source_sha256", None), "episode source_sha256"
            ),
        }
        digest = item["source_sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("episode source_sha256 must be a lowercase SHA-256 digest")
        result.append(item)
    return result


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _finite_mapping(value: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return {str(name): _finite_number(item, f"{label}.{name}") for name, item in value.items()}


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value
