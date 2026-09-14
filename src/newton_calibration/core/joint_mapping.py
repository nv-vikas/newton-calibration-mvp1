from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_UNIT_ALIASES = {
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "turn": "turn",
    "turns": "turn",
    "rev": "turn",
    "revolution": "turn",
    "revolutions": "turn",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
}
_UNIT_DIMENSIONS = {
    "rad": "angle",
    "deg": "angle",
    "turn": "angle",
    "m": "length",
    "cm": "length",
    "mm": "length",
}
_TO_BASE = {
    "rad": 1.0,
    "deg": math.pi / 180.0,
    "turn": 2.0 * math.pi,
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}


def canonical_joint_unit(unit: str) -> str:
    """Return a supported canonical position unit or reject an unknown unit."""

    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("joint unit must be a non-empty string")
    try:
        return _UNIT_ALIASES[unit.strip().casefold()]
    except KeyError as exc:
        supported = ", ".join(sorted(_TO_BASE))
        raise ValueError(f"unsupported joint unit {unit!r}; supported canonical units are {supported}") from exc


def joint_unit_factor(source_unit: str, target_unit: str) -> float:
    """Return the multiplier that converts source positions to target positions."""

    source = canonical_joint_unit(source_unit)
    target = canonical_joint_unit(target_unit)
    if _UNIT_DIMENSIONS[source] != _UNIT_DIMENSIONS[target]:
        raise ValueError(f"incompatible joint units: {source!r} cannot be converted to {target!r}")
    return _TO_BASE[source] / _TO_BASE[target]


@dataclass(frozen=True)
class JointBinding:
    """An explicit, auditable map from one real-data joint to one USD DOF.

    Positions are mapped as::

        q_usd = sign * scale * convert(q_source, source_unit, usd_unit) + offset

    ``offset`` is expressed in ``usd_unit``.  Velocities use the same mapping
    without the offset.  ``scale`` is a dimensionless setup-specific gain; it
    is deliberately separate from unit conversion and direction.
    """

    source_joint: str
    usd_joint: str
    source_unit: str
    usd_unit: str
    sign: int = 1
    scale: float = 1.0
    offset: float = 0.0
    transform_confirmed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_joint, str) or not self.source_joint.strip():
            raise ValueError("source_joint must be a non-empty string")
        if not isinstance(self.usd_joint, str) or not self.usd_joint.strip():
            raise ValueError("usd_joint must be a non-empty string")
        if self.sign not in (-1, 1):
            raise ValueError("joint mapping sign must be exactly -1 or 1")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("joint mapping scale must be finite and greater than zero")
        if not math.isfinite(self.offset):
            raise ValueError("joint mapping offset must be finite")
        if not isinstance(self.transform_confirmed, bool):
            raise TypeError("transform_confirmed must be a boolean")
        source = canonical_joint_unit(self.source_unit)
        target = canonical_joint_unit(self.usd_unit)
        joint_unit_factor(source, target)
        object.__setattr__(self, "source_unit", source)
        object.__setattr__(self, "usd_unit", target)

    def apply_position(self, values: np.ndarray) -> np.ndarray:
        factor = self.sign * self.scale * joint_unit_factor(self.source_unit, self.usd_unit)
        return np.asarray(values, dtype=np.float64) * factor + self.offset

    def apply_velocity(self, values: np.ndarray) -> np.ndarray:
        factor = self.sign * self.scale * joint_unit_factor(self.source_unit, self.usd_unit)
        return np.asarray(values, dtype=np.float64) * factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_joint": self.source_joint,
            "usd_joint": self.usd_joint,
            "source_unit": self.source_unit,
            "usd_unit": self.usd_unit,
            "sign": self.sign,
            "scale": self.scale,
            "offset": self.offset,
            "transform_confirmed": self.transform_confirmed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JointBinding:
        return cls(
            source_joint=str(value["source_joint"]),
            usd_joint=str(value["usd_joint"]),
            source_unit=str(value["source_unit"]),
            usd_unit=str(value["usd_unit"]),
            sign=int(value.get("sign", 1)),
            scale=float(value.get("scale", 1.0)),
            offset=float(value.get("offset", 0.0)),
            transform_confirmed=value["transform_confirmed"],
        )


@dataclass(frozen=True)
class JointMappingProposal:
    source_joint: str
    usd_joint: str | None
    status: str
    match_kind: str | None
    confidence: float
    candidates: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_joint": self.source_joint,
            "usd_joint": self.usd_joint,
            "status": self.status,
            "match_kind": self.match_kind,
            "confidence": self.confidence,
            "candidates": list(self.candidates),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JointMappingReport:
    proposals: tuple[JointMappingProposal, ...]
    unused_usd_joints: tuple[str, ...]

    @property
    def ready(self) -> bool:
        """Whether every source joint has a unique deterministic name match."""

        return bool(self.proposals) and all(item.status == "matched" for item in self.proposals)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(f"{item.source_joint}: {item.reason}" for item in self.proposals if item.status != "matched")

    def require_unambiguous(self) -> dict[str, str]:
        if not self.ready:
            details = "; ".join(self.blockers) or "no source joints were supplied"
            raise ValueError(f"joint mapping is not ready: {details}")
        return {item.source_joint: str(item.usd_joint) for item in self.proposals}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "proposals": [item.to_dict() for item in self.proposals],
            "unused_usd_joints": list(self.unused_usd_joints),
        }


def normalize_joint_name(name: str) -> str:
    """Normalize only spelling separators, never infer robot semantics."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("joint names must be non-empty strings")
    leaf = name.strip().rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return re.sub(r"[^0-9a-z]+", "", leaf.casefold())


def propose_joint_mapping(source_joints: Sequence[str], usd_joints: Sequence[str]) -> JointMappingReport:
    """Propose exact or normalized one-to-one matches without fuzzy guessing.

    Full-name exact matches are preferred.  A normalized match is accepted only
    when it is unique on both sides after exact matches are reserved.  Collisions
    are reported as ambiguity and deliberately block readiness.
    """

    sources = _unique_names(source_joints, "source")
    targets = _unique_names(usd_joints, "USD")
    target_set = set(targets)
    matched: dict[str, tuple[str, str, float]] = {}
    used_targets: set[str] = set()

    for source in sources:
        if source in target_set:
            matched[source] = (source, "exact", 1.0)
            used_targets.add(source)

    remaining_sources = [name for name in sources if name not in matched]
    remaining_targets = [name for name in targets if name not in used_targets]
    source_by_normalized = _group_by_normalized(remaining_sources)
    target_by_normalized = _group_by_normalized(remaining_targets)

    proposals: list[JointMappingProposal] = []
    for source in sources:
        if source in matched:
            target, kind, confidence = matched[source]
            proposals.append(
                JointMappingProposal(source, target, "matched", kind, confidence, (target,), "unique exact match")
            )
            continue
        normalized = normalize_joint_name(source)
        candidates = tuple(target_by_normalized.get(normalized, ()))
        source_collisions = source_by_normalized.get(normalized, ())
        if len(candidates) == 1 and len(source_collisions) == 1:
            target = candidates[0]
            used_targets.add(target)
            proposals.append(
                JointMappingProposal(
                    source,
                    target,
                    "matched",
                    "normalized",
                    0.97,
                    candidates,
                    "unique normalized spelling match",
                )
            )
        elif candidates:
            reason = "normalized name is ambiguous; an explicit source-to-USD binding is required"
            proposals.append(JointMappingProposal(source, None, "ambiguous", None, 0.0, candidates, reason))
        else:
            reason = "no exact or normalized USD joint match; an explicit source-to-USD binding is required"
            proposals.append(JointMappingProposal(source, None, "unmatched", None, 0.0, (), reason))

    return JointMappingReport(tuple(proposals), tuple(name for name in targets if name not in used_targets))


def bindings_from_unambiguous_report(
    report: JointMappingReport,
    *,
    source_units: Mapping[str, str],
    usd_units: Mapping[str, str],
    affine_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    confirmed_transforms: Sequence[str] = (),
) -> tuple[JointBinding, ...]:
    """Turn a safe name proposal into explicit coordinate bindings.

    Units are mandatory even for identity mappings.  Direction, scale, and zero
    offset default to identity but are serialized into every resulting binding.
    Robot-specific sign/zero conventions cannot be inferred from joint names.
    ``confirmed_transforms`` must name every binding whose complete unit,
    direction, scale, and offset have been verified, including identity maps.
    """

    mapping = report.require_unambiguous()
    overrides = affine_overrides or {}
    unknown_overrides = sorted(set(overrides) - set(mapping))
    if unknown_overrides:
        raise ValueError(f"affine overrides reference unknown source joints: {unknown_overrides}")
    confirmed = set(confirmed_transforms)
    unknown_confirmations = sorted(confirmed - set(mapping))
    if unknown_confirmations:
        raise ValueError(f"transform confirmations reference unknown source joints: {unknown_confirmations}")
    result: list[JointBinding] = []
    for source, target in mapping.items():
        if source not in source_units:
            raise ValueError(f"source unit is required for joint {source!r}")
        if target not in usd_units:
            raise ValueError(f"USD unit is required for joint {target!r}")
        override = overrides.get(source, {})
        result.append(
            JointBinding(
                source_joint=source,
                usd_joint=target,
                source_unit=source_units[source],
                usd_unit=usd_units[target],
                sign=int(override.get("sign", 1)),
                scale=float(override.get("scale", 1.0)),
                offset=float(override.get("offset", 0.0)),
                transform_confirmed=source in confirmed,
            )
        )
    return tuple(result)


def _unique_names(values: Sequence[str], label: str) -> list[str]:
    names = list(values)
    for name in names:
        normalize_joint_name(name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label} joint names: {duplicates}")
    return names


def _group_by_normalized(values: Sequence[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for value in values:
        grouped.setdefault(normalize_joint_name(value), []).append(value)
    return {key: tuple(items) for key, items in grouped.items()}
