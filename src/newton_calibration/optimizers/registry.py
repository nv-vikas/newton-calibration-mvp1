from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from numbers import Real
from typing import Any

from newton_calibration.core.models import ParameterSpec
from newton_calibration.core.protocols import OptimizerPlugin

ENTRY_POINT_GROUP = "newton_calibration.optimizers"


class OptimizerRegistryError(RuntimeError):
    """Base error for optimizer discovery and registration failures."""


class OptimizerContractError(ValueError):
    """Raised when a plug-in violates the optimizer/runner boundary."""


@dataclass(frozen=True)
class OptimizerInit:
    """Locked inputs supplied to an optimizer factory.

    ``options`` is reserved for method-specific configuration.  The fit runner's
    iteration/generation budget deliberately does not live here: increasing a
    job's stopping budget must not invalidate an otherwise resumable optimizer
    checkpoint.
    """

    parameters: tuple[ParameterSpec, ...] | Sequence[ParameterSpec]
    population: int
    seed: int
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "options", copy.deepcopy(dict(self.options)))
        if not self.parameters:
            raise ValueError("an optimizer requires at least one parameter")
        if self.population < 1:
            raise ValueError("optimizer population must be positive")


OptimizerFactory = Callable[[OptimizerInit], OptimizerPlugin]


@dataclass(frozen=True)
class OptimizerRegistration:
    name: str
    version: str
    provider: str
    factory: OptimizerFactory

    def create(self, initialization: OptimizerInit) -> OptimizerPlugin:
        provider_initialization = OptimizerInit(
            parameters=initialization.parameters,
            population=initialization.population,
            seed=initialization.seed,
            options=copy.deepcopy(dict(initialization.options)),
        )
        optimizer = self.factory(provider_initialization)
        validate_optimizer_plugin(optimizer, name=self.name)
        return optimizer


_REGISTRY: dict[str, OptimizerRegistration] = {}
_REGISTRY_LOCK = threading.RLock()
_DISCOVERY_COMPLETE = False


def register_optimizer(
    name: str,
    factory: OptimizerFactory,
    *,
    version: str,
    provider: str = "in-process",
    replace: bool = False,
) -> OptimizerRegistration:
    """Register a versioned optimizer factory.

    External packages should expose the same factory through the
    ``newton_calibration.optimizers`` Python entry-point group when the plug-in
    must also be discoverable by a standalone CLI or Docker process.
    """

    normalized_name = _nonempty_text(name, "optimizer name")
    normalized_version = _nonempty_text(version, "optimizer version")
    normalized_provider = _nonempty_text(provider, "optimizer provider")
    if not callable(factory):
        raise TypeError("optimizer factory must be callable")
    registration = OptimizerRegistration(normalized_name, normalized_version, normalized_provider, factory)
    with _REGISTRY_LOCK:
        if normalized_name in _REGISTRY and not replace:
            existing = _REGISTRY[normalized_name]
            raise OptimizerRegistryError(
                f"Optimizer {normalized_name!r} is already registered at version {existing.version!r}"
            )
        _REGISTRY[normalized_name] = registration
    return registration


def get_optimizer_registration(name: str, *, discover: bool = True) -> OptimizerRegistration:
    normalized_name = _nonempty_text(name, "optimizer name")
    with _REGISTRY_LOCK:
        registration = _REGISTRY.get(normalized_name)
    if registration is None and discover:
        discover_optimizer_plugins()
        with _REGISTRY_LOCK:
            registration = _REGISTRY.get(normalized_name)
    if registration is None:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"Unknown optimizer {normalized_name!r}; available optimizers: {available}")
    return registration


def create_optimizer(name: str, initialization: OptimizerInit) -> OptimizerPlugin:
    """Create a registered optimizer without giving it access to the runtime."""

    return get_optimizer_registration(name).create(initialization)


def list_optimizers(*, discover: bool = True) -> dict[str, str]:
    """Return registered optimizer names mapped to their implementation versions."""

    if discover:
        discover_optimizer_plugins()
    with _REGISTRY_LOCK:
        return {name: _REGISTRY[name].version for name in sorted(_REGISTRY)}


def discover_optimizer_plugins(*, force: bool = False) -> dict[str, str]:
    """Load installed optimizer factories from Python package entry points.

    An entry point's name is its optimizer registry name.  Its loaded value must
    be either an :class:`OptimizerRegistration` or a factory accepting one
    :class:`OptimizerInit`.  Factory versions come from the provider package's
    distribution version, or from a ``__optimizer_version__`` attribute when a
    provider explicitly supplies one.
    """

    global _DISCOVERY_COMPLETE
    with _REGISTRY_LOCK:
        if _DISCOVERY_COMPLETE and not force:
            return {name: _REGISTRY[name].version for name in sorted(_REGISTRY)}

        discovered = metadata.entry_points()
        if hasattr(discovered, "select"):
            entry_points = tuple(discovered.select(group=ENTRY_POINT_GROUP))
        else:  # pragma: no cover - compatibility with older importlib.metadata
            entry_points = tuple(discovered.get(ENTRY_POINT_GROUP, ()))

        for entry_point in entry_points:
            loaded = entry_point.load()
            distribution = getattr(entry_point, "dist", None)
            distribution_name = getattr(distribution, "name", None) or "unknown-distribution"
            provider = f"python-entry-point:{distribution_name}"
            if isinstance(loaded, OptimizerRegistration):
                if loaded.name != entry_point.name:
                    raise OptimizerRegistryError(
                        f"Optimizer entry point {entry_point.name!r} loaded registration {loaded.name!r}"
                    )
                version = loaded.version
                factory = loaded.factory
            else:
                if not callable(loaded):
                    raise OptimizerRegistryError(
                        f"Optimizer entry point {entry_point.name!r} must load a factory or OptimizerRegistration"
                    )
                factory = loaded
                version = getattr(loaded, "__optimizer_version__", None)
                if version is None:
                    version = getattr(distribution, "version", None)
                if not version:
                    raise OptimizerRegistryError(
                        f"Optimizer entry point {entry_point.name!r} does not declare an implementation version"
                    )

            existing = _REGISTRY.get(entry_point.name)
            if existing is not None:
                if existing.version == str(version) and existing.provider == provider:
                    continue
                raise OptimizerRegistryError(
                    f"Optimizer entry point {entry_point.name!r} collides with registered provider "
                    f"{existing.provider!r} version {existing.version!r}"
                )
            register_optimizer(entry_point.name, factory, version=str(version), provider=provider)

        _DISCOVERY_COMPLETE = True
        return {name: _REGISTRY[name].version for name in sorted(_REGISTRY)}


def optimizer_config_fingerprint(
    name: str,
    version: str,
    initialization: OptimizerInit,
    *,
    provider: str = "unknown",
) -> str:
    """Fingerprint all optimizer inputs that must match when resuming a job."""

    payload = {
        "name": _nonempty_text(name, "optimizer name"),
        "version": _nonempty_text(version, "optimizer version"),
        "provider": _nonempty_text(provider, "optimizer provider"),
        "population": initialization.population,
        "seed": initialization.seed,
        "options": initialization.options,
        "parameters": [
            {
                "name": parameter.name,
                "lower": parameter.lower,
                "upper": parameter.upper,
                "initial": parameter.initial,
                "unit": parameter.unit,
                "owner": parameter.owner,
                "rationale": parameter.rationale,
            }
            for parameter in initialization.parameters
        ],
    }
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OptimizerContractError("optimizer configuration must be JSON serializable and finite") from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_optimizer_plugin(optimizer: object, *, name: str = "optimizer") -> None:
    """Fail early when an external factory returns an incomplete plug-in."""

    missing = [
        member
        for member in ("ask", "tell", "state_dict", "load_state_dict", "generation", "best")
        if not hasattr(optimizer, member)
    ]
    noncallable = [
        member
        for member in ("ask", "tell", "state_dict", "load_state_dict")
        if hasattr(optimizer, member) and not callable(getattr(optimizer, member))
    ]
    if missing or noncallable:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if noncallable:
            details.append(f"not callable: {', '.join(noncallable)}")
        raise OptimizerContractError(f"Optimizer plug-in {name!r} is invalid ({'; '.join(details)})")


def validate_candidates(
    candidates: Sequence[Mapping[str, Real]],
    parameters: Sequence[ParameterSpec],
    *,
    expected_count: int | None = None,
) -> list[dict[str, float]]:
    """Validate and normalize optimizer candidates before any physics execution."""

    if expected_count is not None and len(candidates) != expected_count:
        raise OptimizerContractError(f"optimizer returned {len(candidates)} candidates; expected {expected_count}")
    parameter_by_name = {parameter.name: parameter for parameter in parameters}
    expected_names = set(parameter_by_name)
    normalized: list[dict[str, float]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise OptimizerContractError(f"candidate {index} is not a parameter mapping")
        actual_names = set(candidate)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise OptimizerContractError(
                f"candidate {index} parameter keys do not match the locked plan; missing={missing}, extra={extra}"
            )
        normalized_candidate: dict[str, float] = {}
        for parameter in parameters:
            value = candidate[parameter.name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise OptimizerContractError(f"candidate {index} parameter {parameter.name!r} is not a real number")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise OptimizerContractError(f"candidate {index} parameter {parameter.name!r} is not finite")
            if not parameter.lower <= numeric_value <= parameter.upper:
                raise OptimizerContractError(
                    f"candidate {index} parameter {parameter.name!r}={numeric_value} is outside "
                    f"[{parameter.lower}, {parameter.upper}]"
                )
            normalized_candidate[parameter.name] = numeric_value
        normalized.append(normalized_candidate)
    return normalized


def validate_scores(scores: Sequence[Real], *, expected_count: int) -> list[float]:
    """Validate objective values before passing them back to a plug-in."""

    if len(scores) != expected_count:
        raise OptimizerContractError(f"runner produced {len(scores)} scores; expected {expected_count}")
    normalized: list[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
            raise OptimizerContractError(f"score {index} is not a finite real number")
        normalized.append(float(score))
    return normalized


def _nonempty_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
