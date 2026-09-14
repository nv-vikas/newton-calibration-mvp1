from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from newton_calibration.core.models import ParameterSpec
from newton_calibration.optimizers import (
    DiagonalCMAES,
    OptimizerContractError,
    OptimizerInit,
    OptimizerRegistryError,
    create_optimizer,
    get_optimizer_registration,
    list_optimizers,
    optimizer_config_fingerprint,
    register_optimizer,
    registry,
    validate_candidates,
)


def _parameters() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec("x", -5.0, 5.0, 4.0, "", "test", ""),
        ParameterSpec("y", -5.0, 5.0, -4.0, "", "test", ""),
    )


def _scores(candidates: list[dict[str, float]]) -> list[float]:
    return [(candidate["x"] - 0.75) ** 2 + (candidate["y"] + 1.25) ** 2 for candidate in candidates]


def test_builtin_registry_preserves_diagonal_cmaes_behavior():
    parameters = _parameters()
    direct = DiagonalCMAES(parameters, population=8, seed=3)
    registered = create_optimizer(
        "diagonal-cma-es",
        OptimizerInit(parameters=parameters, population=8, seed=3),
    )

    assert list_optimizers(discover=False)["diagonal-cma-es"] == "1"
    for _ in range(5):
        direct_candidates = direct.ask()
        registered_candidates = registered.ask()
        assert direct_candidates == registered_candidates
        scores = _scores(direct_candidates)
        direct.tell(direct_candidates, scores)
        registered.tell(registered_candidates, scores)

    assert registered.generation == direct.generation
    assert registered.best == direct.best


def test_optimizer_state_round_trip_reproduces_next_ask():
    initialization = OptimizerInit(parameters=_parameters(), population=8, seed=9)
    original = create_optimizer("diagonal-cma-es", initialization)
    for _ in range(3):
        candidates = original.ask()
        original.tell(candidates, _scores(candidates))

    restored = create_optimizer("diagonal-cma-es", initialization)
    restored.load_state_dict(original.state_dict())

    assert restored.generation == original.generation
    assert restored.best == original.best
    assert restored.ask() == original.ask()


class _ExternalOptimizer:
    def __init__(self, initialization: OptimizerInit):
        self.initialization = initialization
        self._generation = 0
        self._best: tuple[dict[str, float], float] | None = None

    def ask(self) -> list[dict[str, float]]:
        return [
            {parameter.name: parameter.initial for parameter in self.initialization.parameters}
            for _ in range(self.initialization.population)
        ]

    def tell(self, candidates, scores) -> None:
        winner = int(np.argmin(scores))
        if self._best is None or scores[winner] < self._best[1]:
            self._best = (dict(candidates[winner]), float(scores[winner]))
        self._generation += 1

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best(self) -> tuple[dict[str, float], float] | None:
        return self._best

    def state_dict(self) -> dict:
        return {"generation": self._generation, "best": self._best}

    def load_state_dict(self, state) -> None:
        self._generation = int(state["generation"])
        best = state.get("best")
        self._best = (dict(best[0]), float(best[1])) if best else None


def test_external_factory_registration_is_versioned_and_constructible():
    name = "test.minjae-search.v1"
    registration = register_optimizer(name, _ExternalOptimizer, version="2.4.1", replace=True)
    initialization = OptimizerInit(parameters=_parameters(), population=3, seed=11, options={"mode": "robust"})

    optimizer = create_optimizer(name, initialization)

    assert registration.version == "2.4.1"
    assert registration.provider == "in-process"
    assert get_optimizer_registration(name).version == "2.4.1"
    assert isinstance(optimizer, _ExternalOptimizer)
    assert optimizer.initialization.options == {"mode": "robust"}
    with pytest.raises(OptimizerRegistryError, match="already registered"):
        register_optimizer(name, _ExternalOptimizer, version="2.4.2")


def test_installed_entry_point_discovers_external_factory(monkeypatch):
    name = "test.entry-point-search.v1"

    class _EntryPoints(tuple):
        def select(self, *, group):
            assert group == registry.ENTRY_POINT_GROUP
            return self

    entry_point = SimpleNamespace(
        name=name,
        dist=SimpleNamespace(version="7.2.0"),
        load=lambda: _ExternalOptimizer,
    )
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: _EntryPoints((entry_point,)))
    monkeypatch.setattr(registry, "_DISCOVERY_COMPLETE", False)

    registration = get_optimizer_registration(name)

    assert registration.version == "7.2.0"
    assert registration.provider == "python-entry-point:unknown-distribution"
    assert isinstance(registration.create(OptimizerInit(_parameters(), 2, 1)), _ExternalOptimizer)


def test_entry_point_collision_with_another_provider_fails_closed(monkeypatch):
    name = "test.provider-collision.v1"
    register_optimizer(name, _ExternalOptimizer, version="1", provider="trusted-provider", replace=True)

    class _EntryPoints(tuple):
        def select(self, *, group):
            assert group == registry.ENTRY_POINT_GROUP
            return self

    entry_point = SimpleNamespace(
        name=name,
        dist=SimpleNamespace(name="different-provider", version="1"),
        load=lambda: _ExternalOptimizer,
    )
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: _EntryPoints((entry_point,)))

    with pytest.raises(OptimizerRegistryError, match="collides with registered provider"):
        registry.discover_optimizer_plugins(force=True)


def test_provider_receives_defensive_copy_of_locked_options():
    name = "test.options-copy.v1"
    observed = {}

    def mutating_factory(initialization):
        initialization.options["nested"]["value"] = "provider-mutated"
        observed.update(initialization.options)
        return _ExternalOptimizer(initialization)

    register_optimizer(name, mutating_factory, version="1", replace=True)
    source_options = {"nested": {"value": "locked"}}
    initialization = OptimizerInit(_parameters(), population=2, seed=1, options=source_options)

    create_optimizer(name, initialization)

    assert source_options == {"nested": {"value": "locked"}}
    assert initialization.options == {"nested": {"value": "locked"}}
    assert observed == {"nested": {"value": "provider-mutated"}}


def test_optimizer_config_fingerprint_is_deterministic_and_order_sensitive():
    initialization = OptimizerInit(_parameters(), population=8, seed=3, options={"beta": 2, "alpha": 1})
    registration = get_optimizer_registration("diagonal-cma-es", discover=False)

    first = optimizer_config_fingerprint(registration.name, registration.version, initialization)
    second = optimizer_config_fingerprint(
        registration.name,
        registration.version,
        OptimizerInit(_parameters(), population=8, seed=3, options={"alpha": 1, "beta": 2}),
    )
    reordered = optimizer_config_fingerprint(
        registration.name,
        registration.version,
        replace(initialization, parameters=tuple(reversed(_parameters()))),
    )

    assert first == second
    assert first != reordered
    assert first != optimizer_config_fingerprint(
        registration.name,
        registration.version,
        initialization,
        provider="different-provider",
    )
    assert len(first) == 64


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ({"x": 0.0}, "keys do not match"),
        ({"x": 0.0, "y": 0.0, "z": 0.0}, "keys do not match"),
        ({"x": float("nan"), "y": 0.0}, "not finite"),
        ({"x": 6.0, "y": 0.0}, "outside"),
        ({"x": True, "y": 0.0}, "not a real number"),
    ],
)
def test_candidate_contract_rejects_invalid_external_output(candidate, message):
    with pytest.raises(OptimizerContractError, match=message):
        validate_candidates([candidate], _parameters(), expected_count=1)


def test_candidate_contract_normalizes_valid_numeric_values():
    candidates = validate_candidates(
        [{"x": np.float64(0.5), "y": np.int64(-1)}],
        _parameters(),
        expected_count=1,
    )

    assert candidates == [{"x": 0.5, "y": -1.0}]
    assert all(isinstance(value, float) for value in candidates[0].values())


def test_missing_minjae_provider_fails_closed_and_lists_available_optimizers():
    with pytest.raises(KeyError, match="available optimizers:.*diagonal-cma-es"):
        get_optimizer_registration("minjae-nvopt.v1", discover=False)
