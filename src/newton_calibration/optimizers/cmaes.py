from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from newton_calibration.core.models import ParameterSpec


class DiagonalCMAES:
    """Small bounded, seeded diagonal CMA-ES ask/tell plug-in.

    It intentionally implements the optimizer contract in-tree so the container does
    not depend on a particular optimization service. A production plug-in can replace
    it without changing the five-call API or experiment runner.
    """

    def __init__(self, parameters: Sequence[ParameterSpec], population: int = 12, seed: int = 7):
        if population < 4:
            raise ValueError("population must be at least four")
        self.parameters = list(parameters)
        self.population = int(population)
        self.rng = np.random.default_rng(seed)
        self.lower = np.array([p.lower for p in self.parameters], dtype=np.float64)
        self.upper = np.array([p.upper for p in self.parameters], dtype=np.float64)
        self.mean = self._encode(np.array([p.initial for p in self.parameters], dtype=np.float64))
        self.sigma = 0.28
        self.diag = np.ones(len(self.parameters), dtype=np.float64)
        self._generation = 0
        self._best: tuple[dict[str, float], float] | None = None

    def ask(self) -> list[dict[str, float]]:
        samples = self.mean + self.sigma * self.rng.normal(size=(self.population, len(self.parameters))) * self.diag
        samples = np.clip(samples, 0.0, 1.0)
        return [self._as_dict(self._decode(sample)) for sample in samples]

    def tell(self, candidates: Sequence[dict[str, float]], scores: Sequence[float]) -> None:
        if len(candidates) != self.population or len(scores) != self.population:
            raise ValueError("tell() requires exactly one score for every candidate returned by ask()")
        order = np.argsort(np.asarray(scores, dtype=np.float64))
        mu = self.population // 2
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = raw_weights / raw_weights.sum()
        encoded = np.stack([self._encode(self._from_dict(candidates[index])) for index in order[:mu]])
        old_mean = self.mean.copy()
        self.mean = np.sum(encoded * weights[:, None], axis=0)
        centered = encoded - old_mean
        variance = np.sum(np.square(centered) * weights[:, None], axis=0)
        self.diag = np.sqrt(np.maximum(1e-4, 0.8 * np.square(self.diag) + 0.2 * variance / (self.sigma**2)))
        success = float(np.mean(np.asarray(scores)[order[:mu]])) < float(np.mean(scores))
        self.sigma = float(np.clip(self.sigma * (1.03 if success else 0.97), 0.025, 0.5))
        self._generation += 1
        winner = int(order[0])
        if self._best is None or scores[winner] < self._best[1]:
            self._best = (dict(candidates[winner]), float(scores[winner]))

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best(self) -> tuple[dict[str, float], float] | None:
        return self._best

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
            "diag": self.diag.tolist(),
            "generation": self._generation,
            "best": self._best,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.sigma = float(state["sigma"])
        self.diag = np.asarray(state["diag"], dtype=np.float64)
        self._generation = int(state["generation"])
        best = state.get("best")
        self._best = (dict(best[0]), float(best[1])) if best else None
        self.rng.bit_generator.state = state["rng_state"]

    def _encode(self, values: np.ndarray) -> np.ndarray:
        return (values - self.lower) / (self.upper - self.lower)

    def _decode(self, values: np.ndarray) -> np.ndarray:
        return self.lower + values * (self.upper - self.lower)

    def _as_dict(self, values: np.ndarray) -> dict[str, float]:
        return {spec.name: float(value) for spec, value in zip(self.parameters, values)}

    def _from_dict(self, values: dict[str, float]) -> np.ndarray:
        return np.array([values[spec.name] for spec in self.parameters], dtype=np.float64)
