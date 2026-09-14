from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .models import EnvironmentSpec


class ReplayEpisode(Protocol):
    name: str
    split: str
    time_s: np.ndarray
    command_q: np.ndarray
    actual_q: np.ndarray
    actual_dq: np.ndarray
    joints: list[str]
    trial_id: str
    source_sha256: str


class EvidenceAdapter(Protocol):
    def inventory(self) -> dict[str, Any]: ...

    def load_episode(
        self,
        name: str,
        *,
        dt: float,
        max_duration_s: float | None = None,
    ) -> ReplayEpisode: ...


class RuntimeAdapter(Protocol):
    def describe(self) -> EnvironmentSpec: ...

    def attestation(self) -> dict[str, Any]: ...

    def evaluate(
        self,
        candidate: dict[str, float],
        episodes: Sequence[ReplayEpisode],
        objective_weights: dict[str, float],
        *,
        phase: str = "unscoped",
        run_id: str = "",
        plan_sha256: str = "",
        evidence_fingerprint: str = "",
        mapping_fingerprint: str = "",
    ) -> tuple[float, dict[str, float], dict[str, dict[str, float]], bool]: ...

    def close(self) -> None: ...


@runtime_checkable
class OptimizerPlugin(Protocol):
    """Stateful ask/tell optimizer used by the deterministic fit runner.

    Optimizer state must be JSON serializable.  The runner owns candidate
    evaluation and checkpoint persistence; a plug-in only proposes candidates
    and learns from the scores returned to it.
    """

    def ask(self) -> list[dict[str, float]]: ...

    def tell(self, candidates: Sequence[dict[str, float]], scores: Sequence[float]) -> None: ...

    @property
    def generation(self) -> int: ...

    @property
    def best(self) -> tuple[dict[str, float], float] | None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
