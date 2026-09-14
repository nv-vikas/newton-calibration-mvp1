from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_text, utc_now
from .models import jsonable

GENERATION_SCHEMA = "newton.calibration.fit-generation/v1"
CHECKPOINT_SCHEMA = "newton.calibration.fit-checkpoint/v2"
_GENERATION_NAME = re.compile(r"^(?P<generation>[0-9]{6,})\.json$")
_CHECKPOINT_RESERVED_KEYS = {
    "schema",
    "run_id",
    "execution_fingerprint",
    "candidate_id",
    "completed_generations",
    "optimizer_state",
    "updated_at",
}


class FitJournalError(RuntimeError):
    """Raised when authoritative fit records are missing, mixed, or corrupt."""


@dataclass(frozen=True)
class FitJournalState:
    """The last fully committed optimizer generation."""

    completed_generations: int
    next_candidate_id: int
    optimizer_state: dict[str, Any] | None
    generation_paths: tuple[Path, ...]


class FitJournal:
    """Crash-consistent generation journal for a single fit execution.

    A generation JSON file is the commit record.  Candidate history and the
    familiar fit checkpoint are derived conveniences and can always be rebuilt
    from the contiguous authoritative records.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        execution_fingerprint: str,
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = _nonempty_text(run_id, "run_id")
        self.execution_fingerprint = _nonempty_text(execution_fingerprint, "execution_fingerprint")
        self.checkpoint_metadata = _finite_json_mapping(checkpoint_metadata or {}, "checkpoint metadata")
        reserved = sorted(_CHECKPOINT_RESERVED_KEYS.intersection(self.checkpoint_metadata))
        if reserved:
            raise ValueError(f"checkpoint metadata uses reserved keys: {reserved}")
        self.generation_dir = self.run_dir / "fit-generations"
        self.history_path = self.run_dir / "candidate-history.jsonl"
        self.checkpoint_path = self.run_dir / "fit-checkpoint.json"
        self.lock_path = self.run_dir / ".fit-journal.lock"

    def assert_empty(self) -> None:
        """Fail when a non-resume attempt would overwrite or mix prior work."""

        with self._locked():
            occupied: list[Path] = []
            if self.generation_dir.exists() and any(self.generation_dir.iterdir()):
                occupied.append(self.generation_dir)
            for name in (
                "fit-checkpoint.json",
                "candidate-history.jsonl",
                "baseline.json",
                "fit.json",
                "validation.json",
            ):
                path = self.run_dir / name
                if path.exists():
                    occupied.append(path)
            if occupied:
                rendered = ", ".join(str(path) for path in occupied)
                raise FitJournalError(
                    f"fit attempt is not empty ({rendered}); resume it or create a new run/attempt directory"
                )

    def scan(self) -> FitJournalState:
        """Read and validate all contiguous authoritative generation records."""

        with self._locked():
            return self._scan_unlocked()

    def recover(self) -> FitJournalState:
        """Validate the journal and atomically regenerate both derived artifacts."""

        with self._locked():
            state = self._scan_unlocked()
            if state.completed_generations:
                self._refresh_derived_unlocked(state)
            return state

    def verify_snapshot(self) -> FitJournalState:
        """Validate an immutable journal and its derived records without writing.

        Package verification must not create a lock file inside the package.  Its
        caller is therefore responsible for supplying a stable snapshot (the
        package manifest hashes every file before this method is called).
        """

        state = self._scan_unlocked()
        if not state.completed_generations:
            raise FitJournalError("fit journal contains no committed generations")
        self._verify_derived_unlocked(state)
        return state

    def commit_generation(
        self,
        *,
        generation: int,
        candidate_id_start: int,
        candidates: Sequence[Mapping[str, Any]],
        evaluations: Sequence[Any],
        optimizer_state: Mapping[str, Any],
        optimizer_generation: int,
    ) -> FitJournalState:
        """Atomically commit one evaluated, post-``tell`` optimizer generation."""

        generation = _nonnegative_int(generation, "generation")
        candidate_id_start = _nonnegative_int(candidate_id_start, "candidate_id_start")
        optimizer_generation = _nonnegative_int(optimizer_generation, "optimizer_generation")
        normalized_candidates = _finite_json_sequence(candidates, "candidates")
        normalized_evaluations = _finite_json_sequence(evaluations, "evaluations")
        normalized_optimizer_state = _finite_json_mapping(optimizer_state, "optimizer_state")
        if not normalized_candidates:
            raise FitJournalError("a committed generation must contain at least one candidate")
        if len(normalized_candidates) != len(normalized_evaluations):
            raise FitJournalError("candidate and evaluation counts differ")

        with self._locked():
            current = self._scan_unlocked()
            if generation != current.completed_generations:
                raise FitJournalError(
                    f"generation {generation} is not the next contiguous generation {current.completed_generations}"
                )
            if candidate_id_start != current.next_candidate_id:
                raise FitJournalError(
                    f"candidate_id_start {candidate_id_start} does not match next id {current.next_candidate_id}"
                )
            if optimizer_generation != generation + 1:
                raise FitJournalError(
                    f"post-tell optimizer generation {optimizer_generation} must equal {generation + 1}"
                )

            candidate_id_end = candidate_id_start + len(normalized_candidates)
            _validate_evaluations(
                normalized_candidates,
                normalized_evaluations,
                generation=generation,
                candidate_id_start=candidate_id_start,
            )
            state_generation = normalized_optimizer_state.get("generation")
            if state_generation is not None and (
                isinstance(state_generation, bool)
                or not isinstance(state_generation, int)
                or state_generation != optimizer_generation
            ):
                raise FitJournalError("optimizer_state generation does not match the committed post-tell generation")

            payload = {
                "schema": GENERATION_SCHEMA,
                "run_id": self.run_id,
                "execution_fingerprint": self.execution_fingerprint,
                "checkpoint_metadata": self.checkpoint_metadata,
                "generation": generation,
                "optimizer_generation": optimizer_generation,
                "candidate_id_start": candidate_id_start,
                "candidate_id_end": candidate_id_end,
                "candidates": normalized_candidates,
                "evaluations": normalized_evaluations,
                "optimizer_state": normalized_optimizer_state,
                "committed_at": utc_now(),
            }
            record = {**payload, "record_sha256": _payload_sha256(payload)}
            path = self.generation_dir / f"{generation:06d}.json"
            if path.exists():
                raise FitJournalError(f"generation record already exists: {path}")
            atomic_write_json(path, record)

            committed = self._scan_unlocked()
            self._refresh_derived_unlocked(committed)
            return committed

    def _scan_unlocked(self) -> FitJournalState:
        paths = self._generation_paths_unlocked()
        if not paths:
            legacy = [path for path in (self.checkpoint_path, self.history_path) if path.exists()]
            if legacy:
                rendered = ", ".join(str(path) for path in legacy)
                raise FitJournalError(
                    f"legacy fit artifacts exist without authoritative generation records ({rendered}); "
                    "restart in a new run or migrate them explicitly"
                )
            return FitJournalState(0, 0, None, ())

        next_candidate_id = 0
        optimizer_state: dict[str, Any] | None = None
        for expected_generation, path in enumerate(paths):
            record = _load_finite_json_mapping(path)
            self._validate_record(
                record,
                path=path,
                expected_generation=expected_generation,
                expected_candidate_id=next_candidate_id,
            )
            next_candidate_id = int(record["candidate_id_end"])
            optimizer_state = dict(record["optimizer_state"])

        return FitJournalState(len(paths), next_candidate_id, optimizer_state, tuple(paths))

    def _generation_paths_unlocked(self) -> list[Path]:
        if not self.generation_dir.exists():
            return []
        indexed: list[tuple[int, Path]] = []
        unexpected: list[Path] = []
        for path in self.generation_dir.iterdir():
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                continue
            match = _GENERATION_NAME.fullmatch(path.name) if path.is_file() else None
            if match is None:
                unexpected.append(path)
                continue
            indexed.append((int(match.group("generation")), path))
        if unexpected:
            raise FitJournalError(f"unexpected files in generation journal: {[str(path) for path in unexpected]}")
        indexed.sort(key=lambda item: item[0])
        indices = [index for index, _ in indexed]
        expected = list(range(len(indexed)))
        if indices != expected:
            raise FitJournalError(f"generation records are not contiguous: found {indices}, expected {expected}")
        return [path for _, path in indexed]

    def _validate_record(
        self,
        record: dict[str, Any],
        *,
        path: Path,
        expected_generation: int,
        expected_candidate_id: int,
    ) -> None:
        recorded_hash = record.get("record_sha256")
        if not isinstance(recorded_hash, str):
            raise FitJournalError(f"generation record lacks record_sha256: {path}")
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        if not _constant_time_equal(recorded_hash, _payload_sha256(payload)):
            raise FitJournalError(f"generation record hash mismatch: {path}")
        if record.get("schema") != GENERATION_SCHEMA:
            raise FitJournalError(f"unsupported generation record schema in {path}")
        if record.get("run_id") != self.run_id:
            raise FitJournalError(f"generation record run_id mismatch in {path}")
        if record.get("execution_fingerprint") != self.execution_fingerprint:
            raise FitJournalError(f"generation record execution fingerprint mismatch in {path}")
        if record.get("checkpoint_metadata") != self.checkpoint_metadata:
            raise FitJournalError(f"generation record checkpoint metadata mismatch in {path}")

        generation = _record_int(record, "generation", path)
        optimizer_generation = _record_int(record, "optimizer_generation", path)
        candidate_id_start = _record_int(record, "candidate_id_start", path)
        candidate_id_end = _record_int(record, "candidate_id_end", path)
        if generation != expected_generation:
            raise FitJournalError(f"generation value mismatch in {path}: {generation} != {expected_generation}")
        if optimizer_generation != generation + 1:
            raise FitJournalError(f"optimizer generation mismatch in {path}")
        if candidate_id_start != expected_candidate_id:
            raise FitJournalError(f"candidate id range is not contiguous in {path}")
        candidates = record.get("candidates")
        evaluations = record.get("evaluations")
        if not isinstance(candidates, list) or not isinstance(evaluations, list):
            raise FitJournalError(f"candidates/evaluations must be lists in {path}")
        if not candidates or len(candidates) != len(evaluations):
            raise FitJournalError(f"candidate/evaluation counts are invalid in {path}")
        if candidate_id_end != candidate_id_start + len(candidates):
            raise FitJournalError(f"candidate id range length is invalid in {path}")
        if not isinstance(record.get("optimizer_state"), dict):
            raise FitJournalError(f"optimizer_state must be an object in {path}")
        state_generation = record["optimizer_state"].get("generation")
        if state_generation is not None and (
            isinstance(state_generation, bool)
            or not isinstance(state_generation, int)
            or state_generation != optimizer_generation
        ):
            raise FitJournalError(f"optimizer_state generation mismatch in {path}")
        _validate_evaluations(
            candidates,
            evaluations,
            generation=generation,
            candidate_id_start=candidate_id_start,
            path=path,
        )

    def _refresh_derived_unlocked(self, state: FitJournalState) -> None:
        history_text = self._expected_history_text(state)
        atomic_write_text(self.history_path, history_text)

        if state.optimizer_state is None:
            raise FitJournalError("cannot write a checkpoint without committed optimizer state")
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "run_id": self.run_id,
            "execution_fingerprint": self.execution_fingerprint,
            "candidate_id": state.next_candidate_id,
            "completed_generations": state.completed_generations,
            "optimizer_state": state.optimizer_state,
            "updated_at": utc_now(),
            **self.checkpoint_metadata,
        }
        atomic_write_json(self.checkpoint_path, checkpoint)

    def _verify_derived_unlocked(self, state: FitJournalState) -> None:
        if not self.history_path.is_file():
            raise FitJournalError(f"candidate history is missing: {self.history_path}")
        try:
            history_text = self.history_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise FitJournalError(f"cannot read candidate history: {self.history_path}") from exc
        if history_text != self._expected_history_text(state):
            raise FitJournalError("candidate history is not the canonical projection of committed generations")

        checkpoint = _load_finite_json_mapping(self.checkpoint_path)
        updated_at = checkpoint.pop("updated_at", None)
        if not isinstance(updated_at, str) or not updated_at.strip():
            raise FitJournalError("fit checkpoint updated_at must be a non-empty string")
        expected_checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "run_id": self.run_id,
            "execution_fingerprint": self.execution_fingerprint,
            "candidate_id": state.next_candidate_id,
            "completed_generations": state.completed_generations,
            "optimizer_state": state.optimizer_state,
            **self.checkpoint_metadata,
        }
        if checkpoint != expected_checkpoint:
            raise FitJournalError("fit checkpoint is not the canonical projection of committed generations")

    @staticmethod
    def _expected_history_text(state: FitJournalState) -> str:
        history_lines: list[str] = []
        for path in state.generation_paths:
            record = _load_finite_json_mapping(path)
            for evaluation in record["evaluations"]:
                history_lines.append(json.dumps(evaluation, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return "\n".join(history_lines) + ("\n" if history_lines else "")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate_evaluations(
    candidates: Sequence[Any],
    evaluations: Sequence[Any],
    *,
    generation: int,
    candidate_id_start: int,
    path: Path | None = None,
) -> None:
    location = f" in {path}" if path else ""
    for offset, (candidate, evaluation) in enumerate(zip(candidates, evaluations)):
        if not isinstance(candidate, dict) or not isinstance(evaluation, dict):
            raise FitJournalError(f"candidate/evaluation must be JSON objects{location}")
        expected_id = candidate_id_start + offset
        candidate_id = evaluation.get("candidate_id")
        evaluation_generation = evaluation.get("generation")
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int) or candidate_id != expected_id:
            raise FitJournalError(f"evaluation candidate_id is not contiguous{location}")
        if (
            isinstance(evaluation_generation, bool)
            or not isinstance(evaluation_generation, int)
            or evaluation_generation != generation
        ):
            raise FitJournalError(f"evaluation generation mismatch{location}")
        if evaluation.get("parameters") != candidate:
            raise FitJournalError(f"evaluation parameters do not match candidate{location}")


def _finite_json_mapping(value: Any, label: str) -> dict[str, Any]:
    normalized = _finite_json(value, label)
    if not isinstance(normalized, dict):
        raise FitJournalError(f"{label} must be a JSON object")
    return normalized


def _finite_json_sequence(value: Any, label: str) -> list[Any]:
    normalized = _finite_json(value, label)
    if not isinstance(normalized, list):
        raise FitJournalError(f"{label} must be a JSON array")
    return normalized


def _finite_json(value: Any, label: str) -> Any:
    try:
        encoded = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        normalized = json.loads(encoded, parse_constant=_reject_json_constant)
    except (TypeError, ValueError) as exc:
        raise FitJournalError(f"{label} is not finite JSON") from exc
    _assert_finite_tree(normalized, label)
    return normalized


def _load_finite_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FitJournalError(f"cannot read finite JSON generation record {path}") from exc
    _assert_finite_tree(value, str(path))
    if not isinstance(value, dict):
        raise FitJournalError(f"generation record must be a JSON object: {path}")
    return value


def _assert_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FitJournalError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_tree(child, label)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_tree(child, label)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _constant_time_equal(first: str, second: str) -> bool:
    return hmac.compare_digest(first, second)


def _record_int(record: Mapping[str, Any], key: str, path: Path) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FitJournalError(f"{key} must be a non-negative integer in {path}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
