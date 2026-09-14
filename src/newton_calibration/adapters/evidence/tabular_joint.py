from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from newton_calibration.core.evidence_spec import BoundEvidenceSpec, EpisodeFile, LongFormSchema
from newton_calibration.core.io import sha256_file

_TIME_FACTORS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
_REQUIRED_SIGNALS = ("command_q", "actual_q", "actual_dq")


@dataclass(frozen=True)
class ArticulationEpisode:
    name: str
    split: str
    source_path: str
    time_s: np.ndarray
    command_q: np.ndarray
    actual_q: np.ndarray
    actual_dq: np.ndarray
    joints: list[str]
    trial_id: str
    source_sha256: str


@dataclass(frozen=True)
class TabularEvidenceInspection:
    """Read-only inventory used before real joints are bound to USD joints."""

    root: str
    episodes: tuple[EpisodeFile, ...]
    schema: LongFormSchema
    source_joints: tuple[str, ...]
    source_signals: tuple[str, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "episodes": [item.to_dict() for item in self.episodes],
            "schema": self.schema.to_dict(),
            "source_joints": list(self.source_joints),
            "source_signals": list(self.source_signals),
            "fingerprint": self.fingerprint,
        }


def inspect_tabular_evidence(
    *,
    root: str | Path,
    episodes: Sequence[Mapping[str, str]],
    schema: LongFormSchema,
) -> TabularEvidenceInspection:
    """Inventory unbound evidence while freezing an explicit split manifest.

    This is the agent-facing step before joint-name proposals.  It does not
    infer coordinate units, signs, scales, or zero offsets.
    """

    evidence_root = Path(root).expanduser().resolve()
    if not evidence_root.is_dir():
        raise FileNotFoundError(f"evidence root does not exist or is not a directory: {evidence_root}")
    declared = tuple(
        EpisodeFile.from_path(
            evidence_root,
            item["path"],
            name=item["name"],
            split=item["split"],
            trial_id=item["trial_id"],
            format=item.get("format"),
        )
        for item in episodes
    )
    if not declared:
        raise ValueError("at least one evidence episode must be declared")
    names = [item.name for item in declared]
    if len(set(names)) != len(names):
        raise ValueError("episode names must be unique")
    joints: set[str] = set()
    signals: set[str] = set()
    for episode in declared:
        frame = _read_normalized_frame(evidence_root / episode.path, episode, schema)
        joints.update(frame["_joint"].unique().tolist())
        signals.update(frame["_source_signal"].unique().tolist())
    identity = {
        "episodes": [item.to_dict() for item in declared],
        "schema": schema.to_dict(),
        "source_joints": sorted(joints),
        "source_signals": sorted(signals),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return TabularEvidenceInspection(
        root=str(evidence_root),
        episodes=declared,
        schema=schema,
        source_joints=tuple(sorted(joints)),
        source_signals=tuple(sorted(signals)),
        fingerprint=fingerprint,
    )


class TabularJointEvidence:
    """Load arbitrary-joint CSV/Parquet evidence through a locked binding spec."""

    def __init__(self, spec: BoundEvidenceSpec | str | Path, *, verify_files: bool = True):
        self.spec = BoundEvidenceSpec.read(spec) if isinstance(spec, (str, Path)) else spec
        if not isinstance(self.spec, BoundEvidenceSpec):
            raise TypeError("spec must be a BoundEvidenceSpec or a path to one")
        self.spec.assert_ready()
        self.uri = self.spec.root
        self.revision = self.spec.revision
        self.root = Path(self.spec.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"evidence root does not exist: {self.root}")
        self._by_name = {item.name: item for item in self.spec.episodes}
        if verify_files:
            self.verify_files()

    def verify_files(self) -> None:
        for episode in self.spec.episodes:
            path = self._path(episode)
            if not path.is_file():
                raise FileNotFoundError(f"bound evidence episode does not exist: {path}")
            if path.stat().st_size != episode.size_bytes:
                raise RuntimeError(f"bound evidence episode size changed after binding: {episode.name}")
            if sha256_file(path) != episode.sha256:
                raise RuntimeError(f"bound evidence episode content changed after binding: {episode.name}")

    def inventory(self) -> dict[str, Any]:
        source_joint_union: set[str] = set()
        source_signal_union: set[str] = set()
        missing_by_episode: dict[str, list[str]] = {}
        rates: dict[str, list[float]] = {signal: [] for signal in _REQUIRED_SIGNALS}
        signal_map = {item.source_signal: item.canonical_signal for item in self.spec.signal_bindings}
        required_pairs = {
            (binding.source_joint, source_signal)
            for binding in self.spec.joint_bindings
            for source_signal in signal_map
        }

        for episode in self.spec.episodes:
            frame = self._normalized_frame(episode)
            source_joint_union.update(frame["_joint"].unique().tolist())
            source_signal_union.update(frame["_source_signal"].unique().tolist())
            present = set(zip(frame["_joint"], frame["_source_signal"]))
            missing = sorted(f"{joint}/{signal}" for joint, signal in required_pairs - present)
            if missing:
                missing_by_episode[episode.name] = missing
            for source_signal, canonical_signal in signal_map.items():
                rows = frame[frame["_source_signal"] == source_signal]
                for _, joint_rows in rows.groupby("_joint", sort=False):
                    times = np.sort(joint_rows["_time_s"].drop_duplicates().to_numpy(dtype=np.float64))
                    deltas = np.diff(times)
                    deltas = deltas[deltas > 0.0]
                    if len(deltas):
                        rates[canonical_signal].append(float(1.0 / np.median(deltas)))

        diagnostics = self._training_diagnostics(signal_map)

        return {
            "uri": self.uri,
            "revision": self.revision,
            "fingerprint": self.spec.fingerprint,
            "mapping_fingerprint": self.spec.mapping_fingerprint,
            "bound_spec": self.spec.to_dict(),
            "files": [str(self._path(item)) for item in self.spec.episodes],
            "train_episodes": [item.name for item in self.spec.episodes if item.split == "train"],
            "heldout_episodes": [item.name for item in self.spec.episodes if item.split == "heldout"],
            "source_joints": sorted(source_joint_union),
            "joints": [item.usd_joint for item in self.spec.joint_bindings],
            "source_signals": sorted(source_signal_union),
            "signals": list(_REQUIRED_SIGNALS),
            "sample_rates_hz": {signal: float(np.median(values)) for signal, values in rates.items() if values},
            "missing_by_episode": missing_by_episode,
            "required_signals_present": not missing_by_episode,
            "clock_synchronized": self.spec.clock_synchronized,
            "dynamic_excitation_joints": diagnostics["dynamic_excitation_joints"],
            "reversal_joints": diagnostics["reversal_joints"],
            "effort_saturation_joints": list(self.spec.effort_saturation_joints),
            "evidence_quality_by_joint": diagnostics["evidence_quality_by_joint"],
            "ready": not missing_by_episode,
        }

    def _training_diagnostics(self, signal_map: Mapping[str, str]) -> dict[str, Any]:
        """Screen training data for the motions required by this recipe.

        These deterministic checks qualify evidence for parameter inclusion;
        they are deliberately not presented as a numerical identifiability
        proof. Held-out episodes are never inspected here.
        """

        training_frames: list[pd.DataFrame] = []
        for source in self.spec.episodes:
            if source.split != "train":
                continue
            frame = self._normalized_frame(source)
            frame["_signal"] = frame["_source_signal"].map(signal_map)
            training_frames.append(frame)

        by_joint: dict[str, dict[str, float | bool]] = {}
        dynamic: set[str] = set()
        reversal: set[str] = set()
        for binding in self.spec.joint_bindings:
            minimum_excursion, minimum_velocity = _diagnostic_thresholds(binding.usd_unit)
            maximum_command_excursion = 0.0
            maximum_actual_excursion = 0.0
            maximum_abs_velocity = 0.0
            has_dynamic_episode = False
            has_reversal_episode = False
            for frame in training_frames:
                joint_rows = frame[frame["_joint"] == binding.source_joint].copy()
                values: dict[str, np.ndarray] = {}
                for signal in _REQUIRED_SIGNALS:
                    raw = joint_rows.loc[joint_rows["_signal"] == signal, "_value"].to_numpy(dtype=np.float64)
                    if not len(raw):
                        continue
                    values[signal] = binding.apply_velocity(raw) if signal == "actual_dq" else binding.apply_position(raw)
                if set(values) != set(_REQUIRED_SIGNALS):
                    continue
                command_excursion = float(np.ptp(values["command_q"]))
                actual_excursion = float(np.ptp(values["actual_q"]))
                velocity = values["actual_dq"]
                peak_velocity = float(np.max(np.abs(velocity)))
                maximum_command_excursion = max(maximum_command_excursion, command_excursion)
                maximum_actual_excursion = max(maximum_actual_excursion, actual_excursion)
                maximum_abs_velocity = max(maximum_abs_velocity, peak_velocity)
                has_dynamic_episode |= (
                    command_excursion >= minimum_excursion
                    and actual_excursion >= minimum_excursion
                    and peak_velocity >= minimum_velocity
                )
                has_reversal_episode |= (
                    float(np.max(velocity)) >= minimum_velocity
                    and float(np.min(velocity)) <= -minimum_velocity
                )
            if has_dynamic_episode:
                dynamic.add(binding.source_joint)
            if has_reversal_episode:
                reversal.add(binding.source_joint)
            by_joint[binding.source_joint] = {
                "dynamic_excitation": has_dynamic_episode,
                "bidirectional_reversal": has_reversal_episode,
                "maximum_command_excursion": maximum_command_excursion,
                "maximum_actual_excursion": maximum_actual_excursion,
                "maximum_abs_velocity": maximum_abs_velocity,
                "minimum_excursion_threshold": minimum_excursion,
                "minimum_velocity_threshold": minimum_velocity,
            }
        return {
            "dynamic_excitation_joints": sorted(dynamic),
            "reversal_joints": sorted(reversal),
            "evidence_quality_by_joint": by_joint,
        }

    def load_episode(
        self,
        name: str,
        *,
        dt: float,
        command_delay_s: float = 0.0,
        max_duration_s: float | None = None,
    ) -> ArticulationEpisode:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and greater than zero")
        if not np.isfinite(command_delay_s) or command_delay_s < 0.0:
            raise ValueError("command_delay_s must be finite and non-negative")
        if max_duration_s is not None and (not np.isfinite(max_duration_s) or max_duration_s <= 0.0):
            raise ValueError("max_duration_s must be finite and greater than zero")

        try:
            source = self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"episode {name!r} is not declared in the bound split manifest") from exc
        frame = self._normalized_frame(source)
        signal_map = {item.source_signal: item.canonical_signal for item in self.spec.signal_bindings}
        frame["_signal"] = frame["_source_signal"].map(signal_map)

        series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        starts: list[float] = []
        ends: list[float] = []
        for binding in self.spec.joint_bindings:
            for canonical_signal in _REQUIRED_SIGNALS:
                rows = frame[(frame["_joint"] == binding.source_joint) & (frame["_signal"] == canonical_signal)]
                if rows.empty:
                    raise ValueError(
                        f"episode {source.name!r} is missing required signal {binding.source_joint}/{canonical_signal}"
                    )
                times, values = self._deduplicate(rows, source.name, binding.source_joint, canonical_signal)
                series[(binding.source_joint, canonical_signal)] = (times, values)
                starts.append(float(times[0]))
                ends.append(float(times[-1]))

        start_s, end_s = max(starts), min(ends)
        if end_s <= start_s:
            raise ValueError(f"episode {source.name!r} has no common time range across required signals")
        if max_duration_s is not None and end_s - start_s > max_duration_s:
            start_s, end_s = self._most_active_window(series, start_s, end_s, max_duration_s)
        time_s = np.arange(0.0, end_s - start_s, dt, dtype=np.float64)
        if not len(time_s):
            raise ValueError(f"episode {source.name!r} is shorter than the requested timestep {dt}")
        grid_s = start_s + time_s
        command_grid_s = grid_s - command_delay_s

        count = len(self.spec.joint_bindings)
        command = np.empty((len(time_s), count), dtype=np.float64)
        actual = np.empty_like(command)
        velocity = np.empty_like(command)
        for index, binding in enumerate(self.spec.joint_bindings):
            cmd_t, cmd_v = series[(binding.source_joint, "command_q")]
            q_t, q_v = series[(binding.source_joint, "actual_q")]
            dq_t, dq_v = series[(binding.source_joint, "actual_dq")]
            command[:, index] = binding.apply_position(self._zoh(cmd_t, cmd_v, command_grid_s))
            actual[:, index] = binding.apply_position(np.interp(grid_s, q_t, q_v))
            velocity[:, index] = binding.apply_velocity(np.interp(grid_s, dq_t, dq_v))

        return ArticulationEpisode(
            name=source.name,
            split=source.split,
            source_path=str(self._path(source)),
            time_s=time_s,
            command_q=command,
            actual_q=actual,
            actual_dq=velocity,
            joints=[item.usd_joint for item in self.spec.joint_bindings],
            trial_id=source.trial_id,
            source_sha256=source.sha256,
        )

    def _normalized_frame(self, source: EpisodeFile) -> pd.DataFrame:
        return _read_normalized_frame(self._path(source), source, self.spec.schema)

    @staticmethod
    def _deduplicate(
        rows: pd.DataFrame,
        episode: str,
        joint: str,
        signal: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        ordered = rows.sort_values("_time_s", kind="stable")
        grouped = ordered.groupby("_time_s", sort=True)["_value"]
        conflicts = grouped.agg(lambda values: values.nunique(dropna=False)).gt(1)
        if conflicts.any():
            first = float(conflicts[conflicts].index[0])
            raise ValueError(
                f"episode {episode!r} has conflicting duplicate samples for {joint}/{signal} at {first:g}s"
            )
        deduplicated = grouped.first()
        times = deduplicated.index.to_numpy(dtype=np.float64)
        values = deduplicated.to_numpy(dtype=np.float64)
        if not len(times):
            raise ValueError(f"episode {episode!r} has no usable samples for {joint}/{signal}")
        return times, values

    def _path(self, episode: EpisodeFile) -> Path:
        path = (self.root / episode.path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:  # defensive against malformed objects bypassing validation
            raise ValueError(f"bound episode path escapes evidence root: {episode.path}") from exc
        return path

    @staticmethod
    def _zoh(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(times, grid, side="right") - 1
        indices = np.clip(indices, 0, len(values) - 1)
        return values[indices]

    @staticmethod
    def _most_active_window(
        series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
        start_s: float,
        end_s: float,
        duration_s: float,
    ) -> tuple[float, float]:
        coarse_dt = min(0.1, duration_s / 20.0)
        coarse = np.arange(start_s, end_s, coarse_dt, dtype=np.float64)
        if len(coarse) < 3:
            return start_s, end_s
        energy = np.zeros(len(coarse), dtype=np.float64)
        source_joints = dict.fromkeys(joint for joint, signal in series if signal == "command_q")
        for joint in source_joints:
            times, values = series[(joint, "command_q")]
            sampled = TabularJointEvidence._zoh(times, values, coarse)
            energy += np.abs(np.gradient(sampled))
        window = max(2, min(len(coarse), int(duration_s / coarse_dt)))
        score = np.convolve(energy, np.ones(window), mode="valid")
        chosen_start = float(coarse[int(np.argmax(score))])
        return chosen_start, min(end_s, chosen_start + duration_s)


def _read_normalized_frame(path: Path, source: EpisodeFile, schema: LongFormSchema) -> pd.DataFrame:
    if source.format == "parquet":
        frame = pd.read_parquet(path, columns=list(schema.columns))
    elif source.format == "csv":
        frame = pd.read_csv(path, usecols=list(schema.columns))
    else:  # guarded by EpisodeFile, kept defensive for serialized input
        raise ValueError(f"unsupported evidence format {source.format!r}")
    if frame.empty:
        raise ValueError(f"episode {source.name!r} contains no rows")
    if schema.joint_column and schema.signal_column:
        joints = frame[schema.joint_column]
        signals = frame[schema.signal_column]
    else:
        fields = frame[str(schema.field_column)].astype(str)
        parsed = fields.str.rsplit(schema.field_separator, n=1, expand=True)
        if parsed.shape[1] != 2 or parsed.isna().any(axis=None):
            raise ValueError(
                f"episode {source.name!r} has fields that cannot be split into joint/signal using "
                f"{schema.field_separator!r}"
            )
        joints, signals = parsed[0], parsed[1]
    result = pd.DataFrame(
        {
            "_time_s": pd.to_numeric(frame[schema.time_column], errors="coerce") * _TIME_FACTORS[schema.time_unit],
            "_joint": joints.astype(str),
            "_source_signal": signals.astype(str),
            "_value": pd.to_numeric(frame[schema.value_column], errors="coerce"),
        }
    )
    if not np.isfinite(result["_time_s"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"episode {source.name!r} contains non-finite or non-numeric timestamps")
    if not np.isfinite(result["_value"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"episode {source.name!r} contains non-finite or non-numeric values")
    return result


def _diagnostic_thresholds(unit: str) -> tuple[float, float]:
    """Return conservative motion/velocity floors in the bound USD unit."""

    excursion_by_unit = {
        "rad": 1e-3,
        "deg": 0.05729577951308232,
        "turn": 1.5915494309189535e-4,
        "m": 1e-4,
        "cm": 1e-2,
        "mm": 1e-1,
    }
    threshold = excursion_by_unit[unit]
    return threshold, threshold
