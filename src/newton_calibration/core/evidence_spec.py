from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json
from .joint_mapping import JointBinding

_SPLITS = {"train", "heldout"}
_FORMATS = {"csv", "parquet"}
_TIME_UNITS = {"s", "ms", "us", "ns"}
_CANONICAL_SIGNALS = {"command_q", "actual_q", "actual_dq"}


@dataclass(frozen=True)
class EpisodeFile:
    """One immutable evidence episode with an explicit scientific split."""

    name: str
    path: str
    split: str
    format: str
    sha256: str
    size_bytes: int
    trial_id: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("episode name must be non-empty")
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("episode trial_id must be a non-empty capture identifier")
        relative = Path(self.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("episode path must be relative to the evidence root and may not contain '..'")
        if self.split not in _SPLITS:
            raise ValueError(f"episode {self.name!r} has invalid split {self.split!r}; use train or heldout")
        if self.format not in _FORMATS:
            raise ValueError(f"episode {self.name!r} has unsupported format {self.format!r}")
        if not _is_sha256(self.sha256):
            raise ValueError(f"episode {self.name!r} sha256 must be a 64-character lowercase hex digest")
        if self.size_bytes < 0:
            raise ValueError("episode size_bytes must be non-negative")

    @classmethod
    def from_path(
        cls,
        root: str | Path,
        path: str | Path,
        *,
        name: str,
        split: str,
        trial_id: str,
        format: str | None = None,
    ) -> EpisodeFile:
        evidence_root = Path(root).expanduser().resolve()
        source = Path(path)
        source = source.resolve() if source.is_absolute() else (evidence_root / source).resolve()
        try:
            relative = source.relative_to(evidence_root)
        except ValueError as exc:
            raise ValueError(f"episode file {source} is outside evidence root {evidence_root}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"episode file does not exist: {source}")
        selected_format = (format or source.suffix.lstrip(".")).casefold()
        return cls(
            name=name,
            path=relative.as_posix(),
            split=split,
            format=selected_format,
            sha256=sha256_file(source),
            size_bytes=source.stat().st_size,
            trial_id=trial_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "split": self.split,
            "format": self.format,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "trial_id": self.trial_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EpisodeFile:
        return cls(
            name=str(value["name"]),
            path=str(value["path"]),
            split=str(value["split"]),
            format=str(value["format"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            trial_id=str(value["trial_id"]),
        )


@dataclass(frozen=True)
class LongFormSchema:
    """Column contract for generic long-form joint telemetry."""

    time_column: str = "time_ns"
    time_unit: str = "ns"
    value_column: str = "value"
    joint_column: str | None = None
    signal_column: str | None = None
    field_column: str | None = "field"
    field_separator: str = "/"

    def __post_init__(self) -> None:
        if self.time_unit not in _TIME_UNITS:
            raise ValueError(f"unsupported timestamp unit {self.time_unit!r}; use one of {sorted(_TIME_UNITS)}")
        for label, value in (("time_column", self.time_column), ("value_column", self.value_column)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        separate = self.joint_column is not None or self.signal_column is not None
        if separate and not (self.joint_column and self.signal_column):
            raise ValueError("joint_column and signal_column must be supplied together")
        if not separate and not self.field_column:
            raise ValueError("supply either joint/signal columns or a combined field column")
        if self.field_column and not self.field_separator:
            raise ValueError("field_separator must be non-empty for a combined field column")

    @property
    def columns(self) -> tuple[str, ...]:
        result = [self.time_column, self.value_column]
        if self.joint_column and self.signal_column:
            result.extend((self.joint_column, self.signal_column))
        else:
            result.append(str(self.field_column))
        return tuple(dict.fromkeys(result))

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_column": self.time_column,
            "time_unit": self.time_unit,
            "value_column": self.value_column,
            "joint_column": self.joint_column,
            "signal_column": self.signal_column,
            "field_column": self.field_column,
            "field_separator": self.field_separator,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LongFormSchema:
        return cls(
            time_column=str(value.get("time_column", "time_ns")),
            time_unit=str(value.get("time_unit", "ns")),
            value_column=str(value.get("value_column", "value")),
            joint_column=_optional_text(value.get("joint_column")),
            signal_column=_optional_text(value.get("signal_column")),
            field_column=_optional_text(value.get("field_column", "field")),
            field_separator=str(value.get("field_separator", "/")),
        )


@dataclass(frozen=True)
class SignalBinding:
    source_signal: str
    canonical_signal: str

    def __post_init__(self) -> None:
        if not self.source_signal.strip():
            raise ValueError("source_signal must be non-empty")
        if self.canonical_signal not in _CANONICAL_SIGNALS:
            raise ValueError(
                f"canonical signal {self.canonical_signal!r} is unsupported; use {sorted(_CANONICAL_SIGNALS)}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"source_signal": self.source_signal, "canonical_signal": self.canonical_signal}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SignalBinding:
        return cls(str(value["source_signal"]), str(value["canonical_signal"]))


@dataclass(frozen=True)
class EvidenceReadiness:
    ready: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class BoundEvidenceSpec:
    """Portable, serialized contract binding real evidence to USD coordinates."""

    root: str
    revision: str
    episodes: tuple[EpisodeFile, ...]
    schema: LongFormSchema
    joint_bindings: tuple[JointBinding, ...]
    signal_bindings: tuple[SignalBinding, ...]
    # Delay can only be estimated when command and state timestamps share a
    # clock, or when their offset has been independently characterized.  The
    # safe default is false for older manifests that did not record this fact.
    clock_synchronized: bool = False
    # Position-only evidence cannot prove that an effort limit was reached.
    # Callers may explicitly record which source coordinates have independent
    # saturation evidence (for example, a controller saturation flag or
    # measured commanded/limited effort).  This declaration is locked into the
    # evidence fingerprint and is never inferred from tracking error.
    effort_saturation_joints: tuple[str, ...] = ()
    adapter: str = "tabular_joint.v1"
    schema_version: str = "newton.calibration/bound-evidence@2"

    def __post_init__(self) -> None:
        if self.adapter != "tabular_joint.v1":
            raise ValueError(f"unsupported evidence adapter {self.adapter!r}")
        if self.schema_version != "newton.calibration/bound-evidence@2":
            raise ValueError(f"unsupported bound evidence schema {self.schema_version!r}")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("evidence revision must be non-empty")
        if not isinstance(self.clock_synchronized, bool):
            raise TypeError("clock_synchronized must be a boolean")
        if any(not isinstance(name, str) or not name.strip() for name in self.effort_saturation_joints):
            raise ValueError("effort_saturation_joints must contain non-empty source-joint names")
        if len(set(self.effort_saturation_joints)) != len(self.effort_saturation_joints):
            raise ValueError("effort_saturation_joints must not contain duplicates")

    @property
    def readiness(self) -> EvidenceReadiness:
        blockers: list[str] = []
        if not self.episodes:
            blockers.append("no evidence episodes are declared")
        episode_names = [item.name for item in self.episodes]
        if len(set(episode_names)) != len(episode_names):
            blockers.append("episode names are not unique")
        for split in sorted(_SPLITS):
            if not any(item.split == split for item in self.episodes):
                blockers.append(f"no {split} episode is declared")
        train_hashes = {item.sha256 for item in self.episodes if item.split == "train"}
        heldout_hashes = {item.sha256 for item in self.episodes if item.split == "heldout"}
        leaked_hashes = sorted(train_hashes & heldout_hashes)
        if leaked_hashes:
            blockers.append(
                "train and heldout splits contain identical episode bytes: " + ", ".join(leaked_hashes)
            )
        train_trials = {item.trial_id for item in self.episodes if item.split == "train"}
        heldout_trials = {item.trial_id for item in self.episodes if item.split == "heldout"}
        leaked_trials = sorted(train_trials & heldout_trials)
        if leaked_trials:
            blockers.append(
                "train and heldout splits reference the same real capture/trial: " + ", ".join(leaked_trials)
            )
        if not self.joint_bindings:
            blockers.append("no explicit source-to-USD joint bindings are declared")
        source_joints = [item.source_joint for item in self.joint_bindings]
        usd_joints = [item.usd_joint for item in self.joint_bindings]
        if len(set(source_joints)) != len(source_joints):
            blockers.append("source joints are not unique in joint bindings")
        if len(set(usd_joints)) != len(usd_joints):
            blockers.append("USD joints are not unique in joint bindings")
        unconfirmed = sorted(item.source_joint for item in self.joint_bindings if not item.transform_confirmed)
        if unconfirmed:
            blockers.append(
                "joint direction/unit/scale/offset transforms are not explicitly confirmed: "
                + ", ".join(unconfirmed)
            )
        unknown_saturation_joints = sorted(set(self.effort_saturation_joints) - set(source_joints))
        if unknown_saturation_joints:
            blockers.append(
                "effort saturation is declared for unbound source joints: "
                + ", ".join(unknown_saturation_joints)
            )
        source_signals = [item.source_signal for item in self.signal_bindings]
        canonical_signals = [item.canonical_signal for item in self.signal_bindings]
        if len(set(source_signals)) != len(source_signals):
            blockers.append("source signals are not unique in signal bindings")
        if len(set(canonical_signals)) != len(canonical_signals):
            blockers.append("canonical signals are not unique in signal bindings")
        missing = sorted(_CANONICAL_SIGNALS - set(canonical_signals))
        if missing:
            blockers.append(f"canonical signal bindings are missing: {missing}")
        return EvidenceReadiness(not blockers, tuple(blockers))

    def assert_ready(self) -> None:
        result = self.readiness
        if not result.ready:
            raise ValueError("bound evidence is not ready: " + "; ".join(result.blockers))

    @property
    def fingerprint(self) -> str:
        """Hash data identities plus all interpretation choices, excluding locator root."""

        payload = self.to_dict()
        payload.pop("root")
        return _canonical_sha256(payload)

    @property
    def mapping_fingerprint(self) -> str:
        return _canonical_sha256([item.to_dict() for item in self.joint_bindings])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter": self.adapter,
            "root": self.root,
            "revision": self.revision,
            "episodes": [item.to_dict() for item in self.episodes],
            "schema": self.schema.to_dict(),
            "joint_bindings": [item.to_dict() for item in self.joint_bindings],
            "signal_bindings": [item.to_dict() for item in self.signal_bindings],
            "clock_synchronized": self.clock_synchronized,
            "effort_saturation_joints": list(self.effort_saturation_joints),
        }

    def write(self, path: str | Path) -> Path:
        self.assert_ready()
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint
        payload["mapping_fingerprint"] = self.mapping_fingerprint
        return write_json(path, payload)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BoundEvidenceSpec:
        schema_version = str(value.get("schema_version", "newton.calibration/bound-evidence@1"))
        if schema_version == "newton.calibration/bound-evidence@1":
            raise ValueError(
                "bound-evidence@1 cannot be activated safely: it did not require independent trial IDs or "
                "explicit coordinate-transform confirmation; recreate and confirm the binding as bound-evidence@2"
            )
        result = cls(
            root=str(value["root"]),
            revision=str(value["revision"]),
            episodes=tuple(EpisodeFile.from_dict(item) for item in _sequence(value["episodes"], "episodes")),
            schema=LongFormSchema.from_dict(_mapping(value["schema"], "schema")),
            joint_bindings=tuple(
                JointBinding.from_dict(_mapping(item, "joint binding"))
                for item in _sequence(value["joint_bindings"], "joint_bindings")
            ),
            signal_bindings=tuple(
                SignalBinding.from_dict(_mapping(item, "signal binding"))
                for item in _sequence(value["signal_bindings"], "signal_bindings")
            ),
            clock_synchronized=value.get("clock_synchronized", False),
            effort_saturation_joints=tuple(
                str(item)
                for item in _sequence(value.get("effort_saturation_joints", ()), "effort_saturation_joints")
            ),
            adapter=str(value.get("adapter", "tabular_joint.v1")),
            schema_version=schema_version,
        )
        claimed = value.get("fingerprint")
        if claimed is not None and claimed != result.fingerprint:
            raise ValueError("serialized bound evidence fingerprint does not match its contents")
        claimed_mapping = value.get("mapping_fingerprint")
        if claimed_mapping is not None and claimed_mapping != result.mapping_fingerprint:
            raise ValueError("serialized joint mapping fingerprint does not match its contents")
        result.assert_ready()
        return result

    @classmethod
    def read(cls, path: str | Path) -> BoundEvidenceSpec:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return cls.from_dict(_mapping(value, "bound evidence document"))


def bind_evidence_files(
    *,
    root: str | Path,
    episodes: Sequence[Mapping[str, str]],
    schema: LongFormSchema,
    joint_bindings: Sequence[JointBinding],
    signal_bindings: Sequence[SignalBinding],
    revision: str = "local",
    clock_synchronized: bool = False,
    effort_saturation_joints: Sequence[str] = (),
) -> BoundEvidenceSpec:
    """Freeze an explicit episode manifest and its current content digests."""

    evidence_root = Path(root).expanduser().resolve()
    if not evidence_root.is_dir():
        raise FileNotFoundError(f"evidence root does not exist or is not a directory: {evidence_root}")
    episode_files = tuple(
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
    result = BoundEvidenceSpec(
        root=str(evidence_root),
        revision=revision,
        episodes=episode_files,
        schema=schema,
        joint_bindings=tuple(joint_bindings),
        signal_bindings=tuple(signal_bindings),
        clock_synchronized=clock_synchronized,
        effort_saturation_joints=tuple(effort_saturation_joints),
    )
    result.assert_ready()
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an array")
    return value
