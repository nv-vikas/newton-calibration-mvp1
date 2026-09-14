from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from newton_calibration.core.io import sha256_files

HF_REPO_ID = "nvidia/Anchor-Lab"
DEFAULT_REVISION = "647edd5787cd764cdc041103ad282dc59214d919"
HF_DATA_PATTERN = "data/so101_arm_50motion/*.parquet"
HF_ASSET_PATTERN = "robot_assets/so101_no_camera_new_calib.usd"

SO101_JOINTS = ["rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw"]
REQUIRED_SIGNALS = ["actual_q", "command_q", "dq"]


@dataclass(frozen=True)
class SO101Episode:
    name: str
    split: str
    source_path: str
    time_s: np.ndarray
    command_q: np.ndarray
    actual_q: np.ndarray
    actual_dq: np.ndarray
    joints: list[str]


def fetch_anchor_lab_so101(output_dir: str | Path, revision: str = DEFAULT_REVISION) -> dict[str, str]:
    """Download the SO-101 evidence and released USD using Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency message is user-facing
        raise RuntimeError("Install the 'huggingface-hub' dependency to fetch Anchor-Lab.") from exc

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(HF_REPO_ID, revision=revision)
    snapshot = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=info.sha,
        local_dir=root,
        allow_patterns=[HF_DATA_PATTERN, HF_ASSET_PATTERN, "README.md", "robot_assets/README.md"],
    )
    return {"root": str(Path(snapshot)), "revision": info.sha}


class AnchorLabSO101Evidence:
    """Read and normalize Anchor-Lab's long-form SO-101 Parquet evidence."""

    def __init__(self, uri: str | Path, revision: str = "local"):
        self.uri = str(uri)
        self.revision = revision
        self.root = Path(uri).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Evidence path does not exist: {self.root}")
        if self.root.is_file():
            self.files = [self.root]
        else:
            self.files = sorted(self.root.rglob("so101-sysid-50motion-*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No SO-101 50-motion Parquet files found under {self.root}")
        self._by_name = {path.stem: path for path in self.files}

    def inventory(self) -> dict[str, Any]:
        sample = pd.read_parquet(self.files[0], columns=["time_ns", "field"])
        parsed = sample["field"].str.split("/", n=1, expand=True)
        joints = sorted(parsed[0].dropna().unique().tolist())
        signals = sorted(parsed[1].dropna().unique().tolist())
        rates: dict[str, float] = {}
        for signal in ("actual_q", "command_q", "dq"):
            rows = sample[sample["field"].str.endswith(f"/{signal}")]
            if rows.empty:
                continue
            times = np.sort(rows["time_ns"].drop_duplicates().to_numpy(dtype=np.int64))
            delta = np.diff(times) / 1e9
            delta = delta[delta > 0]
            if len(delta):
                rates[signal] = float(1.0 / np.median(delta))
        names = sorted(self._by_name)
        train = [name for name in names if "-train-" in name]
        heldout = [name for name in names if "-heldout-" in name]
        return {
            "uri": self.uri,
            "revision": self.revision,
            "fingerprint": sha256_files(self.files),
            "files": [str(path) for path in self.files],
            "train_episodes": train,
            "heldout_episodes": heldout,
            "joints": joints,
            "signals": signals,
            "sample_rates_hz": rates,
            "required_signals_present": all(signal in signals for signal in REQUIRED_SIGNALS),
        }

    def load_episode(self, name: str, *, dt: float, max_duration_s: float | None = None) -> SO101Episode:
        path = self._resolve_name(name)
        frame = pd.read_parquet(path, columns=["time_ns", "field", "value"])
        frame[["joint", "signal"]] = frame["field"].str.split("/", n=1, expand=True)

        missing = [
            f"{joint}/{signal}"
            for joint in SO101_JOINTS
            for signal in REQUIRED_SIGNALS
            if not ((frame["joint"] == joint) & (frame["signal"] == signal)).any()
        ]
        if missing:
            raise ValueError(f"Episode {path.name} is missing required signals: {missing}")

        series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        starts: list[int] = []
        ends: list[int] = []
        for joint in SO101_JOINTS:
            for signal in REQUIRED_SIGNALS:
                rows = frame[(frame["joint"] == joint) & (frame["signal"] == signal)].sort_values("time_ns")
                times = rows["time_ns"].to_numpy(dtype=np.int64)
                values = rows["value"].to_numpy(dtype=np.float64)
                unique = np.concatenate(([True], np.diff(times) > 0))
                times, values = times[unique], values[unique]
                series[(joint, signal)] = (times, values)
                starts.append(int(times[0]))
                ends.append(int(times[-1]))

        start_ns, end_ns = max(starts), min(ends)
        if end_ns <= start_ns:
            raise ValueError(f"Episode {path.name} has no common time range across required signals")
        duration = (end_ns - start_ns) / 1e9
        if max_duration_s and duration > max_duration_s:
            start_ns, end_ns = self._most_active_window(series, start_ns, end_ns, max_duration_s)
            duration = (end_ns - start_ns) / 1e9
        time_s = np.arange(0.0, duration, dt, dtype=np.float64)
        grid_ns = start_ns + np.rint(time_s * 1e9).astype(np.int64)

        command = np.empty((len(time_s), len(SO101_JOINTS)), dtype=np.float64)
        actual = np.empty_like(command)
        velocity = np.empty_like(command)
        for index, joint in enumerate(SO101_JOINTS):
            cmd_t, cmd_v = series[(joint, "command_q")]
            q_t, q_v = series[(joint, "actual_q")]
            dq_t, dq_v = series[(joint, "dq")]
            command[:, index] = self._zoh(cmd_t, cmd_v, grid_ns)
            actual[:, index] = np.interp(grid_ns, q_t, q_v)
            velocity[:, index] = np.interp(grid_ns, dq_t, dq_v)

        split = "heldout" if "-heldout-" in path.stem else "train"
        return SO101Episode(
            name=path.stem,
            split=split,
            source_path=str(path),
            time_s=time_s,
            command_q=command,
            actual_q=actual,
            actual_dq=velocity,
            joints=list(SO101_JOINTS),
        )

    def _resolve_name(self, name: str) -> Path:
        if name in self._by_name:
            return self._by_name[name]
        matches = [path for key, path in self._by_name.items() if key.endswith(name) or name in key]
        if len(matches) != 1:
            raise KeyError(f"Episode name {name!r} resolved to {len(matches)} files")
        return matches[0]

    @staticmethod
    def _zoh(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(times, grid, side="right") - 1
        indices = np.clip(indices, 0, len(values) - 1)
        return values[indices]

    @staticmethod
    def _most_active_window(
        series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
        start_ns: int,
        end_ns: int,
        duration_s: float,
    ) -> tuple[int, int]:
        coarse_dt = 0.1
        coarse = np.arange(start_ns, end_ns, int(coarse_dt * 1e9), dtype=np.int64)
        if len(coarse) < 3:
            return start_ns, end_ns
        energy = np.zeros(len(coarse), dtype=np.float64)
        for joint in SO101_JOINTS:
            times, values = series[(joint, "command_q")]
            sampled = AnchorLabSO101Evidence._zoh(times, values, coarse)
            energy += np.abs(np.gradient(sampled))
        window = max(2, min(len(coarse), int(duration_s / coarse_dt)))
        score = np.convolve(energy, np.ones(window), mode="valid")
        offset = int(np.argmax(score))
        chosen_start = int(coarse[offset])
        chosen_end = min(end_ns, chosen_start + int(duration_s * 1e9))
        return chosen_start, chosen_end
