from pathlib import Path

import numpy as np
import pandas as pd

from newton_calibration.adapters.evidence import AnchorLabSO101Evidence

JOINTS = ["rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "jaw"]


def _write_episode(path: Path, duration: float = 1.0):
    rows = []
    origin = 1_700_000_000_000_000_000
    for joint_index, joint in enumerate(JOINTS):
        for signal, rate in (("command_q", 20), ("actual_q", 100), ("dq", 100)):
            time_s = np.arange(0.0, duration, 1.0 / rate)
            command = 0.1 * np.sin(2 * np.pi * time_s) + joint_index * 0.01
            values = command if signal != "dq" else np.gradient(command, 1.0 / rate)
            for timestamp, value in zip(time_s, values):
                rows.append(
                    {
                        "time_ns": origin + int(timestamp * 1e9),
                        "time_utc": pd.Timestamp(origin + int(timestamp * 1e9), unit="ns", tz="UTC"),
                        "experiment": path.stem,
                        "field": f"{joint}/{signal}",
                        "value": value,
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_inventory_and_resampling(tmp_path):
    train = tmp_path / "so101-sysid-50motion-train-step-response.parquet"
    heldout = tmp_path / "so101-sysid-50motion-heldout-frequency-sweep.parquet"
    _write_episode(train)
    _write_episode(heldout)
    evidence = AnchorLabSO101Evidence(tmp_path)
    inventory = evidence.inventory()
    assert inventory["required_signals_present"]
    assert len(inventory["train_episodes"]) == 1
    assert len(inventory["heldout_episodes"]) == 1
    episode = evidence.load_episode(train.stem, dt=0.01)
    assert episode.command_q.shape == episode.actual_q.shape == episode.actual_dq.shape
    assert episode.command_q.shape[1] == 6
    assert episode.split == "train"
