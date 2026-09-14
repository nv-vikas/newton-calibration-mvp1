"""Print compact joint ranges for one Anchor-Lab episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from newton_calibration.adapters.evidence import AnchorLabSO101Evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    parser.add_argument("--normalized-duration", type=float)
    args = parser.parse_args()
    episode_path = Path(args.episode)
    frame = pd.read_parquet(episode_path)
    for signal in ("actual_q", "command_q", "dq"):
        rows = frame[frame["field"].str.endswith(f"/{signal}")]
        print(f"\n{signal}")
        print(rows.groupby("field")["value"].agg(["min", "max"]).to_string())
    if args.normalized_duration:
        adapter = AnchorLabSO101Evidence(episode_path)
        episode = adapter.load_episode(
            episode_path.stem,
            dt=1.0 / 120.0,
            max_duration_s=args.normalized_duration,
        )
        print("\nnormalized active-window start")
        print("q ", episode.actual_q[0].tolist())
        print("dq", episode.actual_dq[0].tolist())
        print("command", episode.command_q[0].tolist())
        print("max abs dq", float(abs(episode.actual_dq).max()))


if __name__ == "__main__":
    main()
