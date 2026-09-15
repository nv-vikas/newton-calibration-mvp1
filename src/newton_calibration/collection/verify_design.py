"""Recompute sensitivity scores from recorded simulation traces; not a hardware test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from newton_calibration.core.io import sha256_file, write_json


def verify_design(directory: str | Path) -> dict:
    root = Path(directory)
    search = json.loads((root / "design_search.json").read_text())
    checked = []
    names = [row["parameter"] for row in search["coverage"]]
    aggregate = np.zeros((2, len(names), len(names)))
    for row in search["history"]:
        if "probe_record" not in row:
            continue
        record_path = root / row["probe_record"]
        if sha256_file(record_path) != row["probe_sha256"]:
            raise ValueError("Sensitivity record fingerprint changed")
        record = json.loads(record_path.read_text())
        metadata = record["metadata"]
        command = root / "design_candidates" / "commands" / f"{row['candidate']}.csv"
        predictions = command.with_suffix(".predictions.npz")
        if sha256_file(command) != row["command_sha256"] or sha256_file(command) != metadata["command_sha256"]:
            raise ValueError("Sensitivity input commands changed")
        if sha256_file(predictions) != metadata["simulated_predictions_sha256"]:
            raise ValueError("Sensitivity predictions changed")
        matrices = []
        with np.load(predictions, allow_pickle=False) as traces:
            for anchor in range(2):
                base = traces[f"anchor_{anchor}_baseline"]
                width = base.shape[1] // 2
                noise = np.array(
                    [metadata["assumed_position_noise_rad"]] * width
                    + [metadata["assumed_velocity_noise_rad_s"]] * width
                )
                columns = [
                    ((traces[f"anchor_{anchor}_{name}"] - base) / noise / metadata["perturbation_fraction"]).reshape(-1)
                    for name in record["parameters"]
                ]
                jacobian = np.column_stack(columns)
                matrices.append(jacobian.T @ jacobian / len(base))
                if anchor == 0:
                    repeat = float(np.max(abs(traces["anchor_0_repeat"] - base) / noise))
                    if repeat > 0.1 or not np.isclose(
                        repeat, metadata["repeatability_error_noise_units"], rtol=1e-6, atol=1e-9
                    ):
                        raise ValueError("Sensitivity repeatability record mismatch")
        if not np.allclose(matrices, record["information"], rtol=1e-7, atol=1e-9):
            raise ValueError("Recorded information matrix does not match simulated traces")
        indices = [names.index(name) for name in record["parameters"]]
        addition = np.zeros_like(aggregate)
        for anchor in range(2):
            addition[anchor][np.ix_(indices, indices)] = matrices[anchor]

        def score(value):
            return min(float(np.linalg.slogdet(np.eye(len(names)) + matrix)[1]) for matrix in value)

        gain = score(aggregate + addition) - score(aggregate)
        if not np.isclose(gain, row["information_gain"], rtol=1e-6, atol=1e-8):
            raise ValueError("Recorded candidate information gain changed")
        should_select = gain >= search["thresholds"]["minimum_information_gain"]
        if should_select != (row["decision"] == "selected"):
            raise ValueError("Candidate selection does not match the declared information threshold")
        selected = root / "commands" / command.name
        if row["decision"] == "selected" and sha256_file(selected) != row["command_sha256"]:
            raise ValueError("Selected commands differ from the tested candidate")
        if should_select:
            aggregate += addition
        checked.append(row["candidate"])
    if not checked:
        raise ValueError("No recorded sensitivity traces to verify")
    result = {
        "verified_probe_count": len(checked),
        "verified_candidates": checked,
        "physics": search["physics"],
        "real_evidence_validated": False,
        "check": "Recomputed matrices and selection gains from saved simulation q/dq; checked command/record hashes and baseline repeats",
    }
    write_json(root / "design_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(verify_design(parser.parse_args().directory), indent=2))
