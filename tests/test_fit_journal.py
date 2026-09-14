from __future__ import annotations

import json

import pytest

from newton_calibration.core.fit_journal import FitJournal, FitJournalError
from newton_calibration.core.io import atomic_write_json


def _candidates(start: float, count: int = 2) -> list[dict[str, float]]:
    return [{"x": start + index} for index in range(count)]


def _evaluations(candidates, *, generation: int, candidate_id_start: int):
    return [
        {
            "candidate_id": candidate_id_start + offset,
            "generation": generation,
            "parameters": candidate,
            "score": float(offset + 1),
        }
        for offset, candidate in enumerate(candidates)
    ]


def _commit(journal: FitJournal, generation: int, candidate_id_start: int):
    candidates = _candidates(float(generation))
    return journal.commit_generation(
        generation=generation,
        candidate_id_start=candidate_id_start,
        candidates=candidates,
        evaluations=_evaluations(
            candidates,
            generation=generation,
            candidate_id_start=candidate_id_start,
        ),
        optimizer_state={"generation": generation + 1, "mean": [0.5]},
        optimizer_generation=generation + 1,
    )


def test_generation_records_are_authoritative_and_rebuild_derived_files(tmp_path):
    journal = FitJournal(
        tmp_path,
        run_id="run-1",
        execution_fingerprint="fingerprint-1",
        checkpoint_metadata={"optimizer_name": "minjae", "optimizer_version": "1"},
    )

    first = _commit(journal, 0, 0)
    second = _commit(journal, 1, first.next_candidate_id)
    assert second.completed_generations == 2
    assert second.next_candidate_id == 4

    journal.history_path.write_text("stale\n", encoding="utf-8")
    journal.checkpoint_path.write_text("{}\n", encoding="utf-8")
    recovered = journal.recover()

    history = [json.loads(line) for line in journal.history_path.read_text().splitlines()]
    checkpoint = json.loads(journal.checkpoint_path.read_text())
    assert [item["candidate_id"] for item in history] == [0, 1, 2, 3]
    assert recovered.completed_generations == 2
    assert checkpoint["run_id"] == "run-1"
    assert checkpoint["execution_fingerprint"] == "fingerprint-1"
    assert checkpoint["candidate_id"] == 4
    assert checkpoint["optimizer_state"] == {"generation": 2, "mean": [0.5]}
    assert checkpoint["optimizer_name"] == "minjae"


def test_scan_rejects_different_run_or_execution_fingerprint(tmp_path):
    original = FitJournal(tmp_path, run_id="run-1", execution_fingerprint="fingerprint-1")
    _commit(original, 0, 0)

    with pytest.raises(FitJournalError, match="run_id mismatch"):
        FitJournal(tmp_path, run_id="run-2", execution_fingerprint="fingerprint-1").scan()
    with pytest.raises(FitJournalError, match="fingerprint mismatch"):
        FitJournal(tmp_path, run_id="run-1", execution_fingerprint="fingerprint-2").scan()


def test_scan_rejects_gap_and_hash_corruption(tmp_path):
    journal = FitJournal(tmp_path, run_id="run-1", execution_fingerprint="fingerprint-1")
    _commit(journal, 0, 0)
    first = journal.generation_dir / "000000.json"
    second = journal.generation_dir / "000001.json"
    second.write_bytes(first.read_bytes())
    first.rename(journal.generation_dir / "000002.json")

    with pytest.raises(FitJournalError, match="not contiguous"):
        journal.scan()

    second.rename(first)
    (journal.generation_dir / "000002.json").unlink()
    corrupted = json.loads(first.read_text())
    corrupted["candidate_id_end"] = 999
    atomic_write_json(first, corrupted)
    with pytest.raises(FitJournalError, match="hash mismatch"):
        journal.scan()


def test_commit_rejects_nonfinite_or_misaligned_records(tmp_path):
    journal = FitJournal(tmp_path, run_id="run-1", execution_fingerprint="fingerprint-1")
    candidates = _candidates(0.0)
    evaluations = _evaluations(candidates, generation=0, candidate_id_start=0)
    evaluations[0]["score"] = float("nan")
    with pytest.raises(FitJournalError, match="finite JSON"):
        journal.commit_generation(
            generation=0,
            candidate_id_start=0,
            candidates=candidates,
            evaluations=evaluations,
            optimizer_state={"generation": 1},
            optimizer_generation=1,
        )

    evaluations = _evaluations(candidates, generation=0, candidate_id_start=0)
    evaluations[1]["candidate_id"] = 99
    with pytest.raises(FitJournalError, match="candidate_id is not contiguous"):
        journal.commit_generation(
            generation=0,
            candidate_id_start=0,
            candidates=candidates,
            evaluations=evaluations,
            optimizer_state={"generation": 1},
            optimizer_generation=1,
        )


@pytest.mark.parametrize(
    "artifact",
    ["fit-checkpoint.json", "candidate-history.jsonl", "baseline.json", "fit.json", "validation.json"],
)
def test_assert_empty_rejects_every_existing_attempt_artifact(tmp_path, artifact):
    path = tmp_path / artifact
    path.write_text("existing\n", encoding="utf-8")
    journal = FitJournal(tmp_path, run_id="run-1", execution_fingerprint="fingerprint-1")

    with pytest.raises(FitJournalError, match="not empty"):
        journal.assert_empty()


@pytest.mark.parametrize("artifact", ["fit-checkpoint.json", "candidate-history.jsonl"])
def test_recover_fails_closed_on_legacy_convenience_artifacts(tmp_path, artifact):
    (tmp_path / artifact).write_text("{}\n", encoding="utf-8")
    journal = FitJournal(tmp_path, run_id="run-1", execution_fingerprint="fingerprint-1")

    with pytest.raises(FitJournalError, match="legacy fit artifacts"):
        journal.recover()


def test_atomic_json_rejects_nonfinite_before_replacing_existing_file(tmp_path):
    target = tmp_path / "record.json"
    target.write_text('{"preserved": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not finite JSON"):
        atomic_write_json(target, {"score": float("inf")})

    assert json.loads(target.read_text()) == {"preserved": True}
    assert not list(tmp_path.glob(".record.json.*.tmp"))
