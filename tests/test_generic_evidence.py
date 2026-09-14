import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from newton_calibration.adapters.evidence import TabularJointEvidence, inspect_tabular_evidence
from newton_calibration.core import (
    BoundEvidenceSpec,
    JointBinding,
    LongFormSchema,
    SignalBinding,
    bind_evidence_files,
    bindings_from_unambiguous_report,
    propose_joint_mapping,
)
from newton_calibration.isaaclab import SO101EnvCfg, tuning

SOURCE_JOINTS = ["Shoulder-Pan", "elbow joint", "finger_slide"]
USD_JOINTS = ["/World/Flexiv/shoulder_pan", "elbow_joint", "finger_slide"]


def _write_episode(path: Path, *, conflict: bool = False, phase: float = 0.0) -> None:
    rows: list[dict[str, float | str]] = []
    for joint_index, joint in enumerate(SOURCE_JOINTS):
        for source_signal, rate in (("target", 20), ("position", 100), ("velocity", 100)):
            times = np.arange(0.0, 0.5, 1.0 / rate)
            position_deg = 10.0 * times + joint_index + phase
            values = position_deg if source_signal != "velocity" else np.full_like(times, 10.0)
            for timestamp, value in zip(times, values):
                rows.append(
                    {
                        "timestamp_ms": timestamp * 1000.0,
                        "joint_name": joint,
                        "signal_name": source_signal,
                        "reading": value,
                    }
                )
    if conflict:
        rows.append(
            {
                "timestamp_ms": 0.0,
                "joint_name": SOURCE_JOINTS[0],
                "signal_name": "position",
                "reading": 999.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _bindings() -> tuple[JointBinding, ...]:
    report = propose_joint_mapping(SOURCE_JOINTS, USD_JOINTS)
    assert report.ready
    return bindings_from_unambiguous_report(
        report,
        source_units={joint: "deg" if joint != "finger_slide" else "mm" for joint in SOURCE_JOINTS},
        usd_units={joint: "rad" if joint != "finger_slide" else "m" for joint in USD_JOINTS},
        affine_overrides={
            "Shoulder-Pan": {"sign": -1, "scale": 2.0, "offset": 0.1},
            "finger_slide": {"scale": 0.5, "offset": 0.002},
        },
        confirmed_transforms=SOURCE_JOINTS,
    )


def _spec(tmp_path: Path, *, conflict: bool = False) -> BoundEvidenceSpec:
    _write_episode(tmp_path / "excitation.csv", conflict=conflict)
    _write_episode(tmp_path / "validation.csv", phase=0.25)
    return bind_evidence_files(
        root=tmp_path,
        episodes=(
            {"name": "excitation", "path": "excitation.csv", "split": "train", "trial_id": "capture-1"},
            {"name": "validation", "path": "validation.csv", "split": "heldout", "trial_id": "capture-2"},
        ),
        schema=LongFormSchema(
            time_column="timestamp_ms",
            time_unit="ms",
            value_column="reading",
            joint_column="joint_name",
            signal_column="signal_name",
            field_column=None,
        ),
        joint_bindings=_bindings(),
        signal_bindings=(
            SignalBinding("target", "command_q"),
            SignalBinding("position", "actual_q"),
            SignalBinding("velocity", "actual_dq"),
        ),
        revision="fixture-r1",
        clock_synchronized=True,
        effort_saturation_joints=SOURCE_JOINTS,
    )


def test_deterministic_exact_and_normalized_mapping() -> None:
    report = propose_joint_mapping(SOURCE_JOINTS, USD_JOINTS)
    assert report.ready
    assert report.require_unambiguous() == dict(zip(SOURCE_JOINTS, USD_JOINTS))
    assert [item.match_kind for item in report.proposals] == ["normalized", "normalized", "exact"]


def test_agent_can_inspect_unbound_evidence_before_mapping(tmp_path: Path) -> None:
    _write_episode(tmp_path / "train.csv")
    inspection = inspect_tabular_evidence(
        root=tmp_path,
        episodes=({"name": "train", "path": "train.csv", "split": "train", "trial_id": "capture-1"},),
        schema=LongFormSchema(
            time_column="timestamp_ms",
            time_unit="ms",
            value_column="reading",
            joint_column="joint_name",
            signal_column="signal_name",
            field_column=None,
        ),
    )
    assert inspection.source_joints == tuple(sorted(SOURCE_JOINTS))
    assert inspection.source_signals == ("position", "target", "velocity")
    assert len(inspection.fingerprint) == 64
    assert inspection.episodes[0].split == "train"


def test_ambiguous_normalized_mapping_blocks_readiness() -> None:
    report = propose_joint_mapping(["joint-one"], ["/left/joint_one", "/right/joint-one"])
    assert not report.ready
    assert report.proposals[0].status == "ambiguous"
    assert len(report.proposals[0].candidates) == 2
    with pytest.raises(ValueError, match="explicit source-to-USD binding"):
        report.require_unambiguous()


def test_units_are_explicit_and_dimension_checked() -> None:
    report = propose_joint_mapping(["slide"], ["slide"])
    with pytest.raises(ValueError, match="source unit is required"):
        bindings_from_unambiguous_report(report, source_units={}, usd_units={"slide": "m"})
    with pytest.raises(ValueError, match="incompatible joint units"):
        bindings_from_unambiguous_report(report, source_units={"slide": "deg"}, usd_units={"slide": "m"})


def test_identity_transform_must_also_be_explicitly_confirmed(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    unconfirmed = JointBinding("Shoulder-Pan", USD_JOINTS[0], "deg", "rad")
    changed = BoundEvidenceSpec(
        root=spec.root,
        revision=spec.revision,
        episodes=spec.episodes,
        schema=spec.schema,
        joint_bindings=(unconfirmed, *spec.joint_bindings[1:]),
        signal_bindings=spec.signal_bindings,
        clock_synchronized=spec.clock_synchronized,
        effort_saturation_joints=spec.effort_saturation_joints,
    )
    assert not changed.readiness.ready
    assert any("not explicitly confirmed" in item for item in changed.readiness.blockers)


def test_bound_spec_round_trip_and_fingerprint_excludes_locator(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    path = spec.write(tmp_path / "bound-evidence.json")
    loaded = BoundEvidenceSpec.read(path)
    assert loaded == spec
    assert loaded.fingerprint == spec.fingerprint

    payload = spec.to_dict()
    payload["root"] = "/a/different/mount/location"
    relocated = BoundEvidenceSpec.from_dict(payload)
    assert relocated.fingerprint == spec.fingerprint

    tampered = json.loads(path.read_text())
    tampered["episodes"][0]["split"] = "heldout"
    with pytest.raises(ValueError, match="fingerprint does not match"):
        BoundEvidenceSpec.from_dict(tampered)


def test_inventory_and_affine_mapping_to_usd_order(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    evidence = TabularJointEvidence(spec)
    inventory = evidence.inventory()
    assert inventory["ready"]
    assert inventory["train_episodes"] == ["excitation"]
    assert inventory["heldout_episodes"] == ["validation"]
    assert inventory["joints"] == USD_JOINTS
    assert inventory["mapping_fingerprint"] == spec.mapping_fingerprint
    assert inventory["clock_synchronized"] is True
    assert inventory["dynamic_excitation_joints"] == SOURCE_JOINTS
    assert inventory["reversal_joints"] == []
    assert inventory["effort_saturation_joints"] == SOURCE_JOINTS

    episode = evidence.load_episode("excitation", dt=0.01)
    assert episode.command_q.shape == episode.actual_q.shape == episode.actual_dq.shape
    assert episode.command_q.shape[1] == 3
    assert episode.joints == USD_JOINTS
    # q_usd = -2 * deg_to_rad(q_source) + 0.1
    assert episode.actual_q[0, 0] == pytest.approx(0.1)
    assert episode.actual_dq[0, 0] == pytest.approx(-20.0 * np.pi / 180.0)
    # 2 source mm * 0.5 + 2 mm target offset = 3 mm.
    assert episode.actual_q[0, 2] == pytest.approx(0.003)


def test_changed_evidence_and_conflicting_duplicates_are_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    (tmp_path / "excitation.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="size changed|content changed"):
        TabularJointEvidence(spec)

    conflict_root = tmp_path / "conflict"
    conflict_root.mkdir()
    conflict_spec = _spec(conflict_root, conflict=True)
    evidence = TabularJointEvidence(conflict_spec)
    with pytest.raises(ValueError, match="conflicting duplicate samples"):
        evidence.load_episode("excitation", dt=0.01)


def test_split_must_be_explicit_and_include_holdout(tmp_path: Path) -> None:
    _write_episode(tmp_path / "only.csv")
    with pytest.raises(ValueError, match="no heldout episode"):
        bind_evidence_files(
            root=tmp_path,
            episodes=({"name": "only", "path": "only.csv", "split": "train", "trial_id": "capture-1"},),
            schema=LongFormSchema(
                time_column="timestamp_ms",
                time_unit="ms",
                value_column="reading",
                joint_column="joint_name",
                signal_column="signal_name",
                field_column=None,
            ),
            joint_bindings=_bindings(),
            signal_bindings=(
                SignalBinding("target", "command_q"),
                SignalBinding("position", "actual_q"),
                SignalBinding("velocity", "actual_dq"),
            ),
        )


def test_qualification_metadata_is_locked_and_unknown_saturation_joint_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    payload = spec.to_dict()
    payload["clock_synchronized"] = False
    assert BoundEvidenceSpec.from_dict(payload).fingerprint != spec.fingerprint

    payload = spec.to_dict()
    payload["effort_saturation_joints"] = [*SOURCE_JOINTS, "not_bound"]
    with pytest.raises(ValueError, match="unbound source joints"):
        BoundEvidenceSpec.from_dict(payload)


def test_legacy_bound_evidence_requires_explicit_v2_reconfirmation(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    payload = spec.to_dict()
    payload.pop("clock_synchronized")
    payload.pop("effort_saturation_joints")
    payload_without_root = dict(payload)
    payload_without_root.pop("root")
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload_without_root, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    payload["schema_version"] = "newton.calibration/bound-evidence@1"
    with pytest.raises(ValueError, match="recreate and confirm.*bound-evidence@2"):
        BoundEvidenceSpec.from_dict(payload)


def test_identical_bytes_cannot_cross_train_and_heldout_splits(tmp_path: Path) -> None:
    _write_episode(tmp_path / "train.csv")
    (tmp_path / "heldout.csv").write_bytes((tmp_path / "train.csv").read_bytes())

    with pytest.raises(ValueError, match="train and heldout splits contain identical episode bytes"):
        bind_evidence_files(
            root=tmp_path,
            episodes=(
                {"name": "train", "path": "train.csv", "split": "train", "trial_id": "capture-1"},
                {"name": "heldout", "path": "heldout.csv", "split": "heldout", "trial_id": "capture-2"},
            ),
            schema=LongFormSchema(
                time_column="timestamp_ms",
                time_unit="ms",
                value_column="reading",
                joint_column="joint_name",
                signal_column="signal_name",
                field_column=None,
            ),
            joint_bindings=_bindings(),
            signal_bindings=(
                SignalBinding("target", "command_q"),
                SignalBinding("position", "actual_q"),
                SignalBinding("velocity", "actual_dq"),
            ),
        )


def test_same_real_trial_cannot_cross_splits_after_reexport(tmp_path: Path) -> None:
    _write_episode(tmp_path / "train.csv")
    _write_episode(tmp_path / "heldout.csv", phase=0.4)

    with pytest.raises(ValueError, match="same real capture/trial"):
        bind_evidence_files(
            root=tmp_path,
            episodes=(
                {"name": "train", "path": "train.csv", "split": "train", "trial_id": "capture-1"},
                {"name": "heldout", "path": "heldout.csv", "split": "heldout", "trial_id": "capture-1"},
            ),
            schema=LongFormSchema(
                time_column="timestamp_ms",
                time_unit="ms",
                value_column="reading",
                joint_column="joint_name",
                signal_column="signal_name",
                field_column=None,
            ),
            joint_bindings=_bindings(),
            signal_bindings=(
                SignalBinding("target", "command_q"),
                SignalBinding("position", "actual_q"),
                SignalBinding("velocity", "actual_dq"),
            ),
        )


def test_malformed_json_evidence_contract_fails_closed(tmp_path: Path) -> None:
    malformed = tmp_path / "evidence.json"
    malformed.write_text('{"adapter": "tabular_joint.v1", "episodes": []}', encoding="utf-8")

    with pytest.raises((KeyError, TypeError, ValueError)):
        tuning.analyze(
            env=SO101EnvCfg(usd_path=str(tmp_path / "unused.usda"), runtime="analytic", device="cpu"),
            evidence=malformed,
            workdir=tmp_path / "runs",
        )
