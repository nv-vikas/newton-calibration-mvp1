"""Bounded, durable motion search driven by measured simulator sensitivity.

No optimizer fit, hardware connection, or MVP2/MVP3 behavior lives here.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from newton_calibration.core.io import sha256_file, write_json
from newton_calibration.core.models import jsonable

from .registry import get_catalog


def coverage(matrices, names, request):
    matrices = np.asarray(matrices, dtype=float)
    rows = []
    for index, name in enumerate(names):
        strengths, separation = [], []
        for matrix in matrices:
            diagonal = np.diag(matrix)
            strengths.append(float(diagonal[index]))
            normalizer = np.sqrt(np.maximum(diagonal, 1e-20))
            correlation = matrix / np.outer(normalizer, normalizer)
            inverse = np.linalg.inv(correlation + np.eye(len(names)) * 1e-9)
            separation.append(float(1 / max(inverse[index, index], 1e-20)))
        strength, distinguishability = min(strengths), min(separation)
        rows.append(
            {
                "parameter": name,
                "worst_anchor_strength": strength,
                "worst_anchor_separation": distinguishability,
                "predicted_covered": bool(
                    strength >= request.sensitivity_floor and distinguishability >= request.separation_floor
                ),
            }
        )
    return rows


def _score(matrices):
    return min(float(np.linalg.slogdet(np.eye(len(matrix)) + matrix)[1]) for matrix in matrices)


def candidate_pool(analysis, motion, request):
    # Start broad across joints before deepening any one joint. Reserve holdouts
    # before search; they never enter sensitivity scoring or candidate selection.
    seeds, _ = get_catalog(analysis.evidence_needs["catalog"]).select(
        analysis.evidence_needs, replace(request, max_training_experiments=256), motion.joint_names
    )
    holdouts = [s for s in seeds if s.split == "heldout"]
    seeds = [s for s in seeds if s.split == "train"]
    order = {"servo_sweep@1": 0, "slow_reversal@1": 1, "acceleration_sweep@1": 2, "settling@1": 3}
    seeds.sort(key=lambda s: (order.get(s.recipe_id, 9), motion.joint_names.index(s.usd_joints[0])))
    variants = [(1.0, 1.0, ()), (0.45, 1.0, ()), (1.8, 1.0, ()), (1.0, 0.6, ())]
    variants += [(f, 1.0, pose) for pose in motion.posture_offsets_rad for f in (1.0, 0.45)]
    pool = []
    for variant, (frequency, amplitude, posture) in enumerate(variants):
        for seed in seeds:
            pool.append(
                replace(
                    seed,
                    name=f"candidate_{len(pool) + 1:04d}_{seed.name}",
                    frequency_scale=frequency,
                    amplitude_scale=amplitude,
                    posture_offset_rad=posture,
                    reason=f"Adaptive variant {variant}: test weak parameter directions; retain only useful information",
                )
            )
    return pool, holdouts


def design_campaign(analysis, motion, request, probe, root: Path):
    from .planning import _generate

    pool, holdouts = candidate_pool(analysis, motion, request)
    descriptor = probe.describe()
    if descriptor.get("physics") not in {"Newton", "analytic-contract-test"}:
        raise ValueError("Sensitivity design requires an explicit Newton probe (or labeled test double)")
    if descriptor.get("asset_sha256") != analysis.asset_fingerprint or descriptor.get("scene_id") != motion.scene_id:
        raise ValueError("Sensitivity probe is bound to a different asset or scene")
    requested = [r for r in analysis.evidence_needs["parameters"] if r["disposition"] == "collect"]
    available = {p["name"] for p in descriptor.get("ranges", [])}
    names = [r["parameter"] for r in requested if r["parameter"] in available]
    if not names and pool:
        raise ValueError("No requested parameters have usable probe ranges")
    total = np.zeros((2, len(names), len(names)))
    selected, history, seen_hashes = [], [], set()
    probed = 0
    report = {
        "schema": "newton.motion-search/v1",
        "status": "running",
        "exhausted": False,
        "physics": descriptor["physics"],
        "probe": descriptor,
        "candidate_count": len(pool),
        "history": history,
        "selected": [],
        "remaining_candidates": [],
        "unsupported_probe_parameters": [r["parameter"] for r in requested if r["parameter"] not in available],
        "measurement_gaps": [
            r
            for r in analysis.evidence_needs["parameters"]
            if r["disposition"] not in {"collect", "existing_evidence_eligible"}
        ],
        "thresholds": {
            k: getattr(request, k) for k in ("sensitivity_floor", "separation_floor", "minimum_information_gain")
        },
        "scope": "Predicted conditional coverage; noise and ranges are assumptions; real data must be analyzed afterwards",
    }
    ledger = root / "design_search.json"
    for index, candidate in enumerate(pool):
        report["remaining_candidates"] = [c.name for c in pool[index:]]
        current = coverage(total, names, request)
        weak = {r["parameter"] for r in current if not r["predicted_covered"]}
        if not weak and names:
            report["status"] = "predicted_coverage_reached"
            break
        if probed >= request.max_candidate_probes:
            report["status"] = "probe_budget_reached"
            break
        if len(selected) >= request.max_training_experiments:
            report["status"] = "selection_budget_reached"
            break
        active = [
            r["parameter"]
            for r in requested
            if r["parameter"] in names and set(r["usd_joints"]) & set(candidate.usd_joints)
        ]
        if not set(active) & weak:
            history.append({"candidate": candidate.name, "decision": "covered_targets_skipped"})
            continue
        row = {"candidate": candidate.name, "experiment": jsonable(candidate), "parameters_probed": active}
        try:
            episode = _generate(motion, root / "design_candidates", [candidate])[0]
            path = root / "design_candidates" / episode["command_file"]
            row["command_sha256"] = sha256_file(path)
            # Deduplicate equivalent commands, not platform-dependent CSV float spelling.
            commands = np.loadtxt(path, delimiter=",", skiprows=1)
            rounded = np.round(commands, 9)
            rounded[rounded == 0.0] = 0.0  # Normalize signed zero for equivalent waveforms.
            row["numeric_command_fingerprint"] = hashlib.sha256(rounded.astype("<f8").tobytes()).hexdigest()
            if row["numeric_command_fingerprint"] in seen_hashes:
                row["decision"] = "duplicate_commands_skipped"
                history.append(row)
                continue
            seen_hashes.add(row["numeric_command_fingerprint"])
            probed += 1
            print(
                f"[MOTION-DESIGN] Probe {probed}/{request.max_candidate_probes}: {candidate.name}",
                file=sys.stderr,
                flush=True,
            )
            result = probe(path, candidate, active)
            if result.parameters != active or result.metadata.get("physics") != descriptor["physics"]:
                raise ValueError("Probe parameter order or backend changed")
            local = np.asarray(result.information, dtype=float)
            if local.shape != (2, len(active), len(active)) or not np.isfinite(local).all():
                raise ValueError("Invalid sensitivity information matrix")
            if not np.allclose(local, local.transpose(0, 2, 1), atol=1e-9) or np.linalg.eigvalsh(local).min() < -1e-7:
                raise ValueError("Sensitivity matrix is not positive semidefinite")
            addition = np.zeros_like(total)
            ids = [names.index(name) for name in active]
            for anchor in range(2):
                addition[anchor][np.ix_(ids, ids)] = local[anchor]
            gain = _score(total + addition) - _score(total)
            row.update(
                information_gain=gain,
                rollout_count=result.rollout_count,
                repeatability=result.metadata.get("repeatability_error_noise_units"),
            )
            evidence_path = write_json(root / "design_probes" / f"{candidate.name}.json", result)
            row.update(probe_record=str(evidence_path.relative_to(root)), probe_sha256=sha256_file(evidence_path))
            # Ensure each joint's first informative test is represented, then
            # grow only when measured information gain exceeds the threshold.
            if gain >= request.minimum_information_gain:
                selected.append(replace(candidate, target_parameters=tuple(active)))
                total += addition
                row["decision"] = "selected"
            else:
                row["decision"] = "low_information_gain"
        except (ValueError, TypeError, RuntimeError) as exc:
            row.update(decision="rejected", reason=f"{type(exc).__name__}: {exc}")
        history.append(row)
        report.update(
            probed_candidates=probed, selected=[c.name for c in selected], coverage=coverage(total, names, request)
        )
        write_json(ledger, report)
    else:
        report.update(status="candidate_catalog_exhausted", exhausted=True, remaining_candidates=[])
    report.update(
        probed_candidates=probed,
        selected=[c.name for c in selected],
        coverage=coverage(total, names, request),
        failed_candidates=[r["candidate"] for r in history if r["decision"] == "rejected"],
        omitted_posture_search=not bool(motion.posture_offsets_rad),
    )
    report["completed_rollouts"] = sum(r.get("rollout_count", 0) for r in history)
    report["limits"] = {
        "max_candidate_probes": request.max_candidate_probes,
        "max_training_experiments": request.max_training_experiments,
    }
    report["all_requested_parameters_covered"] = (
        bool(names)
        and all(r["predicted_covered"] for r in report["coverage"])
        and not report["unsupported_probe_parameters"]
        and not report["measurement_gaps"]
    )
    report["backend_failures_require_review"] = bool(report["failed_candidates"])
    if names and all(r["predicted_covered"] for r in report["coverage"]):
        report["status"] = "predicted_coverage_reached"
    if report["exhausted"] and report["failed_candidates"]:
        report.update(status="catalog_visited_with_failures", exhausted=False)
    if not selected and pool:
        report["status"] = "no_informative_motion_selected"
    write_json(ledger, report)
    return selected + (holdouts if selected else []), report
