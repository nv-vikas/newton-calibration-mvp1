from __future__ import annotations

import csv
import io
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from newton_calibration.core.io import atomic_write_text, sha256_file, utc_now, write_json
from newton_calibration.core.models import AnalysisResult

from .contracts import CalibrationRequest, ExperimentSpec
from .generators import generate_waveform
from .registry import get_catalog


@dataclass(frozen=True)
class MotionSpec:
    """A simulation proposal, in USD joint coordinates. NEVER hardware approval.

    A scene adapter supplies the actual starting pose, joint limits and provenance.
    Velocity, acceleration and amplitude caps are explicit *simulation* limits;
    the toolkit does not infer hardware operating limits from a USD.
    """

    joint_names: tuple[str, ...]
    center_rad: tuple[float, ...]
    lower_rad: tuple[float, ...]
    upper_rad: tuple[float, ...]
    amplitude_rad: tuple[float, ...]
    max_velocity_rad_s: tuple[float, ...]
    max_acceleration_rad_s2: tuple[float, ...]
    source: str
    scene_id: str
    command_rate_hz: int = 100
    duration_s: float = 24.0
    hold_s: float = 2.0
    margin_rad: float = 0.15

    def __post_init__(self):
        n = len(self.joint_names)
        if not n or len(set(self.joint_names)) != n or any(not j.strip() for j in self.joint_names):
            raise ValueError("Collection needs unique nonempty USD joint names")
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        if not self.source.strip() or not self.scene_id.strip():
            raise ValueError("Collection needs scene and motion-envelope provenance")
        for name in (
            "center_rad",
            "lower_rad",
            "upper_rad",
            "amplitude_rad",
            "max_velocity_rad_s",
            "max_acceleration_rad_s2",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must have one finite value per joint")
            # Runtime tensor -> NumPy scalars must become immutable, JSON-safe
            # Python values before locking a durable plan.
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if not isinstance(self.command_rate_hz, int) or not 20 <= self.command_rate_hz <= 2000:
            raise ValueError("Command rate must be an integer in [20, 2000] Hz")
        if not np.isfinite([self.duration_s, self.hold_s, self.margin_rad]).all():
            raise ValueError("Collection timing/margin must be finite")
        if not 12 <= self.duration_s <= 120 or self.hold_s < 1 or self.duration_s <= 2 * self.hold_s + 6:
            raise ValueError("Need 12–120 s episodes, endpoint holds, and at least 6 s excitation")
        if self.margin_rad <= 0:
            raise ValueError("Joint-limit margin must be positive")
        center, lower, upper = map(np.asarray, (self.center_rad, self.lower_rad, self.upper_rad))
        if np.any(center <= lower + self.margin_rad) or np.any(center >= upper - self.margin_rad):
            raise ValueError("Starting pose is outside the proposed joint-limit margin")
        for name in ("amplitude_rad", "max_velocity_rad_s", "max_acceleration_rad_s2"):
            if np.any(np.asarray(getattr(self, name)) <= 0):
                raise ValueError(f"{name} must be positive")


@dataclass
class CollectionPlan:
    run_id: str
    workdir: str
    asset_path: str
    asset_sha256: str
    motion_spec: MotionSpec | None
    assistance: dict[str, Any]
    evidence_needs: dict[str, Any] = field(default_factory=dict)
    design: dict[str, Any] = field(default_factory=dict)
    analysis_sha256: str = ""
    schema: str = "newton.collection/v1"
    kind: str = "evidence_collection"
    created_at: str = field(default_factory=utc_now)
    status: str = "needs_scene_setup"
    episodes: list[dict[str, Any]] = field(default_factory=list)
    preview: dict[str, Any] = field(default_factory=lambda: {"requested": True, "status": "pending"})
    command_plan_sha256: str = ""
    real_data: bool = False
    fit_allowed: bool = False
    real_execution_approved: bool = False


class ScenePreview(Protocol):
    """Plug-in gets immutable commands and must return actual Newton artifacts.

    Required return fields: physics='Newton', scene_id, video_path, screen_path,
    backend_record_path, command_plan_sha256. Paths must exist. Video completion
    and screening pass are separate; a failed screen must still retain its video.
    """

    def __call__(self, plan_path: Path, output: Path) -> dict[str, Any]: ...


def prepare_assistance(analysis: AnalysisResult) -> dict[str, Any]:
    env = analysis.environment
    missing = analysis.evidence_spec.get("adapter") == "missing"
    # These are only optimization proposals, never written into the fit environment.
    bounds = {}
    for group in env.joint_groups:
        for suffix in ("stiffness_scale", "damping_scale"):
            name = f"{group}_{suffix}"
            bounds[name] = {
                "proposal": list(env.parameter_bounds.get(name, (0.5, 2.0, 1.0))),
                "source": "declared profile"
                if name in env.parameter_bounds
                else "bounded local search around supplied simulation baseline; requires review",
            }
    return {
        "next_action": "collect_evidence"
        if missing or any(r["disposition"] == "collect" for r in analysis.evidence_needs.get("parameters", []))
        else "review_fitting_readiness",
        "fit_readiness_unchanged": True,
        "joint_mapping": {
            "proposal": dict(env.joint_map),
            "confirmed": env.profile_confirmed,
            "source": "environment profile, NOT observed real feedback",
            "needs": "Verify driver order, units, sign and zero offset against USD; do not infer from matching names",
        },
        "controller": {
            "source": env.controller_profile_source or "not supplied",
            "sim_stiffness": dict(env.base_stiffness_by_joint),
            "sim_damping": dict(env.base_damping_by_joint),
            "confirmed": env.controller_profile_confirmed,
            "needs": "Record real command mode, smoothing, controller rate, payload and feedback timestamps; sim gains are not OEM gains",
        },
        "optimizer_bounds": {
            "proposals": bounds,
            "automatically_applied": False,
            "needs": "Review scale bounds; source absolute armature/friction/effort bounds before fitting. These are not robot motion limits",
        },
        "evidence": {
            "missing": missing,
            "agent_action": "Generate train and held-out commands; screen and record in the supplied scene"
            if missing
            else "Use measured-data analysis; do not substitute simulated traces for evidence",
            "required_logs": [
                "episode_id",
                "actual command-send time",
                "feedback time",
                "commanded joint position",
                "measured joint position",
                "measured or derived velocity",
            ],
            "conditional_logs": {
                "delay": "Synchronized command/measurement clocks with offset and jitter evidence",
                "effort_scale": "Calibrated signed torque/current and an actual saturation observation; do not deliberately saturate the robot",
                "model_diagnosis": "Payload, mode, smoothing and optional aligned video",
            },
            "parameter_requirements": analysis.evidence_needs,
        },
        "operator_checks": [
            "Confirm real-to-USD coordinate mapping",
            "Confirm payload and real controller/interface configuration",
            "Review the proposed start pose, swept volume and motion envelope on site",
            "Configure OEM limits and abort/E-stop supervision before any hardware execution",
        ],
        "not_claimed": [
            "Hardware safety approval",
            "Real evidence already collected",
            "Parameter identifiability proven",
            "Calibration or task transfer validated",
            "Minjae agent installed or invoked",
        ],
    }


def create_collection_plan(
    analysis: AnalysisResult,
    *,
    motion: MotionSpec | None = None,
    preview: ScenePreview | None = None,
    video: bool = True,
) -> CollectionPlan:
    root = Path(analysis.workdir)
    if (root / "collection_plan.json").exists():
        raise ValueError(
            "A collection plan already exists; rerun analyze for a new revision, do not overwrite prior commands"
        )
    if not analysis.asset_fingerprint or sha256_file(analysis.environment.asset_path) != analysis.asset_fingerprint:
        raise ValueError("Cannot collect against a missing or changed USD; rerun analyze")
    result = CollectionPlan(
        analysis.run_id,
        str(root),
        analysis.environment.asset_path,
        analysis.asset_fingerprint,
        motion,
        prepare_assistance(analysis),
        evidence_needs=analysis.evidence_needs,
        analysis_sha256=sha256_file(root / "analysis.json"),
    )
    if not analysis.evidence_needs:
        raise ValueError("Analysis has no parameter/evidence requirements; rerun analyze with the current toolkit")
    write_json(root / "evidence_needs.json", analysis.evidence_needs)
    result.preview = {"requested": video, "status": "pending" if video else "skipped_explicitly"}
    write_json(root / "agent_assistance.json", result.assistance)
    if motion is None:
        result.preview["reason"] = (
            "Scene adapter must supply a starting pose, controlled USD DOFs and a simulation motion envelope"
        )
        write_json(root / "collection_plan.json", result)
        return result
    expected = set(analysis.environment.joint_map.values())
    if set(motion.joint_names) != expected:
        raise ValueError("Motion joints must exactly match the controlled USD joints in the environment profile")
    from newton_calibration.adapters.asset import inspect_usd

    inventory = inspect_usd(analysis.environment.asset_path)
    unsupported = [j.name for j in inventory.joints if j.name in expected and j.kind != "revolute"]
    if unsupported:
        raise ValueError(f"Rad-valued collection currently supports revolute joints only: {unsupported}")
    request = CalibrationRequest(**analysis.collection_request)
    catalog = get_catalog(analysis.evidence_needs["catalog"])
    experiments, deferred = catalog.select(analysis.evidence_needs, request, motion.joint_names)
    result.design = {
        "catalog": analysis.evidence_needs["catalog"],
        "generator_environment": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()},
        "experiments": experiments,
        "deferred_by_budget": deferred,
        "duration_s": len(experiments) * motion.duration_s,
        "all_requested_parameters_calibrated": False,
        "numerical_identifiability_proven": False,
        "requires_real_evidence_reanalysis": True,
        "non_motion_requirements": [
            r
            for r in analysis.evidence_needs["parameters"]
            if r["disposition"] not in {"collect", "existing_evidence_eligible"}
        ],
    }
    write_json(root / "experiment_design.json", result.design)
    if not experiments:
        result.status = "evidence_action_required"
        result.preview.update(
            status="not_applicable",
            reason="No supported new motions: inspect evidence needs for instrumentation, clock, scope or existing-evidence actions",
        )
        _write_instructions(root, result)
        write_json(root / "collection_plan.json", result)
        return result
    try:
        result.episodes = _generate(motion, root, experiments)
    except (ValueError, TypeError) as exc:
        result.status = "generation_failed"
        result.preview.update(status="blocked", reason=f"Motion generation failed: {exc}")
        write_json(root / "collection_plan.json", result)
        return result
    result.status = "commands_generated"
    write_json(root / "collection_plan.json", result)
    # Fixed input to previews: output status changes never change this fingerprint.
    command_path = write_json(root / "command_plan.json", result)
    result.command_plan_sha256 = sha256_file(command_path)
    _write_instructions(root, result)
    verify_commands(command_path)
    if video and preview is None:
        result.status = "preview_pending"
        result.preview.update(
            status="blocked", reason="Bind an Isaac Lab/Newton scene preview adapter; no rendered video exists yet"
        )
    elif video:
        result.status = "preview_running"
        write_json(root / "collection_plan.json", result)
        try:
            artifact = dict(preview(command_path, root / "preview"))
            verify_commands(command_path)
            if sha256_file(command_path) != result.command_plan_sha256:
                raise ValueError("Preview changed the locked command plan")
            if artifact.get("physics") != "Newton" or artifact.get("scene_id") != motion.scene_id:
                raise ValueError("Preview must attest Newton and the requested scene")
            if artifact.get("command_plan_sha256") != result.command_plan_sha256:
                raise ValueError("Preview command fingerprint mismatch")
            for key in ("video_path", "screen_path", "backend_record_path"):
                path = Path(artifact[key]).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"Preview did not produce a local, nonempty {key}")
                artifact[key] = str(path)
                artifact[key + "_sha256"] = sha256_file(path)
            result.status = "preview_complete_review_required"
            result.preview.update(status="complete", **artifact)
        except Exception as exc:  # noqa: BLE001 -- record plug-in failure without losing generated commands
            # Durable failure instead of a lost job or a false successful preview.
            result.status = "preview_failed"
            result.preview.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_json(root / "collection_plan.json", result)
    return result


def _generate(spec: MotionSpec, root: Path, experiments: list[ExperimentSpec]) -> list[dict[str, Any]]:
    n = len(spec.joint_names)
    rate = spec.command_rate_hz
    t = np.arange(round(spec.duration_s * rate) + 1) / rate
    center, lo, hi, requested, vlim, alim = [
        np.asarray(getattr(spec, k))
        for k in (
            "center_rad",
            "lower_rad",
            "upper_rad",
            "amplitude_rad",
            "max_velocity_rad_s",
            "max_acceleration_rad_s2",
        )
    ]
    clearance = np.minimum(center - lo, hi - center) - spec.margin_rad
    amplitude = np.minimum(requested, clearance * 0.8)
    active = np.clip((t - spec.hold_s) / (spec.duration_s - 2 * spec.hold_s), 0, 1)
    window = np.sin(np.pi * active) ** 4  # smooth endpoints, stationary start/end holds
    window[(t <= spec.hold_s) | (t >= spec.duration_s - spec.hold_s)] = 0.0
    episodes = []
    for experiment in experiments:
        name = experiment.name
        offset = np.zeros((len(t), n))
        frequencies = {}
        excited = [spec.joint_names.index(j) for j in experiment.usd_joints]
        for j in excited:
            generated = generate_waveform(
                experiment.recipe_id, t - spec.hold_s, spec.duration_s - 2 * spec.hold_s, j, experiment.variant
            )
            frequencies[spec.joint_names[j]] = list(generated.frequencies_hz)
            # Settling's quintic ramps already have smooth boundaries and true
            # constant-position plateaus. A window would destroy those holds.
            waveform = generated.values if experiment.recipe_id == "settling@1" else window * generated.values
            offset[:, j] = amplitude[j] * waveform
        dq = np.gradient(offset, 1 / rate, axis=0, edge_order=2)
        ddq = np.gradient(dq, 1 / rate, axis=0, edge_order=2)
        # Scale each joint, maintaining spectral content, until all explicit limits hold.
        scale = (
            np.minimum(
                1,
                np.minimum(
                    vlim / np.maximum(np.max(abs(dq), axis=0), 1e-12),
                    alim / np.maximum(np.max(abs(ddq), axis=0), 1e-12),
                ),
            )
            * 0.98
        )
        offset *= scale
        q = center + offset
        dq = np.gradient(q, 1 / rate, axis=0, edge_order=2)
        ddq = np.gradient(dq, 1 / rate, axis=0, edge_order=2)
        span = np.ptp(q, axis=0)
        # Do not pretend almost-zero trajectories provide useful excitation.
        if any(span[j] < min(requested[j] * 0.2, 0.01) for j in excited):
            raise ValueError(f"{name}: motion envelope permits too little excitation; review pose/limits")
        if np.any(q < lo + spec.margin_rad) or np.any(q > hi - spec.margin_rad):
            raise ValueError("Generated command violates position envelope")
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["time_s"]
            + [f"q{i + 1}_rad" for i in range(n)]
            + [f"dq{i + 1}_rad_s" for i in range(n)]
            + [f"ddq{i + 1}_rad_s2" for i in range(n)]
        )
        writer.writerows(np.column_stack([t, q, dq, ddq]).tolist())
        path = atomic_write_text(root / "commands" / f"{name}.csv", buffer.getvalue())
        episodes.append(
            {
                "name": name,
                "split": experiment.split,
                "recipe_id": experiment.recipe_id,
                "target_parameters": list(experiment.target_parameters),
                "required_signals": list(experiment.required_signals),
                "excited_usd_joints": list(experiment.usd_joints),
                "reason": experiment.reason,
                "start_position_rad": list(spec.center_rad),
                "applied_amplitude_scale_by_joint": scale.tolist(),
                "coverage_claim": "Prospective excitation only; measured eligibility and parameter distinguishability remain unproven",
                "command_file": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "duration_s": spec.duration_s,
                "samples": len(t),
                "frequencies_hz": frequencies,
                "range_rad_by_joint": span.tolist(),
                "max_velocity_rad_s_by_joint": abs(dq).max(axis=0).tolist(),
                "max_acceleration_rad_s2_by_joint": abs(ddq).max(axis=0).tolist(),
                "status": "proposed commands, NOT real evidence",
            }
        )
    return episodes


def verify_commands(plan_path: str | Path) -> dict[str, Any]:
    import json

    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    if plan.get("schema") != "newton.collection/v1" or plan.get("real_data") is not False:
        raise ValueError("Not a simulation collection command plan")
    if sha256_file(plan["asset_path"]) != plan["asset_sha256"]:
        raise ValueError("Collection asset fingerprint changed")
    if not plan["episodes"]:
        raise ValueError("No collection commands")
    for episode in plan["episodes"]:
        command = (path.parent / episode["command_file"]).resolve()
        if not command.is_relative_to(path.parent) or sha256_file(command) != episode["sha256"]:
            raise ValueError("Collection command path or fingerprint changed")
    if plan.get("analysis_sha256") and sha256_file(path.parent / "analysis.json") != plan["analysis_sha256"]:
        raise ValueError("Collection analysis fingerprint changed")
    return plan


def _write_instructions(root: Path, plan: CollectionPlan):
    write_json(root / "evidence_requirements.json", plan.assistance["evidence"])
    rows = [
        "# MVP1 evidence collection plan",
        "",
        "Simulation proposal only. No real measurements or hardware approval.",
        "",
        "| Target | Evidence decision | Proposed experiment | Limitation |",
        "|---|---|---|---|",
    ]
    for need in plan.evidence_needs["parameters"]:
        rows.append(
            f"| {need['parameter']} | {need['disposition']} | {', '.join(need['experiment_recipes']) or 'No generated motion'} | {need['limitation']} |"
        )
    rows += [
        "",
        f"Generated: {len(plan.episodes)} episodes. Budget-deferred experiments: {len(plan.design.get('deferred_by_budget', []))}.",
        "",
        "Each CSV's column order is motion_spec.joint_names (USD coordinates), not assumed driver order.",
        "Use experiment_design.json for per-experiment targets, required signals and deferred work.",
    ]
    atomic_write_text(root / "COLLECTION_PLAN.md", "\n".join(rows) + "\n")
    atomic_write_text(
        root / "COLLECT_NEXT.md",
        """# Collection proposal — not calibrated and not hardware-approved

Inspect experiment_design.json and evidence_needs.json first. Every command
identifies its target parameters, required signals, recipe version and actual
excitation. Deferred parameters and budget omissions are not covered. A single
pose with encoders cannot uniquely separate all physical and controller terms.

When supported motions are needed, the toolkit generates train and held-out
commands in **USD coordinates**. The
command-plan fingerprint links these exact files to the Newton preview. Review
the preview's screening report even when the video completes successfully.

Before real execution, confirm the driver-to-USD mapping (including units,
signs, offsets and order), controller mode/smoothing, payload, start pose,
physical swept volume and OEM limits with the robot operator. Do not feed
these CSVs directly to a driver that expects different coordinates. A reviewed
hardware adapter must apply the inverse coordinate transform and enforce limits.
The previous Flexiv ±2-degree runner intentionally rejects larger proposals.

Log actual sent commands and measured feedback, episode IDs, separate clocks,
and controller/payload settings. Preserve held-out episodes; don't fit to them.
See evidence_requirements.json for signal requirements and conditional evidence.
Do not deliberately saturate the robot to satisfy an effort-scale recipe.

After collection, bind the real logs to the USD and run analyze again. Only a
fit-ready analysis may continue through plan → fit → validate → write. Missing
clock/saturation evidence may require an explicit CalibrationRequest with a
narrower target_parameters selection (never silently remove a target); this
collection run does not bypass those gates or prove parameter identifiability.
""",
    )
