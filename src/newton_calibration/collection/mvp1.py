"""MVP1 free-motion evidence requirements and deterministic experiment selection.

Only this module knows actuator parameter semantics. Future domains can provide
their own requirement catalog and selector without changing the preview runner.
These checks establish evidence eligibility, NOT numerical identifiability.
"""

from __future__ import annotations

from typing import Any

from .contracts import CalibrationRequest, ExperimentSpec

VERSION = "mvp1.free-motion-design@1"
SIGNALS = ("command_q", "actual_q", "actual_dq")
FAMILIES = {
    "stiffness_scale": ("servo_sweep@1", "settling@1"),
    "damping_scale": ("servo_sweep@1", "settling@1"),
    "armature": ("acceleration_sweep@1",),
    "friction_nm": ("slow_reversal@1",),
    "effort_scale": (),  # Never command saturation to qualify this parameter.
    "command_delay_s": ("servo_sweep@1",),
}
LIMITATIONS = {
    "stiffness_scale": "Effective closed-loop gain; OEM gains and physical inertia can be confounded.",
    "damping_scale": "Effective damping; controller filtering, friction and delay can be confounded.",
    "armature": "One-pose encoder motion cannot uniquely separate gain, link inertia and motor inertia. Fix independently known terms or obtain calibrated effort and additional reviewed poses.",
    "friction_nm": "Slow reversals support a friction proxy; absolute friction torque needs an independently anchored controller/dynamics model or calibrated effort.",
    "effort_scale": "Needs existing approved saturation evidence and calibrated effort interpretation. No saturation-seeking motion is generated.",
    "command_delay_s": "Effective latency only; clock offsets and controller smoothing must be characterized separately.",
}


def assess_requirements(environment, requested, inventory, eligible, request: CalibrationRequest) -> dict[str, Any]:
    """Use training evidence only to identify missing experiment coverage."""
    mapping = environment.joint_map
    groups = environment.joint_groups
    if not groups:  # Preserve the legacy SO-101 profile without encoding a DOF count.
        groups = {"arm": tuple(j for j in mapping if j != "jaw"), "gripper": tuple(j for j in mapping if j == "jaw")}
    surface = {
        f"{g}_{suffix}": (suffix, tuple(members))
        for g, members in groups.items()
        for suffix in FAMILIES
        if suffix != "command_delay_s"
    }
    surface["command_delay_s"] = ("command_delay_s", tuple(mapping))
    rows = []
    for name in requested:
        if name not in surface:
            raise ValueError(f"No supported MVP1 collection requirement for parameter {name!r}")
        kind, members = surface[name]
        if not members or any(j not in mapping for j in members):
            raise ValueError(f"Incomplete logical-to-USD mapping for {name}")
        observed = name in eligible
        coverage_key = "reversal_joints" if kind == "friction_nm" else "dynamic_excitation_joints"
        covered_joints = set(inventory.get(coverage_key, ())) if inventory.get("train_episodes") else set()
        # Global stream lists alone do not prove per-joint coverage. Inventory's
        # excitation/reversal sets are computed from training command + feedback.
        missing_joints = tuple(j for j in members if j not in covered_joints)
        required = SIGNALS + (("synchronized_command_feedback_timestamps",) if kind == "command_delay_s" else ())
        if kind == "effort_scale":
            required += ("calibrated_effort", "approved_saturation_observation")
        unavailable = [] if request.available_signals is None else sorted(set(SIGNALS) - set(request.available_signals))
        if observed:
            disposition = "existing_evidence_eligible"
        elif kind == "effort_scale":
            disposition = "external_evidence_required"
        elif unavailable or (kind == "command_delay_s" and request.clock_synchronized is False):
            disposition = "instrumentation_blocked"
        elif kind == "command_delay_s" and not missing_joints and inventory.get("train_episodes"):
            disposition = "clock_evidence_required"
        else:
            disposition = "collect"
        rows.append(
            {
                "parameter": name,
                "kind": kind,
                "logical_joints": list(members),
                "usd_joints": [mapping[j] for j in members],
                "missing_usd_joints": [mapping[j] for j in missing_joints],
                "disposition": disposition,
                "experiment_recipes": list(FAMILIES[kind]),
                "required_signals": list(required),
                "unavailable_signals": unavailable,
                "capabilities_confirmed": request.available_signals is not None
                and not unavailable
                and (kind != "command_delay_s" or request.clock_synchronized is True),
                "eligibility_reason": eligible.get(
                    name, "Training evidence does not yet meet the recipe's signal/excitation checks"
                ),
                "limitation": LIMITATIONS[kind],
                "parameter_sharing": "shared across group"
                if len(members) > 1 and kind != "command_delay_s"
                else "single coordinate or global timing",
            }
        )
    return {
        "schema": "newton.evidence-needs/v1",
        "catalog": VERSION,
        "parameters": rows,
        "heldout_needed": not bool(inventory.get("heldout_episodes")),
        "training_evidence_quality_by_joint": inventory.get("evidence_quality_by_joint", {}),
        "missing_signals_by_episode": inventory.get("missing_by_episode", {}),
        "basis": "Training measurements and declared capabilities; held-out motion is not used for experiment selection",
        "numerical_identifiability_proven": False,
    }


def select_experiments(needs: dict, request: CalibrationRequest, joint_order: tuple[str, ...]):
    """Deduplicate shared experiments; never silently drop a budget-limited item."""
    grouped: dict[tuple[str, str], dict] = {}
    for row in needs["parameters"]:
        if row["disposition"] != "collect":
            continue
        for family in row["experiment_recipes"]:
            for joint in row["missing_usd_joints"] or row["usd_joints"]:
                entry = grouped.setdefault((family, joint), {"targets": [], "signals": []})
                entry["targets"].append(row["parameter"])
                entry["signals"].extend(row["required_signals"])
    selected, deferred = [], []
    for (family, joint), entry in grouped.items():
        if joint not in joint_order:
            raise ValueError(f"Experiment joint {joint!r} is absent from the motion envelope")
        name = f"train_{len(selected) + len(deferred) + 1:03d}_joint_{joint_order.index(joint) + 1:02d}_{family.split('@')[0]}"
        experiment = ExperimentSpec(
            name,
            family,
            "train",
            (joint,),
            tuple(dict.fromkeys(entry["targets"])),
            tuple(dict.fromkeys(entry["signals"])),
            "Fill missing training evidence for the named target parameters",
        )
        if len(selected) < request.max_training_experiments:
            selected.append(experiment)
        else:
            deferred.append(experiment)
    # Different trajectories, reserved before fitting. No held-out outcomes are
    # used to adapt train motions. If all evidence exists, don't recollect it.
    if selected or needs["heldout_needed"]:
        targets = tuple(
            row["parameter"]
            for row in needs["parameters"]
            if row["disposition"] in {"collect", "existing_evidence_eligible"}
        )
        involved = {j for row in needs["parameters"] if row["parameter"] in targets for j in row["usd_joints"]}
        joints = tuple(j for j in joint_order if j in involved)
        if joints:
            for variant in range(2):
                selected.append(
                    ExperimentSpec(
                        f"heldout_combined_{variant + 1:02d}",
                        "heldout_multisine@1",
                        "heldout",
                        joints,
                        targets,
                        SIGNALS,
                        "Independent held-out trajectory, not a parameter identification claim",
                        variant,
                    )
                )
    return selected, deferred
