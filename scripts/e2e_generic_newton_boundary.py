"""Exercise the robot-neutral mapping/runtime boundary with real SO-101 evidence.

This is deliberately not a calibration claim.  Anchor-Lab does not publish an
independent actuator-saturation declaration, so the generic recipe correctly
blocks effort-scale identification.  The script instead proves that the
generic surface can bind selected real coordinates to non-contiguous USD DOFs,
leave other DOFs passive, reset the complete articulation between episodes,
and replay the mapped commands in the real Isaac Lab/Newton runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from newton_calibration import JointBinding, LongFormSchema, SignalBinding, bind_evidence_files
from newton_calibration.adapters.evidence import TabularJointEvidence
from newton_calibration.adapters.runtime import create_runtime
from newton_calibration.core.models import jsonable
from newton_calibration.isaaclab import ArticulationEnvCfg

_SOURCE_JOINTS = ("rotation", "elbow", "wrist_roll", "jaw")
_USD_JOINTS = ("shoulder_pan", "elbow_flex", "wrist_roll", "gripper")
_EPISODES = (
    (
        "train-chirp-sweep",
        "so101-sysid-50motion-train-chirp-sweep.parquet",
        "train",
    ),
    (
        "heldout-frequency-sweep",
        "so101-sysid-50motion-heldout-frequency-sweep.parquet",
        "heldout",
    ),
)
_OBJECTIVE = {
    "position_nrmse": 0.60,
    "velocity_nrmse": 0.15,
    "phase_error_s": 0.15,
    "hold_nrmse": 0.10,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify generic real-data-to-USD mapping in Isaac Lab/Newton"
    )
    parser.add_argument("--evidence", required=True, help="Anchor-Lab root or 50-motion directory")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    evidence_root = _resolve_evidence_root(Path(args.evidence))
    episodes = [
        {
            "name": name,
            "path": filename,
            "split": split,
            "trial_id": f"anchor-lab:{name}",
        }
        for name, filename, split in _EPISODES
    ]
    bindings = tuple(
        JointBinding(
            source_joint=source,
            usd_joint=target,
            source_unit="rad",
            usd_unit="rad",
            transform_confirmed=True,
        )
        for source, target in zip(_SOURCE_JOINTS, _USD_JOINTS)
    )
    evidence_spec = bind_evidence_files(
        root=evidence_root,
        episodes=episodes,
        schema=LongFormSchema(
            time_column="time_ns",
            time_unit="ns",
            value_column="value",
            field_column="field",
        ),
        joint_bindings=bindings,
        signal_bindings=(
            SignalBinding("command_q", "command_q"),
            SignalBinding("actual_q", "actual_q"),
            SignalBinding("dq", "actual_dq"),
        ),
        revision="anchor-lab:647edd5787cd764cdc041103ad282dc59214d919",
        clock_synchronized=True,
        # Intentionally empty. The runtime boundary does not pretend the logs
        # prove actuator saturation, and it does not create a calibration plan.
        effort_saturation_joints=(),
    )
    env = ArticulationEnvCfg(
        usd_path=str(Path(args.asset).expanduser().resolve()),
        robot_id="so101-generic-runtime-boundary",
        joint_groups={
            "outer_arm": ("rotation", "elbow", "wrist_roll"),
            "tool": ("jaw",),
        },
        joint_map=dict(zip(_SOURCE_JOINTS, _USD_JOINTS)),
        joint_order=_SOURCE_JOINTS,
        profile_confirmed=True,
        controller_profile_confirmed=True,
        controller_profile_source="SO-101 Anchor-Lab reference profile",
        runtime="isaaclab_newton",
        device=args.device,
        base_stiffness_by_joint={name: 1.7453293 for name in _SOURCE_JOINTS},
        base_damping_by_joint={name: 0.017453292 for name in _SOURCE_JOINTS},
        base_effort_limit_by_joint={name: 10.0 for name in _SOURCE_JOINTS},
        parameter_bounds={
            "outer_arm_effort_scale": (0.005, 0.2, 0.05),
            "outer_arm_armature": (0.001, 0.08, 0.02),
            "outer_arm_friction_nm": (0.0, 0.35, 0.02),
            "tool_effort_scale": (0.001, 0.05, 0.01),
            "tool_armature": (0.0001, 0.03, 0.005),
            "tool_friction_nm": (0.0, 0.25, 0.01),
        },
    )
    evidence = TabularJointEvidence(evidence_spec)
    loaded = [
        evidence.load_episode(name, dt=env.dt, max_duration_s=args.duration)
        for name, _, _ in _EPISODES
    ]
    runtime = create_runtime(env.describe())
    try:
        forward = runtime.evaluate(
            {},
            loaded,
            _OBJECTIVE,
            phase="generic-boundary-forward",
            run_id="generic-boundary",
            plan_sha256="0" * 64,
            evidence_fingerprint=evidence_spec.fingerprint,
            mapping_fingerprint=evidence_spec.mapping_fingerprint,
        )
        forward_attestation = runtime.attestation()
        reverse = runtime.evaluate(
            {},
            list(reversed(loaded)),
            _OBJECTIVE,
            phase="generic-boundary-reverse",
            run_id="generic-boundary",
            plan_sha256="0" * 64,
            evidence_fingerprint=evidence_spec.fingerprint,
            mapping_fingerprint=evidence_spec.mapping_fingerprint,
        )
        reverse_attestation = runtime.attestation()
    finally:
        runtime.close()

    deterministic = bool(
        np.isclose(forward.score, reverse.score, rtol=1e-7, atol=1e-9)
        and forward.stable
        and reverse.stable
        and forward.episodes == reverse.episodes
    )
    expected_runtime_joints = list(_USD_JOINTS)
    boundary_passed = bool(
        deterministic
        and forward_attestation.get("logical_joints") == list(_SOURCE_JOINTS)
        and forward_attestation.get("runtime_joints") == expected_runtime_joints
        and forward_attestation.get("selected_joint_scoped") is True
        and forward_attestation.get("full_state_reset_per_episode") is True
        and reverse_attestation.get("full_state_reset_per_episode") is True
    )
    result = {
        "schema": "newton.calibration.generic-runtime-boundary/v1",
        "passed": boundary_passed,
        "scientific_calibration_claim": False,
        "reason": (
            "Anchor-Lab lacks an independent effort-saturation declaration; "
            "this verifies mapping and runtime behavior, not a generic fitted package."
        ),
        "source_joints": list(_SOURCE_JOINTS),
        "runtime_joints": expected_runtime_joints,
        "passive_runtime_joints": ["shoulder_lift", "wrist_flex"],
        "forward": jsonable(forward),
        "reverse": jsonable(reverse),
        "deterministic_after_episode_reorder": deterministic,
        "attestation": reverse_attestation,
    }
    print(json.dumps(result, indent=2))
    if not boundary_passed:
        raise SystemExit(1)


def _resolve_evidence_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    nested = root / "data" / "so101_arm_50motion"
    if nested.is_dir():
        return nested
    if root.is_dir():
        return root
    raise FileNotFoundError(f"Anchor-Lab evidence directory does not exist: {root}")


if __name__ == "__main__":
    main()
