from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from newton_calibration.adapters.evidence import TabularJointEvidence
from newton_calibration.core import JointBinding, LongFormSchema, SignalBinding, bind_evidence_files
from newton_calibration.isaaclab import (
    ArticulationEnvCfg,
    CalibrationPackageLoadError,
    VerifiedArticulationPackage,
    VerifiedCalibrationPackage,
    tuning,
)

LOGICAL_JOINTS = tuple([f"encoder_axis_{index}" for index in range(1, 8)] + ["left_grip", "right_grip"])
USD_JOINTS = tuple([f"robot_joint_{index}" for index in range(1, 8)] + ["finger_a", "finger_b"])


def _write_episode(path: Path, phase: float) -> None:
    dt = 0.02
    time_s = np.arange(0.0, 0.8, dt)
    rows = []
    for index, joint in enumerate(LOGICAL_JOINTS):
        command = 0.08 * np.sin(2.0 * np.pi * (1.5 + 0.1 * index) * time_s + phase)
        actual = np.empty_like(command)
        actual[0] = command[0]
        for step in range(1, len(time_s)):
            actual[step] = actual[step - 1] + 0.18 * (command[step - 1] - actual[step - 1])
        velocity = np.gradient(actual, dt)
        for signal, values in (("target", command), ("position", actual), ("velocity", velocity)):
            for timestamp, value in zip(time_s, values):
                rows.append(
                    {
                        "time_s": timestamp,
                        "coordinate": joint,
                        "channel": signal,
                        "value": value,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_arbitrary_named_nine_dof_articulation_runs_all_five_calls(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    episodes = []
    for split_index, (split, count) in enumerate((("train", 2), ("heldout", 2))):
        for index in range(count):
            name = f"{split}_{index}"
            _write_episode(evidence_dir / f"{name}.csv", phase=(split_index * count + index) * 0.15)
            episodes.append(
                {"name": name, "path": f"{name}.csv", "split": split, "trial_id": f"capture-{split}-{index}"}
            )

    asset = tmp_path / "unrelated_vendor_names.usda"
    joint_declarations = "\n".join(f'    def PhysicsRevoluteJoint "{joint}" {{}}' for joint in USD_JOINTS)
    asset.write_text(
        '#usda 1.0\n\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])\n{\n'
        + joint_declarations
        + "\n}\n",
        encoding="utf-8",
    )
    bindings = tuple(
        JointBinding(source, target, source_unit="rad", usd_unit="rad", transform_confirmed=True)
        for source, target in zip(LOGICAL_JOINTS, USD_JOINTS)
    )
    evidence_spec = bind_evidence_files(
        root=evidence_dir,
        episodes=episodes,
        schema=LongFormSchema(
            time_column="time_s",
            time_unit="s",
            value_column="value",
            joint_column="coordinate",
            signal_column="channel",
            field_column=None,
        ),
        joint_bindings=bindings,
        signal_bindings=(
            SignalBinding("target", "command_q"),
            SignalBinding("position", "actual_q"),
            SignalBinding("velocity", "actual_dq"),
        ),
        revision="fixture-r1",
        clock_synchronized=True,
        effort_saturation_joints=LOGICAL_JOINTS,
    )
    joint_map = dict(zip(LOGICAL_JOINTS, USD_JOINTS))
    groups = {"manipulator": LOGICAL_JOINTS[:7], "end_effector": LOGICAL_JOINTS[7:]}
    parameter_bounds = {
        "manipulator_effort_scale": (0.25, 1.0, 0.5),
        "manipulator_armature": (0.0, 0.5, 0.02),
        "manipulator_friction_nm": (0.0, 0.5, 0.01),
        "end_effector_effort_scale": (0.25, 1.0, 0.5),
        "end_effector_armature": (0.0, 0.1, 0.005),
        "end_effector_friction_nm": (0.0, 0.2, 0.01),
    }
    env = ArticulationEnvCfg(
        usd_path=str(asset),
        robot_id="vendor-neutral-9dof",
        joint_groups=groups,
        joint_map=joint_map,
        profile_confirmed=True,
        controller_profile_confirmed=True,
        controller_profile_source="fixture controller configuration r1",
        runtime="analytic",
        device="cpu",
        dt=0.02,
        base_stiffness=20.0,
        base_damping=1.0,
        base_effort_limit=20.0,
        base_stiffness_by_joint={name: 20.0 for name in LOGICAL_JOINTS},
        base_damping_by_joint={name: 1.0 for name in LOGICAL_JOINTS},
        base_effort_limit_by_joint={name: 20.0 for name in LOGICAL_JOINTS},
        analytic_inertia_by_joint={name: 1.0 + 0.1 * index for index, name in enumerate(LOGICAL_JOINTS)},
        parameter_bounds=parameter_bounds,
    )

    unsynchronized = tuning.analyze(
        env=env,
        evidence=TabularJointEvidence(replace(evidence_spec, clock_synchronized=False)),
        workdir=tmp_path / "unsynchronized-runs",
    )
    assert not unsynchronized.readiness["requested_parameters_identifiable"]
    assert "command_delay_s" not in unsynchronized.identifiability

    saturation_unobserved = tuning.analyze(
        env=env,
        evidence=TabularJointEvidence(replace(evidence_spec, effort_saturation_joints=())),
        workdir=tmp_path / "unsaturated-runs",
    )
    assert not saturation_unobserved.readiness["requested_parameters_identifiable"]
    assert "manipulator_effort_scale" not in saturation_unobserved.identifiability
    assert "end_effector_effort_scale" not in saturation_unobserved.identifiability

    degree_target = replace(evidence_spec.joint_bindings[0], usd_unit="deg")
    wrong_runtime_units = tuning.analyze(
        env=env,
        evidence=TabularJointEvidence(
            replace(
                evidence_spec,
                joint_bindings=(degree_target, *evidence_spec.joint_bindings[1:]),
            )
        ),
        workdir=tmp_path / "wrong-unit-runs",
    )
    assert not wrong_runtime_units.readiness["joint_map_complete"]
    assert any("requires target units in radians" in item for item in wrong_runtime_units.mapping_report["blockers"])

    analysis = tuning.analyze(
        env=env,
        evidence=TabularJointEvidence(evidence_spec),
        workdir=tmp_path / "runs",
    )
    assert all(analysis.readiness.values()), analysis.warnings
    assert analysis.mapping_report["ready"]
    assert analysis.recipe == "articulation.position_pd.free_space@3"

    plan = tuning.plan(analysis)
    fit = tuning.fit(plan, generations=1, population=4, resume=False)
    validation = tuning.validate(fit)
    package = tuning.write(validation, output=tmp_path / "package")

    manifest = json.loads(Path(package.manifest_path).read_text(encoding="utf-8"))
    patch = Path(package.isaaclab_cfg_path).read_text(encoding="utf-8")
    assert manifest["schema"] == "newton.calibration.package/v2"
    assert manifest["runtime"]["robot_id"] == "vendor-neutral-9dof"
    assert len(manifest["runtime"]["joint_map"]) == 9
    assert "encoder_axis_7" in patch
    assert "finger_b" in patch
    assert Path(package.output_dir, "joint_mapping.json").is_file()
    assert Path(package.output_dir, "evidence_spec.json").is_file()
    with pytest.raises(CalibrationPackageLoadError, match="status|approved"):
        VerifiedArticulationPackage.open(package.output_dir)

    # A CPU contract run cannot be promoted to an activatable Newton package by
    # editing result dataclasses. Only a real runtime attestation can cross that
    # boundary.
    assert manifest["activation_checks"]["authoritative_fit_attestation"] is False
    assert manifest["activation_checks"]["authoritative_validation_attestation"] is False
    with pytest.raises(CalibrationPackageLoadError, match="status|approved"):
        VerifiedCalibrationPackage.open(package.output_dir)

    relabeled_environment = replace(fit.plan.environment, adapter="isaaclab_newton")
    relabeled_plan = replace(fit.plan, environment=relabeled_environment)
    relabeled_fit = replace(fit, plan=relabeled_plan, backend="isaaclab_newton")
    relabeled_validation = replace(
        validation,
        fit=relabeled_fit,
        improvement_pct=35.0,
        regressions=[],
        passed=True,
        gates={name: True for name in validation.gates},
    )
    with pytest.raises(RuntimeError, match="plan changed|durable"):
        tuning.write(relabeled_validation, output=tmp_path / "relabeled-package")
