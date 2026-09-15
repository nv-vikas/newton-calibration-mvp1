from pathlib import Path

import pytest

from newton_calibration.isaaclab import ArticulationEnvCfg, SO101EnvCfg, tuning


@pytest.mark.parametrize("generic", [True, False])
def test_absent_evidence_reports_readiness_and_blocks_fit_plan(tmp_path: Path, generic: bool):
    asset = tmp_path / "robot.usda"
    asset.write_text(
        '#usda 1.0\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {\n'
        'def PhysicsRevoluteJoint "joint1" {}\n}\n'
    )
    env = (
        ArticulationEnvCfg(usd_path=str(asset), joint_groups={"arm": ("joint1",)}, joint_map={"joint1": "joint1"})
        if generic
        else SO101EnvCfg(usd_path=str(asset))
    )
    result = tuning.analyze(env=env, workdir=tmp_path / "runs")
    assert result.evidence_spec["real_samples"] == 0
    assert result.evidence_spec["fit_allowed"] is False
    assert not result.train_episodes and not result.heldout_episodes
    assert not result.identifiable_parameters
    assert result.readiness["required_signals_present"] is False
    assert (Path(result.workdir) / "analysis.json").is_file()
    with pytest.raises(ValueError, match="readiness checks failed"):
        tuning.plan(result, intent="fit")
    assert not (Path(result.workdir) / "plan.json").exists()
    collection = tuning.plan(result)
    assert collection.kind == "evidence_collection"
    assert collection.status == "needs_scene_setup"
    assert collection.preview["requested"] is True
    assert not collection.fit_allowed
    with pytest.raises(TypeError, match="collect and re-analyze"):
        tuning.fit(collection)


def test_missing_asset_still_blocks_absent_evidence(tmp_path: Path):
    env = ArticulationEnvCfg(usd_path=str(tmp_path / "missing.usd"), joint_groups={"arm": ("j",)}, joint_map={"j": "j"})
    result = tuning.analyze(env=env, evidence=None, workdir=tmp_path / "runs")
    assert not result.readiness["asset_exists"]
    assert not result.readiness["asset_profile_valid"]
