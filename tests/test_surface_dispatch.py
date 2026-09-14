from __future__ import annotations

import pytest

import newton_calibration.adapters.surface.isaac_lab as isaac_lab_surface
from newton_calibration.adapters.surface import (
    ArticulationEnvCfg,
    IsaacLabCalibrationAdapter,
    SO101EnvCfg,
    package_loader,
)


class _RuntimeStub:
    def close(self) -> None:
        pass


class _VerifiedStub:
    def __init__(self, cfg):
        self.cfg = cfg
        self.requested_device = None

    def to_env_cfg(self, *, device=None):
        self.requested_device = device
        return self.cfg


@pytest.mark.parametrize(
    "cfg",
    [
        SO101EnvCfg(usd_path="legacy.usda", runtime="analytic", device="cpu"),
        ArticulationEnvCfg(
            usd_path="generic.usda",
            robot_id="example-arm",
            joint_groups={"arm": ("axis",)},
            joint_map={"axis": "axis_joint"},
            joint_order=("axis",),
            profile_confirmed=True,
            runtime="analytic",
            device="cpu",
        ),
    ],
)
def test_shared_surface_uses_schema_dispatch_for_legacy_and_generic(monkeypatch, cfg) -> None:
    verified = _VerifiedStub(cfg)
    calls = []

    def _open(package, *, expected_manifest_sha256=None):
        calls.append((package, expected_manifest_sha256))
        return verified

    monkeypatch.setattr(package_loader.VerifiedCalibrationPackage, "open", _open)
    monkeypatch.setattr(isaac_lab_surface, "create_runtime", lambda environment: _RuntimeStub())

    adapter = IsaacLabCalibrationAdapter.from_calibration(
        "/tmp/calibration-package",
        device="cuda:7",
        expected_manifest_sha256="a" * 64,
    )

    assert adapter.cfg is cfg
    assert verified.requested_device == "cuda:7"
    assert calls == [("/tmp/calibration-package", "a" * 64)]
