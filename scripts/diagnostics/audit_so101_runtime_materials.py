#!/usr/bin/env python3
"""Read-only audit of finalized SO-101 peg/finger materials in Newton/MJWarp.

The script constructs the real Isaac Lab scene, then reads the already-finalized
Newton model and the MuJoCo/MJWarp model derived from it.  It does not step the
controller, write physics properties, or mutate the model.

The JSON report distinguishes:
  * authored/imported Newton per-shape material values;
  * values compiled into the MuJoCo solver geoms;
  * values resident in the actual MJWarp device model;
  * the material that MuJoCo resolves for each peg/finger pair.

The pinned runtime does not necessarily retain MuJoCo-only ``priority`` and
``condim`` custom attributes on the finalized Newton model.  When absent, this
audit reads them from ``solver.mj_model`` (the compiled model produced by
Newton's SolverMuJoCo) and corroborates them against ``solver.mjw_model``.  The
report records that provenance rather than presenting solver defaults as
authored Newton fields.

"friction_path_active" means the native MuJoCo contact path is enabled, contact
constraints are not globally disabled, the pair can collide, the resolved
constraint dimension includes tangential friction, and the resolved sliding
coefficient is positive.  It does not claim that the coefficient is physically
correct or sufficient to retain the peg; that requires a controlled A/B run.

Run from the repository root after the active GPU job has finished::

    docker run --rm --gpus all --ipc=host \
      -e ACCEPT_EULA=Y \
      -e PYTHONPATH=/workspace/newton-calibration-mvp1/src \
      -v "$PWD:/workspace/newton-calibration-mvp1:ro" \
      -v "$PWD/output:/workspace/output" \
      -w /workspace/newton-calibration-mvp1 \
      --entrypoint /workspace/isaaclab/isaaclab.sh \
      newton-calibration-mvp1:latest \
      -p scripts/diagnostics/audit_so101_runtime_materials.py \
      --headless --device cuda:0 \
      --asset /workspace/newton-calibration-mvp1/data/anchor-lab/robot_assets/so101_no_camera_new_calib_task_collision_v4.usda \
      --output /workspace/output/controller_commission/runtime_material_audit_v4.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import traceback
from typing import Any

if os.environ.get("ACCEPT_EULA") != "Y":
    raise RuntimeError("Review the NVIDIA Isaac Sim license and export ACCEPT_EULA=Y before running.")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--asset", required=True, help="SO-101 asset with a supported v3/v4 contact profile"
)
parser.add_argument("--output", required=True, help="Destination JSON report")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import mujoco

from isaaclab_newton.physics.newton_manager import NewtonManager
from newton_calibration.isaaclab.tasks.so101_peg_insertion import PegInsertionMode
from newton_calibration.isaaclab.tasks.so101_peg_insertion.scene import (
    SO101PegInsertionScene,
)


ABS_TOL = 1.0e-6


def _numpy(value: Any) -> np.ndarray:
    """Convert Warp/Torch/NumPy-like storage to a detached host array."""

    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _scalar(value: Any) -> Any:
    """Convert a scalar-like solver value into a JSON scalar."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size != 1:
        return array.tolist()
    item = array.reshape(-1)[0].item()
    return item


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role(
    shape_path: str,
    body_path: str,
    *,
    fixed_shape_expr: str,
    moving_shape_expr: str,
) -> str | None:
    """Classify only shapes declared by the asset-selected contact profile."""

    if re.fullmatch(fixed_shape_expr, shape_path):
        return "fixed_finger_proxy"
    if re.fullmatch(moving_shape_expr, shape_path):
        return "moving_finger_proxy"
    if body_path.rstrip("/") == "/World/Env_0/Peg" or "/Peg/" in shape_path:
        return "peg"
    return None


def _warp_geom_row(array_like: Any, geom_id: int, width: int | None = None) -> np.ndarray:
    """Read a per-geom row from shared or [world, geom, ...] MJWarp storage."""

    array = _numpy(array_like)
    if width is not None:
        if array.ndim == 3:
            return np.asarray(array[0, geom_id], dtype=np.float64)
        if array.ndim == 2 and array.shape[-1] == width:
            return np.asarray(array[geom_id], dtype=np.float64)
    else:
        if array.ndim == 2:
            return np.asarray(array[0, geom_id])
        if array.ndim == 1:
            return np.asarray(array[geom_id])
    raise RuntimeError(
        f"Unexpected per-geom storage shape {array.shape} for geom {geom_id}"
    )


def _resolved_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Apply MuJoCo's priority/equal-priority max material rule."""

    pa = int(a["mjwarp"]["geom_priority"])
    pb = int(b["mjwarp"]["geom_priority"])
    fa = np.asarray(a["mjwarp"]["friction_3"], dtype=np.float64)
    fb = np.asarray(b["mjwarp"]["friction_3"], dtype=np.float64)
    ca = int(a["mjwarp"]["condim"])
    cb = int(b["mjwarp"]["condim"])

    if pa > pb:
        owner = "peg"
        friction_3 = fa
        condim = ca
    elif pb > pa:
        owner = b["role"]
        friction_3 = fb
        condim = cb
    else:
        owner = "componentwise_max_equal_priority"
        friction_3 = np.maximum(fa, fb)
        condim = max(ca, cb)

    # MuJoCo stores one geom coefficient for each of sliding, torsional, and
    # rolling friction, while a contact carries the expanded five-vector.
    friction_5 = [
        float(friction_3[0]),
        float(friction_3[0]),
        float(friction_3[1]),
        float(friction_3[2]),
        float(friction_3[2]),
    ]
    collision_mask_compatible = bool(
        (int(a["mjwarp"]["contype"]) & int(b["mjwarp"]["conaffinity"]))
        or (int(b["mjwarp"]["contype"]) & int(a["mjwarp"]["conaffinity"]))
    )
    return {
        "peg_shape_id": a["shape_id"],
        "peg_shape_path": a["shape_path"],
        "finger_role": b["role"],
        "finger_shape_id": b["shape_id"],
        "finger_shape_path": b["shape_path"],
        "priority_rule": owner,
        "resolved_geom_priority": max(pa, pb),
        "resolved_condim": condim,
        "resolved_friction_3": [float(value) for value in friction_3],
        "resolved_contact_friction_5": friction_5,
        "collision_mask_compatible": collision_mask_compatible,
        "tangential_friction_constraint_configured": bool(
            collision_mask_compatible and condim >= 3 and friction_3[0] > 0.0
        ),
    }


def _audit() -> dict[str, Any]:
    asset = Path(args.asset).expanduser().resolve()
    if not asset.is_file():
        raise FileNotFoundError(asset)

    # Scene reset finalizes the imported model.  No controller or physics step
    # follows; all accesses below are read-only.
    scene = SO101PegInsertionScene(
        usd_path=str(asset),
        device=args.device,
        mode=PegInsertionMode.GRASP_TRANSPORT,
    )
    print("[audit] stage=scene_finalized", flush=True)
    profile = scene.grasp_contact_shape_profile
    if profile is None:
        raise RuntimeError(
            "GRASP_TRANSPORT scene did not resolve an asset-declared grasp-contact profile"
        )
    expected_part_names = tuple(profile.expected_part_names)
    if not expected_part_names or len(set(expected_part_names)) != len(
        expected_part_names
    ):
        raise RuntimeError(
            f"Invalid expected shape-name contract for {profile.profile_id!r}: "
            f"{expected_part_names}"
        )
    print(
        "[audit] stage=contact_profile_resolved "
        f"profile={profile.profile_id} per_channel={len(expected_part_names)}",
        flush=True,
    )
    model = NewtonManager._model
    solver = NewtonManager._solver
    if model is None or solver is None:
        raise RuntimeError("Newton model/solver was not finalized")
    if not hasattr(solver, "mjw_model") or solver.mjw_model is None:
        raise RuntimeError("Expected the live Newton SolverMuJoCo/MJWarp model")
    print("[audit] stage=runtime_handles_resolved", flush=True)

    labels = [str(value) for value in model.shape_label]
    body_labels = [str(value) for value in model.body_label]
    shape_body = _numpy(model.shape_body).astype(np.int64, copy=False)
    shape_mu = _numpy(model.shape_material_mu).astype(np.float64, copy=False)
    shape_mu_torsional = _numpy(model.shape_material_mu_torsional).astype(
        np.float64, copy=False
    )
    shape_mu_rolling = _numpy(model.shape_material_mu_rolling).astype(
        np.float64, copy=False
    )
    namespace = getattr(model, "mujoco", None)
    priority_attribute = (
        getattr(namespace, "geom_priority", None) if namespace is not None else None
    )
    condim_attribute = (
        getattr(namespace, "condim", None) if namespace is not None else None
    )
    shape_priority = (
        _numpy(priority_attribute).astype(np.int64, copy=False)
        if priority_attribute is not None
        else None
    )
    shape_condim = (
        _numpy(condim_attribute).astype(np.int64, copy=False)
        if condim_attribute is not None
        else None
    )
    newton_retains_solver_metadata = bool(
        shape_priority is not None and shape_condim is not None
    )
    if newton_retains_solver_metadata:
        solver_metadata_provenance = "newton.model.mujoco custom attributes"
    else:
        solver_metadata_provenance = (
            "solver.mj_model compiled by Newton SolverMuJoCo; corroborated "
            "against the live solver.mjw_model"
        )
    print(
        "[audit] stage=solver_metadata_provenance "
        f"newton_retained={newton_retains_solver_metadata}",
        flush=True,
    )

    geom_to_shape = _numpy(solver.mjc_geom_to_newton_shape).astype(
        np.int64, copy=False
    )
    if geom_to_shape.ndim != 2 or geom_to_shape.shape[0] != 1:
        raise RuntimeError(
            "This audit requires the one-world reference scene; mapping shape is "
            f"{geom_to_shape.shape}"
        )
    shape_to_geoms: dict[int, list[int]] = {}
    for geom_id, shape_id in enumerate(geom_to_shape[0].tolist()):
        if shape_id >= 0:
            shape_to_geoms.setdefault(int(shape_id), []).append(int(geom_id))
    print("[audit] stage=shape_mapping_loaded", flush=True)

    mj = solver.mj_model
    mjw = solver.mjw_model
    selected: list[dict[str, Any]] = []
    for shape_id, shape_path in enumerate(labels):
        body_id = int(shape_body[shape_id])
        body_path = body_labels[body_id] if body_id >= 0 else "<static-world>"
        role = _role(
            shape_path,
            body_path,
            fixed_shape_expr=profile.fixed_shape_expr,
            moving_shape_expr=profile.moving_shape_expr,
        )
        if role is None:
            continue
        geoms = shape_to_geoms.get(shape_id, [])
        record: dict[str, Any] = {
            "shape_id": shape_id,
            "shape_path": shape_path,
            "body_id": body_id,
            "body_path": body_path,
            "role": role,
            "newton": {
                "geom_priority": (
                    int(shape_priority[shape_id])
                    if shape_priority is not None
                    else None
                ),
                "condim": (
                    int(shape_condim[shape_id])
                    if shape_condim is not None
                    else None
                ),
                "friction_3": [
                    float(shape_mu[shape_id]),
                    float(shape_mu_torsional[shape_id]),
                    float(shape_mu_rolling[shape_id]),
                ],
                "solver_metadata_retained": newton_retains_solver_metadata,
            },
            "solver_geom_ids": geoms,
        }
        if len(geoms) != 1:
            record["mapping_error"] = (
                f"expected exactly one MJ geom, found {len(geoms)}"
            )
            selected.append(record)
            continue
        geom_id = geoms[0]
        mj_friction = np.asarray(mj.geom_friction[geom_id], dtype=np.float64)
        mjw_friction = _warp_geom_row(mjw.geom_friction, geom_id, width=3)
        mjw_priority = int(_warp_geom_row(mjw.geom_priority, geom_id))
        mjw_condim = int(_warp_geom_row(mjw.geom_condim, geom_id))
        mjw_contype = int(_warp_geom_row(mjw.geom_contype, geom_id))
        mjw_conaffinity = int(_warp_geom_row(mjw.geom_conaffinity, geom_id))
        record["mujoco_compiled"] = {
            "geom_id": geom_id,
            "geom_priority": int(mj.geom_priority[geom_id]),
            "condim": int(mj.geom_condim[geom_id]),
            "friction_3": [float(value) for value in mj_friction],
            "contype": int(mj.geom_contype[geom_id]),
            "conaffinity": int(mj.geom_conaffinity[geom_id]),
        }
        record["mjwarp"] = {
            "geom_id": geom_id,
            "geom_priority": mjw_priority,
            "condim": mjw_condim,
            "friction_3": [float(value) for value in mjw_friction],
            "contype": mjw_contype,
            "conaffinity": mjw_conaffinity,
        }
        expected = np.asarray(record["newton"]["friction_3"], dtype=np.float64)
        newton_metadata_matches_mujoco = (
            record["newton"]["geom_priority"]
            == record["mujoco_compiled"]["geom_priority"]
            and record["newton"]["condim"]
            == record["mujoco_compiled"]["condim"]
            if newton_retains_solver_metadata
            else None
        )
        newton_metadata_matches_mjwarp = (
            record["newton"]["geom_priority"] == mjw_priority
            and record["newton"]["condim"] == mjw_condim
            if newton_retains_solver_metadata
            else None
        )
        record["checks"] = {
            "newton_friction_matches_mujoco": bool(
                np.allclose(expected, mj_friction, rtol=0.0, atol=ABS_TOL)
            ),
            "newton_friction_matches_mjwarp": bool(
                np.allclose(expected, mjw_friction, rtol=0.0, atol=ABS_TOL)
            ),
            "newton_solver_metadata_matches_mujoco": (
                newton_metadata_matches_mujoco
            ),
            "newton_solver_metadata_matches_mjwarp": (
                newton_metadata_matches_mjwarp
            ),
            "compiled_mujoco_matches_live_mjwarp": bool(
                np.allclose(mj_friction, mjw_friction, rtol=0.0, atol=ABS_TOL)
                and record["mujoco_compiled"]["geom_priority"] == mjw_priority
                and record["mujoco_compiled"]["condim"] == mjw_condim
                and record["mujoco_compiled"]["contype"] == mjw_contype
                and record["mujoco_compiled"]["conaffinity"] == mjw_conaffinity
            ),
        }
        selected.append(record)

    pegs = [item for item in selected if item["role"] == "peg"]
    fingers = [
        item
        for item in selected
        if item["role"] in {"fixed_finger_proxy", "moving_finger_proxy"}
    ]
    fixed_fingers = [
        item for item in fingers if item["role"] == "fixed_finger_proxy"
    ]
    moving_fingers = [
        item for item in fingers if item["role"] == "moving_finger_proxy"
    ]
    mapped = [item for item in selected if "mjwarp" in item]
    if len(pegs) != 1:
        raise RuntimeError(f"Expected one peg collision shape, found {len(pegs)}")
    expected_count = len(expected_part_names)
    expected_name_set = set(expected_part_names)
    fixed_name_set = {Path(item["shape_path"]).name for item in fixed_fingers}
    moving_name_set = {Path(item["shape_path"]).name for item in moving_fingers}
    if len(fixed_fingers) != expected_count or fixed_name_set != expected_name_set:
        raise RuntimeError(
            f"Expected {expected_count} fixed shapes for {profile.profile_id!r} "
            f"named {sorted(expected_name_set)}, found {len(fixed_fingers)} "
            f"named {sorted(fixed_name_set)}"
        )
    if len(moving_fingers) != expected_count or moving_name_set != expected_name_set:
        raise RuntimeError(
            f"Expected {expected_count} moving shapes for {profile.profile_id!r} "
            f"named {sorted(expected_name_set)}, found {len(moving_fingers)} "
            f"named {sorted(moving_name_set)}"
        )
    if len(mapped) != len(selected):
        raise RuntimeError("At least one target Newton shape lacks a solver geom")
    print(
        "[audit] stage=target_shapes_resolved "
        f"peg={len(pegs)} fixed={len(fixed_fingers)} "
        f"moving={len(moving_fingers)} mapped={len(mapped)}",
        flush=True,
    )

    pairs = [_resolved_pair(pegs[0], finger) for finger in fingers]
    use_mujoco_contacts = bool(getattr(solver, "_use_mujoco_contacts", False))
    run_collision_detection = bool(_scalar(mjw.opt.run_collision_detection))
    disableflags = int(mj.opt.disableflags)
    contact_disable_bit = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    contacts_globally_disabled = bool(disableflags & contact_disable_bit)
    newton_friction_propagated = all(
        item["checks"]["newton_friction_matches_mujoco"]
        and item["checks"]["newton_friction_matches_mjwarp"]
        for item in mapped
    )
    compiled_solver_metadata_is_live = all(
        item["checks"]["compiled_mujoco_matches_live_mjwarp"]
        for item in mapped
    )
    pair_constraints_configured = all(
        item["tangential_friction_constraint_configured"] for item in pairs
    )
    friction_path_active = bool(
        use_mujoco_contacts
        and run_collision_detection
        and not contacts_globally_disabled
        and newton_friction_propagated
        and compiled_solver_metadata_is_live
        and pair_constraints_configured
    )

    unique_pair_materials = sorted(
        {
            (
                item["priority_rule"],
                item["resolved_geom_priority"],
                item["resolved_condim"],
                tuple(item["resolved_friction_3"]),
            )
            for item in pairs
        },
        key=repr,
    )
    return {
        "schema": "newton.calibration/runtime-material-audit@1.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "read-only finalized Newton/MuJoCo/MJWarp material audit; "
            "no controller or physics stepping"
        ),
        "asset": {"path": str(asset), "sha256": _sha256(asset)},
        "grasp_contact_profile": {
            "profile_id": profile.profile_id,
            "attribution": profile.attribution,
            "fixed_shape_expr": profile.fixed_shape_expr,
            "moving_shape_expr": profile.moving_shape_expr,
            "expected_part_names": list(expected_part_names),
            "expected_shapes_per_channel": expected_count,
        },
        "versions": {
            "isaaclab": _version("isaaclab"),
            "isaaclab_newton": _version("isaaclab-newton"),
            "newton": _version("newton"),
            "mujoco": _version("mujoco"),
            "mujoco_warp": _version("mujoco-warp"),
        },
        "runtime": {
            "solver_class": f"{type(solver).__module__}.{type(solver).__qualname__}",
            "device": str(model.device),
            "world_count": int(model.world_count),
            "use_mujoco_contacts": use_mujoco_contacts,
            "run_collision_detection": run_collision_detection,
            "mujoco_disableflags": disableflags,
            "contact_disable_bit": contact_disable_bit,
            "contacts_globally_disabled": contacts_globally_disabled,
        },
        "field_provenance": {
            "friction": (
                "Newton model.shape_material_mu, "
                "shape_material_mu_torsional and shape_material_mu_rolling; "
                "checked against compiled MuJoCo and live MJWarp"
            ),
            "geom_priority_and_condim": solver_metadata_provenance,
            "collision_masks": (
                "solver.mj_model contype/conaffinity compiled by Newton "
                "SolverMuJoCo; checked against live solver.mjw_model"
            ),
            "newton_model_retains_geom_priority_and_condim": (
                newton_retains_solver_metadata
            ),
        },
        "counts": {
            "peg_shapes": len(pegs),
            "fixed_finger_proxy_shapes": len(fixed_fingers),
            "moving_finger_proxy_shapes": len(moving_fingers),
            "resolved_pairs": len(pairs),
        },
        "shapes": selected,
        "resolved_pairs": pairs,
        "unique_resolved_pair_materials": [
            {
                "priority_rule": rule,
                "geom_priority": priority,
                "condim": condim,
                "friction_3": list(friction),
            }
            for rule, priority, condim, friction in unique_pair_materials
        ],
        "checks": {
            "all_target_shapes_map_to_solver": len(mapped) == len(selected),
            "newton_friction_propagated_to_mujoco_and_mjwarp": (
                newton_friction_propagated
            ),
            "compiled_solver_metadata_matches_live_mjwarp": (
                compiled_solver_metadata_is_live
            ),
            "all_pair_collision_masks_compatible": all(
                item["collision_mask_compatible"] for item in pairs
            ),
            "all_pair_tangential_constraints_configured": pair_constraints_configured,
            "friction_path_active": friction_path_active,
        },
        "conclusion": {
            "status": "PASS" if friction_path_active else "FAIL",
            "meaning": (
                "The finalized live solver will create tangential friction "
                "constraints for peg/profiled-grasp-shape contacts. This establishes "
                "configuration and propagation, not physical sufficiency."
                if friction_path_active
                else "The finalized solver does not establish an active friction path; inspect failed checks."
            ),
            "non_claim": (
                "No claim is made that the resolved coefficient matches the real "
                "pad/peg pair or can retain the peg at the requested lift speed."
            ),
        },
    }


def _write_json(destination: Path, payload: dict[str, Any]) -> None:
    """Durably replace the destination with one complete JSON document."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def main() -> None:
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    try:
        report = _audit()
        _write_json(destination, report)
        print(f"AUDIT={destination}", flush=True)
        print(json.dumps(report["checks"], indent=2), flush=True)
        print(json.dumps(report["unique_resolved_pair_materials"], indent=2), flush=True)
        if report["conclusion"]["status"] != "PASS":
            exit_code = 2
    except BaseException as exc:
        # Isaac/Kit shutdown can suppress an in-flight traceback.  Persist and
        # print the failure before closing the app so a run can never appear to
        # have succeeded merely because no output was produced.
        failure = {
            "schema": "newton.calibration/runtime-material-audit-failure@1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ERROR",
            "asset": str(Path(args.asset).expanduser().resolve()),
            "output": str(destination),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json(destination, failure)
            print(f"AUDIT_FAILURE={destination}", flush=True)
        except BaseException as write_exc:
            print(
                "AUDIT_FAILURE_ARTIFACT_WRITE_ERROR="
                f"{type(write_exc).__name__}: {write_exc}",
                flush=True,
            )
        print(json.dumps(failure, indent=2, allow_nan=False), flush=True)
        exit_code = 2
    finally:
        try:
            simulation_app.close()
        except BaseException as close_exc:
            print(
                "AUDIT_APP_CLOSE_ERROR="
                f"{type(close_exc).__name__}: {close_exc}",
                flush=True,
            )
            exit_code = max(exit_code, 3)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
