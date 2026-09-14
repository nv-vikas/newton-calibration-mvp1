# Loading an MVP1 calibration package

`VerifiedSO101Package` is the kitless trust boundary between calibration output
and an Isaac Lab/Newton run. It does not import Isaac Lab while inspecting a
package.

```python
from newton_calibration.isaaclab import VerifiedSO101Package

package = VerifiedSO101Package.open("packages/so101")
env_cfg = package.to_env_cfg(device="cuda:0")
```

The loader fails closed unless the package is the portable MVP1 profile, uses
the `isaaclab_newton` backend, is approved for activation, passed every held-out
gate, contains all 11 in-bounds parameters exactly once, and has mutually
consistent manifest, validation, actuator YAML, timing, and source-USD records.
It reconstructs the semantic joint order as `rotation`, `pitch`, `elbow`,
`wrist_pitch`, `wrist_roll`, `jaw`; JSON object order is never trusted.

The only direct override is the CUDA device. Timestep, gravity, substeps, solver,
joint bindings, actuator values, command delay, asset, and residual are locked
to the validated package.

Packages created before the run plan recorded `asset_fingerprint` are not
silently upgraded or rewritten. They load only when the caller supplies the
original, separately trusted manifest SHA-256; the returned bundle reports
`verification_level="legacy-trusted-manifest"`. Newly generated packages report
`verification_level="complete-v1"`.

## Application surfaces

The package is more than an actuator YAML:

| Surface | Values |
|---|---|
| Isaac Lab explicit PD | Arm/gripper stiffness, damping, and effort caps |
| Newton articulation | Arm/gripper armature and joint friction |
| Toolkit action history | Command delay in physics steps |

The toolkit's `IsaacLabCalibrationAdapter` consumes all three surfaces during
trajectory replay and reapplies runtime values after each replay reset. A custom
Isaac Lab training task must integrate the same three hooks; loading an
`ArticulationCfg` alone cannot implement friction and command delay. A future
task-generic adapter should add configuration patching plus live simulator
readback before training is allowed.

## Tradeoffs

- The calibrated profile is intentionally tied to one asset, timestep, solver,
  controller structure, and evidence scope. Reusing it for another robot or
  setup requires recalibration.
- Strict verification rejects manual edits, including seemingly harmless YAML
  changes. Changes must produce a new package and validation record.
- Command delay changes policy timing and must count physics steps, not policy
  action ticks or task decimation steps.
- The adapter is coupled to the Isaac Lab/Newton APIs used to apply joint
  properties and must be tested when those runtimes change.
- The package proves free-space arm and unloaded-gripper tracking only. It does
  not prove grasp, contact, insertion, or policy transfer.
- Runtime-build metadata is recorded when present but is not yet compared with
  the installed Isaac Lab/Newton build. Deployment must still use the pinned
  MVP1 container until a runtime capability handshake is implemented.
- Embedded hashes detect accidental drift, not authorship. For packages received
  across a trust boundary, pass a separately delivered manifest SHA-256. A
  signed, per-artifact manifest belongs in a future schema revision.
- The v1 digest covers the root USD, not its complete referenced-file closure.
  Assets with external USD, mesh, or texture dependencies need a future
  content-addressed dependency manifest before they are safely relocatable.
