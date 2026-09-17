# Validation record — 2026-09-16

- 39 new offline contract/unit tests passed.
- 73 existing controller-discovery, adaptive-design and collection-assistance
  regression tests passed.
- Ruff checks and Python compilation passed.
- Generated and independently re-read 15 CSVs: 13 training/baseline and 2 held-out.
- Total reference duration: 390 seconds per posture, excluding setup/pauses.
- Checks cover hashes, times, reference displacement/velocity/acceleration,
  fixed-anchor semantics, full TCP rotation, xyzw↔wxyz conversion, profile
  binding, stale/reversed feedback, approval expiry, command-recording and
  stop-on-failure behavior with a synthetic SDK.

**Not performed:** actual RDK import/integration, robot connection, real motion,
Isaac Lab/Newton replay, collision/singularity screening, fitting, parameter
identifiability analysis, real-data held-out validation or video capture.

The mock execution tests are software tests only. Their synthetic approval and
screening records are temporary test fixtures and are not included as hardware
authorization or physics evidence. No real approval is included.

This delivery is an isolated Cartesian data-collection proposal and runner.
It leaves the toolkit's unsupported-Cartesian-fitting guard unchanged. Existing
joint-PD archives, policies and MVP2/MVP3 experimental artifacts are untouched.
