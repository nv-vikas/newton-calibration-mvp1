# Optimizer plug-ins

The calibration runner owns the physics experiment. An optimizer plug-in only
participates in the ask/tell search loop:

1. The toolkit supplies the locked parameter specifications, population, seed,
   and method-specific options.
2. The plug-in proposes a complete population of candidates.
3. The toolkit verifies exact parameter names, finite numeric values, and the
   locked bounds before running any candidate.
4. The toolkit runs Newton and computes the objective score.
5. The plug-in receives the candidates and scores and advances one generation.
6. The toolkit checkpoints the plug-in state and later re-evaluates its best
   candidate before held-out validation.

The optimizer never receives a Newton runtime handle and never decides whether
a calibration package is valid.

## Provider contract

A provider factory receives one `OptimizerInit` and returns an object satisfying
`OptimizerPlugin`:

```python
from newton_calibration.optimizers import OptimizerInit


class MinjaeOptimizer:
    def __init__(self, initialization: OptimizerInit):
        self.initialization = initialization

    def ask(self) -> list[dict[str, float]]:
        ...

    def tell(
        self,
        candidates: list[dict[str, float]],
        scores: list[float],
    ) -> None:
        ...

    @property
    def generation(self) -> int:
        ...

    @property
    def best(self) -> tuple[dict[str, float], float] | None:
        ...

    def state_dict(self) -> dict:
        """Return JSON-serializable state."""
        ...

    def load_state_dict(self, state: dict) -> None:
        ...


def create_optimizer(initialization: OptimizerInit) -> MinjaeOptimizer:
    return MinjaeOptimizer(initialization)
```

The Minjae package makes that factory discoverable in its `pyproject.toml`:

```toml
[project.entry-points."newton_calibration.optimizers"]
"minjae-nvopt.v1" = "minjae_newton.optimizer:create_optimizer"
```

The provider distribution, package version, configuration fingerprint, and full
fit-execution fingerprint become part of the locked plan, fit record,
per-generation journal, checkpoint, and calibration manifest. Resuming with a
different provider, version, evidence set, runtime, parameter space, objective,
or configuration is rejected. The generation ceiling may be increased without
changing the execution fingerprint. The toolkit also re-hashes the evidence
files and SO-101 USD before fitting, validation, and packaging; changing bytes
in place requires a new analysis and plan.

For MVP1, the runtime build is held constant by the pinned Docker image. A
future production worker should add the resolved container/runtime build digest
to the locked execution record rather than relying only on deployment policy.

## Agent selection

Once the provider package is installed in the same Docker worker as the
toolkit, Minjae's agent can select it without receiving direct access to Newton:

```python
calibration_plan = tuning.plan(
    analysis,
    optimizer="minjae-nvopt.v1",
    optimizer_options={"strategy": "adaptive-search"},
)
fit_run = tuning.fit(calibration_plan, resume=True)
```

The equivalent CLI selection is:

```bash
newton-calibration run \
  --asset /workspace/data/so101.usd \
  --evidence /workspace/data/anchor-lab \
  --optimizer minjae-nvopt.v1 \
  --optimizer-options-json '{"strategy":"adaptive-search"}'
```

This repository deliberately does not register a pretend Minjae implementation.
Until the real Minjae package or service adapter is installed, selecting
`minjae-nvopt.v1` fails closed as an unknown optimizer.
