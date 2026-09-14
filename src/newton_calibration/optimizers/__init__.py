from .cmaes import DiagonalCMAES
from .registry import (
    ENTRY_POINT_GROUP,
    OptimizerContractError,
    OptimizerInit,
    OptimizerRegistration,
    OptimizerRegistryError,
    create_optimizer,
    discover_optimizer_plugins,
    get_optimizer_registration,
    list_optimizers,
    optimizer_config_fingerprint,
    register_optimizer,
    validate_candidates,
    validate_optimizer_plugin,
    validate_scores,
)


def _create_diagonal_cmaes(initialization: OptimizerInit) -> DiagonalCMAES:
    return DiagonalCMAES(
        initialization.parameters,
        population=initialization.population,
        seed=initialization.seed,
    )


register_optimizer(
    "diagonal-cma-es",
    _create_diagonal_cmaes,
    version="1",
    provider="newton-calibration",
    replace=True,
)


__all__ = [
    "ENTRY_POINT_GROUP",
    "DiagonalCMAES",
    "OptimizerContractError",
    "OptimizerInit",
    "OptimizerRegistration",
    "OptimizerRegistryError",
    "create_optimizer",
    "discover_optimizer_plugins",
    "get_optimizer_registration",
    "list_optimizers",
    "optimizer_config_fingerprint",
    "register_optimizer",
    "validate_candidates",
    "validate_optimizer_plugin",
    "validate_scores",
]
