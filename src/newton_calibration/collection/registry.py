"""Explicitly installed domain catalogs; artifacts never execute plug-in code."""

from collections.abc import Callable
from dataclasses import dataclass

from . import mvp1


@dataclass(frozen=True)
class ExperimentCatalog:
    id: str
    assess: Callable
    select: Callable


_CATALOGS = {mvp1.VERSION: ExperimentCatalog(mvp1.VERSION, mvp1.assess_requirements, mvp1.select_experiments)}


def register_catalog(catalog: ExperimentCatalog):
    """Trusted application setup only; new domains need explicit versioned catalogs."""
    if "@" not in catalog.id or catalog.id in _CATALOGS or not callable(catalog.assess) or not callable(catalog.select):
        raise ValueError("Catalog registration requires unique versioned ID and assess/select callables")
    _CATALOGS[catalog.id] = catalog


def get_catalog(catalog_id: str) -> ExperimentCatalog:
    if catalog_id not in _CATALOGS:
        raise ValueError(f"No installed experiment catalog: {catalog_id!r}")
    return _CATALOGS[catalog_id]
