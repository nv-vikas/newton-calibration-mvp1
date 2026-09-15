"""Simulation-only evidence collection planning; no robot execution interface."""

from .contracts import CalibrationRequest, ExperimentSpec
from .planning import CollectionPlan, MotionSpec, create_collection_plan, prepare_assistance, verify_commands
from .sensitivity import FiniteDifferenceProbe, PredictionBackend, SensitivityResult

__all__ = [
    "CalibrationRequest",
    "CollectionPlan",
    "ExperimentSpec",
    "FiniteDifferenceProbe",
    "MotionSpec",
    "PredictionBackend",
    "SensitivityResult",
    "create_collection_plan",
    "prepare_assistance",
    "verify_commands",
]
