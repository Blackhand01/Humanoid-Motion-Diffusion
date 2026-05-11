"""Evaluation metric and anomaly scoring modules."""

from embodied_motion_flow.evaluation.anomaly import (
    anomaly_score_reconstruction_plus_biomechanical,
    reconstruction_error_score,
)
from embodied_motion_flow.evaluation.metrics import (
    anomaly_auroc,
    mse_reconstruction,
    physical_constraint_violations,
    temporal_smoothness,
)
from embodied_motion_flow.evaluation.runner import EvaluationOutputs, evaluate_model

__all__ = [
    "EvaluationOutputs",
    "anomaly_auroc",
    "anomaly_score_reconstruction_plus_biomechanical",
    "evaluate_model",
    "mse_reconstruction",
    "physical_constraint_violations",
    "reconstruction_error_score",
    "temporal_smoothness",
]
