"""Evaluation metric and anomaly scoring modules."""

from embodied_motion_flow.evaluation.anomaly import (
    anomaly_score_reconstruction_plus_biomechanical,
    reconstruction_error_score,
)
from embodied_motion_flow.evaluation.metrics import (
    anomaly_auroc,
    beat_alignment_score,
    default_smpl_joint_limits,
    joint_limit_violation_rate,
    mse_reconstruction,
    physical_constraint_violations,
    temporal_smoothness,
    temporal_smoothness_index,
)
from embodied_motion_flow.evaluation.runner import EvaluationOutputs, evaluate_model

__all__ = [
    "EvaluationOutputs",
    "anomaly_auroc",
    "anomaly_score_reconstruction_plus_biomechanical",
    "beat_alignment_score",
    "default_smpl_joint_limits",
    "evaluate_model",
    "joint_limit_violation_rate",
    "mse_reconstruction",
    "physical_constraint_violations",
    "reconstruction_error_score",
    "temporal_smoothness",
    "temporal_smoothness_index",
]
