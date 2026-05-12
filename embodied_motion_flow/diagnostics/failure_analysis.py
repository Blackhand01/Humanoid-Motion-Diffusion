"""Failure-case mining for biomechanical motion diagnostics."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class FailureCaseRecord:
    """Metadata for one saved failure case."""

    epoch: int
    split: str
    sample_index: int
    source_path: str
    reason: str
    tsi: float
    jlvr: float
    self_collision: float
    artifact_path: str


class FailureCaseAnalyzer:
    """Save generated sequences that exceed quantitative failure thresholds.

    Failure categories:
        Limb Explosion: high Temporal Smoothness Index, usually abrupt motion.
        Joint Dislocation: high Joint Limit Violation Rate.
        Self Collision: excessive non-adjacent joint-center proximity.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tsi_threshold: float = 0.25,
        jlvr_threshold: float = 0.02,
        self_collision_threshold: float = 0.01,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "failure_log.csv"
        self.tsi_threshold = tsi_threshold
        self.jlvr_threshold = jlvr_threshold
        self.self_collision_threshold = self_collision_threshold
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.log_path.exists():
            return
        with self.log_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FailureCaseRecord.__dataclass_fields__.keys()))
            writer.writeheader()

    @staticmethod
    def _source_at(source_paths: Sequence[Any] | None, index: int) -> str:
        if source_paths is None or index >= len(source_paths):
            return "unknown"
        return str(source_paths[index])

    def _failure_reasons(self, tsi: float, jlvr: float, self_collision: float) -> list[str]:
        reasons: list[str] = []
        if tsi > self.tsi_threshold:
            reasons.append("Limb Explosion")
        if jlvr > self.jlvr_threshold:
            reasons.append("Joint Dislocation")
        if self_collision > self.self_collision_threshold:
            reasons.append("Self Collision")
        return reasons

    def analyze_batch(
        self,
        motion: torch.Tensor,
        tsi: torch.Tensor,
        jlvr: torch.Tensor,
        self_collision: torch.Tensor | None,
        epoch: int,
        split: str,
        source_paths: Sequence[Any] | None = None,
    ) -> list[FailureCaseRecord]:
        """Save failing sequences from one batch and append a CSV diagnostic log."""
        batch_size = motion.shape[0]
        collision_values = (
            self_collision.detach().cpu()
            if self_collision is not None
            else torch.zeros(batch_size, dtype=torch.float32)
        )
        tsi_values = tsi.detach().cpu()
        jlvr_values = jlvr.detach().cpu()
        motion_cpu = motion.detach().cpu()
        records: list[FailureCaseRecord] = []

        for sample_idx in range(batch_size):
            tsi_value = float(tsi_values[sample_idx].item())
            jlvr_value = float(jlvr_values[sample_idx].item())
            collision_value = float(collision_values[sample_idx].item())
            reasons = self._failure_reasons(tsi_value, jlvr_value, collision_value)
            if not reasons:
                continue

            reason_slug = "_".join(reason.lower().replace(" ", "_") for reason in reasons)
            artifact_path = self.output_dir / f"epoch_{epoch:04d}_{split}_{sample_idx:04d}_{reason_slug}.pt"
            source_path = self._source_at(source_paths, sample_idx)
            torch.save(
                {
                    "motion": motion_cpu[sample_idx],
                    "epoch": epoch,
                    "split": split,
                    "source_path": source_path,
                    "tsi": tsi_value,
                    "jlvr": jlvr_value,
                    "self_collision": collision_value,
                    "reason": "; ".join(reasons),
                },
                artifact_path,
            )
            records.append(
                FailureCaseRecord(
                    epoch=epoch,
                    split=split,
                    sample_index=sample_idx,
                    source_path=source_path,
                    reason="; ".join(reasons),
                    tsi=tsi_value,
                    jlvr=jlvr_value,
                    self_collision=collision_value,
                    artifact_path=str(artifact_path),
                )
            )

        if records:
            with self.log_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(FailureCaseRecord.__dataclass_fields__.keys()))
                for record in records:
                    writer.writerow(asdict(record))
        return records
