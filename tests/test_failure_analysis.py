"""Tests for automatic failure-case diagnostics."""

from __future__ import annotations

from pathlib import Path

import torch

from embodied_motion_flow.diagnostics.failure_analysis import FailureCaseAnalyzer


def test_failure_analyzer_saves_artifact_and_log(tmp_path: Path) -> None:
    analyzer = FailureCaseAnalyzer(
        output_dir=tmp_path,
        tsi_threshold=0.1,
        jlvr_threshold=0.1,
        self_collision_threshold=0.1,
    )
    records = analyzer.analyze_batch(
        motion=torch.zeros(1, 5, 72),
        tsi=torch.tensor([0.2]),
        jlvr=torch.tensor([0.0]),
        self_collision=torch.tensor([0.0]),
        epoch=1,
        split="val",
        source_paths=["sample.pkl"],
    )
    assert len(records) == 1
    assert records[0].reason == "Limb Explosion"
    assert Path(records[0].artifact_path).exists()
    assert "Limb Explosion" in (tmp_path / "failure_log.csv").read_text(encoding="utf-8")
