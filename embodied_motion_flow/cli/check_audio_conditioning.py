"""Check AIST++ audio coverage and frame-level alignment before Phase 2/3 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from embodied_motion_flow.config import load_config
from embodied_motion_flow.data.dataset import build_dataloaders
from embodied_motion_flow.evaluation.metrics import beat_alignment_score
from embodied_motion_flow.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AIST++ audio feature coverage and alignment.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--fail-under", type=float, default=None, help="Minimum clip-level audio coverage required.")
    return parser.parse_args()


def _coverage(dataset: object) -> dict[str, float | int]:
    value = getattr(dataset, "audio_coverage", None)
    if isinstance(value, dict):
        return value
    return {"clips": 0, "clips_with_audio": 0, "clip_fraction": 0.0, "source_motions": 0, "source_motions_with_audio": 0}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config.project.output_dir)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(config.project.log_level, log_file=logs_dir / "check_audio_conditioning.log")
    logger = get_logger("embodied_motion_flow.check_audio_conditioning")

    data_splits = build_dataloaders(config)
    report = {
        "audio_root": config.audio.root_dir,
        "feature_dim": config.audio.feature_dim,
        "train": _coverage(data_splits.train_dataset),
        "val": _coverage(data_splits.val_dataset),
        "test": _coverage(data_splits.test_dataset),
    }

    batch = next(iter(data_splits.train_loader))
    motion = batch["motion"]
    audio = batch.get("audio_context")
    beats = batch.get("beat_indicator")
    has_audio = batch.get("has_audio")
    if isinstance(audio, torch.Tensor):
        report["first_batch_audio_shape"] = list(audio.shape)
        report["first_batch_audio_abs_mean"] = float(audio.abs().mean().item())
    if isinstance(has_audio, torch.Tensor):
        report["first_batch_has_audio_fraction"] = float(has_audio.float().mean().item())
    if isinstance(beats, torch.Tensor):
        report["first_batch_beat_density"] = float(beats.float().mean().item())
        if motion.shape[-1] == 72:
            report["first_batch_bas_reference"] = float(beat_alignment_score(motion, beats, reduce="mean").item())

    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / "audio_conditioning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Audio conditioning report: %s", json.dumps(report, indent=2))
    logger.info("Saved audio conditioning report to %s", report_path)

    threshold = args.fail_under if args.fail_under is not None else config.audio.min_coverage
    if float(report["train"]["clip_fraction"]) < float(threshold):
        raise SystemExit(
            f"Train audio coverage {report['train']['clip_fraction']:.3f} is below required threshold {threshold:.3f}."
        )


if __name__ == "__main__":
    main()
