"""Project logging utility."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str, log_file: Path | None = None) -> None:
    """Configure global logging format and optional file sink."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger."""
    return logging.getLogger(name)
