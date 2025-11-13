"""Centralized logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, log_dir: Path, timestamp: str) -> logging.Logger:
    """Configure a logger that writes to both console and a timestamped file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_name = f"{name}_{timestamp}"

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = log_dir / f"{timestamp}.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

