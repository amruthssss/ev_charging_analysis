"""Central logging utilities for the EV Charging project."""
from __future__ import annotations

import logging
from pathlib import Path

from src.config import LOGS_DIR

LOG_FILE: Path = LOGS_DIR / "ev_platform.log"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that logs to console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
