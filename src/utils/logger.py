"""
utils/logger.py
-----------------
Centralised logging configuration for Soundscape Mapper.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "soundscape_mapper",
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure and return a named logger.

    Parameters
    ----------
    name:
        Logger name.
    level:
        Log level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
    log_file:
        Optional file path.  If provided, a ``FileHandler`` is added.
    fmt:
        Log message format string.
    date_fmt:
        Date/time format string.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "soundscape_mapper") -> logging.Logger:
    """Return a logger by *name*, creating it with INFO level if needed.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
