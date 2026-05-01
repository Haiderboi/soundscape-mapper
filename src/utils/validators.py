"""
utils/validators.py
---------------------
Input validation for audio files and spatial data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac"}


class ValidationError(ValueError):
    """Raised when validation fails."""


def validate_audio_file(path: str | Path) -> Path:
    """Validate that *path* points to a supported audio file.

    Parameters
    ----------
    path:
        Path to the audio file.

    Returns
    -------
    Path
        Resolved path.

    Raises
    ------
    ValidationError
        If the file does not exist or has an unsupported format.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise ValidationError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValidationError(
            f"Unsupported audio format '{path.suffix}'.  "
            f"Supported: {sorted(SUPPORTED_AUDIO_FORMATS)}"
        )
    if path.stat().st_size == 0:
        raise ValidationError(f"Audio file is empty: {path}")
    logger.debug("Audio file valid: %s", path)
    return path


def validate_waveform(y: np.ndarray, sr: int) -> None:
    """Validate a loaded waveform array.

    Raises
    ------
    ValidationError
        If the waveform is empty, contains only NaN/Inf, or has an
        invalid sample rate.
    """
    if y is None or len(y) == 0:
        raise ValidationError("Waveform is empty.")
    if not np.isfinite(y).all():
        raise ValidationError("Waveform contains NaN or Inf values.")
    if sr <= 0:
        raise ValidationError(f"Sample rate must be positive, got {sr}.")


def validate_spatial_dataframe(gdf, required_columns: Optional[List[str]] = None) -> None:
    """Validate a GeoDataFrame for spatial analysis.

    Parameters
    ----------
    gdf:
        GeoDataFrame to validate.
    required_columns:
        Column names that must be present.

    Raises
    ------
    ValidationError
        If the GeoDataFrame is empty, has no geometry, or is missing
        required columns.
    """
    if gdf is None or len(gdf) == 0:
        raise ValidationError("GeoDataFrame is empty.")
    if gdf.geometry is None or gdf.geometry.is_empty.all():
        raise ValidationError("GeoDataFrame has no valid geometry.")
    if required_columns:
        missing = [c for c in required_columns if c not in gdf.columns]
        if missing:
            raise ValidationError(
                f"GeoDataFrame is missing columns: {missing}"
            )
    logger.debug("GeoDataFrame valid: %d rows", len(gdf))


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the top-level project configuration dictionary.

    Raises
    ------
    ValidationError
        If required sections are absent or have invalid values.
    """
    required_sections = ["paths", "audio", "spatial", "ml", "visualization"]
    for section in required_sections:
        if section not in config:
            raise ValidationError(
                f"Missing required config section: '{section}'"
            )

    audio = config.get("audio", {})
    if audio.get("sample_rate", 0) <= 0:
        raise ValidationError("audio.sample_rate must be a positive integer.")
    if audio.get("n_fft", 0) <= 0:
        raise ValidationError("audio.n_fft must be a positive integer.")

    logger.debug("Configuration is valid.")


def validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:
    """Validate geographic coordinates.

    Raises
    ------
    ValidationError
        If latitude or longitude are out of range.
    """
    if not -90 <= latitude <= 90:
        raise ValidationError(
            f"Invalid latitude {latitude}.  Must be in [-90, 90]."
        )
    if not -180 <= longitude <= 180:
        raise ValidationError(
            f"Invalid longitude {longitude}.  Must be in [-180, 180]."
        )
