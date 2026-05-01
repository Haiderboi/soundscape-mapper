"""
audio_processing/preprocessing.py
----------------------------------
Audio loading, resampling, normalisation and silence removal utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Supported audio extensions
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac"}


def load_audio(
    file_path: str | Path,
    sample_rate: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and return the waveform and sample rate.

    Parameters
    ----------
    file_path:
        Path to the audio file (WAV, MP3, or FLAC).
    sample_rate:
        Target sample rate in Hz.  The signal is resampled if necessary.
    mono:
        Convert to mono by averaging channels.
    duration:
        Maximum duration to load in seconds.  ``None`` loads the full file.
    offset:
        Start time in seconds.

    Returns
    -------
    (y, sr)
        Waveform as a 1-D float32 array and the actual sample rate used.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file extension is not supported.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{file_path.suffix}'.  "
            f"Supported: {SUPPORTED_FORMATS}"
        )

    logger.debug("Loading audio: %s", file_path)
    y, sr = librosa.load(
        str(file_path),
        sr=sample_rate,
        mono=mono,
        duration=duration,
        offset=offset,
    )
    logger.info("Loaded %s  sr=%d  samples=%d", file_path.name, sr, len(y))
    return y, sr


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a waveform to *target_sr* Hz.

    Parameters
    ----------
    y:
        Input waveform.
    orig_sr:
        Original sample rate.
    target_sr:
        Target sample rate.

    Returns
    -------
    np.ndarray
        Resampled waveform.
    """
    if orig_sr == target_sr:
        return y
    logger.debug("Resampling %d → %d Hz", orig_sr, target_sr)
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def normalize(y: np.ndarray) -> np.ndarray:
    """Peak-normalise a waveform to the range ``[-1, 1]``.

    Parameters
    ----------
    y:
        Input waveform.

    Returns
    -------
    np.ndarray
        Normalised waveform.  Returns *y* unchanged if it is all zeros.
    """
    peak = np.max(np.abs(y))
    if peak == 0:
        logger.warning("Cannot normalise a silent (all-zero) waveform.")
        return y
    return y / peak


def remove_silence(
    y: np.ndarray,
    sr: int,
    top_db: float = 60.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Remove silent intervals from a waveform.

    Parameters
    ----------
    y:
        Input waveform.
    sr:
        Sample rate (unused directly, kept for API consistency).
    top_db:
        Threshold below the reference level (in dB) that is considered silent.
    frame_length:
        FFT frame length.
    hop_length:
        Hop size between frames.

    Returns
    -------
    np.ndarray
        Waveform with silent segments removed.
    """
    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    if len(intervals) == 0:
        logger.warning("No non-silent intervals found – returning original.")
        return y
    trimmed = np.concatenate([y[start:end] for start, end in intervals])
    logger.debug(
        "Silence removal: %d → %d samples", len(y), len(trimmed)
    )
    return trimmed


def apply_window(
    y: np.ndarray,
    window_type: str = "hann",
) -> np.ndarray:
    """Multiply a waveform by a window function.

    Parameters
    ----------
    y:
        Input waveform.
    window_type:
        Window function name understood by ``scipy.signal.get_window``.

    Returns
    -------
    np.ndarray
        Windowed waveform.
    """
    from scipy.signal import get_window

    window = get_window(window_type, len(y))
    return y * window.astype(y.dtype)


def preprocess_audio(
    file_path: str | Path,
    sample_rate: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    normalise: bool = True,
    silence_removal: bool = True,
    top_db: float = 60.0,
) -> Tuple[np.ndarray, int]:
    """Full preprocessing pipeline: load → resample → normalise → trim silence.

    Parameters
    ----------
    file_path:
        Path to the audio file.
    sample_rate:
        Target sample rate.
    mono:
        Convert to mono.
    duration:
        Maximum duration in seconds.
    normalise:
        Apply peak normalisation.
    silence_removal:
        Remove silent segments.
    top_db:
        Silence threshold in dB (used when *silence_removal* is ``True``).

    Returns
    -------
    (y, sr)
        Preprocessed waveform and sample rate.
    """
    y, sr = load_audio(file_path, sample_rate=sample_rate, mono=mono,
                       duration=duration)
    if normalise:
        y = normalize(y)
    if silence_removal:
        y = remove_silence(y, sr, top_db=top_db)
    return y, sr
