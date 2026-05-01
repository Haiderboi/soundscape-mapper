"""
audio_processing/feature_extraction.py
---------------------------------------
Low-level spectral and temporal feature extraction from audio waveforms.
"""

from __future__ import annotations

import logging
from typing import Dict

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute Mel-frequency cepstral coefficients.

    Parameters
    ----------
    y:
        Audio waveform.
    sr:
        Sample rate.
    n_mfcc:
        Number of MFCC coefficients.
    n_fft:
        FFT window size.
    hop_length:
        Hop size between frames.

    Returns
    -------
    np.ndarray
        MFCC matrix of shape ``(n_mfcc, T)``.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    logger.debug("MFCC shape: %s", mfcc.shape)
    return mfcc


def extract_spectral_centroid(
    y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Spectral centroid (brightness indicator).

    Returns
    -------
    np.ndarray
        Shape ``(1, T)``.
    """
    return librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )


def extract_spectral_bandwidth(
    y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Spectral bandwidth.

    Returns
    -------
    np.ndarray
        Shape ``(1, T)``.
    """
    return librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )


def extract_spectral_rolloff(
    y: np.ndarray,
    sr: int,
    roll_percent: float = 0.85,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Spectral roll-off frequency.

    Returns
    -------
    np.ndarray
        Shape ``(1, T)``.
    """
    return librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=roll_percent, n_fft=n_fft, hop_length=hop_length
    )


def extract_zero_crossing_rate(
    y: np.ndarray, hop_length: int = 512
) -> np.ndarray:
    """Zero-crossing rate (noisiness / tonality indicator).

    Returns
    -------
    np.ndarray
        Shape ``(1, T)``.
    """
    return librosa.feature.zero_crossing_rate(y, hop_length=hop_length)


def extract_rms_energy(
    y: np.ndarray, hop_length: int = 512, frame_length: int = 2048
) -> np.ndarray:
    """Root-mean-square energy per frame.

    Returns
    -------
    np.ndarray
        Shape ``(1, T)``.
    """
    return librosa.feature.rms(y=y, frame_length=frame_length,
                                hop_length=hop_length)


def extract_chroma(
    y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Chroma feature (12 pitch classes).

    Returns
    -------
    np.ndarray
        Shape ``(12, T)``.
    """
    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Power mel-spectrogram.

    Returns
    -------
    np.ndarray
        Shape ``(n_mels, T)``.
    """
    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )


def extract_all_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> Dict[str, np.ndarray]:
    """Extract and aggregate all spectral / temporal features into one dictionary.

    The returned dict has the following keys:

    * ``mfcc`` – shape ``(n_mfcc, T)``
    * ``mfcc_mean`` – shape ``(n_mfcc,)``
    * ``mfcc_std`` – shape ``(n_mfcc,)``
    * ``spectral_centroid_mean`` – scalar
    * ``spectral_bandwidth_mean`` – scalar
    * ``spectral_rolloff_mean`` – scalar
    * ``zcr_mean`` – scalar
    * ``rms_mean`` – scalar
    * ``chroma_mean`` – shape ``(12,)``
    * ``mel_spectrogram`` – shape ``(n_mels, T)``
    """
    features: Dict[str, np.ndarray] = {}

    mfcc = extract_mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft,
                         hop_length=hop_length)
    features["mfcc"] = mfcc
    features["mfcc_mean"] = np.mean(mfcc, axis=1)
    features["mfcc_std"] = np.std(mfcc, axis=1)

    centroid = extract_spectral_centroid(y, sr, n_fft=n_fft,
                                          hop_length=hop_length)
    features["spectral_centroid_mean"] = float(np.mean(centroid))

    bandwidth = extract_spectral_bandwidth(y, sr, n_fft=n_fft,
                                            hop_length=hop_length)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))

    rolloff = extract_spectral_rolloff(y, sr, n_fft=n_fft,
                                        hop_length=hop_length)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))

    zcr = extract_zero_crossing_rate(y, hop_length=hop_length)
    features["zcr_mean"] = float(np.mean(zcr))

    rms = extract_rms_energy(y, hop_length=hop_length, frame_length=n_fft)
    features["rms_mean"] = float(np.mean(rms))

    chroma = extract_chroma(y, sr, n_fft=n_fft, hop_length=hop_length)
    features["chroma_mean"] = np.mean(chroma, axis=1)

    features["mel_spectrogram"] = extract_mel_spectrogram(
        y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    logger.info("Extracted %d feature groups", len(features))
    return features
