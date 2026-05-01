"""
audio_processing/indices.py
-----------------------------
Ecoacoustic indices: ACI, NDSI, ADI, BI, RMS dB and derived sub-indices
(biophony, geophony, anthrophony).

References
----------
* Pieretti et al. (2011) – Acoustic Complexity Index
* Joo et al. (2011) – NDSI
* Villanueva-Rivera et al. (2011) – ADI
* Boelman et al. (2007) – Bioacoustic Index
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _power_spectrogram(
    y: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a power spectrogram and its frequency axis.

    Returns
    -------
    (S_power, freqs)
        *S_power* has shape ``(n_fft//2 + 1, T)``.
        *freqs* has shape ``(n_fft//2 + 1,)``.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=22050, n_fft=n_fft)
    return S, freqs


# ---------------------------------------------------------------------------
# Acoustic Complexity Index
# ---------------------------------------------------------------------------

def compute_aci(
    y: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 256,
    j_step: int = 5,
) -> float:
    """Acoustic Complexity Index (ACI).

    Measures temporal variability of intensity within each frequency band.
    Higher values indicate more complex (biologically rich) soundscapes.

    Parameters
    ----------
    y:
        Audio waveform.
    sr:
        Sample rate (unused directly – kept for API consistency).
    n_fft:
        FFT window size.
    hop_length:
        Hop size between frames.
    j_step:
        Number of time frames per sub-interval *j*.

    Returns
    -------
    float
        ACI value.
    """
    S, _ = _power_spectrogram(y, n_fft=n_fft, hop_length=hop_length)
    S_amp = np.sqrt(S)                   # amplitude spectrogram

    n_freq, n_time = S_amp.shape
    aci_total = 0.0

    for freq_bin in range(n_freq):
        row = S_amp[freq_bin, :]
        for j_start in range(0, n_time, j_step):
            segment = row[j_start: j_start + j_step]
            if len(segment) < 2:
                continue
            d_sum = np.sum(np.abs(np.diff(segment)))
            i_sum = np.sum(segment)
            if i_sum > 0:
                aci_total += d_sum / i_sum

    logger.debug("ACI = %.4f", aci_total)
    return float(aci_total)


# ---------------------------------------------------------------------------
# Normalised Difference Soundscape Index
# ---------------------------------------------------------------------------

def compute_ndsi(
    y: np.ndarray,
    sr: int,
    anthrophony_min: int = 1000,
    anthrophony_max: int = 2000,
    biophony_min: int = 2000,
    biophony_max: int = 11000,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Normalised Difference Soundscape Index (NDSI).

    NDSI = (biophony − anthrophony) / (biophony + anthrophony)

    Values close to +1 indicate biophony-dominated soundscapes;
    values close to −1 indicate anthrophony-dominated soundscapes.

    Returns
    -------
    dict with keys ``ndsi``, ``biophony``, ``anthrophony``.
    """
    S, freqs = _power_spectrogram(y, n_fft=n_fft, hop_length=hop_length)

    anthro_mask = (freqs >= anthrophony_min) & (freqs < anthrophony_max)
    bio_mask = (freqs >= biophony_min) & (freqs < biophony_max)

    anthrophony = float(np.sum(S[anthro_mask, :]))
    biophony = float(np.sum(S[bio_mask, :]))

    denom = biophony + anthrophony
    ndsi = (biophony - anthrophony) / denom if denom > 0 else 0.0

    logger.debug(
        "NDSI=%.4f  biophony=%.2f  anthrophony=%.2f",
        ndsi, biophony, anthrophony,
    )
    return {"ndsi": ndsi, "biophony": biophony, "anthrophony": anthrophony}


# ---------------------------------------------------------------------------
# Acoustic Diversity Index
# ---------------------------------------------------------------------------

def compute_adi(
    y: np.ndarray,
    sr: int,
    max_freq: int = 10000,
    db_threshold: float = -50.0,
    freq_step: int = 1000,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> float:
    """Acoustic Diversity Index (ADI).

    Inspired by Shannon's entropy, computed over presence/absence of sound
    energy in equally-spaced frequency bands.

    Returns
    -------
    float
        ADI value (Shannon entropy in nats).
    """
    S, freqs = _power_spectrogram(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)

    band_edges = range(0, max_freq, freq_step)
    proportions = []
    n_frames = S_db.shape[1]

    for low in band_edges:
        high = low + freq_step
        band_mask = (freqs >= low) & (freqs < high)
        band_data = S_db[band_mask, :]
        if band_data.size == 0:
            continue
        # proportion of time frames with energy above threshold in this band
        presence = np.mean(np.max(band_data, axis=0) > db_threshold)
        proportions.append(presence)

    proportions_arr = np.array(proportions)
    # avoid log(0)
    proportions_arr = proportions_arr[proportions_arr > 0]
    if proportions_arr.size == 0:
        return 0.0

    adi = float(-np.sum(proportions_arr * np.log(proportions_arr)))
    logger.debug("ADI = %.4f", adi)
    return adi


# ---------------------------------------------------------------------------
# Bioacoustic Index
# ---------------------------------------------------------------------------

def compute_bi(
    y: np.ndarray,
    sr: int,
    min_freq: int = 2000,
    max_freq: int = 8000,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> float:
    """Bioacoustic Index (BI).

    Area under the mean dB spectrum curve within the biophony frequency band.

    Returns
    -------
    float
        BI value (area under the mean dB spectrum in the biophony band).
    """
    S, freqs = _power_spectrogram(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)

    band_mask = (freqs >= min_freq) & (freqs <= max_freq)
    band_db = S_db[band_mask, :]

    if band_db.size == 0:
        return 0.0

    mean_db = np.mean(band_db, axis=1)
    # clip to 0 so negative dB values don't contribute
    mean_db_clipped = np.clip(mean_db, a_min=0, a_max=None)
    bi = float(np.sum(mean_db_clipped))
    logger.debug("BI = %.4f", bi)
    return bi


# ---------------------------------------------------------------------------
# RMS dB noise level
# ---------------------------------------------------------------------------

def compute_rms_db(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    ref: float = 1.0,
) -> float:
    """Overall RMS level in dB.

    Returns
    -------
    float
        Mean RMS energy in dB across all frames.
    """
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )
    rms_db = float(librosa.amplitude_to_db(rms, ref=ref).mean())
    logger.debug("RMS dB = %.2f", rms_db)
    return rms_db


# ---------------------------------------------------------------------------
# Geophony proxy
# ---------------------------------------------------------------------------

def compute_geophony(
    y: np.ndarray,
    sr: int,
    geo_min: int = 20,
    geo_max: int = 1000,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> float:
    """Proxy for geophony (wind, water) energy in the low-frequency band.

    Returns total power in the geophony band (default 20–1000 Hz).
    """
    S, freqs = _power_spectrogram(y, n_fft=n_fft, hop_length=hop_length)
    mask = (freqs >= geo_min) & (freqs < geo_max)
    geo = float(np.sum(S[mask, :]))
    logger.debug("Geophony = %.2f", geo)
    return geo


# ---------------------------------------------------------------------------
# Master index computation function
# ---------------------------------------------------------------------------

def compute_all_indices(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Compute all ecoacoustic indices at once.

    Returns
    -------
    dict
        Keys: ``aci``, ``ndsi``, ``biophony``, ``anthrophony``,
        ``geophony``, ``adi``, ``bi``, ``rms_db``.
    """
    indices: Dict[str, float] = {}

    indices["aci"] = compute_aci(y, sr, n_fft=n_fft // 2,
                                  hop_length=hop_length // 2)

    ndsi_result = compute_ndsi(y, sr, n_fft=n_fft, hop_length=hop_length)
    indices.update(ndsi_result)

    indices["geophony"] = compute_geophony(
        y, sr, n_fft=n_fft, hop_length=hop_length
    )
    indices["adi"] = compute_adi(y, sr, n_fft=n_fft, hop_length=hop_length)
    indices["bi"] = compute_bi(y, sr, n_fft=n_fft, hop_length=hop_length)
    indices["rms_db"] = compute_rms_db(y)

    logger.info("Computed indices: %s", list(indices.keys()))
    return indices
