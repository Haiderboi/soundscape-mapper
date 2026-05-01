"""
tests/test_audio_processing.py
--------------------------------
Unit tests for audio_processing modules: preprocessing, feature_extraction,
and indices.

All tests use synthetic (randomly generated) waveforms so that no actual
audio files are required.
"""

import numpy as np
import pytest

# Ensure `src` is on sys.path when running tests from the project root
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_processing.preprocessing import (
    normalize,
    remove_silence,
    resample,
)
from src.audio_processing.feature_extraction import (
    extract_all_features,
    extract_mfcc,
    extract_rms_energy,
    extract_zero_crossing_rate,
)
from src.audio_processing.indices import (
    compute_aci,
    compute_adi,
    compute_all_indices,
    compute_bi,
    compute_ndsi,
    compute_rms_db,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 22050
DURATION_SECS = 2
RNG = np.random.default_rng(42)


@pytest.fixture
def sine_wave():
    """440 Hz sine wave at SR sample rate, 2 seconds."""
    t = np.linspace(0, DURATION_SECS, SR * DURATION_SECS, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32), SR


@pytest.fixture
def white_noise():
    """White noise waveform."""
    y = RNG.standard_normal(SR * DURATION_SECS).astype(np.float32)
    return y, SR


@pytest.fixture
def silent_wave():
    """All-zero (silent) waveform."""
    return np.zeros(SR * DURATION_SECS, dtype=np.float32), SR


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_peak_is_one(self, sine_wave):
        y, _ = sine_wave
        y_norm = normalize(y)
        assert pytest.approx(np.max(np.abs(y_norm)), abs=1e-5) == 1.0

    def test_silent_returns_unchanged(self, silent_wave):
        y, _ = silent_wave
        y_norm = normalize(y)
        np.testing.assert_array_equal(y_norm, y)

    def test_shape_preserved(self, white_noise):
        y, _ = white_noise
        assert normalize(y).shape == y.shape


class TestResample:
    def test_same_rate_returns_same(self, sine_wave):
        y, sr = sine_wave
        y_r = resample(y, sr, sr)
        np.testing.assert_array_equal(y_r, y)

    def test_downsampled_length(self, sine_wave):
        y, sr = sine_wave
        target = sr // 2
        y_r = resample(y, sr, target)
        expected_len = int(len(y) * target / sr)
        # allow ±1 sample tolerance from librosa
        assert abs(len(y_r) - expected_len) <= 2


class TestRemoveSilence:
    def test_non_silent_unchanged_approx(self, sine_wave):
        y, sr = sine_wave
        y_trimmed = remove_silence(y, sr, top_db=70)
        # Some samples may be removed at edges but most should remain
        assert len(y_trimmed) > 0

    def test_silent_returns_original(self, silent_wave):
        y, sr = silent_wave
        y_out = remove_silence(y, sr)
        np.testing.assert_array_equal(y_out, y)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestExtractMFCC:
    def test_shape(self, sine_wave):
        y, sr = sine_wave
        mfcc = extract_mfcc(y, sr, n_mfcc=13)
        assert mfcc.shape[0] == 13
        assert mfcc.shape[1] > 0

    def test_values_finite(self, white_noise):
        y, sr = white_noise
        mfcc = extract_mfcc(y, sr)
        assert np.isfinite(mfcc).all()


class TestExtractRMS:
    def test_silent_is_low(self, silent_wave):
        y, _ = silent_wave
        rms = extract_rms_energy(y)
        assert np.mean(rms) < 1e-6

    def test_louder_is_higher(self, sine_wave, silent_wave):
        y_sine, _ = sine_wave
        y_silent, _ = silent_wave
        assert np.mean(extract_rms_energy(y_sine)) > np.mean(
            extract_rms_energy(y_silent)
        )


class TestExtractAllFeatures:
    def test_keys_present(self, sine_wave):
        y, sr = sine_wave
        feats = extract_all_features(y, sr)
        for key in [
            "mfcc", "mfcc_mean", "mfcc_std",
            "spectral_centroid_mean", "spectral_bandwidth_mean",
            "spectral_rolloff_mean", "zcr_mean", "rms_mean",
            "chroma_mean", "mel_spectrogram",
        ]:
            assert key in feats, f"Missing key: {key}"

    def test_scalar_features_finite(self, white_noise):
        y, sr = white_noise
        feats = extract_all_features(y, sr)
        for key in [
            "spectral_centroid_mean", "spectral_bandwidth_mean",
            "spectral_rolloff_mean", "zcr_mean", "rms_mean",
        ]:
            assert np.isfinite(feats[key]), f"Non-finite value for {key}"


# ---------------------------------------------------------------------------
# Ecoacoustic indices tests
# ---------------------------------------------------------------------------

class TestACI:
    def test_positive(self, white_noise):
        y, sr = white_noise
        aci = compute_aci(y, sr)
        assert aci > 0

    def test_returns_float(self, sine_wave):
        y, sr = sine_wave
        assert isinstance(compute_aci(y, sr), float)


class TestNDSI:
    def test_range(self, white_noise):
        y, sr = white_noise
        result = compute_ndsi(y, sr)
        assert -1.0 <= result["ndsi"] <= 1.0

    def test_keys(self, sine_wave):
        y, sr = sine_wave
        result = compute_ndsi(y, sr)
        assert set(result.keys()) == {"ndsi", "biophony", "anthrophony"}

    def test_nonnegative_components(self, white_noise):
        y, sr = white_noise
        result = compute_ndsi(y, sr)
        assert result["biophony"] >= 0
        assert result["anthrophony"] >= 0


class TestADI:
    def test_nonnegative(self, white_noise):
        y, sr = white_noise
        adi = compute_adi(y, sr)
        assert adi >= 0

    def test_returns_float(self, sine_wave):
        y, sr = sine_wave
        assert isinstance(compute_adi(y, sr), float)


class TestBI:
    def test_nonnegative(self, white_noise):
        y, sr = white_noise
        bi = compute_bi(y, sr)
        assert bi >= 0

    def test_returns_float(self, sine_wave):
        y, sr = sine_wave
        assert isinstance(compute_bi(y, sr), float)


class TestRMSdB:
    def test_returns_float(self, sine_wave):
        y, _ = sine_wave
        assert isinstance(compute_rms_db(y), float)

    def test_silence_is_low(self, silent_wave):
        y, _ = silent_wave
        # very quiet
        assert compute_rms_db(y) < -60


class TestComputeAllIndices:
    def test_all_keys_present(self, white_noise):
        y, sr = white_noise
        result = compute_all_indices(y, sr)
        for key in ["aci", "ndsi", "biophony", "anthrophony",
                    "geophony", "adi", "bi", "rms_db"]:
            assert key in result, f"Missing key: {key}"

    def test_all_finite(self, white_noise):
        y, sr = white_noise
        result = compute_all_indices(y, sr)
        for key, val in result.items():
            assert np.isfinite(val), f"Non-finite value for {key}: {val}"
