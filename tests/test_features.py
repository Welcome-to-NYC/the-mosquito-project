"""Tests for src.features.spectral."""

from __future__ import annotations

import numpy as np

from src.features.spectral import (
    DEFAULT_SR,
    FEATURE_NAMES,
    estimate_f0,
    extract_features,
    harmonic_powers,
    rms_amplitude,
    spectral_shape,
    zero_crossing_rate,
)


def _sine(freq: float, n: int, sr: int = DEFAULT_SR, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)


def test_estimate_f0_recovers_synthesized_sine_in_band():
    n, sr = 1024, 5000
    x = np.stack([_sine(440.0, n, sr), _sine(800.0, n, sr)])
    f0 = estimate_f0(x, sr=sr)
    # Coarse FFT bin = 5000/1024 ≈ 4.88 Hz; allow ±5 Hz tolerance.
    assert abs(f0[0] - 440) < 5
    assert abs(f0[1] - 800) < 5


def test_estimate_f0_clamps_to_band():
    # 50 Hz is below the default 100-1500 Hz band; should not be reported.
    n = 1024
    x = _sine(50.0, n)
    f0 = estimate_f0(x[None, :])
    assert f0[0] >= 100  # clamped to band lower bound or argmax in band


def test_harmonic_powers_concentrates_at_harmonic_indices():
    # Mix f0 = 300 Hz with a strong third harmonic (900 Hz). Power at h1 and h3
    # should dominate over h2/h4/h5.
    n, sr = 2048, 5000
    x = _sine(300, n, sr) + 0.7 * _sine(900, n, sr)
    f0 = np.array([300.0], dtype=np.float32)
    h = harmonic_powers(x[None, :], f0, sr=sr, n_harmonics=5)
    # h shape (1, 5)
    h = h[0]
    assert h[0] > h[1]   # f0 > 2f0
    assert h[2] > h[1]   # 3f0 > 2f0
    assert h[2] > h[3]   # 3f0 > 4f0


def test_spectral_shape_centroid_matches_dominant_freq():
    n, sr = 2048, 5000
    x = _sine(700, n, sr)
    shape = spectral_shape(x[None, :], sr=sr)
    centroid, bw, rolloff = shape[0]
    # Centroid is energy-weighted; for a clean sine it sits very near the freq.
    assert abs(centroid - 700) < 30
    # Bandwidth is small for a single sine, much smaller than centroid.
    assert bw < 100
    # 85% rolloff is at-or-above the carrier.
    assert rolloff >= 700 - 30


def test_rms_unit_for_znormed_signal():
    rng = np.random.RandomState(0)
    x = rng.randn(4, 1024).astype(np.float32)
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    rms = rms_amplitude(x)
    assert np.allclose(rms, 1.0, atol=1e-5)


def test_zero_crossing_rate_known_signal():
    # Signal that flips sign every sample: ZCR -> ~1.
    n = 100
    x = np.tile([1.0, -1.0], n // 2).astype(np.float32)
    zcr = zero_crossing_rate(x[None, :])
    assert abs(zcr[0] - 1.0) < 0.01


def test_zero_crossing_rate_constant_signal_is_zero():
    x = np.full((1, 100), 0.5, dtype=np.float32)
    assert zero_crossing_rate(x)[0] == 0.0


def test_extract_features_shape_and_dtype():
    rng = np.random.RandomState(0)
    X = rng.randn(8, 1024).astype(np.float32)
    feats = extract_features(X, sr=5000)
    assert feats.shape == (8, len(FEATURE_NAMES))
    assert feats.dtype == np.float32


def test_extract_features_distinguishes_pure_tones():
    # Two distinct pure tones produce well-separated feature vectors.
    n, sr = 1024, 5000
    a = _sine(300, n, sr)
    b = _sine(900, n, sr)
    X = np.stack([a, b, a, b])
    feats = extract_features(X, sr=sr)
    # f0 column should match.
    assert abs(feats[0, 0] - 300) < 10
    assert abs(feats[1, 0] - 900) < 10
    # Cosine similarity between same-tone vectors > between different tones.
    def cs(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    assert cs(feats[0], feats[2]) > cs(feats[0], feats[1])


def test_extract_features_rejects_1d():
    import pytest
    with pytest.raises(ValueError):
        extract_features(np.zeros(1024, dtype=np.float32))
