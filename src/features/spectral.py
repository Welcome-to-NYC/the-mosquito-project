"""Spectral features for the W2 classical baseline.

The features are the same ones every wingbeat paper in the W1 reading list
uses, so the baseline is comparable to published numbers:

* fundamental frequency in the wingbeat range (default 100–1500 Hz)
* power at the fundamental and at the next 4 harmonics (2f0, 3f0, 4f0, 5f0)
* spectral centroid, bandwidth, and rolloff (energy distribution shape)
* RMS amplitude
* zero-crossing rate

Everything operates on a 2-D ``(n_segments, win_samples)`` float32 array and
returns a 2-D ``(n_segments, n_features)`` float32 array. Pure numpy — no
torch import, so this works with the LR / XGBoost path that doesn't need
GPUs at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_SR = 5000
DEFAULT_F0_RANGE = (100.0, 1500.0)
DEFAULT_HARMONICS = 5  # f0, 2f0, 3f0, 4f0, 5f0


FEATURE_NAMES: tuple[str, ...] = (
    "f0",
    "power_h1",
    "power_h2",
    "power_h3",
    "power_h4",
    "power_h5",
    "centroid",
    "bandwidth",
    "rolloff_85",
    "rms",
    "zcr",
)


def _power_spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, power) where ``power`` has the same row layout as ``x``.

    ``x`` may be 1-D or 2-D (rows = segments). Output rows match input rows.
    Power is ``|FFT|^2`` of the real FFT — rfft so we don't carry the negative
    half.
    """
    if x.ndim == 1:
        x = x[None, :]
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    power = (spec * spec.conj()).real.astype(np.float32, copy=False)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr).astype(np.float32, copy=False)
    return freqs, power


def estimate_f0(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    f_range: tuple[float, float] = DEFAULT_F0_RANGE,
) -> np.ndarray:
    """Per-segment fundamental frequency via peak of the power spectrum.

    ``x`` shape ``(N, win)`` -> returns ``(N,)`` float32 in Hz. This is the
    crude "argmax in band" estimator, deliberately so: the W2 baseline is
    supposed to be a strong simple-features baseline, not a proper pitch
    tracker.
    """
    freqs, power = _power_spectrum(x, sr)
    in_band = (freqs >= f_range[0]) & (freqs <= f_range[1])
    if not in_band.any():
        return np.full(power.shape[0], 0.0, dtype=np.float32)
    band_power = power[:, in_band]
    band_freqs = freqs[in_band]
    idx = np.argmax(band_power, axis=-1)
    return band_freqs[idx].astype(np.float32, copy=False)


def harmonic_powers(
    x: np.ndarray,
    f0_hz: np.ndarray,
    sr: int = DEFAULT_SR,
    n_harmonics: int = DEFAULT_HARMONICS,
    bin_tolerance: int = 2,
) -> np.ndarray:
    """Power at each of ``n_harmonics`` integer multiples of ``f0_hz``.

    Returns shape ``(N, n_harmonics)``. For each segment we sum the power in
    a small bin window around ``k * f0`` (``k`` from 1..n_harmonics) — the
    tolerance compensates for FFT bin granularity at modest segment lengths.
    """
    freqs, power = _power_spectrum(x, sr)
    n_bins = power.shape[-1]
    df = freqs[1] - freqs[0] if len(freqs) > 1 else sr / 2.0

    out = np.zeros((power.shape[0], n_harmonics), dtype=np.float32)
    for k in range(1, n_harmonics + 1):
        target = k * f0_hz  # (N,)
        center = np.clip(np.round(target / df).astype(int), 0, n_bins - 1)
        for i, c in enumerate(center):
            lo = max(0, c - bin_tolerance)
            hi = min(n_bins, c + bin_tolerance + 1)
            out[i, k - 1] = power[i, lo:hi].sum()
    return out


def spectral_shape(x: np.ndarray, sr: int = DEFAULT_SR, rolloff: float = 0.85) -> np.ndarray:
    """(centroid, bandwidth, rolloff) per segment, shape ``(N, 3)`` in Hz."""
    freqs, power = _power_spectrum(x, sr)
    eps = 1e-12
    total = power.sum(axis=-1, keepdims=True) + eps
    p_norm = power / total
    centroid = (freqs[None, :] * p_norm).sum(axis=-1)
    var = ((freqs[None, :] - centroid[:, None]) ** 2 * p_norm).sum(axis=-1)
    bandwidth = np.sqrt(var)

    cum = np.cumsum(p_norm, axis=-1)
    # Rolloff: lowest frequency below which `rolloff` of the energy lies.
    idx = np.argmax(cum >= rolloff, axis=-1)
    rolloff_freq = freqs[idx]

    return np.stack([centroid, bandwidth, rolloff_freq], axis=-1).astype(np.float32, copy=False)


def rms_amplitude(x: np.ndarray) -> np.ndarray:
    """RMS amplitude per segment, shape ``(N,)``."""
    return np.sqrt(np.mean(x.astype(np.float32, copy=False) ** 2, axis=-1)).astype(np.float32, copy=False)


def zero_crossing_rate(x: np.ndarray) -> np.ndarray:
    """ZCR = fraction of consecutive sample pairs with sign change. ``(N,)``."""
    if x.ndim == 1:
        x = x[None, :]
    signs = np.sign(x)
    # np.sign returns 0 for zero samples; promote zeros to a non-zero sign so
    # they don't artificially break a "crossing" — usually this is a non-issue
    # on z-normed signals, but it makes ZCR well-defined regardless.
    signs = np.where(signs == 0, 1, signs)
    crossings = (signs[..., :-1] * signs[..., 1:]) < 0
    return crossings.mean(axis=-1).astype(np.float32, copy=False)


def extract_features(
    X: np.ndarray,
    sr: int = DEFAULT_SR,
    f_range: tuple[float, float] = DEFAULT_F0_RANGE,
    n_harmonics: int = DEFAULT_HARMONICS,
) -> np.ndarray:
    """End-to-end: ``(N, win)`` signals -> ``(N, len(FEATURE_NAMES))`` features.

    The column order matches :data:`FEATURE_NAMES`. Input is treated as
    z-normalized by the preprocess pipeline — RMS will be ~1 by construction
    for normalized segments and only varies meaningfully when normalize=False.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, win), got {X.shape}")
    f0 = estimate_f0(X, sr=sr, f_range=f_range)
    harmonics = harmonic_powers(X, f0, sr=sr, n_harmonics=n_harmonics)
    shape = spectral_shape(X, sr=sr)
    rms = rms_amplitude(X)
    zcr = zero_crossing_rate(X)

    feats = np.column_stack([
        f0,                  # 1
        harmonics,           # n_harmonics columns (default 5)
        shape,               # 3 (centroid, bandwidth, rolloff)
        rms,                 # 1
        zcr,                 # 1
    ]).astype(np.float32, copy=False)

    return feats
