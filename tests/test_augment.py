"""Tests for the signal augmentation primitives."""

from __future__ import annotations

import numpy as np

from src.data.augment import (
    WingbeatAugment,
    add_pink_noise,
    add_white_noise,
    random_gain,
    random_time_shift,
)


def _sine(freq: float, n: int = 1024, sr: int = 5000) -> np.ndarray:
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_add_white_noise_respects_snr():
    rng = np.random.default_rng(0)
    x = _sine(440, n=4096)
    p_x = (x ** 2).mean()
    for snr in (0, 10, 20):
        noisy = add_white_noise(x, snr_db=snr, rng=rng)
        delta = noisy - x
        p_n = (delta ** 2).mean()
        observed = 10 * np.log10(p_x / p_n)
        # Noise variance is set exactly; with N=4096 samples the realized
        # SNR sits within a few tenths of a dB of the target.
        assert abs(observed - snr) < 1.0, (snr, observed)


def test_add_white_noise_higher_snr_means_smaller_perturbation():
    rng = np.random.default_rng(1)
    x = _sine(300)
    a = add_white_noise(x, snr_db=0, rng=rng)
    b = add_white_noise(x, snr_db=20, rng=rng)
    assert (a - x).std() > (b - x).std()


def test_add_pink_noise_returns_float32():
    rng = np.random.default_rng(2)
    x = _sine(300, n=2048)
    out = add_pink_noise(x, snr_db=10, rng=rng)
    assert out.dtype == np.float32
    assert out.shape == x.shape


def test_random_gain_within_range():
    x = _sine(300)
    rng = np.random.default_rng(3)
    for _ in range(50):
        out = random_gain(x, gain_range=(0.5, 2.0), rng=rng)
        ratio = out.max() / x.max() if x.max() != 0 else 0
        assert 0.5 - 1e-6 <= ratio <= 2.0 + 1e-6


def test_random_time_shift_preserves_length_and_values():
    x = np.arange(100, dtype=np.float32)
    rng = np.random.default_rng(4)
    out = random_time_shift(x, max_shift_frac=0.1, rng=rng)
    assert out.shape == x.shape
    # Circular shift -> the sorted multiset is unchanged.
    assert np.array_equal(np.sort(out), np.sort(x))


def test_random_time_shift_zero_when_max_shift_frac_zero():
    x = np.arange(100, dtype=np.float32)
    out = random_time_shift(x, max_shift_frac=0.0)
    np.testing.assert_array_equal(out, x.astype(np.float32))


def test_wingbeat_augment_returns_float32_same_shape():
    aug = WingbeatAugment(seed=0)
    x = _sine(300)
    out = aug(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32


def test_wingbeat_augment_disabled_pipeline_is_identity():
    aug = WingbeatAugment(p_noise=0.0, p_gain=0.0, p_shift=0.0, seed=0)
    x = _sine(300)
    out = aug(x)
    np.testing.assert_allclose(out, x)


def test_wingbeat_augment_changes_signal_when_enabled():
    aug = WingbeatAugment(p_noise=1.0, p_gain=1.0, p_shift=1.0, seed=0)
    x = _sine(300)
    out = aug(x)
    # With every aug firing, the output must differ from the input.
    assert not np.allclose(out, x, atol=1e-3)


def test_wingbeat_augment_seeded_reproducibility():
    a = WingbeatAugment(seed=7)
    b = WingbeatAugment(seed=7)
    x = _sine(300)
    np.testing.assert_array_equal(a(x), b(x))
