"""Tests for the dataset-agnostic preprocessing primitives."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocess import (
    DEFAULT_HOP,
    DEFAULT_WIN,
    WindowingResult,
    segment,
    stack_segments,
    window_signal,
    znorm,
)


def test_segment_emits_full_windows():
    x = np.arange(2048, dtype=np.float32)
    seg = segment(x, win_samples=1024, hop_samples=512)
    # Starts at 0, 512, 1024 — last one ends at 2048 exactly.
    assert len(seg) == 3
    assert all(s.shape == (1024,) for s in seg)
    assert seg[0][0] == 0
    assert seg[1][0] == 512
    assert seg[2][0] == 1024


def test_segment_drops_short_tail_by_default():
    x = np.arange(1500, dtype=np.float32)  # one full window + 476 leftover
    seg = segment(x, win_samples=1024, hop_samples=512)
    assert len(seg) == 1


def test_segment_pads_short_tail_when_asked():
    x = np.arange(1500, dtype=np.float32)
    seg = segment(x, win_samples=1024, hop_samples=512, drop_last_short=False)
    # First full window + a padded tail starting at 512.
    assert len(seg) == 2
    assert seg[1][:988].tolist() == list(range(512, 1500))
    assert (seg[1][988:] == 0).all()


def test_segment_rejects_2d():
    with pytest.raises(ValueError):
        segment(np.zeros((2, 10), dtype=np.float32))


def test_segment_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        segment(np.zeros(100, dtype=np.float32), win_samples=0, hop_samples=10)


def test_znorm_zero_mean_unit_std():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    z = znorm(x)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1.0) < 1e-6
    assert z.dtype == np.float32


def test_znorm_constant_signal_is_zero():
    x = np.full(64, 7.0, dtype=np.float32)
    z = znorm(x)
    assert (z == 0).all()


def test_window_signal_defaults():
    x = np.random.RandomState(0).randn(4096).astype(np.float32)
    out = window_signal(x)
    assert isinstance(out, WindowingResult)
    # 4096 samples, win=1024, hop=512 → starts at 0..3072 step 512 = 7 windows.
    assert out.segments.shape == (7, DEFAULT_WIN)
    assert out.segments.dtype == np.float32
    # Each row should be approximately z-normed.
    means = out.segments.mean(axis=1)
    stds = out.segments.std(axis=1)
    assert np.allclose(means, 0, atol=1e-5)
    assert np.allclose(stds, 1, atol=1e-5)


def test_window_signal_short_input_returns_empty():
    x = np.zeros(100, dtype=np.float32)
    out = window_signal(x)
    assert out.segments.shape == (0, DEFAULT_WIN)
    assert out.n_dropped == 1


def test_window_signal_no_normalize():
    x = np.full(2048, 7.0, dtype=np.float32)
    out = window_signal(x, normalize=False)
    assert (out.segments == 7.0).all()


def test_stack_segments_carries_source_index():
    a = WindowingResult(segments=np.zeros((2, 4), dtype=np.float32), n_dropped=0)
    b = WindowingResult(segments=np.ones((3, 4), dtype=np.float32), n_dropped=0)
    X, src = stack_segments([a, b])
    assert X.shape == (5, 4)
    assert src.tolist() == [0, 0, 1, 1, 1]


def test_stack_segments_handles_empty_inputs():
    empty = WindowingResult(segments=np.empty((0, 4), dtype=np.float32), n_dropped=1)
    a = WindowingResult(segments=np.ones((2, 4), dtype=np.float32), n_dropped=0)
    X, src = stack_segments([empty, a, empty])
    assert X.shape == (2, 4)
    assert src.tolist() == [1, 1]


def test_stack_segments_all_empty_returns_empty():
    empty = WindowingResult(segments=np.empty((0, 0), dtype=np.float32), n_dropped=1)
    X, src = stack_segments([empty, empty])
    assert X.size == 0
    assert src.size == 0
