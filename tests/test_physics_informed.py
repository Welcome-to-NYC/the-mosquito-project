"""Tests for src.models.physics_informed."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.models.physics_informed import (
    HarmonicAttention,
    LearnableFFT,
    PhysicsInformedCNN,
    TemporalEnvelope,
)


def _sine_batch(batch: int, length: int, freq: float, sr: int = 5000) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) / sr
    sig = torch.sin(2 * math.pi * freq * t)
    return sig.unsqueeze(0).unsqueeze(0).expand(batch, 1, length).contiguous()


# --- LearnableFFT


def test_learnable_fft_output_shape():
    fft = LearnableFFT(n_filters=64, kernel_size=129)
    x = torch.randn(4, 1, 1024)
    y = fft(x)
    assert y.shape == (4, 64, 1024)
    assert y.dtype == torch.float32


def test_learnable_fft_responds_to_target_frequency():
    # A 500 Hz sine should excite the filter whose center is closest to 500 Hz
    # the strongest. With freq_range (100, 1500) and 64 filters, the bin
    # spacing is (1500-100)/63 ≈ 22 Hz; we expect the argmax band to be
    # within one bin of the carrier.
    fft = LearnableFFT(n_filters=64, kernel_size=129, sample_rate=5000, freq_range=(100, 1500))
    x = _sine_batch(2, 1024, 500.0)
    mag = fft(x).mean(dim=-1)             # (B, F)
    target_idx = int(torch.argmin(torch.abs(fft.init_freqs_hz - 500.0)).item())
    pred_idx = int(mag[0].argmax().item())
    assert abs(pred_idx - target_idx) <= 1


def test_learnable_fft_rejects_even_kernel():
    with pytest.raises(ValueError):
        LearnableFFT(kernel_size=128)


def test_learnable_fft_rejects_wrong_input_shape():
    fft = LearnableFFT()
    with pytest.raises(ValueError):
        fft(torch.randn(4, 1024))      # missing channel dim
    with pytest.raises(ValueError):
        fft(torch.randn(4, 3, 1024))   # multi-channel


def test_learnable_fft_grads_flow():
    fft = LearnableFFT(n_filters=16, kernel_size=33)
    x = torch.randn(2, 1, 256, requires_grad=False)
    target = torch.randn(2, 16, 256)
    y = fft(x)
    ((y - target) ** 2).mean().backward()
    assert fft.conv.weight.grad is not None
    assert fft.conv.weight.grad.abs().sum().item() > 0


# --- HarmonicAttention


def test_harmonic_attention_output_shapes():
    freqs = torch.linspace(100, 1500, 64)
    attn = HarmonicAttention(freqs, n_harmonics=5)
    mag = torch.rand(3, 64, 100)
    h, f0 = attn(mag)
    assert h.shape == (3, 5)
    assert f0.shape == (3,)


def test_harmonic_attention_recovers_synthetic_f0():
    # Build a fake magnitude that has a peak at the bin closest to 400 Hz —
    # the fundamental should be recovered up to bin resolution.
    freqs = torch.linspace(100, 1500, 64)
    target = 400.0
    target_idx = int(torch.argmin(torch.abs(freqs - target)).item())
    mag = torch.zeros(2, 64, 50)
    mag[:, target_idx, :] = 1.0
    attn = HarmonicAttention(freqs, n_harmonics=5)
    _, f0 = attn(mag)
    assert (torch.abs(f0 - freqs[target_idx]) < 1e-6).all()


def test_harmonic_attention_ignores_out_of_band_peaks():
    freqs = torch.linspace(100, 2000, 64)
    out_of_band_idx = int(torch.argmin(torch.abs(freqs - 1500.0)).item())  # > 1000 Hz
    in_band_idx = int(torch.argmin(torch.abs(freqs - 600.0)).item())
    mag = torch.zeros(1, 64, 10)
    mag[0, out_of_band_idx, :] = 5.0   # bigger but out of band
    mag[0, in_band_idx, :] = 1.0       # smaller but in band
    attn = HarmonicAttention(freqs, n_harmonics=5, f0_band=(200, 1000))
    _, f0 = attn(mag)
    assert torch.allclose(f0, freqs[in_band_idx])


# --- TemporalEnvelope


def test_temporal_envelope_output_shape():
    env = TemporalEnvelope(feature_dim=32)
    x = torch.randn(4, 1, 1024)
    feat = env(x)
    assert feat.shape == (4, 32)


def test_temporal_envelope_grads_flow():
    env = TemporalEnvelope(feature_dim=16)
    x = torch.randn(2, 1, 1024)
    target = torch.randn(2, 16)
    feat = env(x)
    ((feat - target) ** 2).mean().backward()
    grads = [p.grad is not None for p in env.parameters()]
    assert all(grads)


# --- PhysicsInformedCNN


def test_physics_informed_forward_shape():
    model = PhysicsInformedCNN(n_classes=3)
    x = torch.randn(4, 1, 1024)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_physics_informed_aux_outputs():
    # fft_stride=4 (the default): magnitude time dim 1024 -> 256.
    model = PhysicsInformedCNN(n_classes=3, n_filters=64, n_harmonics=5)
    x = torch.randn(2, 1, 1024)
    aux = model(x, return_aux=True)
    assert isinstance(aux, dict)
    assert aux["logits"].shape == (2, 3)
    assert aux["magnitude"].shape[0] == 2
    assert aux["magnitude"].shape[1] == 64
    assert aux["magnitude"].shape[2] == 256   # 1024 / fft_stride
    assert aux["harmonic_power"].shape == (2, 5)
    assert aux["f0_hz"].shape == (2,)


def test_physics_informed_stride_one_keeps_full_resolution():
    model = PhysicsInformedCNN(n_classes=3, n_filters=32, fft_stride=1)
    x = torch.randn(2, 1, 1024)
    aux = model(x, return_aux=True)
    assert aux["magnitude"].shape == (2, 32, 1024)


def test_physics_informed_param_count_reasonable():
    model = PhysicsInformedCNN(n_classes=3)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Expected ballpark: LearnableFFT 2*64*129 = 16k, plus three small heads.
    # Stay under 30 k so the ESP32 deployment story still holds after INT8.
    assert 5_000 < n < 35_000, f"unexpected param count: {n}"


def test_physics_informed_backward():
    model = PhysicsInformedCNN(n_classes=3)
    x = torch.randn(4, 1, 1024)
    target = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), target)
    loss.backward()
    grads_present = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert all(grads_present)


def test_physics_informed_runs_on_mps_when_available():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    device = torch.device("mps")
    model = PhysicsInformedCNN(n_classes=3).to(device)
    x = torch.randn(4, 1, 1024, device=device)
    out = model(x)
    assert out.device.type == "mps"
    assert out.shape == (4, 3)
