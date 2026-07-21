"""Tests for the 1D-CNN architecture."""

from __future__ import annotations

import pytest
import torch

from src.models.cnn_1d import CNN1D, ConvBlock, num_params


def test_forward_pass_shape():
    model = CNN1D(n_classes=3)
    x = torch.randn(8, 1, 1024)
    y = model(x)
    assert y.shape == (8, 3)
    assert y.dtype == torch.float32


def test_forward_with_different_input_lengths():
    # AdaptiveAvgPool1d(1) means any input length collapses to channel size.
    model = CNN1D(n_classes=3)
    for L in (256, 1024, 4096):
        x = torch.randn(2, 1, L)
        assert model(x).shape == (2, 3)


def test_param_count_is_small():
    model = CNN1D(n_classes=3)
    n = num_params(model)
    # Tight enough that ESP32 INT8 deployment stays viable. 13 K param ceiling
    # leaves headroom but still flags accidental architecture bloat.
    assert n < 13_000, f"unexpected param count: {n}"


def test_param_count_scales_with_channel_width():
    small = num_params(CNN1D(n_classes=3, channels=(8, 16, 32), kernel_sizes=(7, 5, 3), fc_hidden=16))
    large = num_params(CNN1D(n_classes=3, channels=(32, 64, 128), kernel_sizes=(7, 5, 3), fc_hidden=64))
    assert small < large


def test_rejects_misaligned_channels_and_kernels():
    with pytest.raises(ValueError):
        CNN1D(channels=(16, 32, 64), kernel_sizes=(7, 5))


def test_conv_block_rejects_even_kernel():
    with pytest.raises(ValueError):
        ConvBlock(1, 16, kernel_size=6)


def test_forward_rejects_2d_input():
    model = CNN1D()
    with pytest.raises(ValueError):
        model(torch.randn(8, 1024))


def test_backward_pass_runs_and_grads_flow():
    model = CNN1D(n_classes=3)
    x = torch.randn(4, 1, 1024)
    target = torch.randint(0, 3, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), target)
    loss.backward()
    grads_present = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()]
    assert all(grads_present), "some parameters did not receive gradients"


def test_runs_on_mps_when_available():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    device = torch.device("mps")
    model = CNN1D(n_classes=3).to(device)
    x = torch.randn(4, 1, 1024, device=device)
    y = model(x)
    assert y.device.type == "mps"
    assert y.shape == (4, 3)
