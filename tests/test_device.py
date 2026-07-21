"""Smoke tests for src.utils.device.

These run on any host (MPS or CPU). The MPS-specific assertions are guarded
behind ``torch.backends.mps.is_available()`` so the suite stays green on CI
or non-Apple-silicon hosts.
"""

from __future__ import annotations

import os

import torch

from src.utils.device import get_device, mps_memory_summary, to_device


def test_get_device_returns_torch_device():
    d = get_device()
    assert isinstance(d, torch.device)
    assert d.type in {"mps", "cpu"}


def test_get_device_respects_explicit_preference():
    d = get_device(prefer="cpu")
    assert d.type == "cpu"


def test_get_device_enables_mps_fallback_when_mps():
    # Reset and observe the side effect.
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    d = get_device()
    if d.type == "mps":
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_to_device_handles_nested_containers():
    device = torch.device("cpu")
    batch = {
        "x": torch.zeros(2, 3),
        "y": [torch.ones(4), torch.ones(4)],
        "meta": {"id": torch.tensor([1, 2, 3])},
        "label": "not-a-tensor",
        "tup": (torch.zeros(1), 5),
    }
    moved = to_device(batch, device)
    assert moved["x"].device.type == "cpu"
    assert moved["y"][0].device.type == "cpu"
    assert moved["meta"]["id"].device.type == "cpu"
    assert moved["label"] == "not-a-tensor"
    assert isinstance(moved["tup"], tuple)
    assert moved["tup"][1] == 5  # non-tensor passthrough


def test_mps_memory_summary_keys():
    mem = mps_memory_summary()
    if torch.backends.mps.is_available():
        assert {"current_allocated", "driver_allocated"} <= set(mem.keys())
    else:
        assert mem == {}
