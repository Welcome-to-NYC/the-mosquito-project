"""Device helpers tuned for Apple Silicon (MPS).

The project trains on M1 Pro with the PyTorch MPS backend. A handful of ops
are still missing or slow on MPS; we enable the CPU fallback env var on
import so silent crashes turn into transparent (slower) CPU ops.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _enable_mps_fallback() -> None:
    """Allow MPS to fall back to CPU for unsupported ops.

    Must be set before the first MPS tensor is created. Setting it from Python
    works because PyTorch reads it lazily at op-dispatch time.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Order of preference: explicit ``prefer`` -> MPS -> CPU. CUDA is intentionally
    not considered: this project is M1-only.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        _enable_mps_fallback()
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in ``obj`` (tensor / dict / list / tuple) to ``device``.

    Non-tensor leaves are returned as-is, which makes this safe to call on
    arbitrary batch dictionaries from a DataLoader.
    """
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=False)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


def mps_memory_summary() -> dict[str, int]:
    """Snapshot of MPS allocator state (bytes). Empty dict on non-MPS hosts."""
    if not torch.backends.mps.is_available():
        return {}
    return {
        "current_allocated": torch.mps.current_allocated_memory(),
        "driver_allocated": torch.mps.driver_allocated_memory(),
    }
