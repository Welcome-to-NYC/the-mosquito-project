"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch (CPU + MPS).

    ``deterministic=True`` also forces deterministic algorithms where supported.
    Note that on MPS many ops are non-deterministic regardless, so use this as
    a best-effort guarantee, not an absolute one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
