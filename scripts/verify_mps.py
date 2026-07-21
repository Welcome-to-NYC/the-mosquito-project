"""Smoke test the M1 Pro / MPS environment.

Run after `./setup.sh`. Exercises the ops this project actually depends on:
matmul, Conv1d forward+backward, and torch.fft.rfft (used for spectral
features and as a sanity reference for the learnable-FFT layer).

Failures here predict failures inside training; pay attention to which ops
silently fall back to CPU.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

# Add src/ to the import path so we can reuse the project's device helper
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import torch  # noqa: E402

from src.utils.device import get_device, mps_memory_summary  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    set_seed(42)
    failures = 0

    section("environment")
    print(f"  python   : {sys.version.split()[0]}")
    print(f"  torch    : {torch.__version__}")
    print(f"  platform : {sys.platform}")
    print(f"  mps avail: {torch.backends.mps.is_available()}")
    print(f"  mps built: {torch.backends.mps.is_built()}")
    print(f"  fallback : PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '<unset>')}")

    if not torch.backends.mps.is_available():
        _fail("MPS is not available — bail out, this project requires it.")
        return 1

    device = get_device()
    print(f"  device   : {device}")

    section("matmul on MPS")
    try:
        a = torch.randn(512, 512, device=device)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = a @ a.T
        torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000 / 10
        _ok(f"512x512 matmul x10 avg {dt:.2f} ms")
    except Exception as exc:  # pragma: no cover
        _fail(f"matmul: {exc}")
        traceback.print_exc()
        failures += 1

    section("Conv1d forward + backward on MPS")
    # Mirrors the Fanioudakis 2018 first layer: 1 -> 16 channels, kernel 7.
    try:
        x = torch.randn(32, 1, 1024, device=device, requires_grad=False)
        conv = torch.nn.Conv1d(1, 16, kernel_size=7, padding=3).to(device)
        target = torch.randn(32, 16, 1024, device=device)

        t0 = time.perf_counter()
        y = conv(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        _ok(f"Conv1d(1,16,k=7) forward+backward on (32,1,1024) in {dt:.2f} ms")
        _ok(f"output shape {tuple(y.shape)}, loss={loss.item():.4f}")
    except Exception as exc:
        _fail(f"Conv1d: {exc}")
        traceback.print_exc()
        failures += 1

    section("torch.fft.rfft on MPS")
    # Spectral features are core to baselines and to validating the learnable
    # FFT layer. If this falls back, it's slow but correct.
    try:
        x = torch.randn(32, 1024, device=device)
        t0 = time.perf_counter()
        X = torch.fft.rfft(x, dim=-1)
        torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        _ok(f"rfft on (32,1024) in {dt:.2f} ms; output dtype={X.dtype}, shape={tuple(X.shape)}")
        # Round-trip check
        x_back = torch.fft.irfft(X, n=1024, dim=-1)
        err = (x - x_back).abs().max().item()
        if err < 1e-3:
            _ok(f"irfft round-trip max abs err {err:.2e}")
        else:
            _warn(f"irfft round-trip max abs err {err:.2e} (expected < 1e-3)")
    except Exception as exc:
        _fail(f"rfft: {exc}")
        traceback.print_exc()
        failures += 1

    section("memory")
    mem = mps_memory_summary()
    if mem:
        cur_mb = mem["current_allocated"] / (1024**2)
        drv_mb = mem["driver_allocated"] / (1024**2)
        _ok(f"MPS allocator: current={cur_mb:.1f} MB, driver={drv_mb:.1f} MB")
    else:
        _warn("memory summary unavailable")

    section("summary")
    if failures == 0:
        print(f"  {GREEN}ALL CHECKS PASSED{RESET}")
        return 0
    print(f"  {RED}{failures} CHECK(S) FAILED{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
