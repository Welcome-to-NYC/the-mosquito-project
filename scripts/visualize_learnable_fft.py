"""Plot the LearnableFFT cos/sin pairs before and after training.

Two questions this answers:

1. Did the model drift off the Fourier basis it was initialized with?
   If the learned weights still look like windowed sinusoids, the
   physics-informed prior is doing useful work — the network refined
   each band's window / phase but didn't throw away the structure.
   If the weights become noisy, the prior is being ignored.

2. Which frequency bands matter? Plot the per-filter L2 norm. A dead
   filter (norm collapsed to ~0) means the network learned to skip
   that frequency band entirely.

Run::

    python scripts/visualize_learnable_fft.py --exp exp_physics_informed_w6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.physics_informed import LearnableFFT, PhysicsInformedCNN  # noqa: E402


def _load_model(exp_dir: Path) -> PhysicsInformedCNN:
    cfg_path = exp_dir / "config_used.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    kwargs = (cfg.get("model") or {}).get("kwargs") or {}
    kwargs.setdefault("n_classes", 3)

    model = PhysicsInformedCNN(**kwargs)
    state = torch.load(exp_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model


def _filter_freq_response(weight: np.ndarray, sample_rate: int, n_fft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute the magnitude of each filter's DTFT over [0, sr/2]."""
    spec = np.fft.rfft(weight, n=n_fft, axis=-1)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, mag


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="experiment dir under experiments/")
    parser.add_argument("--n-show", type=int, default=8,
                        help="how many cos filter pairs to plot in the time-domain panel")
    args = parser.parse_args(argv)

    exp_dir = ROOT / "experiments" / args.exp
    if not exp_dir.is_dir():
        raise SystemExit(f"not a directory: {exp_dir}")

    print(f"-- loading {exp_dir.name}")
    model = _load_model(exp_dir)
    fft = model.fft

    # Build a freshly-initialized reference for "what the basis looked like at t=0".
    init_fft = LearnableFFT(
        n_filters=fft.n_filters,
        kernel_size=fft.kernel_size,
        sample_rate=fft.sample_rate,
        freq_range=fft.freq_range,
        stride=fft.stride,
    )
    learned = fft.conv.weight.detach().cpu().numpy()[:, 0, :]   # (2F, K)
    initial = init_fft.conv.weight.detach().cpu().numpy()[:, 0, :]
    n_filters = fft.n_filters
    cos_init = initial[0::2]   # (F, K)
    cos_learned = learned[0::2]
    sin_learned = learned[1::2]
    sample_rate = fft.sample_rate
    init_freqs = fft.init_freqs_hz.cpu().numpy()

    # ---- panel 1: time-domain comparison for a few cos filters
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    n_show = min(args.n_show, n_filters)
    chosen = np.linspace(0, n_filters - 1, n_show, dtype=int)
    t = (np.arange(fft.kernel_size) - fft.kernel_size // 2) / sample_rate * 1000  # ms

    for i in chosen:
        axes[0, 0].plot(t, cos_init[i], alpha=0.4, color="tab:gray")
        axes[0, 0].plot(t, cos_learned[i], alpha=0.7, label=f"{init_freqs[i]:.0f} Hz")
    axes[0, 0].set_title("Cos filters: initial (gray) vs learned")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("weight")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].grid(alpha=0.3)

    # ---- panel 2: per-filter weight norm
    norms_init = np.linalg.norm(initial, axis=-1)[0::2]      # cos filter norms
    norms_learned = np.linalg.norm(learned, axis=-1)[0::2]
    axes[0, 1].plot(init_freqs, norms_init, "o-", label="initial", alpha=0.6)
    axes[0, 1].plot(init_freqs, norms_learned, "x-", label="learned")
    axes[0, 1].set_title("L2 norm per cos filter")
    axes[0, 1].set_xlabel("center freq (Hz)")
    axes[0, 1].set_ylabel("‖w‖₂")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # ---- panel 3: frequency response of cos filters (heatmap)
    freqs_axis, mag = _filter_freq_response(cos_learned, sample_rate, n_fft=2048)
    im = axes[1, 0].imshow(
        mag, aspect="auto", origin="lower",
        extent=(freqs_axis[0], freqs_axis[-1], 0, n_filters),
        cmap="magma",
    )
    axes[1, 0].set_title("Learned cos-filter frequency response (per band)")
    axes[1, 0].set_xlabel("frequency (Hz)")
    axes[1, 0].set_ylabel("filter index")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.85)

    # ---- panel 4: drift — how far each filter moved from its init
    drift = np.linalg.norm(cos_learned - cos_init, axis=-1)
    axes[1, 1].bar(np.arange(n_filters), drift, color="tab:purple")
    axes[1, 1].set_title("Drift from initialization (‖cos_learned − cos_init‖₂)")
    axes[1, 1].set_xlabel("filter index (low → high freq)")
    axes[1, 1].set_ylabel("drift")
    axes[1, 1].grid(alpha=0.3, axis="y")

    out_path = exp_dir / "learnable_fft_basis.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"-- saved {out_path}")

    # Also dump the numeric drift table for quick inspection
    table_path = exp_dir / "learnable_fft_drift.csv"
    with table_path.open("w") as f:
        f.write("filter_idx,init_freq_hz,init_norm,learned_norm,drift\n")
        for i in range(n_filters):
            f.write(
                f"{i},{init_freqs[i]:.2f},{norms_init[i]:.4f},"
                f"{norms_learned[i]:.4f},{drift[i]:.4f}\n"
            )
    print(f"-- saved {table_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
