"""Render the deployed LFFTDetector as a clean architecture graph (Netron-style).

Shapes and per-layer params are pulled from the ACTUAL model via forward hooks,
not hand-typed, so the figure can't drift from the code.

    python scripts/plot_model_graph.py
-> docs/model_architecture.png  (+ _dark.png)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.improve_audio_detector_1d import LFFTDetector  # noqa: E402


def trace():
    m = LFFTDetector(n_filters=48, n_classes=2).eval()
    sh = {}
    for n in ["fft", "b1", "b2", "gap", "fc"]:
        getattr(m, n).register_forward_hook(
            lambda mod, i, o, k=n: sh.__setitem__(k, tuple(o.shape)))
    with torch.no_grad():
        m(torch.randn(1, 1, 1024))
    npar = {n: sum(p.numel() for p in getattr(m, n).parameters())
            for n in ["fft", "b1", "b2", "gap", "fc"]}
    tot = sum(p.numel() for p in m.parameters())
    return sh, npar, tot


THEMES = {
    "light": dict(
        bg="#ffffff", ink="#1e2330", dim="#5b6373", edge="#8b93a3",
        amber="#d98a1f", amber_fill="#fbf1df",
        slate="#334155", slate_fill="#eef1f7",
        neut="#aab2c0", neut_fill="#f4f5f8",
        teal="#0e9384", teal_fill="#e6f4f1",
        rose="#c9455f", rose_fill="#fbedf0",
    ),
    "dark": dict(
        bg="#0b0e15", ink="#eae8e2", dim="#9aa4b6", edge="#5c6579",
        amber="#f5b13d", amber_fill="#241d10",
        slate="#9db2d8", slate_fill="#161d2c",
        neut="#5c6579", neut_fill="#141926",
        teal="#46caba", teal_fill="#0f1f1d",
        rose="#e0687e", rose_fill="#221217",
    ),
}


def draw(theme_name, sh, npar, tot):
    C = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7.4, 11.2))
    fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 10); ax.set_ylim(0, 107); ax.axis("off")

    MONO = {"family": "monospace"}
    cx = 5.35

    def box(y, h, w, title, sub, fill, edge, tp, lw=1.6, tsize=12.5, ssize=9.6):
        b = FancyBboxPatch((cx - w/2, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.9",
                           linewidth=lw, edgecolor=edge, facecolor=fill,
                           mutation_aspect=0.5, zorder=3)
        ax.add_patch(b)
        ty = y + h/2 + (0.9 if sub else 0)
        ax.text(cx, ty, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=tp, zorder=4)
        if sub:
            ax.text(cx, y + h/2 - 1.05, sub, ha="center", va="center",
                    fontsize=ssize, color=C["dim"], zorder=4, **MONO)

    def arrow(y0, y1, label):
        a = FancyArrowPatch((cx, y0), (cx, y1), arrowstyle="-|>",
                            mutation_scale=14, lw=1.5, color=C["edge"], zorder=2)
        ax.add_patch(a)
        ax.text(cx + 0.45, (y0 + y1)/2, label, ha="left", va="center",
                fontsize=9.2, color=C["dim"], **MONO)

    def ptag(y, w, p):
        if p == 0:
            txt = "0 params"
        else:
            txt = f"{p:,} p"
        ax.text(cx + w/2 - 0.35, y + 0.55, txt, ha="right", va="bottom",
                fontsize=8.2, color=C["dim"], alpha=.9, **MONO)

    # ---- layout (top y=95 downward) ----
    # input pill
    box(90.5, 4.4, 6.6, "Raw waveform", "1 × 1024  ·  5 kHz  ·  ~0.2 s",
        C["teal_fill"], C["teal"], C["ink"], tsize=12.5)
    arrow(90.3, 84.6, "1 × 1024")

    # LearnableFFT (hero)
    box(77.5, 6.9, 8.2, "LearnableFFT front-end",
        "Conv1d 1→48  k=129  s=4   +  |·| magnitude",
        C["amber_fill"], C["amber"], C["ink"], lw=2.4, tsize=13.5)
    ax.text(cx - 8.2/2 + 0.35, 77.5 + 6.9 - 0.5, "Fourier basis, then trained",
            ha="left", va="top", fontsize=8.4, color=C["amber"], style="italic")
    ptag(77.5, 8.2, npar["fft"])
    arrow(77.3, 71.6, f"{sh['fft'][1]} × {sh['fft'][2]}   spectrogram")

    # conv block 1
    box(65.0, 5.6, 8.0, "Conv block 1", "Conv1d 48→32  k=5 · BN · ReLU · MaxPool2",
        C["slate_fill"], C["slate"], C["ink"])
    ptag(65.0, 8.0, npar["b1"])
    arrow(64.8, 59.4, f"{sh['b1'][1]} × {sh['b1'][2]}")

    # conv block 2
    box(52.8, 5.6, 8.0, "Conv block 2", "Conv1d 32→32  k=3 · BN · ReLU · MaxPool2",
        C["slate_fill"], C["slate"], C["ink"])
    ptag(52.8, 8.0, npar["b2"])
    arrow(52.6, 47.2, f"{sh['b2'][1]} × {sh['b2'][2]}")

    # GAP
    box(41.0, 5.2, 7.2, "Global avg pool", "mean over time  ·  dropout",
        C["neut_fill"], C["neut"], C["ink"], tsize=12.5)
    ptag(41.0, 7.2, npar["gap"])
    arrow(40.8, 35.4, f"{sh['gap'][1]}-d embedding")

    # Linear
    box(29.8, 5.2, 7.2, "Linear classifier", "Linear 32 → 2  ·  softmax",
        C["neut_fill"], C["neut"], C["ink"], tsize=12.5)
    ptag(29.8, 7.2, npar["fc"])
    arrow(29.6, 24.2, f"{sh['fc'][1]} logits")

    # output
    box(19.4, 4.6, 7.6, "mosquito  /  not-mosquito", "",
        C["rose_fill"], C["rose"], C["ink"], tsize=12.5)

    # ---- side group brackets ----
    def bracket(y0, y1, label, col):
        x = cx - 4.9
        ax.plot([x, x-0.35, x-0.35, x], [y0, y0, y1, y1],
                color=col, lw=1.3, zorder=1)
        ax.text(x-0.7, (y0+y1)/2, label, rotation=90, ha="center", va="center",
                fontsize=9.5, color=col, fontweight="bold")
    bracket(84.4, 77.5, "FRONT-END", C["amber"])
    bracket(70.6, 29.8, "CNN CLASSIFIER", C["slate"])

    # ---- title + footer ----
    ax.text(0.9, 105.6, "LearnableFFT Detector", fontsize=19, fontweight="bold",
            color=C["ink"], ha="left", va="top")
    ax.text(0.9, 102.4, "on-device mosquito wingbeat classifier", fontsize=11,
            color=C["dim"], ha="left", va="top")

    ax.text(9.4, 105.6, f"{tot:,}", fontsize=20, fontweight="bold",
            color=C["amber"], ha="right", va="top", **MONO)
    ax.text(9.4, 102.2, f"parameters  ·  {tot*4/1024:.0f} KB fp32",
            fontsize=9.5, color=C["dim"], ha="right", va="top", **MONO)

    ax.text(0.9, 13.8, "Every stage is conv1d + one magnitude op — hand-portable to ESP32,\n"
                       "no inference runtime.  Verified bit-exact vs PyTorch (1200/1200).",
            fontsize=9.6, color=C["dim"], ha="left", va="top")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = ROOT / "docs" / (f"model_architecture{'' if theme_name=='light' else '_dark'}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=C["bg"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", out)


def main():
    sh, npar, tot = trace()
    print("shapes:", sh)
    for t in ("light", "dark"):
        draw(t, sh, npar, tot)


if __name__ == "__main__":
    main()
