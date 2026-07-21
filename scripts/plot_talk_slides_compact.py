"""Condensed 3-slide version of the talk deck (product / method / proof).

Reuses the palette + helpers from plot_talk_slides. Light theme, 16:9.

    python scripts/plot_talk_slides_compact.py
        -> docs/talk_slides_compact.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_talk_slides as T  # noqa: E402
from plot_talk_slides import (  # noqa: E402
    GROUND, PANEL, PANEL2, LINE, INK, DIM, FAINT, SIGNAL, SIG_SOFT,
    HARM, HARM_SOFT, MUT, MONO, page, rbox, chip, bullet)

T.NPAGES = 3
OUT = ROOT / "docs" / "talk_slides_compact.pdf"


def sechead(ax, num, title):
    ax.text(1.1, 8.15, num, ha="left", va="center", fontsize=12, color=SIGNAL,
            family=MONO, fontweight="bold")
    ax.text(1.6, 8.15, title.upper(), ha="left", va="center", fontsize=12,
            color=DIM, family=MONO)


def colhead(ax, x, txt):
    ax.text(x, 4.5, txt.upper(), ha="left", va="center", fontsize=10.5,
            color=HARM, family=MONO)


# ---------- Slide 1: THE PRODUCT ----------
def c1(ax):
    ax.plot([1.1, 1.7], [8.15, 8.15], color=SIGNAL, lw=1.4)
    ax.text(1.85, 8.13, "WINGBEAT CLASSIFIER · ON-DEVICE", ha="left", va="center",
            fontsize=11.5, color=SIGNAL, family=MONO)
    ax.text(1.08, 7.25, "Mosquito,", ha="left", va="center", fontsize=46,
            color=INK, fontweight="bold")
    ax.text(6.0, 7.25, "or not.", ha="left", va="center", fontsize=46,
            color=SIGNAL, fontweight="bold")
    ax.text(1.12, 6.25, "A machine-learning model that identifies a mosquito from its wingbeat, "
            "running on a four-dollar chip.", ha="left", va="center", fontsize=14.5, color=DIM)
    x = 1.1
    for num, lab in [("91.6%", "on-chip acc"), ("ESP32", "~$4 chip"), ("23,394", "parameters")]:
        x = chip(ax, x, 5.0, num, lab)
    ax.plot([1.1, 14.9], [4.75, 4.75], color=LINE, lw=1)

    # left column: what it does
    colhead(ax, 1.1, "What it does")
    bullet(ax, 1.15, 3.75, "Reads an insect's wingbeat", size=15)
    bullet(ax, 1.15, 3.1, "Decides: mosquito, or not", size=15)
    bullet(ax, 1.15, 2.45, "Runs on-device, on an ESP32", size=15)

    # right column: data
    colhead(ax, 8.2, "Trained on public data")
    ax.text(8.2, 3.85, "HumBugDB", ha="left", va="center", fontsize=15.5, color=INK, fontweight="bold")
    ax.text(10.4, 3.85, "mosquito · 20+ species", ha="left", va="center", fontsize=12, color=DIM)
    ax.text(8.2, 3.4, "Aedes · Culex · Anopheles · Mansonia", ha="left", va="center",
            fontsize=11.5, color=FAINT, style="italic")
    ax.text(8.2, 2.65, "InsectSound1000", ha="left", va="center", fontsize=15.5, color=INK, fontweight="bold")
    ax.text(11.15, 2.65, "non-mosquito", ha="left", va="center", fontsize=12, color=DIM)
    ax.text(8.2, 2.2, "gall midge · fungus gnat · hoverfly", ha="left", va="center",
            fontsize=11.5, color=FAINT, style="italic")
    ax.text(1.1, 1.35, "5 kHz · 1024-sample windows (0.2 s) · recording-level split, no leakage",
            ha="left", va="center", fontsize=10.5, color=FAINT, family=MONO)


# ---------- Slide 2: THE METHOD ----------
def c2(ax, fig):
    sechead(ax, "01", "The method")
    ax.text(1.1, 7.5, "Frequency + harmonic overtones", ha="left", va="top", fontsize=32,
            color=INK, fontweight="bold")
    body = ("Some insects share the exact same wingbeat\n"
            "frequency, so pitch alone can't tell them apart.\n"
            "The answer is harmonics: the overtones above\n"
            "that pitch, like a flute versus a trumpet.\n"
            "Our LearnableFFT captures them, and the\n"
            "CNN classifies.")
    ax.text(1.1, 6.1, body, ha="left", va="top", fontsize=13.5, color=DIM, linespacing=1.4)

    # spectrum inset (right, below the title)
    axs = fig.add_axes([0.56, 0.35, 0.39, 0.29])
    axs.set_facecolor(PANEL)
    for sp in axs.spines.values():
        sp.set_color(LINE)
    f = np.linspace(0, 2600, 700)
    HARMF = [500, 1000, 1500, 2000, 2500]
    def curve(a):
        y = np.zeros_like(f)
        for k, amp in zip(HARMF, a):
            w = 42 + HARMF.index(k) * 7
            y += amp * np.exp(-0.5 * ((f - k) / w) ** 2)
        return y
    m = curve([1.0, 0.62, 0.40, 0.22, 0.10])
    o = curve([1.0, 0.26, 0.11, 0.05, 0.03])
    axs.axvspan(430, 560, color=SIGNAL, alpha=0.10)
    axs.fill_between(f, o, color=HARM, alpha=0.14); axs.plot(f, o, color=HARM, lw=2.0)
    axs.fill_between(f, m, color=SIGNAL, alpha=0.14); axs.plot(f, m, color=SIGNAL, lw=2.2)
    axs.set_xlim(0, 2600); axs.set_ylim(0, 1.15); axs.set_yticks([])
    axs.set_xticks(HARMF); axs.set_xticklabels(["f₀", "2f₀", "3f₀", "4f₀", "5f₀"],
                                               color=FAINT, fontsize=8.5, family=MONO)
    axs.tick_params(colors=LINE, length=3)
    axs.text(1300, 1.02, "mosquito vs other insect · same f₀", color=DIM, fontsize=8.5, family=MONO)

    # architecture flow (bottom)
    nodes = [("Input", "1 × 1024", False), ("LearnableFFT", "48 × 256", True),
             ("Conv ×2", "32 × 64", False), ("GAP + Linear", "→ 2", False)]
    x0, y0, w, h, gap = 1.1, 1.35, 3.15, 1.55, 0.55
    for i, (nm, shp, key) in enumerate(nodes):
        x = x0 + i * (w + gap)
        rbox(ax, x, y0, w, h, SIG_SOFT if key else PANEL, SIGNAL if key else LINE,
             2.0 if key else 1.3)
        ax.text(x + w / 2, y0 + h - 0.42, nm, ha="center", va="center", fontsize=13.5,
                color=INK, fontweight="bold")
        ax.text(x + w / 2, y0 + 0.5, shp, ha="center", va="center", fontsize=13,
                color=SIGNAL if key else HARM, family=MONO, fontweight="bold")
        if i < 3:
            ax.text(x + w + gap / 2, y0 + h / 2, "→", ha="center", va="center",
                    fontsize=18, color=FAINT)
    ax.text(15.0, 0.7, "conv1d only · 23,394 params · 91 KB", ha="right", va="center",
            fontsize=10.5, color=FAINT, family=MONO)


# ---------- Slide 3: THE PROOF ----------
def c3(ax):
    sechead(ax, "02", "The proof")
    ax.text(1.1, 7.2, "Grounded, and tested", ha="left", va="top", fontsize=34,
            color=INK, fontweight="bold")
    # left: research
    items = [("Over a million insects sorted into hundreds of\n"
              "types by wingbeat spectra — overtones split the\n"
              "same-frequency ones.", "Yamoa · Brydegaard 2025, Sci Rep (optical)"),
             ("Mosquitoes carry rich, clean harmonics —\n"
              "six-plus overtones on integer multiples.",
              "Arthur et al. 2014, JASA (acoustic)")]
    y = 4.35
    for txt, cite in items:
        ax.add_patch(Rectangle((1.1, y), 0.06, 1.55, color=HARM, zorder=3))
        rbox(ax, 1.16, y, 6.3, 1.55, PANEL, LINE, 1.1)
        ax.text(1.5, y + 1.28, txt, ha="left", va="top", fontsize=11.5, color=INK, linespacing=1.35)
        ax.text(1.5, y + 0.3, cite, ha="left", va="center", fontsize=9, color=FAINT, family=MONO)
        y -= 1.8

    # right: synthetic bars
    ax.text(8.5, 6.05, "Our synthetic stress test", ha="left", va="center", fontsize=13,
            color=INK, fontweight="bold")
    ax.text(8.5, 5.6, "same pitch, different overtones only", ha="left", va="center",
            fontsize=10.5, color=FAINT, family=MONO)
    bx, bw = 8.5, 6.0

    def bar(y, label, val, frac, col):
        ax.text(bx, y + 0.5, label, ha="left", va="center", fontsize=14, color=INK)
        ax.text(bx + bw, y + 0.5, val, ha="right", va="center", fontsize=14, color=col,
                family=MONO, fontweight="bold")
        rbox(ax, bx, y - 0.28, bw, 0.42, PANEL2, LINE, 1.0)
        ax.add_patch(FancyBboxPatch((bx, y - 0.28), bw * frac, 0.42,
                     boxstyle="round,pad=0,rounding_size=0.1", linewidth=0,
                     facecolor=col, zorder=3))
    bar(4.55, "Frequency only", "51%", 0.51, MUT)
    bar(3.35, "Learnable FFT + CNN", "95–100%", 0.97, SIGNAL)

    # bottom: on-chip strip
    rbox(ax, 1.1, 0.75, 13.8, 1.15, PANEL, LINE, 1.3)
    ax.text(1.5, 1.55, "ON CHIP", ha="left", va="center", fontsize=10, color=HARM, family=MONO)
    ax.text(1.5, 1.1, "ESP32-CAM · hand-coded C++ (no runtime) · 108 KB SRAM (33%) · "
            "bit-exact vs PyTorch 1200/1200", ha="left", va="center", fontsize=12, color=DIM)
    ax.text(14.5, 1.32, "91.6%", ha="right", va="center", fontsize=22, color=SIGNAL,
            family=MONO, fontweight="bold")


def main():
    OUT.parent.mkdir(exist_ok=True)
    with PdfPages(OUT) as pdf:
        for i, fn in enumerate([c1, c2, c3], start=1):
            fig, ax = page()
            if i == 2:
                c2(ax, fig)
            else:
                fn(ax)
            T.deco(ax, i)
            pdf.savefig(fig, facecolor=GROUND); plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
