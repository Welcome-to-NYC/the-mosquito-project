"""Render the technical-talk deck (7 slides, minus the sound->light slide) to a
single 16:9 PDF that matches the on-screen deck's dark, instrument-panel look.

    python scripts/plot_talk_slides.py   ->  docs/talk_slides.pdf
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
OUT = ROOT / "docs" / "talk_slides.pdf"

# palette (light / print)
GROUND="#ffffff"; PANEL="#f4f5f8"; PANEL2="#e9ecf2"; LINE="#d7dbe4"
INK="#1a1f2b"; DIM="#586074"; FAINT="#8a92a2"
SIGNAL="#c8811a"; SIG_SOFT="#f8eeda"; HARM="#0e9384"; HARM_SOFT="#e2f2ef"; MUT="#aeb6c4"
MONO="DejaVu Sans Mono"

NPAGES = 7


def page():
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.patch.set_facecolor(GROUND); ax.set_facecolor(GROUND)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def deco(ax, n):
    ax.text(15.4, 8.62, "WINGBEAT · ON-DEVICE", ha="right", va="center",
            fontsize=8.5, color=FAINT, family=MONO)
    ax.text(8, 0.34, f"{n} / {NPAGES}", ha="center", va="center",
            fontsize=8.5, color=FAINT, family=MONO)


def eyebrow(ax, num, txt):
    ax.text(1.1, 8.05, num, ha="left", va="center", fontsize=12, color=SIGNAL,
            family=MONO, fontweight="bold")
    ax.text(1.1 + 0.55, 8.05, txt.upper(), ha="left", va="center", fontsize=12,
            color=DIM, family=MONO)


def rbox(ax, x, y, w, h, fc, ec, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14",
                 linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1, zorder=2))


def chip(ax, x, y, num, label):
    w = 0.55 + 0.20*len(num) + 0.11*len(label)
    rbox(ax, x, y, w, 0.7, PANEL, LINE, 1.2)
    ax.text(x+0.28, y+0.35, num, ha="left", va="center", fontsize=16, color=INK,
            family=MONO, fontweight="bold")
    ax.text(x+0.28+0.22*len(num)+0.15, y+0.35, label.upper(), ha="left", va="center",
            fontsize=9, color=FAINT, family=MONO)
    return x + w + 0.35


def bullet(ax, x, y, txt, color=INK, size=17):
    ax.add_patch(plt.Circle((x+0.08, y+0.02), 0.07, color=SIGNAL, zorder=3))
    ax.text(x+0.45, y, txt, ha="left", va="center", fontsize=size, color=color)


# ---------------- slides ----------------
def s1(ax):
    ax.plot([1.1, 1.7], [7.05, 7.05], color=SIGNAL, lw=1.4)
    ax.text(1.85, 7.03, "WINGBEAT CLASSIFIER · ON-DEVICE", ha="left", va="center",
            fontsize=12.5, color=SIGNAL, family=MONO)
    ax.text(1.05, 5.35, "Mosquito,", ha="left", va="center", fontsize=68,
            color=INK, fontweight="bold")
    ax.text(1.05, 3.85, "or not.", ha="left", va="center", fontsize=68,
            color=SIGNAL, fontweight="bold")
    ax.text(1.12, 2.55, "A machine-learning model that identifies a mosquito from its\n"
            "wingbeat — running on a four-dollar chip.", ha="left", va="top",
            fontsize=17, color=DIM)
    x = 1.1
    for num, lab in [("91.6%", "on-chip acc"), ("ESP32", "~$4 chip"), ("23,394", "parameters")]:
        x = chip(ax, x, 0.95, num, lab)


def s2(ax):
    eyebrow(ax, "01", "What we built")
    ax.text(1.1, 6.7, "A lean model,\na precise job.", ha="left", va="top",
            fontsize=40, color=INK, fontweight="bold", linespacing=1.02)
    bullet(ax, 1.15, 3.55, "Reads an insect's wingbeat")
    bullet(ax, 1.15, 2.85, "Decides: mosquito, or not")
    bullet(ax, 1.15, 2.15, "Runs on-device, on an ESP32 (~$4)")
    # big stat right
    ax.text(12.4, 5.1, "91.6%", ha="center", va="center", fontsize=96,
            color=SIGNAL, family=MONO, fontweight="bold")
    ax.text(12.4, 3.5, "accuracy on public test data", ha="center", va="center",
            fontsize=15, color=DIM)


def s3(ax):
    eyebrow(ax, "05", "The model")
    ax.text(1.1, 7.3, "LearnableFFT + CNN", ha="left", va="top", fontsize=40,
            color=INK, fontweight="bold")
    # architecture: 4 blocks with real shapes/params
    nodes = [
        ("Input", "raw wingbeat", "1 × 1024", "0.2 s @ 5 kHz", False),
        ("LearnableFFT", "Conv1d = Fourier basis", "48 × 256", "48 filt · k129 · s4", True),
        ("Conv ×2", "48→32→32 · BN · pool", "32 × 64", "k5, k3 · MaxPool2", False),
        ("GAP + Linear", "pool, then classify", "→ 2", "mosquito / not", False),
    ]
    x0, y0, w, h, gap = 1.1, 4.15, 3.15, 1.95, 0.55
    for i, (nm, op, shp, sub, key) in enumerate(nodes):
        x = x0 + i*(w+gap)
        rbox(ax, x, y0, w, h, SIG_SOFT if key else PANEL, SIGNAL if key else LINE,
             2.2 if key else 1.4)
        ax.text(x+w/2, y0+h-0.42, nm, ha="center", va="center", fontsize=15,
                color=INK, fontweight="bold")
        ax.text(x+w/2, y0+h-0.82, op, ha="center", va="center", fontsize=9.5,
                color=DIM, family=MONO)
        # shape tag
        cc = SIGNAL if key else HARM
        ax.text(x+w/2, y0+0.72, shp, ha="center", va="center", fontsize=14,
                color=cc, family=MONO, fontweight="bold")
        ax.text(x+w/2, y0+0.34, sub, ha="center", va="center", fontsize=8.5,
                color=FAINT, family=MONO)
        if i < 3:
            ax.text(x+w+gap/2, y0+h/2, "→", ha="center", va="center", fontsize=20,
                    color=FAINT)
    ax.text(1.1, 3.35, "The LearnableFFT starts as a real Fourier transform, then trains — tuning to the\n"
            "frequencies and harmonics that separate insects. Everything is conv1d plus one\n"
            "magnitude op, so it ports to C++ by hand, with no inference runtime.",
            ha="left", va="top", fontsize=14.5, color=DIM, linespacing=1.4)
    ax.text(1.1, 1.35, "23,394 parameters  ·  91 KB (fp32)", ha="left", va="center",
            fontsize=13, color=SIGNAL, family=MONO, fontweight="bold")


def s4(ax):
    eyebrow(ax, "02", "The data")
    ax.text(1.1, 7.55, "Trained on public data", ha="left", va="top", fontsize=37,
            color=INK, fontweight="bold")
    # two dataset cards with real species
    for x, tag, nm in [(1.1, "MOSQUITOES", "HumBugDB"), (8.1, "NON-MOSQUITO", "InsectSound1000")]:
        rbox(ax, x, 3.2, 6.4, 3.05, PANEL, LINE, 1.4)
        ax.text(x+0.4, 5.85, tag, ha="left", va="center", fontsize=10.5, color=HARM, family=MONO)
        ax.text(x+0.4, 5.38, nm, ha="left", va="center", fontsize=19, color=INK, fontweight="bold")
    # mosquito genera + examples
    ax.text(1.5, 4.55, "Aedes · Culex · Anopheles · Mansonia", ha="left", va="center",
            fontsize=13.5, color=INK, fontweight="bold")
    ax.text(1.5, 4.02, "20+ species · Anopheles most common", ha="left", va="center",
            fontsize=10.5, color=DIM, family=MONO)
    ax.text(1.5, 3.6, "e.g. Ae. albopictus · Cx. quinquefasciatus · An. gambiae", ha="left",
            va="center", fontsize=9.5, color=FAINT, style="italic")
    # non-mosquito 3 species
    sp = [("gall midge", "Aphidoletes aphidimyza"),
          ("fungus gnat", "Bradysia difformis"),
          ("hoverfly", "Episyrphus balteatus")]
    yy = 4.7
    for common, sci in sp:
        ax.add_patch(plt.Circle((8.6, yy+0.02), 0.05, color=HARM, zorder=4))
        ax.text(8.85, yy, common, ha="left", va="center", fontsize=12.5, color=INK, fontweight="bold")
        ax.text(10.7, yy, sci, ha="left", va="center", fontsize=10.5, color=FAINT, style="italic")
        yy -= 0.52
    # recording spec + performance
    ax.text(1.1, 2.5, "5 kHz  ·  1024-sample windows (0.2 s)  ·  recording-level split, no leakage",
            ha="left", va="center", fontsize=12.5, color=FAINT, family=MONO)
    ax.text(1.1, 1.72, "Negatives are small midges, gnats and a hoverfly — mosquito-adjacent, not silence.",
            ha="left", va="center", fontsize=14, color=DIM)
    ax.text(1.1, 1.05, "On this data, the model reaches ", ha="left", va="center",
            fontsize=15, color=INK)
    ax.text(6.05, 1.05, "91.6% accuracy.", ha="left", va="center",
            fontsize=15, color=SIGNAL, fontweight="bold")


def s5(ax):
    eyebrow(ax, "03", "The hard case")
    ax.text(1.1, 6.9, "Same frequency,\ndifferent insect.", ha="left", va="top",
            fontsize=44, color=INK, fontweight="bold", linespacing=1.02)
    ax.text(1.1, 4.05, "We started small: a compact CNN that learns the wingbeat frequency.\n"
            "But a completely different insect can share the exact same frequency,\n"
            "and then frequency alone fails.",
            ha="left", va="top", fontsize=17, color=DIM, linespacing=1.4)
    # visual row
    y = 1.7
    rbox(ax, 1.1, y, 3.6, 0.95, PANEL, LINE, 1.3)
    ax.text(2.9, y+0.47, "Mosquito  ≈ same Hz", ha="center", va="center",
            fontsize=14.5, color=INK, family=MONO)
    ax.text(5.15, y+0.47, "=", ha="center", va="center", fontsize=26, color=SIGNAL)
    rbox(ax, 5.6, y, 3.6, 0.95, PANEL, LINE, 1.3)
    ax.text(7.4, y+0.47, "Other insect  ≈ same Hz", ha="center", va="center",
            fontsize=14.5, color=INK, family=MONO)
    ax.text(9.75, y+0.47, "→", ha="center", va="center", fontsize=24, color=MUT)
    ax.text(10.2, y+0.47, "frequency-only  ✗", ha="left", va="center",
            fontsize=16, color=MUT)


def s6(ax, fig):
    eyebrow(ax, "04", "The solution")
    ax.text(1.1, 6.9, "Frequency +\nharmonic overtones", ha="left", va="top",
            fontsize=36, color=INK, fontweight="bold", linespacing=1.05)
    ax.text(1.1, 3.7, "Same note on a flute and a trumpet —\n"
            "same pitch, different timbre. That timbre\n"
            "is the overtones. And a mosquito's are\n"
            "especially strong.", ha="left", va="top", fontsize=16.5,
            color=DIM, linespacing=1.4)
    # spectrum inset (right)
    axs = fig.add_axes([0.525, 0.17, 0.4, 0.6])
    axs.set_facecolor(PANEL)
    for sp in axs.spines.values():
        sp.set_color(LINE)
    f = np.linspace(0, 2600, 700)
    HARMF = [500, 1000, 1500, 2000, 2500]
    def curve(a):
        y = np.zeros_like(f)
        for k, amp in zip(HARMF, a):
            w = 42 + HARMF.index(k)*7
            y += amp*np.exp(-0.5*((f-k)/w)**2)
        return y
    m = curve([1.0, 0.62, 0.40, 0.22, 0.10])
    o = curve([1.0, 0.26, 0.11, 0.05, 0.03])
    axs.axvspan(430, 560, color=SIGNAL, alpha=0.10)
    axs.fill_between(f, o, color=HARM, alpha=0.14); axs.plot(f, o, color=HARM, lw=2.2)
    axs.fill_between(f, m, color=SIGNAL, alpha=0.14); axs.plot(f, m, color=SIGNAL, lw=2.4)
    axs.set_xlim(0, 2600); axs.set_ylim(0, 1.15)
    axs.set_yticks([])
    axs.set_xticks(HARMF); axs.set_xticklabels(["f₀", "2f₀", "3f₀", "4f₀", "5f₀"],
                                               color=FAINT, fontsize=9, family=MONO)
    axs.tick_params(colors=LINE, length=3)
    axs.text(495, 1.07, "shared f₀", ha="center", color=DIM, fontsize=9, family=MONO)
    # legend
    ax.text(9.55, 6.7, "■ mosquito", color=SIGNAL, fontsize=12, family=MONO, ha="left")
    ax.text(11.6, 6.7, "■ other insect", color=HARM, fontsize=12, family=MONO, ha="left")


def s7(ax):
    eyebrow(ax, "06", "Evidence")
    ax.text(1.1, 7.0, "Established research —\nand our own test", ha="left", va="top",
            fontsize=34, color=INK, fontweight="bold", linespacing=1.05)
    # left research items
    def research(y, h, lines, cite):
        ax.add_patch(Rectangle((1.1, y), 0.06, h, color=HARM, zorder=3))
        rbox(ax, 1.16, y, 6.5, h, PANEL, LINE, 1.1)
        ax.text(1.55, y+h-0.32, lines, ha="left", va="top", fontsize=12.5, color=INK, linespacing=1.4)
        ax.text(1.55, y+0.32, cite, ha="left", va="center", fontsize=9, color=FAINT, family=MONO)
    research(3.35, 1.95,
             "Over a million insects, sorted into hundreds\n"
             "of types by their wingbeat spectra — overtones\n"
             "set apart the same-frequency ones.",
             "Yamoa · Brydegaard 2025, Sci Rep (optical/lidar)")
    research(1.2, 1.55,
             "Mosquitoes carry rich, clean harmonics —\n"
             "six-plus overtones on integer multiples.",
             "Arthur et al. 2014, JASA (acoustic)")
    # right bars
    bx, bw = 8.7, 6.0
    def bar(y, label, val, frac, col):
        ax.text(bx, y+0.5, label, ha="left", va="center", fontsize=15, color=INK)
        ax.text(bx+bw, y+0.5, val, ha="right", va="center", fontsize=15, color=col,
                family=MONO, fontweight="bold")
        rbox(ax, bx, y-0.28, bw, 0.42, PANEL2, LINE, 1.0)
        ax.add_patch(FancyBboxPatch((bx, y-0.28), bw*frac, 0.42,
                     boxstyle="round,pad=0,rounding_size=0.1", linewidth=0,
                     facecolor=col, zorder=3))
    bar(4.6, "Frequency only", "51%", 0.51, MUT)
    bar(3.2, "Learnable FFT + CNN", "95–100%", 0.97, SIGNAL)
    ax.text(bx, 2.35, "Our synthetic stress test — same pitch, different overtones only.",
            ha="left", va="center", fontsize=12.5, color=FAINT)


def s8(ax):
    eyebrow(ax, "07", "On-chip")
    ax.text(1.1, 7.3, "Running on real silicon", ha="left", va="top", fontsize=40,
            color=INK, fontweight="bold")
    specs = [
        ("HARDWARE", "ESP32-CAM · 240 MHz dual-core · 4 MB flash"),
        ("INFERENCE", "hand-coded C++ · no TFLite runtime"),
        ("MEMORY", "108 KB SRAM used · 33% of budget"),
        ("VERIFIED", "chip = PyTorch on 1200 / 1200 · max logit err 7e-6"),
    ]
    y = 5.55
    for tag, val in specs:
        ax.add_patch(Rectangle((1.1, y-0.5), 0.06, 0.98, color=HARM, zorder=3))
        ax.text(1.35, y+0.17, tag, ha="left", va="center", fontsize=10, color=HARM, family=MONO)
        ax.text(1.35, y-0.28, val, ha="left", va="center", fontsize=12.5, color=INK)
        y -= 1.18
    # right 2x2 metric stats
    mets = [("91.6%", "on-chip accuracy", True), ("90.6%", "mosquito recall", False),
            ("94.7%", "fly rejection", False), ("91.7%", "background reject", False)]
    pos = [(8.8, 3.95), (12.0, 3.95), (8.8, 1.75), (12.0, 1.75)]
    for (num, lab, key), (x, yy) in zip(mets, pos):
        rbox(ax, x, yy, 3.0, 1.95, SIG_SOFT if key else PANEL, SIGNAL if key else LINE,
             2.0 if key else 1.3)
        ax.text(x+1.5, yy+ch_top(1.95), num, ha="center", va="center", fontsize=29,
                color=SIGNAL if key else INK, family=MONO, fontweight="bold")
        ax.text(x+1.5, yy+0.45, lab, ha="center", va="center", fontsize=12, color=DIM)
    ax.text(1.1, 1.0, "Same-rig background is still rejected — the model learned the insect, not the recording setup.",
            ha="left", va="center", fontsize=11, color=FAINT)


def ch_top(h):
    return h - 0.62


def main():
    OUT.parent.mkdir(exist_ok=True)
    # order matches the speech: build → data → hard case → harmonics → model → evidence
    order = [s1, s2, s4, s5, None, s3, s7]
    with PdfPages(OUT) as pdf:
        for i, fn in enumerate(order, start=1):
            fig, ax = page()
            if i == 5:
                s6(ax, fig)
            else:
                fn(ax)
            deco(ax, i)
            pdf.savefig(fig, facecolor=GROUND); plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
