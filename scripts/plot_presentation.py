"""Presentation-quality slide figures for the mosquito-detector project.

Generates standalone high-DPI PNGs in docs/slides/, one per talking point,
styled for projection to a non-ML audience.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

# palette
GREEN = "#2e7d32"; BLUE = "#1565c0"; ORANGE = "#ef6c00"
GRAY = "#90a4ae"; RED = "#c62828"; DARK = "#263238"; LIGHT = "#eceff1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.edgecolor": "#b0bec5",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.8,
    "figure.dpi": 170, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load(name):
    return json.loads((EXP / name / "results.json").read_text())


def barlabels(ax, bars, fmt="{:.0%}", dy=0.012, fs=12):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=fs, fontweight="bold", color=DARK)


# ============================================================ SLIDE 1: hero
fig = plt.figure(figsize=(12, 6.2)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.text(0.5, 0.90, "Audio Mosquito Detector on an ESP32 Chip",
        ha="center", fontsize=27, fontweight="bold", color=DARK)
ax.text(0.5, 0.80, "Runs entirely on an ESP32 microcontroller — no cloud, no phone",
        ha="center", fontsize=15, color="#546e7a")
# big number
ax.text(0.24, 0.44, "91.6%", ha="center", fontsize=78, fontweight="bold", color=GREEN)
ax.text(0.24, 0.25, "detection accuracy\n(measured on the chip)", ha="center", fontsize=15, color="#455a64")
# stat cards
cards = [("91 KB", "model size\n(fp32)", BLUE), ("23,394", "parameters", BLUE),
         ("90%", "mosquitoes\ncaught", GREEN), ("2.4 / s", "on-chip\nthroughput", ORANGE)]
x0 = 0.50; cw = 0.115; gap = 0.01
for i, (big, small, col) in enumerate(cards):
    x = x0 + i*(cw+gap)
    ax.add_patch(FancyBboxPatch((x, 0.34), cw, 0.28, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=LIGHT, ec=col, lw=2, transform=ax.transAxes))
    ax.text(x+cw/2, 0.535, big, ha="center", fontsize=17, fontweight="bold", color=col)
    ax.text(x+cw/2, 0.40, small, ha="center", fontsize=11, color="#546e7a")
ax.text(0.5, 0.10, "Trained on real mosquito recordings (HumBugDB) + real fly wingbeats (InsectSound1000)",
        ha="center", fontsize=12, color="#78909c")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(OUT / "slide1_hero.png"); plt.close()
print("slide1_hero.png")


# ============================================================ SLIDE: data sources
fig = plt.figure(figsize=(12, 6.2)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.text(0.5, 0.93, "The data: real recordings", ha="center", fontsize=25, fontweight="bold", color=DARK)
ax.text(0.5, 0.855, "Two questions the model must tell apart: mosquito, or not?",
        ha="center", fontsize=14, color="#546e7a")
# MOSQUITO box
ax.add_patch(FancyBboxPatch((0.05, 0.30), 0.42, 0.44, boxstyle="round,pad=0.01,rounding_size=0.02",
                            fc="#e8f5e9", ec=GREEN, lw=2.5, transform=ax.transAxes))
ax.text(0.26, 0.685, "MOSQUITO", ha="center", fontsize=18, fontweight="bold", color=GREEN)
ax.text(0.26, 0.60, "HumBugDB", ha="center", fontsize=15, fontweight="bold", color=DARK)
ax.text(0.26, 0.435, "real mosquito recordings,\nmade with smartphones outdoors\n\n34 mosquito species\n\n(public dataset, NeurIPS 2021)",
        ha="center", va="center", fontsize=11.5, color="#455a64")
# NOT MOSQUITO box
ax.add_patch(FancyBboxPatch((0.53, 0.30), 0.42, 0.44, boxstyle="round,pad=0.01,rounding_size=0.02",
                            fc="#e3f2fd", ec=BLUE, lw=2.5, transform=ax.transAxes))
ax.text(0.74, 0.685, "NOT MOSQUITO", ha="center", fontsize=18, fontweight="bold", color=BLUE)
ax.text(0.555, 0.585,
        "• Background noise\n   (same recording devices\n    as the mosquitoes)\n\n"
        "• Real fly wingbeats\n   InsectSound1000, 3 fly\n   species (public, 2024)",
        ha="left", va="top", fontsize=11.5, color="#455a64")
ax.text(0.5, 0.15, "Training and test use SEPARATE recordings — so the accuracy is honest, not memorised",
        ha="center", fontsize=12, color=DARK, style="italic",
        bbox=dict(boxstyle="round,pad=0.45", fc=LIGHT, ec=ORANGE, lw=1.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(OUT / "slide_data.png"); plt.close()
print("slide_data.png")


# ============================================================ SLIDE 2: pipeline + verification
fig = plt.figure(figsize=(13, 4.6)); fig.patch.set_facecolor("white")
axp = fig.add_axes([0, 0, 1, 1]); axp.axis("off")
axp.text(0.5, 0.88, "How it works", ha="center", fontsize=23, fontweight="bold", color=DARK)
steps = [("Sound\n0.128 s", "#455a64"), ("Learnable FFT\n(frequency)", BLUE),
         ("Tiny CNN", BLUE), ("Mosquito?\nYes / No", GREEN)]
n = len(steps); bw = 0.19; bh = 0.30; y = 0.36
for i, (label, col) in enumerate(steps):
    x = 0.04 + i*0.245
    axp.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.01,rounding_size=0.03",
                                 fc="white", ec=col, lw=3))
    axp.text(x+bw/2, y+bh/2, label, ha="center", va="center", fontsize=15, fontweight="bold", color=col)
    if i < n-1:
        axp.add_patch(FancyArrowPatch((x+bw+0.008, y+bh/2), (x+0.245, y+bh/2),
                                      arrowstyle="-|>", mutation_scale=22, color="#b0bec5", lw=2.5))
axp.text(0.5, 0.16, "Runs entirely on the ESP32 chip — no cloud, no phone, no internet",
         ha="center", fontsize=14, color="#546e7a", style="italic")
axp.set_xlim(0, 1); axp.set_ylim(0, 1)
fig.savefig(OUT / "slide2_pipeline.png"); plt.close()
print("slide2_pipeline.png")


# ============================================================ SLIDE 3: detection + no shortcut
det = load("exp_audio_detector_1d_improved")["LearnableFFT"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))
a1.set_title("Detection performance", loc="left")
m = ["accuracy", "catches\nmosquitoes", "rejects\nbackground", "rejects\nflies"]
v = [det["acc"], det["mosq_recall"], det["neg_by_source"]["humbug_bg"], det["neg_by_source"]["insectsound_fly"]]
cols = [GREEN, GREEN, BLUE, BLUE]
bars = a1.bar(m, v, color=cols, width=0.62); barlabels(a1, bars)
a1.axhline(0.9, ls="--", c=GRAY, lw=1.2); a1.text(3.4, 0.905, "90%", color=GRAY, fontsize=10)
a1.set_ylim(0, 1.05); a1.set_ylabel("rate")

a2.set_title('Not a "cheat" — proof', loc="left")
mm = ["background\n(SAME device\nas mosquitoes)", "flies\n(different\ndevice)"]
vv = [det["neg_by_source"]["humbug_bg"], det["neg_by_source"]["insectsound_fly"]]
bars = a2.bar(mm, vv, color=[ORANGE, BLUE], width=0.5); barlabels(a2, bars)
a2.set_ylim(0, 1.05); a2.set_ylabel("correctly rejected")
a2.text(0.5, 0.13, "Rejects background from the SAME device\n=> learned the wingbeat, not the recording setup",
        transform=a2.transAxes, ha="center", fontsize=10.5, color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=ORANGE, lw=1.3))
fig.tight_layout()
fig.savefig(OUT / "slide3_detection.png"); plt.close()
print("slide3_detection.png")


# ============================================================ SLIDE: how the score is computed
fig = plt.figure(figsize=(13.5, 6.4)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.text(0.5, 0.945, "How the score is calculated", ha="center", fontsize=24, fontweight="bold", color=DARK)
ax.text(0.5, 0.875, "We tested 1,200 sounds on the chip and counted right vs wrong",
        ha="center", fontsize=13, color="#546e7a")

GREENC, REDC = "#c8e6c9", "#ffcdd2"
cw, cwh = 0.155, 0.175
gx, gy = 0.135, 0.29   # bottom-left of the 2x2 grid
# cells[r][c]: r0=top=actual mosquito, r1=bottom=actual not; c0=pred not, c1=pred mosquito
cells = [
    [("58", "missed", REDC), ("558", "correct", GREENC)],
    [("541", "correct", GREENC), ("43", "false\nalarm", REDC)],
]
for r in range(2):
    for c in range(2):
        num, note, col = cells[r][c]
        x = gx + c * cw
        y = gy + (1 - r) * cwh
        ax.add_patch(FancyBboxPatch((x, y), cw*0.94, cwh*0.9, boxstyle="round,pad=0.005,rounding_size=0.012",
                                    fc=col, ec="#90a4ae", lw=1.2))
        ax.text(x + cw*0.47, y + cwh*0.56, num, ha="center", fontsize=21, fontweight="bold", color=DARK)
        ax.text(x + cw*0.47, y + cwh*0.22, note, ha="center", fontsize=9.5, color="#5d4037")
# column headers (predicted)
ax.text(0.5*(gx+gx+2*cw), 0.685, "the model's answer", ha="center", fontsize=11, color=BLUE, fontweight="bold")
ax.text(gx + cw*0.47, 0.655, "“not mosquito”", ha="center", fontsize=10.5, color="#455a64")
ax.text(gx + cw + cw*0.47, 0.655, "“mosquito”", ha="center", fontsize=10.5, color="#455a64")
# row headers (actual)
ax.text(gx - 0.048, gy + cwh + cwh*0.45, "really\nMOSQUITO", ha="center", va="center", fontsize=10, color=GREEN, fontweight="bold")
ax.text(gx - 0.048, gy + cwh*0.45, "really\nNOT", ha="center", va="center", fontsize=10, color=BLUE, fontweight="bold")
ax.text(0.045, 0.455, "the truth", ha="center", va="center", fontsize=11, color=DARK, fontweight="bold", rotation=90)

# ---- right: the four scores ----
rx = 0.55
rows = [
    ("Accuracy", "everything it got right", "(558 + 541) / 1200", "91.6%", GREEN),
    ("Catch rate  (recall)", "of real mosquitoes, how many caught", "558 / 616", "91%", BLUE),
    ("Few false alarms  (precision)", "when it says “mosquito”, how often right", "558 / 601", "93%", BLUE),
    ("F1  (balance of the two)", "one number so neither can be faked", "combines the two above", "0.92", ORANGE),
]
ax.text(rx, 0.815, "The four scores", fontsize=15, fontweight="bold", color=DARK)
for i, (name, desc, formula, val, col) in enumerate(rows):
    y = 0.635 - i*0.16
    ax.add_patch(FancyBboxPatch((rx, y-0.045), 0.42, 0.145, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=LIGHT, ec=col, lw=1.6))
    ax.text(rx+0.018, y+0.062, name, fontsize=12.5, fontweight="bold", color=col)
    ax.text(rx+0.018, y+0.024, desc, fontsize=9.3, color="#546e7a")
    ax.text(rx+0.018, y-0.020, formula, fontsize=9.5, color="#455a64", family="monospace")
    ax.text(rx+0.405, y+0.020, val, fontsize=18, fontweight="bold", color=col, ha="right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(OUT / "slide_metrics.png"); plt.close()
print("slide_metrics.png")


# ============================================================ SLIDE: what the FFT does
fig = plt.figure(figsize=(13.5, 6.0)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.text(0.5, 0.94, "Why we analyse frequency (the “FFT” step)", ha="center",
        fontsize=22, fontweight="bold", color=DARK)
ax.text(0.5, 0.875, "A mosquito is defined by its pitch — so we let the model see pitch directly",
        ha="center", fontsize=13, color="#546e7a")

# left: raw waveform (messy)
axw = fig.add_axes([0.05, 0.42, 0.37, 0.34]); axw.set_facecolor("#fafafa")
tt = np.linspace(0, 1, 800)
raw = (np.sin(2*np.pi*12*tt) + 0.5*np.sin(2*np.pi*29*tt) + 0.3*np.sin(2*np.pi*47*tt))
axw.plot(tt, raw, color="#90a4ae", lw=1.0)
axw.set_title("Raw sound wave", fontsize=13, color=DARK, fontweight="bold")
axw.set_xticks([]); axw.set_yticks([])
for s in axw.spines.values():
    s.set_edgecolor("#cfd8dc")
ax.text(0.235, 0.36, "just a wiggly line — hard to read", ha="center",
        fontsize=10.5, color="#78909c", style="italic")

# arrow
ax.annotate("", xy=(0.55, 0.585), xytext=(0.44, 0.585),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=26, color=BLUE, lw=3))
ax.text(0.495, 0.635, "FFT", ha="center", fontsize=13, fontweight="bold", color=BLUE)

# right: frequency bars (clear peak)
axf = fig.add_axes([0.58, 0.42, 0.37, 0.34]); axf.set_facecolor("#fafafa")
freqs = np.arange(0, 12)
mags = np.array([0.2,0.3,0.5,1.0,3.0,5.5,3.2,1.2,0.6,0.4,0.3,0.2])
cols = ["#cfd8dc"]*len(freqs)
cols[5] = GREEN; cols[4] = GREEN; cols[6] = GREEN
axf.bar(freqs, mags, color=cols, width=0.75)
axf.set_title("Broken into pitches (frequencies)", fontsize=13, color=DARK, fontweight="bold")
axf.annotate("mosquito\nband", xy=(5, 5.5), xytext=(8.5, 4.6),
             fontsize=10, color=GREEN, ha="center", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
axf.set_xticks([]); axf.set_yticks([])
for s in axf.spines.values():
    s.set_edgecolor("#cfd8dc")
ax.text(0.765, 0.36, "one clear peak = the mosquito’s pitch", ha="center",
        fontsize=10.5, color=GREEN, style="italic")

# bottom takeaways
ax.text(0.5, 0.25,
        "Mosquitoes hum at a specific pitch (~400–800 Hz); flies and background do not.",
        ha="center", fontsize=13, color=DARK)
ax.text(0.5, 0.16,
        "Our “LearnableFFT” does this split on-chip — and fine-tunes which pitches to watch.",
        ha="center", fontsize=13, color=DARK)
ax.text(0.5, 0.06,
        "That one idea took the model from 84% to 91% — at no extra size cost.",
        ha="center", fontsize=13.5, color=ORANGE, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(OUT / "slide_fft.png"); plt.close()
print("slide_fft.png")


# ============================================================ SLIDE 4: improvement journey
imp = load("exp_audio_detector_1d_improved")
order = ["baseline (8,16,24)", "bigger (16,32,64)", "LearnableFFT"]
names = ["Baseline\n(tiny 1D)", "Bigger\nmodel", "+ Learnable FFT\n(frequency)"]
accs = [imp[o]["acc"] for o in order]
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_title("Getting to 90%: the frequency front-end did it", loc="left")
x = np.arange(len(names))
bars = ax.bar(x, accs, color=[GRAY, GRAY, GREEN], width=0.55)
barlabels(ax, bars, dy=0.008)
ax.plot(x, accs, "o-", color=DARK, lw=2, ms=8, alpha=0.5)
ax.axhline(0.9, ls="--", c=ORANGE, lw=1.5); ax.text(2.35, 0.905, "90% goal", color=ORANGE, fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.75, 0.96); ax.set_ylabel("accuracy")
for i, o in enumerate(order):
    ax.text(i, 0.765, f"{imp[o]['params']:,} params", ha="center", fontsize=10, color="#607d8b")
fig.tight_layout()
fig.savefig(OUT / "slide4_improvement.png"); plt.close()
print("slide4_improvement.png")


# ============================================================ SLIDE 5: what's solved vs hard
fig, ax = plt.subplots(figsize=(11.5, 5.4))
ax.set_title("Detection is solved. Fine species ID is data-limited.", loc="left")
labels = ["Is it a\nmosquito?", "Which genus?\n(Aedes/Culex/…)", "Exact species\n(our result)", "Exact species\n(lab claim)"]
vals = [0.92, 0.90, 0.73, 0.96]
cols = [GREEN, GREEN, ORANGE, GRAY]
bars = ax.bar(labels, vals, color=cols, width=0.62); barlabels(ax, bars)
ax.axhline(0.9, ls="--", c=GRAY, lw=1.0)
ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy")
ax.text(3, 0.845, "inflated by\ntest leakage", ha="center", fontsize=9.5, color=RED)
ax.annotate("same-species wingbeats\noverlap — needs temperature /\ntime metadata to go higher",
            xy=(2, 0.73), xytext=(1.4, 0.40), fontsize=10.5, color=DARK,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=ORANGE, lw=1.2))
fig.tight_layout()
fig.savefig(OUT / "slide5_ceiling.png"); plt.close()
print("slide5_ceiling.png")


# ============================================================ SLIDE 6: size comparison
fig, ax = plt.subplots(figsize=(11.5, 5.0))
ax.set_title("How small is our model?", loc="left")
models = ["Our model\n(91 KB)", "MobileNet\n(phone AI)", "ResNet-18\n(typical vision)"]
kb = [91, 14000, 45000]  # KB (fp32)
colors = [GREEN, GRAY, GRAY]
y = np.arange(len(models))
bars = ax.barh(y, kb, color=colors, height=0.55)
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(models)
ax.set_xlabel("size (KB, log scale)")
ax.set_xlim(50, 120000)
ax.invert_yaxis()
for b, v in zip(bars, kb):
    lab = f"{v/1000:.0f} MB" if v >= 1000 else f"{v} KB"
    ax.text(v*1.18, b.get_y()+b.get_height()/2, lab, va="center", fontsize=12, fontweight="bold", color=DARK)
ax.text(0.98, 0.08, "~150x smaller than a phone-AI model (MobileNet)", transform=ax.transAxes,
        ha="right", fontsize=12, color=GREEN, style="italic", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "slide6_size.png"); plt.close()
print("slide6_size.png")

print(f"\n== {len(list(OUT.glob('*.png')))} slides in {OUT}")
