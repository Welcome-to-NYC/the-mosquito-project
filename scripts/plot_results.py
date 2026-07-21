"""Summary charts for the mosquito-project experiments (English labels)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def load(name):
    return json.loads((EXP / name / "results.json").read_text())


# ---------------------------------------------------------------- FIGURE 1: audio detector
det = load("exp_audio_mosquito_detector")  # list of {config, params, acc, mosq_recall, neg_by_source}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
fig.suptitle("Honest Audio Mosquito Detector (recording-level split)", fontweight="bold")

configs = [r["config"].split()[0] for r in det]
x = np.arange(len(configs)); w = 0.25
acc = [r["acc"] for r in det]
mrec = [r["mosq_recall"] for r in det]
nrej = [np.mean(list(r["neg_by_source"].values())) for r in det]
ax[0].bar(x - w, acc, w, label="accuracy", color="#2b6cb0")
ax[0].bar(x, mrec, w, label="mosquito recall", color="#38a169")
ax[0].bar(x + w, nrej, w, label="non-mosq rejection", color="#dd6b20")
ax[0].set_xticks(x); ax[0].set_xticklabels(configs); ax[0].set_ylim(0, 1.05)
ax[0].axhline(0.9, ls="--", c="gray", lw=0.8)
ax[0].set_title("Performance"); ax[0].legend(fontsize=8)
for i in range(len(configs)):
    ax[0].text(x[i]-w, acc[i]+.01, f"{acc[i]:.2f}", ha="center", fontsize=7)

# rejection by source
src_names = list(det[0]["neg_by_source"].keys())
for j, r in enumerate(det):
    vals = [r["neg_by_source"][s] for s in src_names]
    ax[1].bar(np.arange(len(src_names)) + j*0.35, vals, 0.35,
              label=r["config"].split()[0])
ax[1].set_xticks(np.arange(len(src_names)) + 0.17)
ax[1].set_xticklabels(["HumBugDB\nbackground\n(same rig)", "InsectSound\nflies"], fontsize=8)
ax[1].set_ylim(0, 1.08); ax[1].axhline(0.9, ls="--", c="gray", lw=0.8)
ax[1].set_title("Rejection by negative source\n(both high => no rig shortcut)")
ax[1].legend(fontsize=8)
for j, r in enumerate(det):
    for k, s in enumerate(src_names):
        ax[1].text(k + j*0.35, r["neg_by_source"][s]+.01, f"{r['neg_by_source'][s]:.2f}",
                   ha="center", fontsize=7)

# model size vs accuracy
params = [r["params"] for r in det]
ax[2].plot(params, acc, "o-", color="#2b6cb0", ms=9)
for i, r in enumerate(det):
    ax[2].annotate(f"{r['fp32_kb']:.0f} KB\n{acc[i]:.3f}", (params[i], acc[i]),
                   textcoords="offset points", xytext=(0, 10), fontsize=8, ha="center")
ax[2].set_xscale("log"); ax[2].set_xlabel("parameters"); ax[2].set_ylabel("accuracy")
ax[2].set_ylim(0.85, 0.95); ax[2].set_title("Size vs accuracy (deployable)")
fig.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT / "fig1_audio_detector.png"); plt.close()
print("saved fig1_audio_detector.png")


# ---------------------------------------------------------------- FIGURE 2: big picture
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
fig.suptitle("Project Findings: what is honestly achievable", fontweight="bold")

# (a) audio size sweep
sweep = load("exp_audio_size_sweep")
sp = [r["params"] for r in sweep]; sa = [r["audio_test_acc"] for r in sweep]
suc = [r["ucr_optical_acc"] for r in sweep]
ax[0].plot(sp, sa, "o-", label="audio test (in-modality)", color="#38a169", ms=8)
ax[0].plot(sp, suc, "s--", label="UCR optical (cross-modality)", color="#e53e3e", ms=8)
ax[0].set_xscale("log"); ax[0].set_xlabel("parameters"); ax[0].set_ylabel("accuracy")
ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8)
ax[0].set_title("Audio task: tiny model suffices\nbut optical transfer fails at any size")

# (b) modality gap
xm = load("exp_cross_modality")
labels = ["audio\n(in-modality)", "UCR optical\n(cross-modality)"]
vals = [xm["audio_test_acc"], xm["ucr_optical_acc"]]
bars = ax[1].bar(labels, vals, color=["#38a169", "#e53e3e"], width=0.6)
ax[1].axhline(0.5, ls=":", c="gray", lw=1); ax[1].text(1.4, 0.5, "chance", fontsize=8, color="gray")
ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("accuracy")
ax[1].set_title(f"Audio -> Optical transfer FAILS\ngap = {xm['modality_gap']:+.2f}")
for b, v in zip(bars, vals):
    ax[1].text(b.get_x()+b.get_width()/2, v+.01, f"{v:.2f}", ha="center", fontweight="bold")

# (c) species ceiling: detection vs species; honest vs literature
labels2 = ["mosquito\ndetection", "species\n(UCR\nhonest)", "species\n(Wingbeats\nhonest)", "species\n(literature\nclaim)"]
vals2 = [0.967, 0.73, 0.774, 0.96]
colors2 = ["#38a169", "#dd6b20", "#dd6b20", "#a0aec0"]
bars = ax[2].bar(labels2, vals2, color=colors2, width=0.65)
ax[2].axhline(0.9, ls="--", c="gray", lw=0.8); ax[2].set_ylim(0, 1.05)
ax[2].set_ylabel("accuracy")
ax[2].set_title("Detection is solved; species hits ~0.75 ceiling\n(literature 0.96 not reproducible honestly)")
for b, v in zip(bars, vals2):
    ax[2].text(b.get_x()+b.get_width()/2, v+.01, f"{v:.2f}", ha="center", fontsize=8)
ax[2].text(3, 0.80, "confound\ninflated", ha="center", fontsize=7, color="#e53e3e")
fig.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(OUT / "fig2_findings.png"); plt.close()
print("saved fig2_findings.png")

print("== done. charts in docs/")
