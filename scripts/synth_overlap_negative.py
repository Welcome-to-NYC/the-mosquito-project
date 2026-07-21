"""Synthetic 'overlapping hard negative' — a stand-in for chironomid midges.

No public dataset has chironomid wingbeat recordings whose fundamental (~500 Hz)
overlaps mosquitoes. So we synthesise a controlled negative that SHARES the
fundamental with mosquitoes but differs in HARMONIC structure and temporal
envelope — exactly the difference the optical/lidar literature (Brydegaard;
Gonzalez-Perez 2022) says separates same-wingbeat-frequency insects.

Purpose (honest):
  * NOT a claim about real chironomids — it encodes the *assumption* that they
    differ in overtones. It is a controlled demonstration.
  * Shows two things:
      (A) a FREQUENCY-ONLY detector (fundamental peak) CANNOT separate them —
          both sit at ~450-550 Hz. This is the trap acoustic mosquito traps hit.
      (B) our spectrum/harmonic-aware model CAN, IF the overtone difference is
          real. We sweep the overtone gap to see how different they must be.

Run:
    python scripts/synth_overlap_negative.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 8000
DUR = 0.256                       # 2048 samples
N = int(SR * DUR)
OUT = ROOT / "docs" / "synth_overlap.png"


def wingbeat(f0, harm_amps, am_depth, am_rate, jitter, seed):
    """One synthetic wingbeat: fundamental + harmonics, freq jitter, amplitude
    modulation (wing kinematics), pink-ish noise. harm_amps = list of relative
    amplitudes for [f0, 2f0, 3f0, 4f0, 5f0]."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SR
    # slow frequency drift + per-sample jitter
    f = f0 * (1 + jitter * (rng.standard_normal() * 0.5))
    sig = np.zeros(N)
    phase0 = rng.uniform(0, 2*np.pi)
    for k, a in enumerate(harm_amps, start=1):
        sig += a * np.sin(2*np.pi*(k*f)*t + k*phase0)
    # amplitude modulation (wingstroke envelope)
    env = 1 + am_depth * np.sin(2*np.pi*am_rate*t + rng.uniform(0, 2*np.pi))
    sig *= env
    # onset/offset taper + noise
    sig *= np.hanning(N)
    sig += 0.04 * rng.standard_normal(N)
    return (sig / (np.abs(sig).max() + 1e-9)).astype(np.float32)


def make_set(kind, n, gap, seed0):
    """kind='mosquito' or 'midge'. gap in [0,1] scales the overtone difference."""
    X = []
    for i in range(n):
        s = seed0 + i
        rng = np.random.default_rng(s)
        if kind == "mosquito":
            f0 = rng.uniform(430, 560)                 # Aedes/Culex band
            # mosquito: strong 2nd/3rd harmonics (rich overtones)
            base = np.array([1.0, 0.62, 0.40, 0.22, 0.10])
        else:  # midge: OVERLAPPING f0, but different overtone rolloff
            f0 = rng.uniform(450, 560)                 # overlaps mosquito!
            # midge synthetic: overtones weaker by `gap` (steeper rolloff) +
            # a slightly different 2nd/3rd balance
            base = np.array([1.0, 0.62-0.45*gap, 0.40-0.30*gap, 0.22-0.10*gap, 0.10])
            base = np.clip(base, 0.02, None)
        base = base * rng.uniform(0.9, 1.1, size=5)     # per-sample variation
        am = rng.uniform(0.15, 0.45)
        amr = rng.uniform(8, 22)
        X.append(wingbeat(f0, base, am, amr, jitter=0.03, seed=s))
    return np.array(X, np.float32)


def logspec(x, n_fft=256, hop=128):
    win = np.hanning(n_fft)
    nfr = 1 + (len(x)-n_fft)//hop
    fr = np.stack([x[i*hop:i*hop+n_fft]*win for i in range(nfr)])
    S = np.abs(np.fft.rfft(fr, axis=1))**2
    lp = np.log(S + 1e-8).mean(0)           # average spectrum over time (F bins)
    return (lp - lp.mean())/(lp.std()+1e-8)


def fundamental(x):
    Y = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    fr = np.fft.rfftfreq(len(x), 1/SR)
    b = (fr>=200)&(fr<=1200)
    return fr[b][np.argmax(Y[b])]


def acc_from(feats_m, feats_c):
    """Simple nearest-centroid / logistic-ish separability via a 1-NN split."""
    from numpy.linalg import norm
    X = np.vstack([feats_m, feats_c]); y = np.array([0]*len(feats_m)+[1]*len(feats_c))
    rng = np.random.default_rng(0); idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]; cut = int(0.7*len(X))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    c0 = Xtr[ytr==0].mean(0); c1 = Xtr[ytr==1].mean(0)
    pred = np.array([0 if norm(x-c0)<norm(x-c1) else 1 for x in Xte])
    return float((pred==yte).mean())


def main():
    print("== synthesising mosquito vs midge-like (overlapping fundamental)")
    for gap in [0.0, 0.3, 0.6, 0.9]:
        M = make_set("mosquito", 300, gap, 0)
        C = make_set("midge", 300, gap, 100000)
        # feature A: fundamental frequency only
        fmA = np.array([[fundamental(x)] for x in M]); fcA = np.array([[fundamental(x)] for x in C])
        accA = acc_from(fmA, fcA)
        # feature B: full log spectrum (harmonic-aware, like our FFT+CNN sees)
        fmB = np.array([logspec(x) for x in M]); fcB = np.array([logspec(x) for x in C])
        accB = acc_from(fmB, fcB)
        f0m = fmA.mean(); f0c = fcA.mean()
        print(f"  overtone gap={gap:.1f} | f0 mosq {f0m:.0f}Hz vs midge {f0c:.0f}Hz (overlap) "
              f"| FREQ-ONLY acc {accA:.2f} | HARMONIC(spectrum) acc {accB:.2f}")

    # plot mean spectra at gap=0.6 to show the harmonic difference
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        M = make_set("mosquito", 200, 0.6, 0); C = make_set("midge", 200, 0.6, 100000)
        fr = np.fft.rfftfreq(256, 1/SR)
        Sm = np.mean([np.log(np.abs(np.fft.rfft(x[:256]*np.hanning(256)))**2+1e-8) for x in M],0)
        Sc = np.mean([np.log(np.abs(np.fft.rfft(x[:256]*np.hanning(256)))**2+1e-8) for x in C],0)
        plt.figure(figsize=(9,4.2))
        plt.plot(fr, Sm, color="#dc2626", lw=2, label="mosquito (synthetic)")
        plt.plot(fr, Sc, color="#2563eb", lw=2, label="midge-like negative (synthetic)")
        plt.axvspan(430,560, color="#16a34a", alpha=.08)
        plt.text(495, Sm.max(), "shared\nfundamental\n~500 Hz", ha="center", fontsize=9, color="#15803d")
        plt.xlim(0,3000); plt.xlabel("frequency (Hz)"); plt.ylabel("log power")
        plt.title("Same fundamental, different harmonics — why frequency-alone fails")
        plt.legend(); plt.tight_layout(); plt.savefig(OUT, dpi=140)
        print(f"\n-- saved {OUT}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
