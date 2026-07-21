"""Build the cross-modality experiment datasets.

The honest question: **does a mosquito-vs-fly classifier trained ONLY on
audio transfer to OPTICAL wingbeat sensors?**

To answer it without the model cheating on dataset identity, we deliberately
use different sources for each class in train vs test:

  TRAIN / VAL / TEST-in-modality  (all AUDIO):
    class 0 mosquito       <- HumBugDB mosquito recordings
    class 1 non_mosquito   <- InsectSound1000 Diptera (hoverfly/gall-midge/fungus-gnat)

  CROSS-MODALITY test A  (OPTICAL, the headline):
    UCR InsectWingbeat  ->  mosquito species vs Fruit/House flies
    (mosq and fly are processed identically here => no within-test artifact)

  CROSS-MODALITY test B  (OPTICAL, mosquito recall only):
    Wingbeats  ->  all mosquito, processed through OUR STFT (matches train)

Everything is mapped to a common (F_BINS, T_FRAMES) log-power spectrogram via
src/data/spectro.py so audio and optical live in one representation.

Splits are RECORDING-LEVEL (group by recording id) so no recording leaks
across train/val/test — otherwise the in-modality number is inflated and the
modality gap is understated.

Run:
    python scripts/build_cross_modality.py
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import spectro  # noqa: E402
from src.data.metadata import load_humbugdb  # noqa: E402
from src.data.ucr_loader import load_ts, MOSQUITO_LABELS, FLY_LABELS  # noqa: E402

OUT = ROOT / "data" / "processed"
IS1000 = ROOT / "data" / "raw" / "insectsound1000"
WINGBEATS = ROOT / "data" / "raw" / "wingbeats"
UCR_TEST = ROOT / "data" / "raw" / "insect_wingbeat_ucr" / "InsectWingbeat" / "InsectWingbeat_eq_TEST.ts"

RNG = np.random.default_rng(42)


def _load_wav(path: Path, sr: int = spectro.SR) -> np.ndarray | None:
    import librosa
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"   skip {path.name}: {e}")
        return None


def _wav_to_segments(y: np.ndarray, max_seg: int) -> list[np.ndarray]:
    segs = spectro.segment_waveform(y, max_segments=max_seg)
    return [spectro.wav_segment_to_spec(s) for s in segs]


# ---------------------------------------------------------------- audio: HumBugDB mosquito
def build_humbug_mosquito(max_recordings: int, max_seg_per_rec: int):
    print("== HumBugDB mosquito")
    df = load_humbugdb()
    df = df[df["label"] == "mosquito"].reset_index(drop=True)
    # group by recording_id
    rec_ids = df["recording_id"].unique().tolist()
    RNG.shuffle(rec_ids)
    rec_ids = rec_ids[:max_recordings]
    specs, groups = [], []
    t0 = time.time()
    for i, rid in enumerate(rec_ids):
        rows = df[df["recording_id"] == rid]
        got = 0
        for _, r in rows.iterrows():
            y = _load_wav(Path(r["path"]))
            if y is None:
                continue
            for s in _wav_to_segments(y, max_seg_per_rec - got):
                specs.append(s); groups.append(rid); got += 1
            if got >= max_seg_per_rec:
                break
        if (i + 1) % 25 == 0:
            print(f"   {i+1}/{len(rec_ids)} recs, {len(specs)} segs ({time.time()-t0:.0f}s)")
    return np.asarray(specs, dtype=np.float32), np.asarray(groups, dtype=object)


# ---------------------------------------------------------------- audio: InsectSound Diptera
def build_insectsound_diptera(max_seg_per_file: int):
    print("== InsectSound1000 Diptera")
    wavs = sorted(IS1000.glob("*.wav"))
    print(f"   {len(wavs)} wav files on disk")
    specs, groups = [], []
    t0 = time.time()
    for i, w in enumerate(wavs):
        # recording id = filename minus segment index: <date>_<species>_<rec>_s<seg>_ch0
        m = re.match(r"(.+)_s\d+_ch\d+\.wav$", w.name)
        rid = "is1000:" + (m.group(1) if m else w.stem)
        y = _load_wav(w)
        if y is None:
            continue
        for s in _wav_to_segments(y, max_seg_per_file):
            specs.append(s); groups.append(rid)
        if (i + 1) % 300 == 0:
            print(f"   {i+1}/{len(wavs)} files, {len(specs)} segs ({time.time()-t0:.0f}s)")
    return np.asarray(specs, dtype=np.float32), np.asarray(groups, dtype=object)


# ---------------------------------------------------------------- optical: UCR
def build_ucr_optical():
    print("== UCR optical (mosquito vs fly)")
    ts = load_ts(UCR_TEST)
    X, y_raw, classes = ts.X, ts.y, ts.classes
    specs, labels = [], []
    for i in range(len(X)):
        name = classes[y_raw[i]].lower() if 0 <= y_raw[i] < len(classes) else ""
        if name in MOSQUITO_LABELS:
            lab = 0
        elif name in FLY_LABELS:
            lab = 1
        else:
            continue
        specs.append(spectro.ucr_instance_to_spec(X[i]))
        labels.append(lab)
    return np.asarray(specs, dtype=np.float32), np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------- optical: Wingbeats mosquito
def build_wingbeats_optical(max_files: int):
    print("== Wingbeats optical (mosquito recall)")
    wavs = list(WINGBEATS.rglob("*.wav"))
    RNG.shuffle(wavs)
    wavs = wavs[:max_files]
    specs = []
    t0 = time.time()
    for i, w in enumerate(wavs):
        y = _load_wav(w)
        if y is None:
            continue
        # each Wingbeats clip is one 0.625 s event -> one segment
        specs.append(spectro.wav_segment_to_spec(y[: spectro.SEG_SAMPLES]))
        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{len(wavs)} ({time.time()-t0:.0f}s)")
    return np.asarray(specs, dtype=np.float32), np.zeros(len(specs), dtype=np.int64)


def _group_split(groups: np.ndarray, val_frac=0.15, test_frac=0.15):
    """Split unique groups into train/val/test index masks."""
    uniq = np.array(sorted(set(groups.tolist())))
    RNG.shuffle(uniq)
    n = len(uniq)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_g = set(uniq[:n_test].tolist())
    val_g = set(uniq[n_test:n_test + n_val].tolist())
    train_g = set(uniq[n_test + n_val:].tolist())
    tr = np.array([g in train_g for g in groups])
    va = np.array([g in val_g for g in groups])
    te = np.array([g in test_g for g in groups])
    return tr, va, te


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--humbug-recs", type=int, default=400)
    p.add_argument("--humbug-seg-per-rec", type=int, default=25)
    p.add_argument("--is-seg-per-file", type=int, default=4)
    p.add_argument("--wingbeats-files", type=int, default=3000)
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    # --- audio positives + negatives ---
    mosq_X, mosq_g = build_humbug_mosquito(args.humbug_recs, args.humbug_seg_per_rec)
    fly_X, fly_g = build_insectsound_diptera(args.is_seg_per_file)
    print(f"   audio mosquito segs: {len(mosq_X)}, audio fly segs: {len(fly_X)}")

    # recording-level split within each class, then merge
    m_tr, m_va, m_te = _group_split(mosq_g)
    f_tr, f_va, f_te = _group_split(fly_g)

    def _pack(mask_m, mask_f):
        X = np.concatenate([mosq_X[mask_m], fly_X[mask_f]], axis=0)
        y = np.concatenate([np.zeros(mask_m.sum(), dtype=np.int64),
                            np.ones(mask_f.sum(), dtype=np.int64)])
        idx = RNG.permutation(len(X))
        return X[idx], y[idx]

    Xtr, ytr = _pack(m_tr, f_tr)
    Xva, yva = _pack(m_va, f_va)
    Xte, yte = _pack(m_te, f_te)
    classes = np.array(["mosquito", "non_mosquito"], dtype=object)

    for split, (X, y) in {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}.items():
        path = OUT / f"xmod_audio_{split}.npz"
        np.savez_compressed(path, X=X, y=y, classes=classes)
        print(f"   saved {path.name}: X{X.shape} mosq={int((y==0).sum())} fly={int((y==1).sum())}")

    # --- optical cross-modality tests ---
    ucr_X, ucr_y = build_ucr_optical()
    np.savez_compressed(OUT / "xmod_optical_ucr.npz", X=ucr_X, y=ucr_y, classes=classes)
    print(f"   saved xmod_optical_ucr.npz: X{ucr_X.shape} mosq={int((ucr_y==0).sum())} fly={int((ucr_y==1).sum())}")

    wb_X, wb_y = build_wingbeats_optical(args.wingbeats_files)
    np.savez_compressed(OUT / "xmod_optical_wingbeats.npz", X=wb_X, y=wb_y, classes=classes)
    print(f"   saved xmod_optical_wingbeats.npz: X{wb_X.shape} (all mosquito)")

    print("== done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
