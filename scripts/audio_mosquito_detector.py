"""Honest audio-only mosquito detector: "mosquito vs not-mosquito".

Scope (per current team decision): binary detection only, audio is fine.

The trap to avoid is the source shortcut — if every mosquito comes from
HumBugDB and every negative comes from a different rig, the model learns "which
recording device" not "is there a wingbeat". We defuse it by putting HumBugDB's
OWN background/ambient audio (same microphone, same sessions as its mosquitoes)
into the negative class alongside real fly wingbeats:

    class 1  mosquito      <- HumBugDB mosquito recordings
    class 0  not_mosquito  <- HumBugDB background/ambient  (same rig!)
                            +  InsectSound1000 Diptera      (real fly wingbeats)

Because mosquito AND non-mosquito both appear from the HumBugDB rig, "device
identity" is no longer predictive of the label — the model must use the signal.
Splits are recording-level. We report overall accuracy AND recall on each
negative sub-source separately, so a rig shortcut would show up as one
sub-source being trivially rejected while another fails.

Run:
    python scripts/audio_mosquito_detector.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_cross_modality import SpecCNN, spec_augment, _eval  # noqa: E402
from src.data import spectro  # noqa: E402
from src.data.metadata import load_humbugdb  # noqa: E402
from src.evaluation.metrics import evaluate, summarize  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_audio_mosquito_detector"
IS1000 = ROOT / "data" / "raw" / "insectsound1000"


def _load_wav(path, sr=spectro.SR):
    import librosa
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception:
        return None


def _segs(y, max_seg):
    return [spectro.wav_segment_to_spec(s)
            for s in spectro.segment_waveform(y, max_segments=max_seg)]


def build(max_mosq_rec, max_bg_rec, seg_per_rec, max_fly_files, seed):
    rng = np.random.default_rng(seed)
    df = load_humbugdb()
    X, y, groups, srcs = [], [], [], []

    def add_humbug(label_name, bin_label, max_rec, src):
        sub = df[df["label"] == label_name]
        recs = list(sub["recording_id"].unique()); rng.shuffle(recs)
        n = 0
        for rid in recs[:max_rec]:
            got = 0
            for _, r in sub[sub["recording_id"] == rid].iterrows():
                sig = _load_wav(r["path"])
                if sig is None:
                    continue
                for s in _segs(sig, seg_per_rec - got):
                    X.append(s); y.append(bin_label); groups.append(rid); srcs.append(src); got += 1
                if got >= seg_per_rec:
                    break
            n += got
        print(f"   {src}: {n} segs from {min(max_rec,len(recs))} recordings")

    print("== HumBugDB mosquito (positive)")
    add_humbug("mosquito", 1, max_mosq_rec, "humbug_mosq")
    print("== HumBugDB background (negative, same rig)")
    add_humbug("background", 0, max_bg_rec, "humbug_bg")

    print("== InsectSound flies (negative, real wingbeat)")
    wavs = sorted(IS1000.glob("*.wav")); rng.shuffle(wavs)
    n = 0
    for w in wavs[:max_fly_files]:
        sig = _load_wav(w)
        if sig is None:
            continue
        rid = "is1000:" + w.name.rsplit("_s", 1)[0]
        for s in _segs(sig, 4):
            X.append(s); y.append(0); groups.append(rid); srcs.append("insectsound_fly"); n += 1
    print(f"   insectsound_fly: {n} segs")

    return (np.asarray(X, np.float32), np.asarray(y, np.int64),
            np.asarray(groups, object), np.asarray(srcs, object))


def group_split(groups, seed, val=0.15, test=0.15):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist()))); rng.shuffle(uniq)
    nt = int(len(uniq) * test); nv = int(len(uniq) * val)
    tg, vg = set(uniq[:nt]), set(uniq[nt:nt+nv])
    te = np.array([g in tg for g in groups]); va = np.array([g in vg for g in groups])
    tr = ~(te | va)
    return tr, va, te


def train(Xtr, ytr, Xva, yva, channels, epochs, bs, lr, device, seed=42):
    set_seed(seed)
    model = SpecCNN(n_classes=2, channels=channels, dropout=0.4).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnt = np.bincount(ytr, minlength=2); w = (1.0/cnt)[ytr]
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ytr), True)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr)),
        batch_size=bs, sampler=sampler, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best = -1, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = spec_augment(xb.clone())
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        vp, _ = _eval(model, Xva, yva, device)
        f1 = evaluate(yva, vp, classes=["not_mosq", "mosq"])["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1; best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return model, npar, best_f1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")
    t0 = time.time()
    X, y, groups, srcs = build(400, 250, 25, 636, args.seed)
    print(f"   total {len(X)} segs (mosq {int((y==1).sum())} / not {int((y==0).sum())}), {time.time()-t0:.0f}s")

    tr, va, te = group_split(groups, args.seed)
    classes = ["not_mosquito", "mosquito"]

    # size sweep for deployability
    rows = []
    for label, ch in [("small (8,16,24)", (8,16,24)), ("mid (16,32,64)", (16,32,64))]:
        model, npar, bf1 = train(X[tr], y[tr], X[va], y[va], ch, args.epochs, args.batch_size, args.lr, device)
        tp, tprob = _eval(model, X[te], y[te], device)
        m = evaluate(y[te], tp, y_score=tprob, classes=classes)
        # per negative sub-source recall (of true negatives, how many correctly called not_mosquito)
        sub = {}
        for s in ["humbug_bg", "insectsound_fly"]:
            mask = (srcs[te] == s) & (y[te] == 0)
            if mask.sum():
                sub[s] = float((tp[mask] == 0).mean())
        mosq_recall = float((tp[(y[te]==1)] == 1).mean())
        rows.append({"config": label, "params": npar, "fp32_kb": round(npar*4/1024,1),
                     "acc": m["accuracy"], "macro_f1": m["macro_f1"],
                     "mosq_recall": mosq_recall, "neg_by_source": sub})
        print(f"\n### {label}: params {npar:,}")
        print(summarize(m))
        print(f"   mosquito recall           : {mosq_recall:.4f}")
        print(f"   not-mosq recall by source : "
              + "  ".join(f"{k}={v:.3f}" for k, v in sub.items()))

    print("\n" + "="*66)
    print("AUDIO MOSQUITO DETECTOR (honest, recording-level split)")
    print("="*66)
    print(f"{'config':<18}{'params':>8}{'KB':>7}{'acc':>8}{'f1':>8}{'mosq_rec':>10}")
    for r in rows:
        print(f"{r['config']:<18}{r['params']:>8}{r['fp32_kb']:>7.1f}{r['acc']:>8.4f}{r['macro_f1']:>8.4f}{r['mosq_recall']:>10.4f}")
    print("\nnegative rejection by sub-source (should be high for BOTH = no rig shortcut):")
    for r in rows:
        print(f"  {r['config']:<18} " + "  ".join(f"{k}={v:.3f}" for k,v in r['neg_by_source'].items()))

    (OUT/"results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n-- saved {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
