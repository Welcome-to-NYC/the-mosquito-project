"""1D raw-waveform mosquito detector for ESP32 deployment (binary, audio-only).

Same honest task/data as scripts/audio_mosquito_detector.py (the 2D version),
but on the raw 1D waveform at the exact spec the chip already runs:
5 kHz, 1024-sample windows, per-window z-norm, CNN1D(8,16,24). This lets us
reuse the existing on-chip C++ inference (firmware/wingbeat_stream) unchanged
except for the class count.

    class 1 mosquito     <- HumBugDB mosquito
    class 0 not_mosquito <- HumBugDB background (same rig) + InsectSound flies

Recording-level split; per-negative-source rejection reported (rig-shortcut
check). Saves best.pt + a balanced streaming test set for eval_on_chip.

Run:
    python scripts/train_audio_detector_1d.py --epochs 30
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

from src.models.cnn_1d import CNN1D  # noqa: E402
from src.data.metadata import load_humbugdb  # noqa: E402
from src.evaluation.metrics import evaluate, summarize  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_audio_detector_1d"
IS1000 = ROOT / "data" / "raw" / "insectsound1000"
SR = 5000
WIN = 1024
HOP = 512
EPS = 1e-8

KWARGS = dict(n_classes=2, in_channels=1, channels=[8, 16, 24],
              kernel_sizes=[7, 5, 3], fc_hidden=16, dropout=0.2)


def _load(path):
    import librosa
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        return y.astype(np.float32)
    except Exception:
        return None


def _windows(sig, max_w):
    out = []
    if len(sig) < WIN:
        sig = np.pad(sig, (0, WIN - len(sig)))
    for i in range(0, len(sig) - WIN + 1, HOP):
        w = sig[i:i+WIN]
        w = (w - w.mean()) / (w.std() + EPS)
        out.append(w.astype(np.float32))
        if len(out) >= max_w:
            break
    return out


def build(max_mosq_rec, max_bg_rec, win_per_rec, max_fly_files, seed):
    rng = np.random.default_rng(seed)
    df = load_humbugdb()
    X, y, groups, srcs = [], [], [], []

    def add_hb(label, binlab, max_rec, src):
        sub = df[df["label"] == label]
        recs = list(sub["recording_id"].unique()); rng.shuffle(recs)
        n = 0
        for rid in recs[:max_rec]:
            got = 0
            for _, r in sub[sub["recording_id"] == rid].iterrows():
                sig = _load(r["path"])
                if sig is None:
                    continue
                for w in _windows(sig, win_per_rec - got):
                    X.append(w); y.append(binlab); groups.append(rid); srcs.append(src); got += 1
                if got >= win_per_rec:
                    break
            n += got
        print(f"   {src}: {n} windows")

    print("== HumBugDB mosquito (pos)")
    add_hb("mosquito", 1, max_mosq_rec, "humbug_mosq")
    print("== HumBugDB background (neg, same rig)")
    add_hb("background", 0, max_bg_rec, "humbug_bg")

    print("== InsectSound flies (neg)")
    wavs = sorted(IS1000.glob("*.wav")); rng.shuffle(wavs)
    n = 0
    for w in wavs[:max_fly_files]:
        sig = _load(w)
        if sig is None:
            continue
        rid = "is1000:" + w.name.rsplit("_s", 1)[0]
        for win in _windows(sig, 4):
            X.append(win); y.append(0); groups.append(rid); srcs.append("insectsound_fly"); n += 1
    print(f"   insectsound_fly: {n} windows")
    return (np.asarray(X, np.float32), np.asarray(y, np.int64),
            np.asarray(groups, object), np.asarray(srcs, object))


def gsplit(groups, seed, val=0.15, test=0.15):
    rng = np.random.default_rng(seed)
    u = np.array(sorted(set(groups.tolist()))); rng.shuffle(u)
    nt = int(len(u)*test); nv = int(len(u)*val)
    tg, vg = set(u[:nt]), set(u[nt:nt+nv])
    te = np.array([g in tg for g in groups]); va = np.array([g in vg for g in groups])
    return ~(te | va), va, te


@torch.no_grad()
def _eval(model, X, device, bs=256):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i+bs], device=device).unsqueeze(1)
        out.append(torch.softmax(model(xb), -1).cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")
    t0 = time.time()
    X, y, groups, srcs = build(400, 250, 25, 636, args.seed)
    print(f"   total {len(X)} (mosq {int((y==1).sum())} / not {int((y==0).sum())}), {time.time()-t0:.0f}s")

    tr, va, te = gsplit(groups, args.seed)
    classes = ["not_mosquito", "mosquito"]
    model = CNN1D(**KWARGS).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   params {npar:,}")

    cnt = np.bincount(y[tr], minlength=2); w = (1.0/cnt)[y[tr]]
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), int(tr.sum()), True)
    ds = torch.utils.data.TensorDataset(torch.as_tensor(X[tr]).unsqueeze(1), torch.as_tensor(y[tr]))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best = -1, None
    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        vp = _eval(model, X[va], device).argmax(-1)
        f1 = evaluate(y[va], vp, classes=classes)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1; best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs:
            print(f"   epoch {ep}: val_f1 {f1:.4f}")
    model.load_state_dict(best)

    prob = _eval(model, X[te], device); pred = prob.argmax(-1)
    m = evaluate(y[te], pred, y_score=prob, classes=classes)
    print("\n---- 1D audio detector, honest split ----")
    print(summarize(m))
    sub = {}
    for s in ["humbug_bg", "insectsound_fly"]:
        mask = (srcs[te] == s) & (y[te] == 0)
        if mask.sum():
            sub[s] = float((pred[mask] == 0).mean())
    mosq_recall = float((pred[y[te]==1] == 1).mean())
    print(f"   mosquito recall           : {mosq_recall:.4f}")
    print(f"   not-mosq rejection by src : " + "  ".join(f"{k}={v:.3f}" for k,v in sub.items()))

    # save model + balanced streaming test set (2 per class-ish, spread sources)
    torch.save({"model": best, "classes": classes, "kwargs": KWARGS}, OUT / "best.pt")
    # streaming test: take a manageable balanced subset
    rng = np.random.default_rng(0)
    te_idx = np.where(te)[0]
    sel = rng.choice(te_idx, size=min(2000, len(te_idx)), replace=False)
    np.savez_compressed(OUT / "stream_test.npz",
                        X=X[sel].astype(np.float32), y=y[sel].astype(np.int64),
                        source=srcs[sel], classes=np.array(classes, dtype=object))
    result = {"n_params": npar, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
              "mosq_recall": mosq_recall, "neg_by_source": sub,
              "confusion": m["confusion"].tolist()}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n-- saved {OUT/'best.pt'}, stream_test.npz, results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
