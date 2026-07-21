"""Wingbeats 6-species: isolate WHY our honest number (0.77) is below published SOTA.

We change two knobs independently — spectrogram resolution and split type — with
the SAME big model, to see which one drives the gap:

  resolution: coarse (N_FFT 256, ~129x40)  vs  fine (N_FFT 512, ~257x78)
  split:      random (leaky)                vs  session-independent (honest)

4 runs. Reading:
  fine/session - coarse/session  = how much resolution alone buys
  */random - */session           = how much leakage inflates
  fine/random                    = closest to the published-SOTA recipe

Run:
    python scripts/wingbeats_4condition.py --per-species 3000 --epochs 25
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

from src.data.metadata import load_wingbeats  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_wingbeats_4condition"
SPECIES = ["ae_aegypti", "ae_albopictus", "an_arabiensis",
           "an_gambiae", "c_pipiens", "c_quinquefasciatus"]
SP2IDX = {s: i for i, s in enumerate(SPECIES)}
SR = 8000
SEG = 5000
EPS = 1e-8

# two resolution settings
RES = {
    "coarse": dict(n_fft=256, hop=128),   # ~129 x 40
    "fine":   dict(n_fft=512, hop=64),    # ~257 x 78
}


def spec(sig, n_fft, hop):
    win = np.hanning(n_fft).astype(np.float32)
    if len(sig) < SEG:
        sig = np.pad(sig, (0, SEG - len(sig)))
    sig = sig[:SEG]
    nfr = 1 + (len(sig) - n_fft) // hop
    frames = np.stack([sig[i*hop:i*hop+n_fft] * win for i in range(nfr)])
    s = np.fft.rfft(frames, axis=1)
    logp = np.log(s.real**2 + s.imag**2 + EPS).T.astype(np.float32)
    mu, sd = logp.mean(), logp.std() + EPS
    return ((logp - mu) / sd).astype(np.float32)


class BigCNN2D(nn.Module):
    def __init__(self, n_classes=6, channels=(32, 64, 128, 128), dropout=0.3):
        super().__init__()
        layers, prev = [], 1
        for c in channels:
            layers += [nn.Conv2d(prev, c, 3, padding=1), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            prev = c
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels[-1], 64); self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc2(self.act(self.drop(self.fc1(x))))


def build(per_species, per_session, n_fft, hop, seed):
    import librosa
    rng = np.random.default_rng(seed)
    df = load_wingbeats()
    X, y, groups = [], [], []
    for sp in SPECIES:
        sub = df[df["species"] == sp]
        sessions = list(sub["recording_id"].unique()); rng.shuffle(sessions)
        got = 0
        for sess in sessions:
            if got >= per_species:
                break
            paths = list(sub[sub["recording_id"] == sess]["path"]); rng.shuffle(paths)
            for p in paths[:per_session]:
                if got >= per_species:
                    break
                try:
                    sig, _ = librosa.load(p, sr=SR, mono=True)
                except Exception:
                    continue
                X.append(spec(sig, n_fft, hop)); y.append(SP2IDX[sp]); groups.append(sess); got += 1
        print(f"   {sp}: {got}", flush=True)
    return np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(groups, object)


def random_split(n, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); cut = int(n*(1-test_frac))
    return idx[:cut], idx[cut:]


def session_split(groups, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist()))); rng.shuffle(uniq)
    cut = int(len(uniq)*(1-test_frac)); tg = set(uniq[:cut].tolist())
    tr = np.array([i for i, g in enumerate(groups) if g in tg])
    te = np.array([i for i, g in enumerate(groups) if g not in tg])
    return tr, te


def train_eval(X, y, tr, te, epochs, bs, lr, device, seed=42):
    set_seed(seed)
    model = BigCNN2D(n_classes=6).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    Xtr = torch.as_tensor(X[tr], device=device); ytr = torch.as_tensor(y[tr], device=device)
    Xte = torch.as_tensor(X[te], device=device)
    n = len(tr)
    for ep in range(1, epochs+1):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pr = []
        for i in range(0, len(Xte), bs):
            pr.append(model(Xte[i:i+bs]).argmax(-1).cpu().numpy())
    pred = np.concatenate(pr)
    acc = float((pred == y[te]).mean())
    return acc, npar


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-species", type=int, default=3000)
    p.add_argument("--per-session", type=int, default=6)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    device = get_device(); OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")
    results = {}
    for rname, rp in RES.items():
        print(f"\n== building {rname} spectrograms (n_fft={rp['n_fft']}, hop={rp['hop']})", flush=True)
        t0 = time.time()
        X, y, groups = build(args.per_species, args.per_session, rp["n_fft"], rp["hop"], args.seed)
        print(f"   X {X.shape}, {len(set(groups.tolist()))} sessions, {time.time()-t0:.0f}s", flush=True)
        for sname in ["random", "session"]:
            if sname == "random":
                tr, te = random_split(len(X), args.seed)
            else:
                tr, te = session_split(groups, args.seed)
            acc, npar = train_eval(X, y, tr, te, args.epochs, args.batch_size, args.lr, device)
            key = f"{rname}/{sname}"
            results[key] = {"acc": round(acc, 4), "params": npar,
                            "input": list(X.shape[1:]), "n_test": int(len(te))}
            print(f"   >> {key:18s} acc {acc:.4f}  (input {X.shape[1]}x{X.shape[2]}, test {len(te)})", flush=True)

    print("\n" + "="*60)
    print("WINGBEATS 4-CONDITION (big model, 6-species)")
    print("="*60)
    print(f"{'condition':<20}{'accuracy':>10}")
    for k, v in results.items():
        print(f"{k:<20}{v['acc']:>10.4f}")
    # deltas
    try:
        res_gain = results["fine/session"]["acc"] - results["coarse/session"]["acc"]
        leak_gain_c = results["coarse/random"]["acc"] - results["coarse/session"]["acc"]
        leak_gain_f = results["fine/random"]["acc"] - results["fine/session"]["acc"]
        print(f"\nresolution effect (fine-coarse, honest): {res_gain:+.3f}")
        print(f"leakage effect (random-session, coarse): {leak_gain_c:+.3f}")
        print(f"leakage effect (random-session, fine)  : {leak_gain_f:+.3f}")
        print(f"best (fine/random, ~published recipe)  : {results['fine/random']['acc']:.3f}")
        print(f"our honest (fine/session)              : {results['fine/session']['acc']:.3f}")
    except KeyError:
        pass
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n-- saved {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
