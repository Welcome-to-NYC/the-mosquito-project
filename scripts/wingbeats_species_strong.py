"""Does Wingbeats 6-species reach ~90%+ HONESTLY with a richer pipeline?

Our first Wingbeats test (8 kHz, 129x40 spectrogram, tiny CNN) gave 68.5% on a
session-honest split — same ballpark as UCR (73%). But the literature quotes
~96%, using the raw high-resolution waveform + big models. This script tests
whether the honest Wingbeats ceiling actually rises with a richer input and a
bigger model, to settle: is the 96% about DATA richness (Wingbeats raw signal
carries more than UCR's pre-compressed 200x20), or about setup/leak?

Changes vs the first test:
  * finer spectrogram: n_fft=512, hop=64  -> 257 freq x ~78 time (vs 129x40)
  * bigger 2D CNN: channels (32,64,128,128)
  * SESSION-honest split only (we already know random ~ honest+2.5pp here)

If honest accuracy climbs well past 70% -> Wingbeats data is genuinely richer
than UCR and the ~70% was our lossy pipeline. If it stays ~70% -> the 96% was
setup/leak, and honest optical species tops out ~70% regardless of dataset.

Run:
    python scripts/wingbeats_species_strong.py --per-species 3000 --epochs 30
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
from src.evaluation.metrics import evaluate, summarize  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_wingbeats_species_strong"
SPECIES = ["ae_aegypti", "ae_albopictus", "an_arabiensis",
           "an_gambiae", "c_pipiens", "c_quinquefasciatus"]
SP2IDX = {s: i for i, s in enumerate(SPECIES)}

SR = 8000
N_FFT = 512
HOP = 64
SEG = 5000
EPS = 1e-8


def fine_spec(sig):
    win = np.hanning(N_FFT).astype(np.float32)
    if len(sig) < SEG:
        sig = np.pad(sig, (0, SEG - len(sig)))
    sig = sig[:SEG]
    n_frames = 1 + (len(sig) - N_FFT) // HOP
    frames = np.stack([sig[i*HOP:i*HOP+N_FFT] * win for i in range(n_frames)])
    spec = np.fft.rfft(frames, axis=1)
    logp = np.log(spec.real**2 + spec.imag**2 + EPS).T.astype(np.float32)  # (257, n_frames)
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


def build(per_species, per_session, seed):
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
                X.append(fine_spec(sig)); y.append(SP2IDX[sp]); groups.append(sess); got += 1
        print(f"   {sp}: {got} clips")
    return np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(groups, object)


def session_split(groups, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist()))); rng.shuffle(uniq)
    cut = int(len(uniq) * (1 - test_frac))
    tg = set(uniq[:cut].tolist())
    tr = np.array([i for i, g in enumerate(groups) if g in tg])
    te = np.array([i for i, g in enumerate(groups) if g not in tg])
    return tr, te


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-species", type=int, default=3000)
    p.add_argument("--per-session", type=int, default=6)
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
    X, y, groups = build(args.per_species, args.per_session, args.seed)
    print(f"   X {X.shape}, {len(set(groups.tolist()))} sessions, {time.time()-t0:.0f}s")

    tr, te = session_split(groups, args.seed)
    model = BigCNN2D(n_classes=6).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   model params {n_params:,}  (input {X.shape[1]}x{X.shape[2]})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    Xtr = torch.as_tensor(X[tr], device=device); ytr = torch.as_tensor(y[tr], device=device)
    Xte = torch.as_tensor(X[te], device=device)
    n = len(tr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i+args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        sched.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pr = []
                for i in range(0, len(Xte), args.batch_size):
                    pr.append(model(Xte[i:i+args.batch_size]).argmax(-1).cpu().numpy())
                acc = float((np.concatenate(pr) == y[te]).mean())
            print(f"   epoch {epoch}: honest test_acc {acc:.4f}")

    model.eval()
    with torch.no_grad():
        pr = []
        for i in range(0, len(Xte), args.batch_size):
            pr.append(model(Xte[i:i+args.batch_size]).argmax(-1).cpu().numpy())
    preds = np.concatenate(pr)
    m = evaluate(y[te], preds, classes=SPECIES)
    print("\n---- Wingbeats 6-species, STRONG pipeline, HONEST split ----")
    print(summarize(m))
    print("\n== comparison ==")
    print(f"   first (tiny, 129x40):  0.685")
    print(f"   strong ({X.shape[1]}x{X.shape[2]}, {n_params//1000}K): {m['accuracy']:.4f}")
    print(f"   UCR SOTA reference:    ~0.73")

    (OUT / "results.json").write_text(json.dumps(
        {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "per_class": m["per_class"],
         "n_params": n_params, "input_shape": list(X.shape[1:])}, indent=2, default=str))
    print(f"\n-- saved {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
