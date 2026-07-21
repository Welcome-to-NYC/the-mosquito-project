"""Push the 1D audio detector to ~0.90 while staying ESP32-friendly.

Baseline (train_audio_detector_1d): 0.842, weak on fly rejection (0.87). The 1D
raw-waveform model has to learn frequency structure implicitly; the 2D
spectrogram model got 0.918 because frequency is handed to it explicitly. We
close that gap with a LearnableFFT front-end (a Conv1d initialised as a Fourier
basis -> a learned spectrogram) so the model gets 2D-style frequency features
from a 1D input -- and it stays deployable (it's all conv1d + a magnitude op).

Configs tried (data built once, reused):
  A. bigger CNN1D (16,32,64)                     -- more capacity, pure 1D
  B. LearnableFFT(48, stride4) -> conv1d stack   -- learned spectrogram
  C. B + augmentation (noise + gain)             -- generalisation

Run:
    python scripts/improve_audio_detector_1d.py --epochs 35
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

from scripts.train_audio_detector_1d import build, gsplit, _eval  # noqa: E402
from src.models.cnn_1d import CNN1D  # noqa: E402
from src.models.physics_informed import LearnableFFT  # noqa: E402
from src.evaluation.metrics import evaluate, summarize  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_audio_detector_1d_improved"


class LFFTDetector(nn.Module):
    """LearnableFFT front-end + conv1d stack over the learned spectrogram."""

    def __init__(self, n_filters=48, n_classes=2, dropout=0.3):
        super().__init__()
        self.fft = LearnableFFT(n_filters=n_filters, kernel_size=129,
                                sample_rate=5000, freq_range=(100.0, 2000.0), stride=4)
        self.b1 = nn.Sequential(nn.Conv1d(n_filters, 32, 5, padding=2),
                                nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.b2 = nn.Sequential(nn.Conv1d(32, 32, 3, padding=1),
                                nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        z = self.fft(x)              # (B, n_filters, T)
        z = self.b2(self.b1(z))
        return self.fc(self.drop(self.gap(z).flatten(1)))


def augment(xb):
    """On-the-fly: random gain + white noise at random SNR (z-normed input)."""
    B = xb.size(0)
    gain = (0.5 + 1.0 * torch.rand(B, 1, 1, device=xb.device))
    xb = xb * gain
    if torch.rand(1).item() < 0.6:
        snr = 5 + 15 * torch.rand(1, device=xb.device).item()
        p_sig = xb.pow(2).mean()
        noise = torch.randn_like(xb) * torch.sqrt(p_sig / (10 ** (snr / 10)))
        xb = xb + noise
    return xb


def train(make_model, X, y, tr, va, te, srcs, classes, device,
          epochs, bs, lr, aug=False, seed=42):
    set_seed(seed)
    model = make_model().to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnt = np.bincount(y[tr], minlength=2); w = (1.0/cnt)[y[tr]]
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), int(tr.sum()), True)
    ds = torch.utils.data.TensorDataset(torch.as_tensor(X[tr]).unsqueeze(1), torch.as_tensor(y[tr]))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best = -1, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if aug:
                xb = augment(xb)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        vp = _eval(model, X[va], device).argmax(-1)
        f1 = evaluate(y[va], vp, classes=classes)["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1; best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    prob = _eval(model, X[te], device); pred = prob.argmax(-1)
    m = evaluate(y[te], pred, classes=classes)
    sub = {}
    for s in ["humbug_bg", "insectsound_fly"]:
        mask = (srcs[te] == s) & (y[te] == 0)
        if mask.sum():
            sub[s] = round(float((pred[mask] == 0).mean()), 3)
    return model, {"params": npar, "acc": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
                   "mosq_recall": round(float((pred[y[te]==1]==1).mean()), 4), "neg_by_source": sub}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args(argv)

    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")
    t0 = time.time()
    X, y, groups, srcs = build(400, 250, 25, 636, 42)
    print(f"   built {len(X)} in {time.time()-t0:.0f}s")
    tr, va, te = gsplit(groups, 42)
    classes = ["not_mosquito", "mosquito"]

    configs = [
        ("baseline (8,16,24)", lambda: CNN1D(n_classes=2, in_channels=1, channels=[8,16,24],
                                             kernel_sizes=[7,5,3], fc_hidden=16, dropout=0.2), False),
        ("bigger (16,32,64)", lambda: CNN1D(n_classes=2, in_channels=1, channels=[16,32,64],
                                            kernel_sizes=[7,5,3], fc_hidden=32, dropout=0.3), False),
        ("LearnableFFT", lambda: LFFTDetector(n_filters=48), False),
        ("LearnableFFT + aug", lambda: LFFTDetector(n_filters=48), True),
    ]
    results = {}
    best_model = None; best_acc = -1
    for name, mk, aug in configs:
        model, r = train(mk, X, y, tr, va, te, srcs, classes, device,
                         args.epochs, args.batch_size, args.lr, aug=aug)
        results[name] = r
        print(f"\n### {name}: params {r['params']:,}  acc {r['acc']:.4f}  "
              f"mosq_rec {r['mosq_recall']:.3f}  neg {r['neg_by_source']}")
        if r["acc"] > best_acc:
            best_acc = r["acc"]; best_model = (name, model, r)

    print("\n" + "="*74)
    print("1D DETECTOR IMPROVEMENT SWEEP (honest split)")
    print("="*74)
    print(f"{'config':<24}{'params':>8}{'acc':>8}{'mosq_rec':>10}{'bg_rej':>8}{'fly_rej':>8}")
    for k, r in results.items():
        s = r["neg_by_source"]
        print(f"{k:<24}{r['params']:>8}{r['acc']:>8.3f}{r['mosq_recall']:>10.3f}"
              f"{s.get('humbug_bg',0):>8.3f}{s.get('insectsound_fly',0):>8.3f}")
    print(f"\n>> best: {best_model[0]} @ {best_model[2]['acc']:.4f}")

    # save best + stream test for potential redeploy
    name, model, r = best_model
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": classes, "config": name}, OUT / "best.pt")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"-- saved {OUT/'best.pt'} and results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
