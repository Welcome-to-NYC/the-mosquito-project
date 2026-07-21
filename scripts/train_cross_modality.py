"""Train a mosquito-vs-fly classifier on AUDIO only, test on OPTICAL.

The whole point is to measure the MODALITY GAP honestly:

    in-modality  = accuracy on held-out AUDIO test  (HumBugDB mosq vs InsectSound fly)
    cross-modality = accuracy on OPTICAL test        (UCR mosq vs fly)
    gap          = in-modality - cross-modality

A large gap means "audio-trained wingbeat discrimination does NOT transfer to
optical"; a small gap means the friend's idea (train audio, deploy optical)
actually works. Either result is a real finding.

Anti-overfitting: recording-level splits (done in build step), dropout, weight
decay, SpecAugment-style masking, early stopping on audio-val macro-F1, and we
report train vs val vs optical so any overfit is visible in the numbers.

Run:
    python scripts/train_cross_modality.py --epochs 40
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

from src.evaluation.metrics import evaluate, summarize  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "exp_cross_modality"


class SpecCNN(nn.Module):
    """Small 2D CNN over a (F, T) log-power spectrogram. ~50-100K params."""

    def __init__(self, n_classes=2, channels=(16, 32, 64), dropout=0.4):
        super().__init__()
        layers, prev = [], 1
        for c in channels:
            layers += [
                nn.Conv2d(prev, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = c
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels[-1], 32)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)          # (B,1,F,T)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc2(self.act(self.fc1(x)))


def spec_augment(x: torch.Tensor, n_freq=2, n_time=2, max_f=16, max_t=6):
    """In-batch SpecAugment: zero out random freq/time bands. x: (B,F,T)."""
    B, F, T = x.shape
    for _ in range(n_freq):
        f = torch.randint(0, max_f, (1,)).item()
        f0 = torch.randint(0, max(1, F - f), (1,)).item()
        x[:, f0:f0 + f, :] = 0.0
    for _ in range(n_time):
        t = torch.randint(0, max_t, (1,)).item()
        t0 = torch.randint(0, max(1, T - t), (1,)).item()
        x[:, :, t0:t0 + t] = 0.0
    return x


def _load(split):
    z = np.load(PROC / f"xmod_{split}.npz", allow_pickle=True)
    return z["X"].astype(np.float32), z["y"].astype(np.int64), list(z["classes"])


@torch.no_grad()
def _eval(model, X, y, device, bs=256):
    model.eval()
    probs = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], device=device)
        probs.append(torch.softmax(model(xb), -1).cpu().numpy())
    probs = np.concatenate(probs)
    return probs.argmax(-1), probs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")

    Xtr, ytr, classes = _load("audio_train")
    Xva, yva, _ = _load("audio_val")
    Xte, yte, _ = _load("audio_test")
    ucr_X, ucr_y, _ = _load("optical_ucr")
    wb_X, wb_y, _ = _load("optical_wingbeats")
    print(f"   train {Xtr.shape} val {Xva.shape} test {Xte.shape}")
    print(f"   ucr {ucr_X.shape} wingbeats {wb_X.shape}")

    model = SpecCNN(n_classes=2, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   model params: {n_params:,}")

    # class-balanced sampling
    cls_count = np.bincount(ytr, minlength=2)
    w = (1.0 / cls_count)[ytr]
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double), num_samples=len(ytr), replacement=True)
    Xtr_t = torch.as_tensor(Xtr)
    ytr_t = torch.as_tensor(ytr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t),
        batch_size=args.batch_size, sampler=sampler, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1, best_state, best_epoch = -1.0, None, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, correct, run = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = spec_augment(xb.clone())
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(xb); tot += len(xb)
            correct += (logits.argmax(-1) == yb).sum().item()
        sched.step()
        vp, _ = _eval(model, Xva, yva, device)
        vm = evaluate(yva, vp, classes=classes)
        val_f1 = vm["macro_f1"]
        print(f"epoch {epoch:3d}  train_loss {run/tot:.4f}  train_acc {correct/tot:.4f}  "
              f"val_acc {vm['accuracy']:.4f}  val_f1 {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    print(f"== best val macro-F1 {best_val_f1:.4f} @ epoch {best_epoch}")
    model.load_state_dict(best_state)

    # ---- final evaluation on all sets ----
    def report(name, X, y, binary=True):
        pred, prob = _eval(model, X, y, device)
        m = evaluate(y, pred, y_score=prob if binary else None, classes=classes)
        print(f"\n---- {name} ----")
        print(summarize(m))
        return m

    m_val = report("AUDIO val (in-modality)", Xva, yva)
    m_test = report("AUDIO test (in-modality)", Xte, yte)
    m_ucr = report("OPTICAL UCR (cross-modality PRIMARY)", ucr_X, ucr_y)
    # Wingbeats is all-mosquito: report recall on class 0 only
    wb_pred, _ = _eval(model, wb_X, wb_y, device)
    wb_recall = float((wb_pred == 0).mean())

    gap = m_test["accuracy"] - m_ucr["accuracy"]
    print("\n" + "=" * 60)
    print("MODALITY GAP SUMMARY")
    print("=" * 60)
    print(f"  in-modality  (audio test)  accuracy : {m_test['accuracy']:.4f}")
    print(f"  cross-modality (UCR optical) accuracy: {m_ucr['accuracy']:.4f}")
    print(f"  => modality gap                      : {gap:+.4f}")
    pc = {r["class"]: r for r in m_ucr["per_class"]}
    print(f"  UCR balanced accuracy                : {m_ucr.get('balanced_accuracy', float('nan')):.3f}")
    print(f"  UCR mosquito recall / fly recall     : "
          f"{pc.get('mosquito',{}).get('recall',float('nan')):.3f} / "
          f"{pc.get('non_mosquito',{}).get('recall',float('nan')):.3f}")
    print(f"  Wingbeats optical mosquito recall    : {wb_recall:.4f}")

    result = {
        "n_params": n_params,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "audio_val_acc": m_val["accuracy"],
        "audio_test_acc": m_test["accuracy"],
        "audio_test_macro_f1": m_test["macro_f1"],
        "ucr_optical_acc": m_ucr["accuracy"],
        "ucr_confusion": m_ucr["confusion"].tolist(),
        "ucr_per_class": m_ucr["per_class"],
        "wingbeats_mosq_recall": wb_recall,
        "modality_gap": gap,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str))
    torch.save({"model": best_state, "classes": list(classes)}, OUT / "best.pt")
    print(f"\n-- saved {OUT/'results.json'} and best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
