"""Standalone validation experiment on UCR InsectWingbeat.

Answers two questions:

1. Given the 10-class UCR data (4 mosquito species × 2 sexes + 2 fly
   species), can a small classifier learn the discrimination?
2. Specifically, what is the mosquito↔fly confusion rate?

If the answer is "yes, well separated", that's strong evidence that
optical-modality mosquito vs fly discrimination is possible at all —
which is the missing piece our main W6 / student model could never
validate (no optical non-mosquito data in our training set).

Model: tiny 2D CNN over the (200 freq bands × T time steps) spectrogram.
Same family as our 1D-CNN architecturally; just 2D filters.

Run:
    python scripts/ucr_mosquito_vs_fly.py --epochs 15
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ucr_loader import binary_mosquito_vs_fly, load_ts  # noqa: E402
from src.evaluation.metrics import evaluate, summarize  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


class TinyCNN2D(nn.Module):
    """A small 2D CNN for (n_freq × n_time) spectrogram input. Same compute
    family as our 1D-CNN baseline — 3 conv blocks, adaptive pool, FC.
    """

    def __init__(self, n_classes: int, in_freq: int = 200, in_time: int = 20,
                 channels=(16, 32, 64), dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = 1
        for c in channels:
            layers += [
                nn.Conv2d(prev, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = c
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels[-1], 32)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T) → add channel dim → (B, 1, F, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc2(self.act(self.fc1(x)))


def _normalize_per_sample(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = X.mean(axis=(1, 2), keepdims=True)
    sd = X.std(axis=(1, 2), keepdims=True) + eps
    return ((X - mu) / sd).astype(np.float32)


def _train_one(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    classes: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    n = len(X_train)
    Xt = torch.as_tensor(X_train, device=device)
    yt = torch.as_tensor(y_train, device=device)
    Xs = torch.as_tensor(X_test, device=device)
    ys = torch.as_tensor(y_test, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        running_loss = 0.0
        running_n = 0
        running_correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            running_n += len(xb)
            running_correct += (logits.argmax(-1) == yb).sum().item()
        scheduler.step()

        # Eval on test
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xs), batch_size):
                preds.append(model(Xs[i : i + batch_size]).argmax(-1).cpu().numpy())
            y_pred = np.concatenate(preds)
        test_acc = float((y_pred == y_test).mean())
        print(f"epoch {epoch:3d}  train_loss {running_loss/running_n:.4f}  "
              f"train_acc {running_correct/running_n:.4f}  test_acc {test_acc:.4f}")

    # Final eval with full metrics
    model.eval()
    with torch.no_grad():
        probs = []
        for i in range(0, len(Xs), batch_size):
            probs.append(torch.softmax(model(Xs[i : i + batch_size]), dim=-1).cpu().numpy())
        probs = np.concatenate(probs)
    y_pred = probs.argmax(-1)
    metrics = evaluate(y_test, y_pred, y_score=probs, classes=classes)
    print()
    print(summarize(metrics))
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path",
                        default="data/raw/insect_wingbeat_ucr/InsectWingbeat/InsectWingbeat_eq_TRAIN.ts")
    parser.add_argument("--test-path",
                        default="data/raw/insect_wingbeat_ucr/InsectWingbeat/InsectWingbeat_eq_TEST.ts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None,
                        help="cap training instances (for quick smoke runs)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "exp_ucr_validation" / "results.json")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    print(f"== device: {device}")

    print("== loading TRAIN")
    t0 = time.time()
    train_ts = load_ts(args.train_path, max_instances=args.max_train)
    print(f"   {train_ts.X.shape} in {time.time()-t0:.1f}s")

    print("== loading TEST")
    t0 = time.time()
    test_ts = load_ts(args.test_path)
    print(f"   {test_ts.X.shape} in {time.time()-t0:.1f}s")
    classes = train_ts.classes
    in_freq, in_time = train_ts.n_dims, train_ts.max_len

    # Normalize per-sample to handle the wild dynamic range we saw
    # (X values were spanning [-70, +70] in raw form).
    X_train = _normalize_per_sample(train_ts.X)
    X_test = _normalize_per_sample(test_ts.X)

    # --- Experiment A: full 10-class ---
    print("\n========== 10-class (all UCR labels) ==========")
    model10 = TinyCNN2D(n_classes=len(classes), in_freq=in_freq, in_time=in_time).to(device)
    n_params = sum(p.numel() for p in model10.parameters() if p.requires_grad)
    print(f"   model params: {n_params:,}")
    metrics10 = _train_one(model10, X_train, train_ts.y, X_test, test_ts.y,
                           classes, args.epochs, args.batch_size, args.lr, device)

    # --- Experiment B: binary mosquito vs fly ---
    print("\n========== BINARY mosquito vs fly ==========")
    y_train_bin, train_mask = binary_mosquito_vs_fly(train_ts)
    y_test_bin, test_mask = binary_mosquito_vs_fly(test_ts)
    X_train_bin = X_train[train_mask]
    X_test_bin = X_test[test_mask]
    print(f"   train: {X_train_bin.shape}, classes: mosq {(y_train_bin==0).sum()} / fly {(y_train_bin==1).sum()}")
    print(f"   test:  {X_test_bin.shape}, classes: mosq {(y_test_bin==0).sum()} / fly {(y_test_bin==1).sum()}")
    model_bin = TinyCNN2D(n_classes=2, in_freq=in_freq, in_time=in_time).to(device)
    metrics_bin = _train_one(model_bin, X_train_bin, y_train_bin, X_test_bin, y_test_bin,
                             ["mosquito", "fly"], args.epochs, args.batch_size, args.lr, device)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "10class": {
            "accuracy": metrics10["accuracy"],
            "macro_f1": metrics10["macro_f1"],
            "auc_macro": metrics10["auc_macro"],
            "per_class": metrics10["per_class"],
            "confusion": metrics10["confusion"].tolist(),
            "classes": metrics10["classes"],
        },
        "binary_mosq_vs_fly": {
            "accuracy": metrics_bin["accuracy"],
            "macro_f1": metrics_bin["macro_f1"],
            "auc_macro": metrics_bin["auc_macro"],
            "per_class": metrics_bin["per_class"],
            "confusion": metrics_bin["confusion"].tolist(),
            "classes": metrics_bin["classes"],
        },
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\n-- saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
