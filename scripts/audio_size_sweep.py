"""Model-size sweep on the AUDIO mosquito-vs-fly task.

Purpose: answer "how small can the model be and still separate mosquito from
fly on AUDIO?" — because if a tiny model already solves the audio task, the
same-size model is a plausible target for the optical task too (the friend's
argument: audio feasibility implies optical feasibility at similar size).

For each capacity we report:
  * params
  * audio TEST accuracy / macro-F1  (in-modality, recording-level held out)
  * UCR OPTICAL accuracy            (cross-modality, for reference)
  * Wingbeats optical mosq recall   (cross-modality, for reference)

The audio column is the headline the user asked for; the optical columns show
whether the modality gap shrinks or grows with capacity (bonus).

Run:
    python scripts/audio_size_sweep.py --epochs 30
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

from scripts.train_cross_modality import SpecCNN, spec_augment, _load, _eval  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_audio_size_sweep"

# (label, channels, fc_hidden-via-dropout kept default) — vary width/depth.
CONFIGS = [
    ("xs  (8,16)",       (8, 16)),
    ("s   (8,16,24)",    (8, 16, 24)),
    ("m   (16,32,64)",   (16, 32, 64)),
    ("l   (32,64,128)",  (32, 64, 128)),
]


def train_one(channels, Xtr, ytr, Xva, yva, device, epochs, lr, wd, dropout, seed):
    set_seed(seed)
    model = SpecCNN(n_classes=2, channels=channels, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cls_count = np.bincount(ytr, minlength=2)
    w = (1.0 / cls_count)[ytr]
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double), num_samples=len(ytr), replacement=True)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr)),
        batch_size=128, sampler=sampler, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = spec_augment(xb.clone())
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        vp, _ = _eval(model, Xva, yva, device)
        f1 = evaluate(yva, vp, classes=["mosquito", "non_mosquito"])["macro_f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, n_params, best_f1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")

    Xtr, ytr, classes = _load("audio_train")
    Xva, yva, _ = _load("audio_val")
    Xte, yte, _ = _load("audio_test")
    ucr_X, ucr_y, _ = _load("optical_ucr")
    wb_X, wb_y, _ = _load("optical_wingbeats")

    rows = []
    for label, channels in CONFIGS:
        t0 = time.time()
        model, n_params, best_f1 = train_one(
            channels, Xtr, ytr, Xva, yva, device,
            args.epochs, args.lr, args.weight_decay, args.dropout, args.seed)

        te_pred, te_prob = _eval(model, Xte, yte, device)
        m_te = evaluate(yte, te_pred, y_score=te_prob, classes=classes)
        ucr_pred, _ = _eval(model, ucr_X, ucr_y, device)
        m_ucr = evaluate(ucr_y, ucr_pred, classes=classes)
        wb_pred, _ = _eval(model, wb_X, wb_y, device)
        wb_recall = float((wb_pred == 0).mean())

        # fp32 KB of the weights
        kb = n_params * 4 / 1024.0
        row = {
            "config": label,
            "channels": list(channels),
            "params": n_params,
            "fp32_kb": round(kb, 1),
            "audio_test_acc": round(m_te["accuracy"], 4),
            "audio_test_macro_f1": round(m_te["macro_f1"], 4),
            "ucr_optical_acc": round(m_ucr["accuracy"], 4),
            "wingbeats_mosq_recall": round(wb_recall, 4),
            "train_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"\n### {label}: params={n_params:,} ({kb:.1f} KB)")
        print(f"    audio test acc {row['audio_test_acc']:.4f}  f1 {row['audio_test_macro_f1']:.4f}")
        print(f"    UCR optical acc {row['ucr_optical_acc']:.4f}  wingbeats recall {row['wingbeats_mosq_recall']:.4f}")

    # summary table
    print("\n" + "=" * 78)
    print("AUDIO-ONLY SIZE SWEEP (mosquito vs fly)")
    print("=" * 78)
    print(f"{'config':<16}{'params':>8}{'KB':>8}{'audio_acc':>11}{'audio_f1':>10}{'ucr_acc':>9}{'wb_rec':>8}")
    for r in rows:
        print(f"{r['config']:<16}{r['params']:>8}{r['fp32_kb']:>8.1f}"
              f"{r['audio_test_acc']:>11.4f}{r['audio_test_macro_f1']:>10.4f}"
              f"{r['ucr_optical_acc']:>9.4f}{r['wingbeats_mosq_recall']:>8.4f}")

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\n-- saved {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
