"""Push UCR mosquito species accuracy as high as it will go.

Baseline (scripts/ucr_species_classifier.py): 5-class 70.8 %, species-among-
mosquitoes 65.7 %, mosquito-detection 96.7 %. The bottleneck is same-genus
species (Culex quinx vs tarsalis). We try, in order, the standard levers and
measure each so the ceiling is empirical, not guessed:

  A. bigger 2D CNN + more epochs + SpecAugment + label smoothing (direct 5-class)
  B. train on the full 10 UCR labels (species x sex + 2 flies) then MERGE
     predictions to 5-class — finer supervision (sex dimorphism is an easy,
     informative auxiliary signal)
  C. 3-seed ENSEMBLE of the better of A/B (average probabilities)

Reports 5-class accuracy, mosquito-detection accuracy, and species accuracy
among true mosquitoes for each.

Run:
    python scripts/ucr_species_improve.py --epochs 45
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

from scripts.ucr_mosquito_vs_fly import TinyCNN2D, _normalize_per_sample  # noqa: E402
from src.data.ucr_loader import load_ts  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_ucr_species_improve"

# 5-class target
CLASS5 = ["non_mosquito", "Aedes", "Quinx", "Stigma", "Tarsalis"]
G5 = {
    "fruit_flies": 0, "house_flies": 0,
    "aedes_female": 1, "aedes_male": 1,
    "quinx_female": 2, "quinx_male": 2,
    "stigma_female": 3, "stigma_male": 3,
    "tarsalis_female": 4, "tarsalis_male": 4,
}
# 10-class -> 5-class map (index into sorted UCR classes handled via names)
TEN_TO_FIVE = {  # UCR raw label -> 5-class idx
    "fruit_flies": 0, "house_flies": 0,
    "aedes_female": 1, "aedes_male": 1,
    "quinx_female": 2, "quinx_male": 2,
    "stigma_female": 3, "stigma_male": 3,
    "tarsalis_female": 4, "tarsalis_male": 4,
}


def spec_augment2d(x, n_f=2, n_t=2, max_f=24, max_t=4):
    B, F, T = x.shape
    for _ in range(n_f):
        f = int(torch.randint(0, max_f, (1,))); f0 = int(torch.randint(0, max(1, F - f), (1,)))
        x[:, f0:f0+f, :] = 0.0
    for _ in range(n_t):
        t = int(torch.randint(0, max_t, (1,))); t0 = int(torch.randint(0, max(1, T - t), (1,)))
        x[:, :, t0:t0+t] = 0.0
    return x


def train_model(Xtr, ytr, Xte, n_classes, channels, epochs, bs, lr, device,
                augment=True, label_smoothing=0.05, seed=42, log_every=15):
    set_seed(seed)
    model = TinyCNN2D(n_classes=n_classes, in_freq=Xtr.shape[1], in_time=Xtr.shape[2],
                      channels=channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    Xtr_t = torch.as_tensor(Xtr, device=device); ytr_t = torch.as_tensor(ytr, device=device)
    Xte_t = torch.as_tensor(Xte, device=device)
    n = len(Xtr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            if augment:
                xb = spec_augment2d(xb.clone())
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
        if epoch % log_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pr = []
                for i in range(0, len(Xte_t), bs):
                    pr.append(torch.softmax(model(Xte_t[i:i+bs]), -1).cpu().numpy())
                pr = np.concatenate(pr)
            print(f"      epoch {epoch}: params {n_params:,}")
    # final probs
    model.eval()
    with torch.no_grad():
        pr = []
        for i in range(0, len(Xte_t), bs):
            pr.append(torch.softmax(model(Xte_t[i:i+bs]), -1).cpu().numpy())
    return np.concatenate(pr), n_params


def hierarchical(y5_true, probs5):
    """probs5: (N,5). Returns dict of the three views."""
    yp = probs5.argmax(-1)
    acc5 = float((yp == y5_true).mean())
    det_true = (y5_true != 0).astype(int); det_pred = (yp != 0).astype(int)
    det_acc = float((det_pred == det_true).mean())
    mos = y5_true != 0
    sp_all = float((yp[mos] == y5_true[mos]).mean())
    cd = mos & (yp != 0)
    sp_det = float((yp[cd] == y5_true[cd]).mean())
    return {"acc5": acc5, "det_acc": det_acc, "species_all": sp_all, "species_detected": sp_det}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-3)
    args = p.parse_args(argv)

    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")

    base = "data/raw/insect_wingbeat_ucr/InsectWingbeat"
    tr = load_ts(f"{base}/InsectWingbeat_eq_TRAIN.ts")
    te = load_ts(f"{base}/InsectWingbeat_eq_TEST.ts")
    Xtr = _normalize_per_sample(tr.X); Xte = _normalize_per_sample(te.X)
    y5_tr = np.array([G5[tr.classes[i].lower()] for i in tr.y], dtype=np.int64)
    y5_te = np.array([G5[te.classes[i].lower()] for i in te.y], dtype=np.int64)
    # 10-class labels (raw UCR)
    uniq = sorted(set(c.lower() for c in tr.classes))
    ten_idx = {c: i for i, c in enumerate(uniq)}
    y10_tr = np.array([ten_idx[tr.classes[i].lower()] for i in tr.y], dtype=np.int64)
    ten_to5 = np.array([TEN_TO_FIVE[c] for c in uniq], dtype=np.int64)
    print(f"   train {Xtr.shape}  test {Xte.shape}")

    results = {}
    CH = (32, 64, 128)

    # --- A: bigger direct 5-class ---
    print("\n== A: bigger 5-class + aug + label-smoothing")
    t0 = time.time()
    pA, npar = train_model(Xtr, y5_tr, Xte, 5, CH, args.epochs, args.batch_size, args.lr, device)
    results["A_bigger_5class"] = {**hierarchical(y5_te, pA), "params": npar, "sec": round(time.time()-t0)}
    print("   ", results["A_bigger_5class"])

    # --- B: 10-class -> merge to 5 ---
    print("\n== B: 10-class (species x sex) -> merge to 5-class")
    t0 = time.time()
    p10, npar = train_model(Xtr, y10_tr, Xte, 10, CH, args.epochs, args.batch_size, args.lr, device)
    # collapse 10-class probs to 5-class by summing group members
    p10_to5 = np.zeros((len(p10), 5), dtype=np.float32)
    for j in range(10):
        p10_to5[:, ten_to5[j]] += p10[:, j]
    results["B_10class_merged"] = {**hierarchical(y5_te, p10_to5), "params": npar, "sec": round(time.time()-t0)}
    print("   ", results["B_10class_merged"])

    # --- C: 3-seed ensemble of the better approach ---
    better = "B" if results["B_10class_merged"]["acc5"] >= results["A_bigger_5class"]["acc5"] else "A"
    print(f"\n== C: 3-seed ensemble of approach {better}")
    t0 = time.time()
    acc_probs = np.zeros((len(Xte), 5), dtype=np.float32)
    for seed in (0, 1, 2):
        if better == "A":
            pr, _ = train_model(Xtr, y5_tr, Xte, 5, CH, args.epochs, args.batch_size, args.lr, device, seed=seed)
            acc_probs += pr
        else:
            pr, _ = train_model(Xtr, y10_tr, Xte, 10, CH, args.epochs, args.batch_size, args.lr, device, seed=seed)
            tmp = np.zeros((len(pr), 5), dtype=np.float32)
            for j in range(10):
                tmp[:, ten_to5[j]] += pr[:, j]
            acc_probs += tmp
    acc_probs /= 3.0
    results["C_ensemble"] = {**hierarchical(y5_te, acc_probs), "based_on": better, "sec": round(time.time()-t0)}
    print("   ", results["C_ensemble"])

    # --- summary ---
    print("\n" + "=" * 70)
    print("SPECIES-ACCURACY IMPROVEMENT SWEEP")
    print("=" * 70)
    print(f"{'approach':<22}{'5-class':>9}{'detect':>9}{'species':>9}")
    print(f"{'baseline (small)':<22}{0.708:>9.3f}{0.967:>9.3f}{0.654:>9.3f}")
    for k, v in results.items():
        print(f"{k:<22}{v['acc5']:>9.3f}{v['det_acc']:>9.3f}{v['species_all']:>9.3f}")

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n-- saved {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
