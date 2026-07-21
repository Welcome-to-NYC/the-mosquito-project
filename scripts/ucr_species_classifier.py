"""Optical mosquito species + non-mosquito classifier on UCR InsectWingbeat.

Answers: "can ONE optical model say 'not a mosquito' OR 'mosquito, and which
species'?" — all within a single clean optical dataset (no cross-source /
cross-modality confound, unlike our earlier audio->optical experiment).

Label space (10 UCR labels -> 5 classes, sexes merged):

    0  non_mosquito   <- Fruit_flies + House_flies
    1  Aedes          <- Aedes_female + Aedes_male
    2  Quinx          <- Culex quinquefasciatus (Quinx_female/male)
    3  Stigma         <- Stigma_female + Stigma_male
    4  Tarsalis       <- Culex tarsalis (Tarsalis_female/male)

Perfectly balanced: 5000 per class train / test.

We report three views:
  * flat 5-class accuracy / per-class / confusion
  * mosquito DETECTION (any species vs non_mosquito) — the coarse gate
  * SPECIES accuracy among true mosquitoes — the fine head

Run:
    python scripts/ucr_species_classifier.py --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ucr_mosquito_vs_fly import TinyCNN2D, _normalize_per_sample  # noqa: E402
from src.data.ucr_loader import load_ts  # noqa: E402
from src.evaluation.metrics import evaluate, summarize  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_ucr_species"

# UCR label -> 5-class index + name
GROUP = {
    "fruit_flies": (0, "non_mosquito"), "house_flies": (0, "non_mosquito"),
    "aedes_female": (1, "Aedes"), "aedes_male": (1, "Aedes"),
    "quinx_female": (2, "Quinx"), "quinx_male": (2, "Quinx"),
    "stigma_female": (3, "Stigma"), "stigma_male": (3, "Stigma"),
    "tarsalis_female": (4, "Tarsalis"), "tarsalis_male": (4, "Tarsalis"),
}
CLASS_NAMES = ["non_mosquito", "Aedes", "Quinx", "Stigma", "Tarsalis"]


def remap(ts):
    y5 = np.array([GROUP[ts.classes[i].lower()][0] for i in ts.y], dtype=np.int64)
    return y5


def _train(model, Xtr, ytr, Xte, yte, epochs, bs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xtr_t = torch.as_tensor(Xtr, device=device); ytr_t = torch.as_tensor(ytr, device=device)
    Xte_t = torch.as_tensor(Xte, device=device)
    n = len(Xtr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        run = correct = seen = 0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            run += loss.item()*len(xb); seen += len(xb)
            correct += (logits.argmax(-1) == yb).sum().item()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xte_t), bs):
                preds.append(model(Xte_t[i:i+bs]).argmax(-1).cpu().numpy())
            yp = np.concatenate(preds)
        acc = float((yp == yte).mean())
        print(f"epoch {epoch:3d}  train_loss {run/seen:.4f}  train_acc {correct/seen:.4f}  test_acc {acc:.4f}")
    model.eval()
    with torch.no_grad():
        probs = []
        for i in range(0, len(Xte_t), bs):
            probs.append(torch.softmax(model(Xte_t[i:i+bs]), -1).cpu().numpy())
        probs = np.concatenate(probs)
    return probs.argmax(-1), probs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")

    base = "data/raw/insect_wingbeat_ucr/InsectWingbeat"
    tr = load_ts(f"{base}/InsectWingbeat_eq_TRAIN.ts")
    te = load_ts(f"{base}/InsectWingbeat_eq_TEST.ts")
    Xtr = _normalize_per_sample(tr.X); Xte = _normalize_per_sample(te.X)
    ytr = remap(tr); yte = remap(te)
    print(f"   train {Xtr.shape}  test {Xte.shape}  classes {CLASS_NAMES}")

    model = TinyCNN2D(n_classes=5, in_freq=tr.n_dims, in_time=tr.max_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   params {n_params:,}")

    t0 = time.time()
    yp, probs = _train(model, Xtr, ytr, Xte, yte, args.epochs, args.batch_size, args.lr, device)
    print(f"   trained in {time.time()-t0:.0f}s")

    # ---- view 1: flat 5-class ----
    m = evaluate(yte, yp, y_score=probs, classes=CLASS_NAMES)
    print("\n---- 5-class (non_mosquito + 4 species) ----")
    print(summarize(m))

    # ---- view 2: mosquito detection (coarse) ----
    det_true = (yte != 0).astype(int)   # 1 = mosquito
    det_pred = (yp != 0).astype(int)
    det = evaluate(det_true, det_pred, classes=["non_mosquito", "mosquito"])
    print("\n---- mosquito DETECTION (any species vs non_mosquito) ----")
    print(summarize(det))

    # ---- view 3: species accuracy among TRUE mosquitoes ----
    mos_mask = yte != 0
    sp_acc = float((yp[mos_mask] == yte[mos_mask]).mean())
    # among mosquitoes that were correctly detected as mosquito
    correctly_detected = mos_mask & (yp != 0)
    sp_acc_detected = float((yp[correctly_detected] == yte[correctly_detected]).mean())
    print("\n---- SPECIES accuracy (among true mosquitoes) ----")
    print(f"  species correct / all true mosquitoes      : {sp_acc:.4f}")
    print(f"  species correct / detected-as-mosquito only: {sp_acc_detected:.4f}")

    result = {
        "n_params": n_params,
        "flat5_accuracy": m["accuracy"],
        "flat5_macro_f1": m["macro_f1"],
        "flat5_per_class": m["per_class"],
        "flat5_confusion": m["confusion"].tolist(),
        "mosquito_detection_acc": det["accuracy"],
        "mosquito_detection_per_class": det["per_class"],
        "species_acc_all_mosquitoes": sp_acc,
        "species_acc_detected_only": sp_acc_detected,
        "class_names": CLASS_NAMES,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str))
    torch.save({"model": model.state_dict(), "classes": CLASS_NAMES}, OUT / "best.pt")
    print(f"\n-- saved {OUT/'results.json'} and best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
