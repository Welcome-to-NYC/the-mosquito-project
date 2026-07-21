"""Wingbeats 6-species classification: leaky (random) vs honest (session) split.

Purpose: settle whether the ~96% mosquito-species accuracy people quote on
Wingbeats is real signal or a recording-session shortcut. We use ONE dataset
(Wingbeats optical, 6 species), ONE model, ONE training recipe, and change
ONLY the train/test split:

  * RANDOM split  : shuffle all clips, 80/20. Clips from the same recording
                    session land in both train and test -> the model can
                    memorise session artefacts. This is the inflated number.
  * SESSION split : hold out whole recording sessions (group by recording_id).
                    No session appears in both -> the model must actually use
                    the wingbeat. This is the honest number.

The gap between the two IS the confound. If honest >> chance and close to
random, species is genuinely learnable; if honest collapses, the 96% was a
shortcut.

Run:
    python scripts/wingbeats_species_honest.py --per-species 2500 --epochs 25
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

from scripts.ucr_mosquito_vs_fly import TinyCNN2D  # noqa: E402
from src.data import spectro  # noqa: E402
from src.data.metadata import load_wingbeats  # noqa: E402
from src.evaluation.metrics import evaluate, summarize  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

OUT = ROOT / "experiments" / "exp_wingbeats_species_honest"
SPECIES = ["ae_aegypti", "ae_albopictus", "an_arabiensis",
           "an_gambiae", "c_pipiens", "c_quinquefasciatus"]
SP2IDX = {s: i for i, s in enumerate(SPECIES)}


def build(per_species: int, per_session: int, seed: int):
    """Sample clips spread across many sessions, build spectrograms."""
    import librosa
    rng = np.random.default_rng(seed)
    df = load_wingbeats()
    X, y, groups = [], [], []
    for sp in SPECIES:
        sub = df[df["species"] == sp]
        sessions = list(sub["recording_id"].unique())
        rng.shuffle(sessions)
        got = 0
        for sess in sessions:
            if got >= per_species:
                break
            rows = sub[sub["recording_id"] == sess]
            paths = list(rows["path"])
            rng.shuffle(paths)
            for p in paths[:per_session]:
                if got >= per_species:
                    break
                try:
                    sig, _ = librosa.load(p, sr=spectro.SR, mono=True)
                except Exception:
                    continue
                X.append(spectro.wav_segment_to_spec(sig[: spectro.SEG_SAMPLES]))
                y.append(SP2IDX[sp])
                groups.append(sess)
                got += 1
        print(f"   {sp}: {got} clips from {len(set(g for g,yy in zip(groups,y) if yy==SP2IDX[sp]))} sessions")
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.int64),
            np.asarray(groups, dtype=object))


def train_eval(Xtr, ytr, Xte, yte, epochs, bs, lr, device, seed=42):
    set_seed(seed)
    model = TinyCNN2D(n_classes=6, in_freq=Xtr.shape[1], in_time=Xtr.shape[2],
                      channels=(16, 32, 64)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t = torch.as_tensor(Xtr, device=device); ytr_t = torch.as_tensor(ytr, device=device)
    Xte_t = torch.as_tensor(Xte, device=device)
    n = len(Xtr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte_t), bs):
            preds.append(model(Xte_t[i:i+bs]).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def random_split(n, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    return idx[:cut], idx[cut:]


def session_split(groups, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq)
    cut = int(len(uniq) * (1 - test_frac))
    train_g = set(uniq[:cut].tolist())
    tr = np.array([i for i, g in enumerate(groups) if g in train_g])
    te = np.array([i for i, g in enumerate(groups) if g not in train_g])
    return tr, te


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-species", type=int, default=2500)
    p.add_argument("--per-session", type=int, default=6)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== device {device}")
    print("== building Wingbeats sample")
    t0 = time.time()
    X, y, groups = build(args.per_species, args.per_session, args.seed)
    print(f"   X {X.shape}, {len(set(groups.tolist()))} sessions total, {time.time()-t0:.0f}s")

    results = {}
    for split_name, splitter in [("random_LEAKY", "random"), ("session_HONEST", "session")]:
        if splitter == "random":
            tr, te = random_split(len(X), args.seed)
        else:
            tr, te = session_split(groups, args.seed)
        preds = train_eval(X[tr], y[tr], X[te], y[te], args.epochs, args.batch_size, args.lr, device)
        m = evaluate(y[te], preds, classes=SPECIES)
        results[split_name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                               "per_class": m["per_class"], "n_test": int(len(te))}
        print(f"\n---- {split_name} (train {len(tr)} / test {len(te)}) ----")
        print(summarize(m))

    print("\n" + "=" * 60)
    print("WINGBEATS 6-SPECIES: LEAKY vs HONEST SPLIT")
    print("=" * 60)
    r = results["random_LEAKY"]["accuracy"]
    h = results["session_HONEST"]["accuracy"]
    print(f"  random  (leaky)  accuracy : {r:.4f}")
    print(f"  session (honest) accuracy : {h:.4f}")
    print(f"  => confound inflation      : {r - h:+.4f}")
    print(f"  chance (6 classes)         : {1/6:.4f}")

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n-- saved {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
