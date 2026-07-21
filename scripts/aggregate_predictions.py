"""Per-recording aggregation.

Segment-level metrics treat every 200 ms window independently, which
double-counts evidence from a single recording. At deployment we'd
aggregate predictions across all segments from one capture and emit one
label per recording. This script does that aggregation on the test set
and compares against the segment-level number.

Aggregation methods:
  * majority: take the modal predicted class
  * mean_prob: mean of per-segment softmax, argmax

Both should improve over segment-level metrics, usually by 1–3 pp.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate, summarize  # noqa: E402


def _load_model(exp_dir: Path, n_classes: int):
    import yaml
    from src.training.config import load_config
    from src.training.train import _build_model
    from src.utils.device import get_device

    tmp = exp_dir / "_agg_cfg.yaml"
    tmp.write_text((exp_dir / "config_used.yaml").read_text())
    try:
        cfg = load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    device = get_device()
    model = _build_model(cfg, n_classes).to(device)
    ckpt = torch.load(exp_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


def _predict_probs(model, device, X: np.ndarray) -> np.ndarray:
    bs = 256
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            chunk = torch.as_tensor(X[i : i + bs], dtype=torch.float32, device=device).unsqueeze(1)
            out.append(torch.softmax(model(chunk), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def aggregate(
    recording_ids: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
) -> tuple[dict, dict]:
    """Return (majority_metrics, mean_prob_metrics)."""
    rec_to_idx: dict[str, list[int]] = {}
    for i, r in enumerate(recording_ids):
        rec_to_idx.setdefault(str(r), []).append(i)

    n_classes = probs.shape[-1]
    majority_true: list[int] = []
    majority_pred: list[int] = []
    mean_true: list[int] = []
    mean_pred: list[int] = []
    mean_probs: list[np.ndarray] = []

    for rec, idxs in rec_to_idx.items():
        labels = y_true[idxs]
        # All segments from one recording should share a label by construction;
        # if they don't (shouldn't happen given leakage-safe split), take majority.
        rec_label = int(Counter(labels).most_common(1)[0][0])

        seg_preds = probs[idxs].argmax(axis=-1)
        majority_true.append(rec_label)
        majority_pred.append(int(Counter(seg_preds).most_common(1)[0][0]))

        mp = probs[idxs].mean(axis=0)
        mean_true.append(rec_label)
        mean_pred.append(int(mp.argmax()))
        mean_probs.append(mp)

    return (
        {"y_true": np.array(majority_true), "y_pred": np.array(majority_pred)},
        {"y_true": np.array(mean_true), "y_pred": np.array(mean_pred),
         "y_score": np.stack(mean_probs)},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "aggregation.json")
    args = parser.parse_args(argv)

    z = np.load(ROOT / "data" / "processed" / "test.npz", allow_pickle=True)
    X = z["X"]
    y = z["y"]
    classes = list(z["classes"])
    recs = z["recording_id"]
    n_classes = len(classes)
    n_recordings = len(set(recs))
    print(f"-- test: {len(y)} segments from {n_recordings} recordings, "
          f"classes {classes}")

    rows = []
    for name in args.exp:
        exp_dir = ROOT / "experiments" / name
        if not exp_dir.is_dir():
            print(f"skip {name}: not a directory")
            continue
        print(f"\n=== {name} ===")
        model, device = _load_model(exp_dir, n_classes)
        probs = _predict_probs(model, device, X)
        seg_pred = probs.argmax(axis=-1)

        seg_metrics = evaluate(y, seg_pred, y_score=probs, classes=classes)
        majority, mean_prob = aggregate(recs, y, probs)
        maj_metrics = evaluate(majority["y_true"], majority["y_pred"], classes=classes)
        mp_metrics = evaluate(mean_prob["y_true"], mean_prob["y_pred"],
                              y_score=mean_prob["y_score"], classes=classes)

        print(f"  segment-level     : acc {seg_metrics['accuracy']:.4f}  "
              f"macro_f1 {seg_metrics['macro_f1']:.4f}  AUC {seg_metrics['auc_macro']:.4f}")
        print(f"  recording majority: acc {maj_metrics['accuracy']:.4f}  "
              f"macro_f1 {maj_metrics['macro_f1']:.4f}")
        print(f"  recording mean-pr.: acc {mp_metrics['accuracy']:.4f}  "
              f"macro_f1 {mp_metrics['macro_f1']:.4f}  AUC {mp_metrics['auc_macro']:.4f}")

        rows.append({
            "exp": name,
            "n_segments": int(len(y)),
            "n_recordings": int(n_recordings),
            "segment_acc": float(seg_metrics["accuracy"]),
            "segment_macro_f1": float(seg_metrics["macro_f1"]),
            "majority_acc": float(maj_metrics["accuracy"]),
            "majority_macro_f1": float(maj_metrics["macro_f1"]),
            "meanprob_acc": float(mp_metrics["accuracy"]),
            "meanprob_macro_f1": float(mp_metrics["macro_f1"]),
            "meanprob_auc": float(mp_metrics["auc_macro"]) if mp_metrics["auc_macro"] else None,
            "delta_meanprob_vs_segment": float(mp_metrics["accuracy"] - seg_metrics["accuracy"]),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\n-- saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
