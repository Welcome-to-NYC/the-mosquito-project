"""Per-source breakdown of a trained model's test predictions.

The W2/W3 baselines all train on a mix of Wingbeats (sterile lab capture)
+ HumBugDB (field) + ESC-50 (background). If the model is actually
learning "wingbeat acoustic signature", per-source recall on the mosquito
class should be similar across Wingbeats and HumBugDB. If it's instead
learning "is this a Wingbeats clip", Wingbeats mosquito recall will be
near 100 % and HumBugDB mosquito recall much lower.

Run::

    python scripts/cross_source_eval.py --exp exp_cnn_1d_baseline
    python scripts/cross_source_eval.py --exp exp_baseline_xgb           # XGBoost variant

Reports a tidy table per source × class with support, predictions, and
recall, plus the overall confusion matrix sliced by source.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_test_arrays():
    z = np.load(ROOT / "data" / "processed" / "test.npz", allow_pickle=True)
    return {
        "X": z["X"],
        "y": z["y"],
        "classes": list(z["classes"]),
        "source": np.array(z["source"]),
        "species": np.array(z["species"]) if "species" in z.files else None,
        "recording_id": np.array(z["recording_id"]) if "recording_id" in z.files else None,
    }


def _predict_torch(exp_dir: Path, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Inference for any torch model under src.models.

    Reads ``config_used.yaml`` to pick the model class (cnn_1d /
    physics_informed / future). Pushes X through in batches and returns
    integer predictions per sample.
    """
    import torch
    import yaml

    from src.training.config import FullConfig, load_config
    from src.training.train import _build_model
    from src.utils.device import get_device

    cfg_path = exp_dir / "config_used.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path} — can't determine model class")
    # config_used.yaml is the resolved dataclass dump from train.py, so it
    # round-trips through load_config cleanly when written back.
    tmp = exp_dir / "_cse_cfg.yaml"
    tmp.write_text(cfg_path.read_text())
    try:
        cfg: FullConfig = load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    device = get_device()
    model = _build_model(cfg, n_classes).to(device)
    ckpt = torch.load(exp_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds: list[np.ndarray] = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(X), bs):
            chunk = torch.as_tensor(X[i : i + bs], dtype=torch.float32, device=device).unsqueeze(1)
            out = model(chunk).argmax(dim=-1).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds)


def _predict_sklearn(exp_dir: Path, X: np.ndarray) -> np.ndarray:
    """Inference for the LR / XGBoost baselines via the pickled estimator.
    Re-extracts the same 11-d spectral features the model was trained on.
    """
    from src.features.spectral import extract_features

    feat = np.empty((len(X), 11), dtype=np.float32)
    bs = 4096
    for i in range(0, len(X), bs):
        feat[i : i + bs] = extract_features(X[i : i + bs])

    pkl = next(exp_dir.glob("model_*.pkl"))
    with pkl.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    if "scaler" in bundle:
        feat = bundle["scaler"].transform(feat)
    return model.predict(feat)


def _predict(exp_dir: Path, X: np.ndarray, n_classes: int) -> np.ndarray:
    if any(exp_dir.glob("model_*.pkl")):
        print(f"-- using sklearn path (found {next(exp_dir.glob('model_*.pkl')).name})")
        return _predict_sklearn(exp_dir, X)
    if (exp_dir / "best.pt").exists():
        print("-- using torch path (best.pt)")
        return _predict_torch(exp_dir, X, n_classes)
    raise SystemExit(f"no recognizable model artifact under {exp_dir}/")


def _per_source_metrics(y_true: np.ndarray, y_pred: np.ndarray, sources: np.ndarray, classes: list[str]) -> None:
    n_classes = len(classes)
    print()
    print("=== per source × class ===")
    print(f"{'source':<10s} {'class':<22s} {'support':>8s} {'pred=this':>10s} {'recall':>8s}")
    src_class_keys = sorted({(s, c) for s, c in zip(sources, y_true)})
    for s, c in src_class_keys:
        mask_true = (sources == s) & (y_true == c)
        n_support = int(mask_true.sum())
        n_correct = int(((y_pred == c) & mask_true).sum())
        recall = n_correct / max(n_support, 1)
        print(f"{s:<10s} {classes[c]:<22s} {n_support:>8d} {n_correct:>10d} {recall:>8.4f}")

    print()
    print("=== mosquito recall by source (the smoking-gun row) ===")
    mosq_idx = classes.index("mosquito") if "mosquito" in classes else 1
    for s in sorted(set(sources)):
        mask = (sources == s) & (y_true == mosq_idx)
        if not mask.any():
            continue
        recall = ((y_pred == mosq_idx) & mask).sum() / mask.sum()
        print(f"  {s:<10s}: support={mask.sum():>7d}  recall={recall:.4f}")

    print()
    print("=== prediction distribution per source-class slice (rows=true, cols=pred) ===")
    for s in sorted(set(sources)):
        mask_s = sources == s
        sub_t = y_true[mask_s]
        sub_p = y_pred[mask_s]
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(sub_t, sub_p):
            cm[t, p] += 1
        present_rows = (cm.sum(axis=1) > 0)
        print(f"\n  source = {s} (n={mask_s.sum()})")
        header = " " * 22 + "".join(f"{c[:9]:>10s}" for c in classes)
        print(header)
        for i in range(n_classes):
            if not present_rows[i]:
                continue
            cells = "".join(f"{cm[i, j]:>10d}" for j in range(n_classes))
            print(f"  {classes[i][:20]:<20s}  {cells}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="experiment dir name under experiments/")
    args = parser.parse_args(argv)

    exp_dir = ROOT / "experiments" / args.exp
    if not exp_dir.is_dir():
        raise SystemExit(f"not a directory: {exp_dir}")

    print(f"== loading test.npz")
    test = _load_test_arrays()
    classes = test["classes"]
    print(f"   shape={test['X'].shape}  classes={classes}  sources={sorted(set(test['source']))}")

    print(f"== predicting via {exp_dir.name}")
    y_pred = _predict(exp_dir, test["X"], n_classes=len(classes))

    overall_acc = float((y_pred == test["y"]).mean())
    print(f"   overall test accuracy: {overall_acc:.4f}")

    _per_source_metrics(test["y"], y_pred, test["source"], classes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
