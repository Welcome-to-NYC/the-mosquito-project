"""Classical baseline: spectral features + Logistic Regression / XGBoost.

This is the "what does a simple model with hand-engineered FFT features get
us?" reference number. Every later model — 1D-CNN, MosquitoSong+,
physics-informed CNN — has to beat this to justify itself.

Usage::

    python -m src.models.baseline_lr                       # all defaults
    python -m src.models.baseline_lr --model lr            # LR only
    python -m src.models.baseline_lr --model xgb           # XGBoost only
    python -m src.models.baseline_lr --feature-cache       # save features for reuse

Outputs go under ``experiments/exp_baseline_<model>/``:
  features_train.npy / features_val.npy / features_test.npy  (when --feature-cache)
  model_lr.pkl  /  model_xgb.pkl
  results.json   — full metrics dict (per-split, per-class)
  summary.txt    — printable digest from src.evaluation.metrics.summarize
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from src.evaluation.metrics import evaluate, summarize
from src.features.spectral import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    z = np.load(path, allow_pickle=True)
    X = z["X"]
    y = z["y"]
    classes = list(z["classes"])
    return X, y, classes


def _batched_extract(X: np.ndarray, sr: int, batch: int = 4096) -> np.ndarray:
    """Featurize ``X`` in batches; tqdm gives a progress bar that doesn't stall
    on the rfft call for a 472k-row matrix.
    """
    out = np.empty((X.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    for i in tqdm(range(0, len(X), batch), desc="features", leave=False):
        out[i : i + batch] = extract_features(X[i : i + batch], sr=sr)
    return out


def _featurize_partition(name: str, npz_path: Path, sr: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    print(f"-- featurizing {name}: {npz_path}")
    X, y, classes = _load_npz(npz_path)
    t0 = time.perf_counter()
    feats = _batched_extract(X, sr=sr)
    print(f"   {feats.shape} features in {time.perf_counter() - t0:.1f}s")
    return feats, y, classes


def train_lr(
    X_train: np.ndarray, y_train: np.ndarray, max_iter: int = 1000
) -> tuple[LogisticRegression, StandardScaler]:
    """LogisticRegression with class_weight='balanced'. The scaler is returned so
    inference uses the same standardization as training.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    print(f"-- LR training on {Xs.shape}")
    t0 = time.perf_counter()
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
        multi_class="auto",
    )
    clf.fit(Xs, y_train)
    print(f"   fit in {time.perf_counter() - t0:.1f}s")
    return clf, scaler


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, n_classes: int):
    """XGBoost with sample_weight from class_weight='balanced'. We use
    XGBClassifier with `objective='multi:softprob'` so we get per-class
    probabilities back for AUC.
    """
    import xgboost as xgb  # noqa: WPS433

    print(f"-- XGBoost training on {X_train.shape}")
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    print(f"   fit in {time.perf_counter() - t0:.1f}s")
    return clf


def predict_lr(clf: LogisticRegression, scaler: StandardScaler, X: np.ndarray):
    Xs = scaler.transform(X)
    return clf.predict(Xs), clf.predict_proba(Xs)


def predict_xgb(clf, X: np.ndarray):
    return clf.predict(X), clf.predict_proba(X)


def _evaluate_split(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
) -> dict:
    res = evaluate(y_true, y_pred, y_score=y_proba, classes=classes)
    print(f"\n=== {name} ===")
    print(summarize(res))
    return res


def _save_results(out_dir: Path, model_name: str, results_per_split: dict, classes: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for split, res in results_per_split.items():
        serializable[split] = {
            "accuracy": res["accuracy"],
            "balanced_accuracy": res["balanced_accuracy"],
            "macro_f1": res["macro_f1"],
            "weighted_f1": res["weighted_f1"],
            "auc_macro": res["auc_macro"],
            "per_class": res["per_class"],
            "confusion": res["confusion"].tolist(),
            "classes": res["classes"],
        }
    serializable["model"] = model_name

    (out_dir / "results.json").write_text(json.dumps(serializable, indent=2))

    summary_blocks = [f"# {model_name} baseline\n"]
    for split, res in results_per_split.items():
        summary_blocks.append(f"\n## {split}\n")
        summary_blocks.append(summarize(res))
    (out_dir / "summary.txt").write_text("\n".join(summary_blocks))

    print(f"\n-- saved {out_dir}/results.json + summary.txt")


def run(
    model_kinds: Iterable[str],
    sr: int,
    feature_cache: bool,
    processed_dir: Path = PROCESSED,
    experiments_dir: Path = EXPERIMENTS,
) -> dict:
    train_path = processed_dir / "train.npz"
    val_path = processed_dir / "val.npz"
    test_path = processed_dir / "test.npz"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise SystemExit(f"missing {p}; run preprocess_pipeline first")

    Xt, yt, classes = _featurize_partition("train", train_path, sr)
    Xv, yv, _ = _featurize_partition("val", val_path, sr)
    Xs, ys, _ = _featurize_partition("test", test_path, sr)

    if feature_cache:
        cache = experiments_dir / "exp_baseline_features"
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "features_train.npy", Xt)
        np.save(cache / "features_val.npy", Xv)
        np.save(cache / "features_test.npy", Xs)
        print(f"-- cached features to {cache}/")

    print(f"-- classes: {classes}")
    print(f"-- train class counts: {np.bincount(yt).tolist()}")

    out: dict = {}
    if "lr" in model_kinds:
        out_dir = experiments_dir / "exp_baseline_lr"
        clf, scaler = train_lr(Xt, yt)
        results = {}
        for split, X_, y_ in (("train", Xt, yt), ("val", Xv, yv), ("test", Xs, ys)):
            preds, probs = predict_lr(clf, scaler, X_)
            results[split] = _evaluate_split(split, y_, preds, probs, classes)
        with (out_dir / "model_lr.pkl").open("wb") if False else open("/dev/null", "w") as _:
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "model_lr.pkl").open("wb") as f:
            pickle.dump({"model": clf, "scaler": scaler, "classes": classes,
                         "feature_names": list(FEATURE_NAMES)}, f)
        _save_results(out_dir, "logistic_regression", results, classes)
        out["lr"] = results

    if "xgb" in model_kinds:
        out_dir = experiments_dir / "exp_baseline_xgb"
        clf = train_xgb(Xt, yt, n_classes=len(classes))
        results = {}
        for split, X_, y_ in (("train", Xt, yt), ("val", Xv, yv), ("test", Xs, ys)):
            preds, probs = predict_xgb(clf, X_)
            results[split] = _evaluate_split(split, y_, preds, probs, classes)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "model_xgb.pkl").open("wb") as f:
            pickle.dump({"model": clf, "classes": classes,
                         "feature_names": list(FEATURE_NAMES)}, f)
        _save_results(out_dir, "xgboost", results, classes)
        out["xgb"] = results

    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lr", "xgb", "both"], default="both")
    parser.add_argument("--sr", type=int, default=5000, help="signal sample rate (must match preprocess)")
    parser.add_argument("--feature-cache", action="store_true",
                        help="save extracted feature arrays under experiments/exp_baseline_features/")
    args = parser.parse_args(list(argv) if argv is not None else None)

    kinds = ("lr", "xgb") if args.model == "both" else (args.model,)
    run(model_kinds=kinds, sr=args.sr, feature_cache=args.feature_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
