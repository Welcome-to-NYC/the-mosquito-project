"""Classification metrics shared by every model in the project.

Designed so the same call works for binary and multi-class problems:

    >>> from src.evaluation.metrics import evaluate, summarize
    >>> result = evaluate(y_true, y_pred, y_score, classes=["bg", "mosq", "insect"])
    >>> print(summarize(result))

``y_pred`` is the predicted class index per sample (shape ``(N,)``).
``y_score`` is optional but enables AUC-ROC; expected shape is ``(N,)`` for
binary and ``(N, C)`` for multi-class softmax / sigmoid outputs.

We deliberately *don't* depend on torch here — metrics take numpy arrays so
they're equally usable for the LR / XGBoost baselines and for the CNN
training loops.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _to_int_array(y: Sequence | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def evaluate(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_score: np.ndarray | None = None,
    classes: Sequence[str] | None = None,
) -> dict:
    """Compute the standard metric bundle for one prediction set.

    Returns a dict with these keys:

    ``accuracy``           : float
    ``balanced_accuracy``  : recall macro-averaged across classes
    ``macro_f1``           : float — macro-averaged F1
    ``weighted_f1``        : float — F1 weighted by support
    ``per_class``          : list[dict] of (class, precision, recall, f1, support)
    ``confusion``          : 2-D ndarray, rows=true, cols=pred
    ``classes``            : list[str] — same order as confusion rows/cols
    ``auc_macro``          : float | None — macro AUC if ``y_score`` provided

    ``classes`` defaults to ``["0", "1", ...]``. If you pass class names,
    they're propagated to the confusion matrix axes and per-class table.
    """
    y_true = _to_int_array(y_true, "y_true")
    y_pred = _to_int_array(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch")

    n_classes = int(max(y_true.max(initial=-1), y_pred.max(initial=-1)) + 1) if len(y_true) else 0
    if classes is None:
        classes = [str(i) for i in range(n_classes)]
    elif len(classes) < n_classes:
        raise ValueError(f"got {len(classes)} class names but data references {n_classes} classes")
    classes = list(classes)
    label_indices = list(range(len(classes)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_indices, zero_division=0.0
    )

    per_class = [
        {
            "class": classes[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(classes))
    ]

    cm = confusion_matrix(y_true, y_pred, labels=label_indices)

    # macro_f1 / weighted_f1 / balanced_accuracy via the same per-class arrays.
    macro_f1 = float(np.mean(f1))
    total_support = float(support.sum()) or 1.0
    weighted_f1 = float(np.sum(f1 * support) / total_support)
    balanced_acc = float(np.mean(recall))

    auc_macro: float | None = None
    if y_score is not None and len(y_true) > 0:
        auc_macro = _safe_auc(y_true, y_score, label_indices)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion": cm,
        "classes": classes,
        "auc_macro": auc_macro,
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, labels: list[int]) -> float | None:
    """Macro AUC across classes; returns None if it can't be computed.

    AUC needs at least two classes present in ``y_true`` and a probability
    column per class. Edge cases (single class present, NaN scores) return
    None rather than crashing — the caller decides whether that's a fatal
    error.
    """
    score = np.asarray(y_score)
    n_present = len(set(y_true.tolist()))
    if n_present < 2:
        return None
    try:
        if score.ndim == 1:
            # Binary case
            if len(labels) != 2:
                return None
            return float(roc_auc_score(y_true, score))
        if score.ndim == 2:
            return float(
                roc_auc_score(
                    y_true,
                    score,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
    except ValueError:
        return None
    return None


def summarize(result: Mapping) -> str:
    """Human-readable digest of an :func:`evaluate` result."""
    lines = []
    lines.append("=== metrics ===")
    lines.append(f"accuracy          : {result['accuracy']:.4f}")
    lines.append(f"balanced_accuracy : {result['balanced_accuracy']:.4f}")
    lines.append(f"macro_f1          : {result['macro_f1']:.4f}")
    lines.append(f"weighted_f1       : {result['weighted_f1']:.4f}")
    if result.get("auc_macro") is not None:
        lines.append(f"auc_macro         : {result['auc_macro']:.4f}")
    lines.append("")
    lines.append("per-class:")
    lines.append(f"  {'class':<22s} {'precision':>9s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for row in result["per_class"]:
        lines.append(
            f"  {row['class'][:22]:<22s} {row['precision']:>9.4f} {row['recall']:>8.4f} "
            f"{row['f1']:>8.4f} {row['support']:>8d}"
        )
    lines.append("")
    lines.append("confusion (rows=true, cols=pred):")
    cm = result["confusion"]
    classes = result["classes"]
    header = " " * 22 + "".join(f"{c[:8]:>9s}" for c in classes)
    lines.append(header)
    for i, row in enumerate(cm):
        cells = "".join(f"{v:>9d}" for v in row)
        lines.append(f"  {classes[i][:20]:<20s}  {cells}")
    return "\n".join(lines)
