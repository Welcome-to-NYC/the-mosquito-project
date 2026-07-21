"""Tests for src.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import evaluate, summarize


def test_evaluate_perfect_prediction_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    out = evaluate(y_true, y_pred, classes=["bg", "mosq"])
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["weighted_f1"] == 1.0
    assert out["balanced_accuracy"] == 1.0
    np.testing.assert_array_equal(out["confusion"], [[2, 0], [0, 2]])


def test_evaluate_per_class_table_layout():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    out = evaluate(y_true, y_pred, classes=["bg", "mosq", "insect"])
    classes_in_order = [r["class"] for r in out["per_class"]]
    assert classes_in_order == ["bg", "mosq", "insect"]
    # Recall for "mosq": 2/2 = 1.0
    mosq = next(r for r in out["per_class"] if r["class"] == "mosq")
    assert mosq["recall"] == 1.0
    # Precision for "mosq": predicted 3 times, only 2 correct = 2/3
    assert abs(mosq["precision"] - 2 / 3) < 1e-9


def test_evaluate_handles_unused_class():
    # 'insect' (index 2) appears in the labels list but never as a target/pred.
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    out = evaluate(y_true, y_pred, classes=["bg", "mosq", "insect"])
    assert len(out["per_class"]) == 3
    insect = out["per_class"][2]
    assert insect["support"] == 0
    assert insect["f1"] == 0.0
    assert out["confusion"].shape == (3, 3)


def test_evaluate_auc_binary_with_scores():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.7, 0.9])  # well-separated
    out = evaluate(y_true, y_pred, y_score=y_score, classes=["bg", "mosq"])
    assert out["auc_macro"] is not None
    assert out["auc_macro"] == 1.0


def test_evaluate_auc_multiclass_ovr():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    # Soft scores per-class, peaked on the true class
    y_score = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
    ])
    out = evaluate(y_true, y_pred, y_score=y_score, classes=["a", "b", "c"])
    assert out["auc_macro"] is not None
    assert out["auc_macro"] > 0.99


def test_evaluate_auc_returns_none_when_single_class_present():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    out = evaluate(y_true, y_pred, y_score=y_score, classes=["bg", "mosq"])
    assert out["auc_macro"] is None


def test_evaluate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        evaluate([0, 1], [0])


def test_evaluate_rejects_2d_input():
    with pytest.raises(ValueError):
        evaluate(np.zeros((2, 3), dtype=int), np.zeros((2, 3), dtype=int))


def test_evaluate_default_class_names():
    out = evaluate([0, 1, 2], [0, 1, 2])
    assert out["classes"] == ["0", "1", "2"]


def test_evaluate_too_few_classes_raises():
    with pytest.raises(ValueError):
        evaluate([0, 1, 2], [0, 1, 2], classes=["only_two", "names"])


def test_summarize_runs_and_includes_expected_lines():
    out = evaluate([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], classes=["bg", "mosq", "insect"])
    s = summarize(out)
    assert "accuracy" in s
    assert "macro_f1" in s
    assert "confusion" in s
    assert "bg" in s and "mosq" in s and "insect" in s


def test_summarize_includes_auc_when_present():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_score = np.array([0.1, 0.9])
    out = evaluate(y_true, y_pred, y_score=y_score, classes=["a", "b"])
    s = summarize(out)
    assert "auc_macro" in s
