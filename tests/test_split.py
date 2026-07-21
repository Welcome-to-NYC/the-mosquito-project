"""Tests for leakage-safe train/val/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split import (
    Splits,
    split_by_recording,
    split_with_esc50_folds,
    summarize,
)


def _make_synth(n_recordings: int = 60, segs_per_rec: int = 5, n_labels: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    labels = ["mosquito", "non_mosquito_insect", "background"][:n_labels]
    for rid in range(n_recordings):
        label = labels[rng.randint(n_labels)]
        for s in range(segs_per_rec):
            rows.append({
                "path": f"/tmp/r{rid:03d}_s{s}.wav",
                "source": "synth",
                "label": label,
                "species": None,
                "recording_id": f"synth:{rid}",
                "fold": pd.NA,
                "raw_class": label,
            })
    return pd.DataFrame(rows)


def test_splits_no_leakage():
    df = _make_synth()
    splits = split_by_recording(df, val_frac=0.2, test_frac=0.2, seed=1)
    splits.assert_no_leakage()


def test_splits_cover_all_rows_exactly_once():
    df = _make_synth()
    splits = split_by_recording(df, val_frac=0.2, test_frac=0.2, seed=1)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == len(df)


def test_splits_respect_requested_fractions_approximately():
    df = _make_synth(n_recordings=200, segs_per_rec=5)
    splits = split_by_recording(df, val_frac=0.2, test_frac=0.2, seed=42)
    sizes = splits.sizes()
    total = sum(sizes.values())
    assert 0.15 <= sizes["test"] / total <= 0.25, sizes
    assert 0.15 <= sizes["val"] / total <= 0.25, sizes
    assert 0.50 <= sizes["train"] / total <= 0.70, sizes


def test_splits_each_partition_has_each_label():
    df = _make_synth(n_recordings=120, segs_per_rec=3, n_labels=3, seed=7)
    splits = split_by_recording(df, val_frac=0.2, test_frac=0.2, seed=7)
    for partition in (splits.train, splits.val, splits.test):
        assert set(partition["label"].unique()) == {"mosquito", "non_mosquito_insect", "background"}


def test_splits_empty_input_returns_empty_splits():
    df = _make_synth(n_recordings=0, segs_per_rec=0)
    splits = split_by_recording(df)
    assert len(splits.train) == 0
    assert len(splits.val) == 0
    assert len(splits.test) == 0


def test_splits_reject_bad_fractions():
    df = _make_synth()
    with pytest.raises(ValueError):
        split_by_recording(df, val_frac=0.0, test_frac=0.1)
    with pytest.raises(ValueError):
        split_by_recording(df, val_frac=0.5, test_frac=0.6)


def test_summarize_emits_expected_lines():
    df = _make_synth()
    splits = split_by_recording(df, val_frac=0.2, test_frac=0.2)
    summary = summarize(splits)
    assert "split sizes:" in summary
    assert "per-class counts:" in summary
    assert "mosquito" in summary


def test_esc50_aware_split_uses_fixed_folds_for_esc50():
    # ESC-50 rows: 50 recordings, fold in 1..5
    rng = np.random.RandomState(1)
    esc_rows = []
    for rid in range(50):
        esc_rows.append({
            "path": f"/tmp/esc{rid}.wav",
            "source": "esc50",
            "label": "background" if rng.rand() > 0.1 else "non_mosquito_insect",
            "species": None,
            "recording_id": f"esc50:{rid}",
            "fold": (rid % 5) + 1,
            "raw_class": "x",
        })
    # Other rows: 60 recordings, no fold
    other_rows = []
    for rid in range(60):
        for s in range(3):
            other_rows.append({
                "path": f"/tmp/o{rid}_{s}.wav",
                "source": "humbugdb",
                "label": "mosquito",
                "species": "ae aegypti",
                "recording_id": f"humbugdb:{rid}",
                "fold": pd.NA,
                "raw_class": "mosquito",
            })
    df = pd.DataFrame(esc_rows + other_rows)
    df["fold"] = df["fold"].astype("Int64")

    splits = split_with_esc50_folds(df, val_fold=4, test_fold=5, val_frac_other=0.2, test_frac_other=0.2, seed=3)

    # Every ESC-50 row from fold 5 ends up in test, fold 4 in val, others in train.
    esc_test = splits.test[splits.test["source"] == "esc50"]
    esc_val = splits.val[splits.val["source"] == "esc50"]
    esc_train = splits.train[splits.train["source"] == "esc50"]
    assert (esc_test["fold"] == 5).all()
    assert (esc_val["fold"] == 4).all()
    assert set(esc_train["fold"].dropna().tolist()) <= {1, 2, 3}
    splits.assert_no_leakage()


def test_esc50_aware_split_rejects_invalid_folds():
    df = pd.DataFrame({
        "path": ["x"],
        "source": ["esc50"],
        "label": ["background"],
        "species": [None],
        "recording_id": ["esc50:1"],
        "fold": pd.array([1], dtype="Int64"),
        "raw_class": ["x"],
    })
    with pytest.raises(ValueError):
        split_with_esc50_folds(df, val_fold=4, test_fold=4)
    with pytest.raises(ValueError):
        split_with_esc50_folds(df, val_fold=0, test_fold=5)
