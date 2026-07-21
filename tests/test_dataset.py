"""Tests for the NPZ-backed PyTorch Dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import WingbeatNpz, make_loader, save_partition_npz


def _write_fixture(tmp_path: Path, n: int = 10, win: int = 64, classes=("bg", "mosq")) -> Path:
    rng = np.random.RandomState(0)
    X = rng.randn(n, win).astype(np.float32)
    y = rng.randint(0, len(classes), size=n).astype(np.int64)
    out = tmp_path / "fixture.npz"
    return save_partition_npz(out, X, y, list(classes))


def test_save_partition_npz_shape_and_classes(tmp_path: Path):
    path = _write_fixture(tmp_path)
    loaded = np.load(path, allow_pickle=True)
    assert loaded["X"].shape == (10, 64)
    assert loaded["y"].dtype == np.int64
    assert list(loaded["classes"]) == ["bg", "mosq"]


def test_save_partition_npz_with_optional_metadata(tmp_path: Path):
    X = np.zeros((3, 8), dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.int64)
    rid = np.array(["a", "b", "a"], dtype=object)
    src = np.array(["humbugdb", "humbugdb", "esc50"], dtype=object)
    sp = np.array(["", "ae aegypti", ""], dtype=object)
    out = save_partition_npz(tmp_path / "x.npz", X, y, ["bg", "mosq"], rid, src, sp)
    z = np.load(out, allow_pickle=True)
    assert list(z["recording_id"]) == ["a", "b", "a"]
    assert list(z["source"]) == ["humbugdb", "humbugdb", "esc50"]


def test_save_partition_npz_rejects_bad_shape(tmp_path: Path):
    with pytest.raises(ValueError):
        save_partition_npz(tmp_path / "bad.npz", np.zeros(10), np.zeros(10), ["a"])


def test_save_partition_npz_rejects_length_mismatch(tmp_path: Path):
    with pytest.raises(ValueError):
        save_partition_npz(tmp_path / "bad.npz", np.zeros((5, 8)), np.zeros(4), ["a"])


def test_dataset_len_and_getitem(tmp_path: Path):
    path = _write_fixture(tmp_path, n=10, win=64)
    ds = WingbeatNpz(path)
    assert len(ds) == 10
    sig, lbl = ds[0]
    assert isinstance(sig, torch.Tensor)
    assert isinstance(lbl, torch.Tensor)
    assert sig.shape == (1, 64)         # channel dim added
    assert sig.dtype == torch.float32
    assert lbl.dtype == torch.int64
    assert ds.num_classes == 2


def test_dataset_transform_applied(tmp_path: Path):
    path = _write_fixture(tmp_path, n=4, win=8)
    ds = WingbeatNpz(path, transform=lambda x: x * 0)
    sig, _ = ds[0]
    assert (sig == 0).all()


def test_dataset_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        WingbeatNpz(tmp_path / "nope.npz")


def test_make_loader_yields_correct_shapes(tmp_path: Path):
    path = _write_fixture(tmp_path, n=20, win=64)
    ds = WingbeatNpz(path)
    loader = make_loader(ds, batch_size=4, shuffle=False, num_workers=0)
    sig, lbl = next(iter(loader))
    assert sig.shape == (4, 1, 64)
    assert lbl.shape == (4,)


def test_make_loader_num_workers_zero_disables_persistent(tmp_path: Path):
    path = _write_fixture(tmp_path)
    ds = WingbeatNpz(path)
    loader = make_loader(ds, num_workers=0)
    assert loader.persistent_workers is False
