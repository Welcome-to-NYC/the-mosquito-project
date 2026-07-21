"""Tests for the YAML training config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.config import (
    DataConfig,
    FullConfig,
    ModelConfig,
    TrainConfig,
    TrainEarlyStopConfig,
    TrainSamplerConfig,
    TrainSchedulerConfig,
    load_config,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body)
    return p


def test_empty_config_returns_all_defaults(tmp_path: Path):
    cfg = load_config(_write(tmp_path, ""))
    assert isinstance(cfg, FullConfig)
    assert cfg.train.epochs == 30
    assert cfg.train.batch_size == 64
    assert cfg.model.name == "cnn_1d"
    assert cfg.exp.name == "exp"


def test_partial_config_inherits_defaults(tmp_path: Path):
    cfg = load_config(_write(tmp_path, "train:\n  epochs: 5\n  lr: 0.0005\n"))
    assert cfg.train.epochs == 5
    assert cfg.train.lr == 0.0005
    # Untouched fields take defaults.
    assert cfg.train.batch_size == 64
    assert cfg.train.weight_decay == 1e-4


def test_nested_train_subsections_round_trip(tmp_path: Path):
    yaml = """
train:
  scheduler:
    kind: reduce_on_plateau
    factor: 0.25
    patience: 3
  early_stop:
    metric: val_loss
    mode: min
    patience: 5
"""
    cfg = load_config(_write(tmp_path, yaml))
    assert isinstance(cfg.train.scheduler, TrainSchedulerConfig)
    assert cfg.train.scheduler.factor == 0.25
    assert cfg.train.scheduler.patience == 3
    assert isinstance(cfg.train.early_stop, TrainEarlyStopConfig)
    assert cfg.train.early_stop.metric == "val_loss"
    assert cfg.train.early_stop.mode == "min"


def test_unknown_top_level_key_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_config(_write(tmp_path, "nonsense:\n  foo: 1\n"))


def test_unknown_inner_key_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown keys in TrainConfig"):
        load_config(_write(tmp_path, "train:\n  hyper_typo: 0.1\n"))


def test_data_paths_default_to_data_processed(tmp_path: Path):
    cfg = load_config(_write(tmp_path, ""))
    assert cfg.data.train_npz.endswith("train.npz")
    assert cfg.data.val_npz.endswith("val.npz")
    assert cfg.data.test_npz.endswith("test.npz")


def test_model_kwargs_round_trip(tmp_path: Path):
    yaml = """
model:
  name: cnn_1d
  kwargs:
    n_classes: 3
    channels: [16, 32, 64]
    dropout: 0.3
"""
    cfg = load_config(_write(tmp_path, yaml))
    assert cfg.model.name == "cnn_1d"
    assert cfg.model.kwargs["n_classes"] == 3
    assert cfg.model.kwargs["channels"] == [16, 32, 64]


def test_output_dir_property(tmp_path: Path):
    yaml = "exp:\n  name: my_run\n  output_dir: outputs\n"
    cfg = load_config(_write(tmp_path, yaml))
    assert str(cfg.output_dir).endswith("outputs/exp_my_run")


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
