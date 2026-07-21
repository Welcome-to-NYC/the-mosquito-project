"""Tests for the dataset registry / download orchestration that don't hit the network.

We only validate config parsing and the local-only branches (manual sources,
list output, presence detection). Full end-to-end download is exercised by
running the script against real sources, not by the unit suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import download as dl


def test_config_loads_and_has_expected_datasets():
    cfg = dl.load_config()
    expected = {"esc50", "wingbeats", "humbugdb", "insects", "fruitflies"}
    assert expected <= set(cfg.keys())
    for name, spec in cfg.items():
        assert "source" in spec, f"{name} missing source"
        assert "type" in spec["source"], f"{name} missing source.type"
        assert spec["source"]["type"] in {"url", "kaggle", "zenodo", "manual"}


def test_is_present_treats_gitkeep_only_as_empty(tmp_path: Path):
    (tmp_path / ".gitkeep").touch()
    assert dl.is_present(tmp_path) is False

    (tmp_path / "real_file.txt").write_text("hi")
    assert dl.is_present(tmp_path) is True


def test_is_present_returns_false_for_missing_dir(tmp_path: Path):
    assert dl.is_present(tmp_path / "does_not_exist") is False


def test_run_one_handles_manual_without_network(capsys, tmp_path: Path, monkeypatch):
    # Point the registry's target dir into tmp_path so we don't pollute data/raw.
    cfg = dl.load_config()
    spec = dict(cfg["insects"])
    spec["target_dir"] = str(tmp_path / "insects")
    dl.run_one("insects", spec, force=False)
    out = capsys.readouterr().out
    assert "manual setup required" in out
    assert "notes" in out


def test_kaggle_available_when_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(dl.shutil, "which", lambda _: None)
    ok, reason = dl.kaggle_available()
    assert ok is False
    assert "Kaggle CLI" in reason


def test_cmd_list_prints_every_dataset(capsys):
    cfg = dl.load_config()
    rc = dl.cmd_list(cfg)
    assert rc == 0
    out = capsys.readouterr().out
    for name in cfg:
        assert name in out


def test_strip_single_top_dir_hoists_lone_subdir(tmp_path: Path):
    # Mirror the ESC-50 layout: dest/<TopDir>/{audio/,meta/}
    top = tmp_path / "ESC-50-master"
    (top / "audio").mkdir(parents=True)
    (top / "meta").mkdir()
    (top / "audio" / "x.wav").touch()
    (top / "meta" / "esc50.csv").touch()

    dl.strip_single_top_dir(tmp_path)

    assert (tmp_path / "audio" / "x.wav").exists()
    assert (tmp_path / "meta" / "esc50.csv").exists()
    assert not top.exists()


def test_strip_single_top_dir_is_noop_with_multiple_children(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    dl.strip_single_top_dir(tmp_path)
    assert (tmp_path / "a").exists()
    assert (tmp_path / "b").exists()


def test_strip_single_top_dir_ignores_gitkeep(tmp_path: Path):
    (tmp_path / ".gitkeep").touch()
    top = tmp_path / "only"
    top.mkdir()
    (top / "file.txt").write_text("x")
    dl.strip_single_top_dir(tmp_path)
    assert (tmp_path / "file.txt").exists()
    assert not top.exists()
