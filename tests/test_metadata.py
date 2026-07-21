"""Tests for the per-dataset metadata loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import metadata as md


def _write_esc50_fixture(root: Path) -> None:
    audio = root / "audio"
    audio.mkdir(parents=True)
    meta = root / "meta"
    meta.mkdir()
    csv = meta / "esc50.csv"
    df = pd.DataFrame([
        {"filename": "1-100-A-0.wav", "fold": 1, "target": 0, "category": "dog",
         "esc10": True, "src_file": 100, "take": "A"},
        {"filename": "2-200-A-13.wav", "fold": 2, "target": 13, "category": "crickets",
         "esc10": False, "src_file": 200, "take": "A"},
        {"filename": "3-300-B-7.wav", "fold": 3, "target": 7, "category": "insects",
         "esc10": False, "src_file": 300, "take": "B"},
    ])
    df.to_csv(csv, index=False)
    for f in df["filename"]:
        (audio / f).touch()


def test_load_esc50_returns_uniform_schema(tmp_path: Path):
    _write_esc50_fixture(tmp_path)
    df = md.load_esc50(tmp_path)
    assert list(df.columns) == list(md.COLUMNS)
    assert len(df) == 3
    assert df.loc[df["raw_class"] == "dog", "label"].iloc[0] == "background"
    assert df.loc[df["raw_class"] == "crickets", "label"].iloc[0] == "non_mosquito_insect"
    assert df.loc[df["raw_class"] == "insects", "label"].iloc[0] == "non_mosquito_insect"
    # recording_id == "esc50:<filename>" — see docstring; ESC-50's official
    # folds already handle src_file grouping with four documented exceptions.
    assert df["recording_id"].nunique() == 3
    assert df["recording_id"].iloc[0] == "esc50:1-100-A-0.wav"


def test_load_esc50_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        md.load_esc50(tmp_path)


def _write_humbugdb_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv = root / "neurips_2021_zenodo_0_0_1.csv"
    df = pd.DataFrame([
        {"id": 1, "length": 0.5, "name": "a.wav", "sample_rate": 8000, "record_datetime": "2020",
         "sound_type": "mosquito", "species": "ae aegypti", "gender": "F", "fed": "no",
         "plurality": "Single", "age": "adult", "method": "lab", "mic_type": "phone",
         "device_type": "X", "country": "USA", "district": "GA", "province": "X",
         "place": "X", "location_type": "lab"},
        {"id": 2, "length": 1.2, "name": "b.wav", "sample_rate": 8000, "record_datetime": "2020",
         "sound_type": "background", "species": pd.NA, "gender": pd.NA, "fed": pd.NA,
         "plurality": pd.NA, "age": pd.NA, "method": pd.NA, "mic_type": "phone",
         "device_type": "X", "country": "USA", "district": "GA", "province": "X",
         "place": "X", "location_type": "field"},
        {"id": 3, "length": 2.0, "name": "c.wav", "sample_rate": 8000, "record_datetime": "2020",
         "sound_type": "audio", "species": pd.NA, "gender": pd.NA, "fed": pd.NA,
         "plurality": pd.NA, "age": pd.NA, "method": pd.NA, "mic_type": "phone",
         "device_type": "X", "country": "USA", "district": "GA", "province": "X",
         "place": "X", "location_type": "field"},
    ])
    df.to_csv(csv, index=False)


def test_load_humbugdb_classifies_sound_types(tmp_path: Path):
    _write_humbugdb_fixture(tmp_path)
    df = md.load_humbugdb(tmp_path)
    assert list(df.columns) == list(md.COLUMNS)
    labels = df.set_index("raw_class")["label"]
    assert labels["mosquito"] == "mosquito"
    assert labels["background"] == "background"
    assert labels["audio"] == "background"
    # Species is preserved only for the mosquito row.
    species_for_mosq = df.loc[df["raw_class"] == "mosquito", "species"].iloc[0]
    assert species_for_mosq == "ae aegypti"
    # ...and cleared for the others.
    assert df.loc[df["raw_class"] == "background", "species"].isna().all()


def test_load_humbugdb_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        md.load_humbugdb(tmp_path)


def test_load_wingbeats_empty_when_dir_missing(tmp_path: Path):
    df = md.load_wingbeats(tmp_path / "nope")
    assert list(df.columns) == list(md.COLUMNS)
    assert len(df) == 0


def test_load_wingbeats_walks_species_session_wav(tmp_path: Path):
    # Standard Wingbeats layout: <root>/Wingbeats/<species>/<session>/*.wav
    base = tmp_path / "Wingbeats"
    s1 = base / "Ae. aegypti" / "D_16_12_12_19_46_13"
    s2 = base / "Ae. aegypti" / "D_16_12_12_19_57_52"
    s3 = base / "C. quinquefasciatus" / "D_17_01_01_10_00_00"
    for s in (s1, s2, s3):
        s.mkdir(parents=True)
    (s1 / "a.wav").touch()
    (s1 / "b.wav").touch()  # same session as a.wav -> same recording_id
    (s2 / "c.wav").touch()
    (s3 / "d.wav").touch()

    df = md.load_wingbeats(tmp_path)

    assert len(df) == 4
    assert (df["label"] == "mosquito").all()
    assert set(df["species"]) == {"ae_aegypti", "c_quinquefasciatus"}
    # Two wavs from session 1 share the same recording_id; three sessions total.
    assert df["recording_id"].nunique() == 3
    s1_ids = df[df["path"].str.contains("D_16_12_12_19_46_13")]["recording_id"].unique()
    assert len(s1_ids) == 1


def test_load_wingbeats_handles_extra_age_dir(tmp_path: Path):
    # An. gambiae layout: extra age-bucket level between species and session.
    # The loader uses each wav's parent dir as the session id, so this Just Works.
    base = tmp_path / "Wingbeats"
    age_5d = base / "An. gambiae" / "Anopheles gambiae_ 5d-7d" / "D_session_a"
    age_15d = base / "An. gambiae" / "Anopheles gambiae_15d-17d" / "D_session_b"
    age_5d.mkdir(parents=True)
    age_15d.mkdir(parents=True)
    (age_5d / "x.wav").touch()
    (age_15d / "y.wav").touch()

    df = md.load_wingbeats(tmp_path)
    assert len(df) == 2
    # Different sessions -> different recording ids even within the same species.
    assert df["recording_id"].nunique() == 2


def test_load_wingbeats_steps_into_extra_wingbeats_dir(tmp_path: Path):
    nested = tmp_path / "Wingbeats" / "Ae. aegypti" / "D_session"
    nested.mkdir(parents=True)
    (nested / "x.wav").touch()
    df = md.load_wingbeats(tmp_path)
    assert len(df) == 1
    # And when the root *is* already the species dir, that should still work.
    flat = tmp_path / "flat" / "Ae. aegypti" / "D_session"
    flat.mkdir(parents=True)
    (flat / "y.wav").touch()
    df2 = md.load_wingbeats(tmp_path / "flat")
    assert len(df2) == 1


def test_load_all_concats_with_consistent_schema(tmp_path: Path):
    _write_esc50_fixture(tmp_path / "esc50")
    _write_humbugdb_fixture(tmp_path / "humbugdb")
    # Point loaders at the fixture roots by patching the module-level RAW.
    by_source = {
        "esc50": lambda: md.load_esc50(tmp_path / "esc50"),
        "humbugdb": lambda: md.load_humbugdb(tmp_path / "humbugdb"),
    }
    frames = [fn() for fn in by_source.values()]
    out = pd.concat(frames, ignore_index=True)
    assert list(out.columns) == list(md.COLUMNS)
    assert {"esc50", "humbugdb"} <= set(out["source"])


def test_load_all_drops_unlabeled_by_default(tmp_path: Path, monkeypatch):
    # Construct two rows where one has label=None to verify the drop path.
    df = pd.DataFrame({c: [None] * 2 for c in md.COLUMNS})
    df["label"] = ["mosquito", None]

    def fake_loader():
        return df

    monkeypatch.setattr(md, "LOADERS", {"fake": fake_loader})
    out = md.load_all(["fake"], drop_unlabeled=True)
    assert len(out) == 1
    out_keep = md.load_all(["fake"], drop_unlabeled=False)
    assert len(out_keep) == 2
