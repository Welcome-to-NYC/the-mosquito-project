"""Walk data/raw/ and report what's actually on disk.

Prints, per dataset:
  * presence (placeholder-only vs populated)
  * file counts by extension
  * total size on disk
  * a sampled audio summary: dominant sample rate, channel count, mean
    duration. Sampling avoids reading every file when datasets are huge.

Run it after a download to sanity-check the result, or after preprocessing
to confirm input shape assumptions hold across sources.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "configs" / "datasets.yaml"


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def fmt_size(n: int) -> str:
    """Human-readable byte size (3 significant digits)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:6.2f} {u}"
        f /= 1024
    return f"{f:.2f} TB"


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.name != ".gitkeep":
            yield p


def summarize_dir(root: Path, audio_sample: int = 30) -> dict:
    """Return a stats dict for the contents of ``root``."""
    n = 0
    by_ext: collections.Counter[str] = collections.Counter()
    total_bytes = 0
    audio_files: list[Path] = []

    for p in iter_files(root):
        n += 1
        ext = p.suffix.lower()
        by_ext[ext] += 1
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass
        if ext in AUDIO_EXTS:
            audio_files.append(p)

    audio_summary: dict = {}
    if audio_files:
        audio_summary = _summarize_audio(audio_files, audio_sample)

    return {
        "file_count": n,
        "by_ext": dict(by_ext.most_common()),
        "total_bytes": total_bytes,
        "audio": audio_summary,
    }


def _summarize_audio(files: list[Path], sample_size: int) -> dict:
    """Sample up to ``sample_size`` audio files and aggregate rate/channels/duration.

    Imports soundfile lazily so this script still works (with empty audio
    section) on machines that don't have audio libraries installed.
    """
    try:
        import soundfile as sf  # noqa: WPS433
    except ImportError:
        return {"error": "soundfile not installed; pip install soundfile"}

    rng = random.Random(0)
    sample = files if len(files) <= sample_size else rng.sample(files, sample_size)

    sample_rates: collections.Counter[int] = collections.Counter()
    channels: collections.Counter[int] = collections.Counter()
    durations: list[float] = []
    failed = 0

    for p in sample:
        try:
            info = sf.info(str(p))
        except Exception:  # noqa: BLE001  (we want any failure to count)
            failed += 1
            continue
        sample_rates[info.samplerate] += 1
        channels[info.channels] += 1
        if info.samplerate:
            durations.append(info.frames / info.samplerate)

    return {
        "sampled": len(sample),
        "failed": failed,
        "sample_rates": dict(sample_rates.most_common()),
        "channels": dict(channels.most_common()),
        "duration_mean_s": (sum(durations) / len(durations)) if durations else 0.0,
        "duration_min_s": min(durations) if durations else 0.0,
        "duration_max_s": max(durations) if durations else 0.0,
    }


def known_datasets() -> dict:
    if not CONFIG.exists():
        return {}
    with CONFIG.open() as f:
        return yaml.safe_load(f).get("datasets", {})


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-sample", type=int, default=30, help="audio files sampled per dataset")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = known_datasets()
    seen: set[str] = set()
    rows: list[tuple[str, dict, str]] = []

    # Walk every directory under data/raw/, including ones that aren't in
    # the registry (so a manually-dropped dataset still shows up).
    if RAW.exists():
        for d in sorted(RAW.iterdir()):
            if d.is_dir():
                rows.append((d.name, summarize_dir(d, args.audio_sample), cfg.get(d.name, {}).get("description", "")))
                seen.add(d.name)

    # Mention any registered datasets that haven't been downloaded yet.
    for name in cfg:
        if name not in seen:
            rows.append((name, {"file_count": 0, "by_ext": {}, "total_bytes": 0, "audio": {}}, cfg[name].get("description", "")))

    print(f"== data/raw inventory ({RAW})\n")
    for name, stats, desc in rows:
        print(f"-- {name}")
        if desc:
            print(f"   {desc}")
        n = stats["file_count"]
        if n == 0:
            print(f"   (empty — placeholder only)\n")
            continue
        print(f"   files       : {n}")
        print(f"   total size  : {fmt_size(stats['total_bytes'])}")
        ext_str = ", ".join(f"{e or '<none>'}={c}" for e, c in stats["by_ext"].items())
        print(f"   extensions  : {ext_str}")
        a = stats["audio"]
        if a:
            if a.get("error"):
                print(f"   audio       : {a['error']}")
            else:
                rates = ", ".join(f"{r}Hz×{c}" for r, c in a["sample_rates"].items())
                chans = ", ".join(f"{c}ch×{n}" for c, n in a["channels"].items())
                print(
                    f"   audio       : sampled {a['sampled']} / failed {a['failed']}; "
                    f"rates [{rates}]; channels [{chans}]; "
                    f"dur mean {a['duration_mean_s']:.2f}s "
                    f"(range {a['duration_min_s']:.2f}-{a['duration_max_s']:.2f}s)"
                )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
