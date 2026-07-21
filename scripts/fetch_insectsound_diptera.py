"""Fetch only the Diptera (fly) classes from the InsectSound1000 Kaggle mirror.

InsectSound1000 (Branding et al., Sci. Data 2024) is a 91 GB monolithic
archive of 12 insect species. We only want the three DIPTERA families, which
are the biologically meaningful non-mosquito negatives (Nematocera whose
wingbeat tones overlap the mosquito band):

    Episyrphus_balteatus     hoverfly       (Syrphidae)
    Aphidoletes_aphidimyza   gall midge     (Cecidomyiidae)
    Bradysia_difformis       fungus gnat    (Sciaridae)

The OpenAgrar original is behind a proof-of-work anti-bot wall, so we use the
Kaggle mirror (hesi0ne/insectsound1000) which supports per-file download.

Strategy (resumable):
  1. Enumerate the full file manifest via paged listing (~830 pages, ~7 min),
     cache it to disk so we never re-page.
  2. Filter to the 3 Diptera species, cap N files per species.
  3. Download in parallel (thread pool), skipping files already on disk.

Run:
    python scripts/fetch_insectsound_diptera.py --per-species 800 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "hesi0ne/insectsound1000"
OUT_DIR = ROOT / "data" / "raw" / "insectsound1000"
MANIFEST = OUT_DIR / "_manifest.json"

DIPTERA = ("Episyrphus_balteatus", "Aphidoletes_aphidimyza", "Bradysia_difformis")


def _load_token() -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("KAGGLE_API_TOKEN"):
                os.environ["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return
    raise SystemExit("KAGGLE_API_TOKEN not found in env or .env")


CKPT = OUT_DIR / "_manifest_ckpt.json"


def _list_page(api, token, retries=6):
    """One page with exponential backoff on 429 / transient errors."""
    for attempt in range(retries):
        try:
            return api.dataset_list_files(DATASET, page_token=token, page_size=200)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg or "5" == msg[:1]:
                wait = min(60, 15 * (attempt + 1))
                print(f"   rate-limited, backoff {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(5)
    raise RuntimeError("page fetch failed after retries")


def enumerate_manifest(api) -> list[str]:
    """Page through the full file list, cache all names to MANIFEST.

    Resumable: checkpoints (names, next_token) to CKPT so a 429 crash can
    pick up where it left off instead of restarting.
    """
    if MANIFEST.exists():
        names = json.loads(MANIFEST.read_text())
        print(f"== manifest cached: {len(names)} files")
        return names
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    token = None
    if CKPT.exists():
        ck = json.loads(CKPT.read_text())
        names, token = ck["names"], ck["token"]
        print(f"== resuming enumeration from {len(names)} files")
    print("== enumerating full file list (paged, throttled)")
    page = 0
    t0 = time.time()
    while True:
        res = _list_page(api, token)
        batch = [f.name for f in res.files]
        names.extend(batch)
        page += 1
        token = res.nextPageToken
        if page % 25 == 0:
            print(f"   page {page}: {len(names)} files ({time.time()-t0:.0f}s)")
            CKPT.write_text(json.dumps({"names": names, "token": token}))
        time.sleep(0.4)  # stay under the rate limit
        if not token or not batch:
            break
    MANIFEST.write_text(json.dumps(names))
    CKPT.unlink(missing_ok=True)
    print(f"== manifest saved: {len(names)} files in {time.time()-t0:.0f}s")
    return names


def _rec_id(name: str) -> str:
    """Recording id = filename minus _s<seg>_ch<n>.wav."""
    m = re.match(r"(.+)_s\d+_ch\d+\.wav$", Path(name).name)
    return m.group(1) if m else Path(name).stem


def pick_diptera(names: list[str], per_species: int, channel: str = "ch0",
                 seg_per_rec: int = 6) -> list[str]:
    """Select up to `per_species` files per species, spread across recordings.

    One InsectSound recording is cut into 60+ contiguous 2.5 s segments
    (s0..s60), each a separate file. Taking all segments of one recording
    gives poor diversity, so we take at most `seg_per_rec` segments per
    recording and spread across as many recordings as possible — this keeps
    the recording-level train/val/test split meaningful.
    """
    out: list[str] = []
    for sp in DIPTERA:
        cand = [n for n in names
                if f"_{sp}_" in n and (not channel or n.endswith(f"_{channel}.wav"))]
        by_rec: dict[str, list[str]] = {}
        for n in cand:
            by_rec.setdefault(_rec_id(n), []).append(n)
        recs = sorted(by_rec)
        chosen: list[str] = []
        # round-robin over recordings so we breadth-first sample sessions
        for k in range(seg_per_rec):
            for r in recs:
                if k < len(by_rec[r]) and len(chosen) < per_species:
                    chosen.append(sorted(by_rec[r])[k])
            if len(chosen) >= per_species:
                break
        print(f"   {sp}: {len(chosen)} files from {len(recs)} recordings")
        out.extend(chosen)
    return out


def download_one(api, name: str) -> tuple[str, bool, str]:
    local = OUT_DIR / Path(name).name
    if local.exists() and local.stat().st_size > 0:
        return name, True, "exists"
    for attempt in range(6):
        try:
            api.dataset_download_file(DATASET, name, path=str(OUT_DIR), force=True, quiet=True)
            if local.exists() and local.stat().st_size > 0:
                return name, True, "ok"
            alt = OUT_DIR / name
            if alt.exists():
                alt.rename(local)
                return name, True, "ok(moved)"
            return name, False, "missing-after-dl"
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                wait = min(90, 30 + 20 * attempt)
                time.sleep(wait)
                continue
            if attempt >= 3:
                return name, False, f"err:{e}"
            time.sleep(3 * (attempt + 1))
    return name, False, "err:429-exhausted"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-species", type=int, default=400)
    p.add_argument("--workers", type=int, default=1, help="concurrent downloads (keep low; Kaggle rate-limits)")
    p.add_argument("--delay", type=float, default=1.8, help="seconds between dispatches")
    p.add_argument("--seg-per-rec", type=int, default=6)
    p.add_argument("--channel", default="ch0", help="keep one mic channel only")
    p.add_argument("--species", default=None,
                   help="comma list to restrict which Diptera species to fetch "
                        "(default: all 3). Use to target only the missing ones.")
    args = p.parse_args(argv)

    global DIPTERA
    if args.species:
        wanted = tuple(s.strip() for s in args.species.split(","))
        DIPTERA = tuple(s for s in DIPTERA if s in wanted) or wanted

    _load_token()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    names = enumerate_manifest(api)
    targets = pick_diptera(names, args.per_species, args.channel, args.seg_per_rec)
    print(f"== downloading {len(targets)} files, {args.workers} workers, {args.delay}s delay -> {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = fail = 0
    t0 = time.time()
    # Plain serial loop — the kagglesdk client is NOT thread-safe, so a
    # ThreadPoolExecutor deadlocks. Direct sequential calls run ~2.6 s/file.
    for i, n in enumerate(targets, 1):
        name, ok, msg = download_one(api, n)
        if ok:
            done += 1
        else:
            fail += 1
            if fail <= 30:
                print(f"   FAIL {Path(name).name}: {msg}", flush=True)
        if i % 25 == 0 or i == len(targets):
            rate = i / (time.time() - t0)
            eta = (len(targets) - i) / rate if rate else 0
            print(f"   {i}/{len(targets)}  ok={done} fail={fail}  "
                  f"{rate:.2f} files/s  ETA {eta:.0f}s", flush=True)

    # Final tally per species.
    print("\n== on-disk tally")
    for sp in DIPTERA:
        n = len(list(OUT_DIR.glob(f"*_{sp}_*_{args.channel}.wav")))
        print(f"   {sp}: {n} wav")
    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*.wav")) / 1e6
    print(f"   total: {total_mb:.0f} MB, {done} ok / {fail} fail")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
