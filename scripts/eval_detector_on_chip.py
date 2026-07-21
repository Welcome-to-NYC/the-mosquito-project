"""Stream the 1D binary detector's test set to the ESP32 and verify on-chip.

Protocol (firmware/wingbeat_stream): host -> "INFR" + 1024 fp32; chip ->
"RSLT" + N_CLASSES fp32 + 1 byte argmax. Reports chip<->PyTorch agreement,
on-chip accuracy, and per-negative-source rejection (the honesty check) — all
measured on real silicon.

Run:
    python scripts/eval_detector_on_chip.py
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cnn_1d import CNN1D  # noqa: E402

PORT = "/dev/cu.wchusbserial1120"
BAUD = 460800
EXP = ROOT / "experiments" / "exp_audio_detector_1d"
INPUT_LEN = 1024


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=PORT)
    p.add_argument("--limit", type=int, default=1500)
    args = p.parse_args(argv)

    ck = torch.load(EXP / "best.pt", map_location="cpu", weights_only=False)
    classes = ck["classes"]; NC = len(classes)
    reply_bytes = 4 + NC * 4 + 1
    model = CNN1D(**ck["kwargs"]).eval(); model.load_state_dict(ck["model"])

    z = np.load(EXP / "stream_test.npz", allow_pickle=True)
    X = z["X"].astype(np.float32); y = z["y"].astype(np.int64); src = z["source"]
    if args.limit and args.limit < len(X):
        idx = np.random.default_rng(0).choice(len(X), args.limit, replace=False)
        X, y, src = X[idx], y[idx], src[idx]
    n = len(X)
    print(f"== {n} windows, {NC} classes {classes}")

    ser = serial.Serial(args.port, BAUD, timeout=2.0)
    time.sleep(0.3); ser.reset_input_buffer()

    chip_pred = np.zeros(n, np.int64); pt_pred = np.zeros(n, np.int64)
    chip_log = np.zeros((n, NC), np.float32); pt_log = np.zeros((n, NC), np.float32)
    t0 = time.time()
    for i in range(n):
        ser.write(b"INFR"); ser.write(X[i].astype("<f4").tobytes()); ser.flush()
        with torch.no_grad():
            pl = model(torch.from_numpy(X[i]).unsqueeze(0).unsqueeze(0)).numpy()[0]
        pt_log[i] = pl; pt_pred[i] = int(pl.argmax())
        r = bytearray(); dl = time.time() + 5
        while len(r) < reply_bytes:
            if time.time() > dl:
                raise TimeoutError(f"sample {i}: got {len(r)}/{reply_bytes}: {bytes(r)!r}")
            c = ser.read(reply_bytes - len(r))
            if c:
                r.extend(c)
        if r[:4] != b"RSLT":
            raise RuntimeError(f"sample {i}: bad magic {bytes(r[:4])!r}")
        chip_log[i] = struct.unpack(f"<{NC}f", bytes(r[4:4+NC*4])); chip_pred[i] = r[4+NC*4]
        if (i+1) % 200 == 0 or i == n-1:
            rate = (i+1)/(time.time()-t0)
            print(f"   {i+1}/{n}  {rate:.1f}/s")
    ser.close()

    agree = float((chip_pred == pt_pred).mean())
    maxdiff = float(np.abs(chip_log - pt_log).max())
    chip_acc = float((chip_pred == y).mean())
    print("\n" + "="*56)
    print("ON-CHIP DETECTOR VERIFICATION")
    print("="*56)
    print(f"  chip <-> PyTorch agreement : {int((chip_pred==pt_pred).sum())}/{n} = {agree:.4f}")
    print(f"  max |chip - pt| logit      : {maxdiff:.6f}")
    print(f"  on-chip accuracy           : {chip_acc:.4f}")
    print(f"  on-chip mosquito recall    : {float((chip_pred[y==1]==1).mean()):.4f}")
    for s in ["humbug_bg", "insectsound_fly"]:
        mask = (src == s) & (y == 0)
        if mask.sum():
            print(f"  on-chip reject {s:16s}: {float((chip_pred[mask]==0).mean()):.4f}")
    # confusion
    cm = np.zeros((NC, NC), int)
    for t, pr in zip(y, chip_pred):
        cm[t, pr] += 1
    print("  on-chip confusion (row=true):")
    print("            " + "  ".join(f"{c[:8]:>8}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:>10} " + "  ".join(f"{v:>8d}" for v in cm[i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
