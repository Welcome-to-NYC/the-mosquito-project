"""End-to-end on-silicon evaluation of the W11 distilled student.

Stream the entire `data/processed/test.npz` to the ESP32 over USB serial
one window at a time, collect the chip's logits + argmax, and compare
against (a) PyTorch reference inference on the same model and (b) ground
truth labels. Prints a final report:

    on-chip accuracy             : X / N (Y.YY %)
    PyTorch on-laptop accuracy   : X / N (Y.YY %)
    chip ↔ PyTorch agreement     : X / N (Y.YY %)
    max |chip_logit - pt_logit|  : Z
    mean throughput              : K samples / sec
    per-class confusion (chip)   : ...

Protocol matches `firmware/wingbeat_stream/wingbeat_stream.ino`:
  host -> chip : 4096 bytes (1024 fp32 little-endian)
  chip -> host :   17 bytes ("RSLT" + 3 fp32 + 1 byte argmax)

Wait for the chip's "READY\\n" banner on stdout-style serial before
starting the stream.
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

DEFAULT_PORT = "/dev/cu.wchusbserial1120"
DEFAULT_BAUD = 460800
DEFAULT_CKPT = ROOT / "experiments" / "exp_cnn1d_tiny_distilled" / "best.pt"
DEFAULT_TEST = ROOT / "data" / "processed" / "test.npz"

STUDENT_KWARGS = dict(
    n_classes=3, in_channels=1,
    channels=[8, 16, 24], kernel_sizes=[7, 5, 3],
    fc_hidden=16, dropout=0.2,
)

N_CLASSES = 3
INPUT_LEN = 1024
REPLY_BYTES = 4 + N_CLASSES * 4 + 1  # "RSLT" + 3 fp32 + 1 byte = 17


def wait_for_ready(ser: serial.Serial, timeout: float = 10.0) -> str:
    """Drain serial until we see a line ending with 'READY'."""
    deadline = time.time() + timeout
    buf = bytearray()
    while time.time() < deadline:
        chunk = ser.read(256)
        if chunk:
            buf.extend(chunk)
            text = buf.decode("utf-8", errors="replace")
            if "READY" in text:
                return text
        else:
            time.sleep(0.05)
    raise TimeoutError(f"chip did not report READY within {timeout}s. got: {bytes(buf)!r}")


def reset_chip(ser: serial.Serial) -> None:
    """Toggle RTS to hard-reset the ESP32 (matches esptool's hard_reset)."""
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    time.sleep(0.05)
    ser.reset_input_buffer()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p.add_argument("--limit", type=int, default=None,
                   help="only evaluate the first N test samples (for smoke runs)")
    p.add_argument("--stratified", type=int, default=3000,
                   help="cap at N samples drawn stratified-by-class (default 3000; "
                        "use --stratified 0 to disable and evaluate the whole set)")
    p.add_argument("--no-pytorch", action="store_true",
                   help="skip PyTorch reference inference (chip-only)")
    args = p.parse_args(argv)

    # Force line-buffered stdout so progress prints show up when tee'd.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # ----- load test set -----
    print(f"== loading {args.test}")
    z = np.load(args.test, allow_pickle=True)
    X_all = z["X"].astype(np.float32)
    y_all = z["y"].astype(np.int64)
    classes = list(z["classes"])
    print(f"   full test set: {len(X_all)} samples, classes {classes}")

    # ----- subsample -----
    if args.limit is not None:
        sel = np.arange(min(args.limit, len(X_all)))
    elif args.stratified > 0 and args.stratified < len(X_all):
        rng = np.random.default_rng(42)
        per_class = max(1, args.stratified // len(classes))
        sel_parts = []
        for c in range(len(classes)):
            idx = np.where(y_all == c)[0]
            take = min(per_class, len(idx))
            sel_parts.append(rng.choice(idx, size=take, replace=False))
        sel = np.concatenate(sel_parts)
        rng.shuffle(sel)
        print(f"   stratified sample: {per_class} per class -> {len(sel)} total")
    else:
        sel = np.arange(len(X_all))
    X = X_all[sel]
    y = y_all[sel]
    n = len(X)
    print(f"   evaluating {n} samples")

    # ----- prepare PyTorch model (sample-by-sample to keep memory tiny) -----
    model = None
    if not args.no_pytorch:
        print(f"== loading PyTorch model from {args.ckpt}")
        model = CNN1D(**STUDENT_KWARGS).eval()
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])

    # ----- open serial, drain any stale bytes -----
    # We use a self-syncing 4-byte magic header on every frame ("INFR"),
    # so the chip can recover from any prior state without a hard reset.
    print(f"== opening {args.port} @ {args.baud}")
    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.2)
    drained = ser.read(8192)
    if drained:
        try:
            print(f"   drained {len(drained)} stale bytes: {drained[:80]!r}...")
        except Exception:
            pass
    ser.reset_input_buffer()

    # ----- stream samples (interleaved: chip + PyTorch per sample) -----
    print(f"== streaming {n} samples")
    chip_preds = np.zeros(n, dtype=np.int64)
    chip_logits = np.zeros((n, N_CLASSES), dtype=np.float32)
    pt_preds = np.zeros(n, dtype=np.int64) if model is not None else None
    pt_logits = np.zeros((n, N_CLASSES), dtype=np.float32) if model is not None else None
    t_start = time.time()
    t_last_report = t_start

    for i in range(n):
        payload = X[i].astype("<f4").tobytes()  # 4096 bytes little-endian fp32
        assert len(payload) == INPUT_LEN * 4

        ser.write(b"INFR")
        ser.write(payload)
        ser.flush()

        # While chip is computing (~67 ms), run PyTorch on this sample on the host.
        if model is not None:
            with torch.no_grad():
                xt = torch.from_numpy(X[i]).unsqueeze(0).unsqueeze(0)  # (1, 1, 1024)
                pl = model(xt).numpy()[0]
            pt_logits[i] = pl
            pt_preds[i] = int(pl.argmax())

        # Read 17-byte reply with a generous per-sample timeout.
        reply = bytearray()
        deadline = time.time() + 5.0
        while len(reply) < REPLY_BYTES:
            if time.time() > deadline:
                raise TimeoutError(f"sample {i}: reply timeout, got {len(reply)}/{REPLY_BYTES} bytes: {bytes(reply)!r}")
            chunk = ser.read(REPLY_BYTES - len(reply))
            if chunk:
                reply.extend(chunk)
        if reply[:4] != b"RSLT":
            raise RuntimeError(f"sample {i}: bad magic {bytes(reply[:4])!r}, sync lost")
        logits = struct.unpack("<3f", bytes(reply[4:16]))
        pred = reply[16]
        chip_logits[i] = logits
        chip_preds[i] = pred

        if time.time() - t_last_report > 2.0 or i == n - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_s = (n - i - 1) / rate if rate > 0 else 0
            print(f"   {i+1:5d}/{n}  {rate:5.1f} samp/s  elapsed {elapsed:5.1f}s  ETA {eta_s:5.1f}s")
            t_last_report = time.time()

    elapsed_total = time.time() - t_start
    ser.close()

    # ----- report -----
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    chip_acc = float((chip_preds == y).mean())
    print(f"on-chip accuracy             : {chip_acc:.4f}  ({(chip_preds==y).sum()}/{n})")
    if pt_preds is not None:
        pt_acc = float((pt_preds == y).mean())
        agree = float((chip_preds == pt_preds).mean())
        max_logit_diff = float(np.abs(chip_logits - pt_logits).max())
        print(f"PyTorch on-laptop accuracy   : {pt_acc:.4f}  ({(pt_preds==y).sum()}/{n})")
        print(f"chip ↔ PyTorch agreement     : {agree:.4f}  ({(chip_preds==pt_preds).sum()}/{n})")
        print(f"max |chip_logit - pt_logit|  : {max_logit_diff:.6f}")
    print(f"mean throughput              : {n / elapsed_total:.1f} samples/sec  ({elapsed_total*1000/n:.2f} ms/sample wall)")

    # per-class breakdown (chip)
    print()
    print("per-class on-chip recall:")
    for c, name in enumerate(classes):
        mask = (y == c)
        if mask.sum():
            rec = float((chip_preds[mask] == c).mean())
            print(f"   {name:>22s}  recall {rec:.4f}  ({(chip_preds[mask]==c).sum()}/{mask.sum()})")

    # confusion (chip)
    print()
    print("on-chip confusion matrix (row=true, col=pred):")
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y, chip_preds):
        cm[t, p] += 1
    header = "                       " + "  ".join(f"{c[:6]:>6s}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = "  ".join(f"{v:>6d}" for v in cm[i])
        print(f"   {c:>20s}   {row}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
