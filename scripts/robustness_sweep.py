"""Test-time noise robustness sweep.

Applies WingbeatAugment-style noise at a sequence of SNRs to the test set
and reports how the model's mosquito F1 / accuracy degrades. Documents
the operating envelope — at what noise level does the model stop being
useful?

Run::

    python scripts/robustness_sweep.py --exp exp_cnn1d_tiny_distilled exp_physics_informed_w6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augment import add_pink_noise, add_white_noise  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402


def _load_model(exp_dir: Path, n_classes: int):
    import yaml
    from src.training.config import load_config
    from src.training.train import _build_model
    from src.utils.device import get_device

    tmp = exp_dir / "_rob_cfg.yaml"
    tmp.write_text((exp_dir / "config_used.yaml").read_text())
    try:
        cfg = load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    device = get_device()
    model = _build_model(cfg, n_classes).to(device)
    ckpt = torch.load(exp_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device


def _apply_noise(X: np.ndarray, snr_db: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    if not np.isfinite(snr_db):
        return X.astype(np.float32, copy=False)
    out = np.empty_like(X, dtype=np.float32)
    fn = add_white_noise if kind == "white" else add_pink_noise
    for i in range(len(X)):
        out[i] = fn(X[i], snr_db=snr_db, rng=rng)
    return out


def _predict(model, device, X: np.ndarray) -> np.ndarray:
    bs = 256
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            chunk = torch.as_tensor(X[i : i + bs], dtype=torch.float32, device=device).unsqueeze(1)
            out.append(model(chunk).argmax(dim=-1).cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="+", required=True)
    parser.add_argument("--snrs", type=float, nargs="+",
                        default=[float("inf"), 20.0, 10.0, 0.0, -10.0])
    parser.add_argument("--kind", choices=["white", "pink"], default="white")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "robustness.json")
    args = parser.parse_args(argv)

    z = np.load(ROOT / "data" / "processed" / "test.npz", allow_pickle=True)
    X = z["X"]
    y = z["y"]
    classes = list(z["classes"])
    sources = z["source"]
    n_classes = len(classes)

    results: list[dict] = []
    for name in args.exp:
        exp_dir = ROOT / "experiments" / name
        if not exp_dir.is_dir():
            print(f"skip {name}")
            continue
        print(f"\n=== {name} ({args.kind} noise) ===")
        print(f"{'SNR (dB)':>10s}  {'acc':>7s}  {'mosq F1':>8s}  {'bg F1':>7s}  {'ins F1':>7s}  "
              f"{'HBdb rec':>9s}")

        model, device = _load_model(exp_dir, n_classes)
        for snr in args.snrs:
            rng = np.random.default_rng(args.seed)
            X_noisy = _apply_noise(X, snr, args.kind, rng)
            y_pred = _predict(model, device, X_noisy)
            metrics = evaluate(y, y_pred, classes=classes)
            mosq_idx = classes.index("mosquito")
            humbug_mask = (sources == "humbugdb") & (y == mosq_idx)
            humbug_recall = float(((y_pred == mosq_idx) & humbug_mask).sum() / max(humbug_mask.sum(), 1))

            per_class = {r["class"]: r["f1"] for r in metrics["per_class"]}
            snr_str = "clean" if not np.isfinite(snr) else f"{snr:.0f}"
            print(f"{snr_str:>10s}  "
                  f"{metrics['accuracy']:>7.4f}  "
                  f"{per_class['mosquito']:>8.4f}  "
                  f"{per_class['background']:>7.4f}  "
                  f"{per_class['non_mosquito_insect']:>7.4f}  "
                  f"{humbug_recall:>9.4f}")
            results.append({
                "exp": name,
                "kind": args.kind,
                "snr_db": snr if np.isfinite(snr) else None,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "mosq_f1": per_class["mosquito"],
                "bg_f1": per_class["background"],
                "insect_f1": per_class["non_mosquito_insect"],
                "humbug_mosq_recall": humbug_recall,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n-- saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
