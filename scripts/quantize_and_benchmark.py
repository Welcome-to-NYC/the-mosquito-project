"""Compare deployment-relevant numbers across experiments.

For every ``--exp`` passed, the script reports:

* parameter count
* fp32 state-dict size on disk (KB)
* hypothetical INT8 weight size (KB) — simply ``params × 1 byte``,
  since on ESP32 the actual quantization is handled by the TFLite
  Micro runtime once the model is exported in W11.5
* fp32 CPU inference latency per batch (mean over N batches)
* fp32 test accuracy / mosquito F1 / cross-source mosq recall

Why CPU and not MPS for latency: the deployment target (ESP32) is
CPU-only, and we want a number that approximates the production
inference cost. M1 Pro CPU is ~10× faster than ESP32, so divide the
reported number by ~10 to get a rough deployment estimate.

Run::

    python scripts/quantize_and_benchmark.py \\
        --exp exp_cnn_1d_baseline exp_cnn_1d_balanced exp_cnn_1d_aug \\
              exp_physics_informed_w6 exp_cnn1d_tiny_distilled
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_model_for_exp(exp_dir: Path, n_classes: int = 3) -> tuple[nn.Module, dict]:
    import yaml

    from src.training.config import load_config
    from src.training.train import _build_model

    cfg_path = exp_dir / "config_used.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")
    tmp = exp_dir / "_bench_cfg.yaml"
    tmp.write_text(cfg_path.read_text())
    try:
        cfg = load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    model = _build_model(cfg, n_classes)
    state = torch.load(exp_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, {"model_name": cfg.model.name, "exp_name": cfg.exp.name}


def _state_dict_kb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024.0


def _int8_kb(n_params: int) -> float:
    return n_params / 1024.0


def _measure_latency(model: nn.Module, x: torch.Tensor, n_warmup: int = 5, n_iter: int = 30) -> float:
    """Mean per-batch inference time in milliseconds (CPU, single thread)."""
    model.eval()
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(x)
        return (time.perf_counter() - t0) * 1000.0 / n_iter


def _evaluate_acc_and_per_source(model: nn.Module, exp_dir: Path) -> dict:
    """Run model on test.npz on CPU and return overall accuracy + per-source
    mosquito recall (the cross-source-shortcut metric).
    """
    from src.evaluation.metrics import evaluate

    z = np.load(ROOT / "data" / "processed" / "test.npz", allow_pickle=True)
    X = z["X"]
    y = z["y"]
    classes = list(z["classes"])
    sources = z["source"]

    preds: list[np.ndarray] = []
    bs = 512
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            chunk = torch.as_tensor(X[i : i + bs], dtype=torch.float32).unsqueeze(1)
            preds.append(model(chunk).argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds)

    metrics = evaluate(y, y_pred, classes=classes)
    mosq_idx = classes.index("mosquito") if "mosquito" in classes else 1
    per_src = {}
    for s in sorted(set(sources)):
        mask = (sources == s) & (y == mosq_idx)
        if not mask.any():
            continue
        per_src[s] = float(((y_pred == mosq_idx) & mask).sum() / mask.sum())

    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mosq_f1": next(r["f1"] for r in metrics["per_class"] if r["class"] == "mosquito"),
        "bg_f1": next(r["f1"] for r in metrics["per_class"] if r["class"] == "background"),
        "insect_f1": next(r["f1"] for r in metrics["per_class"] if r["class"] == "non_mosquito_insect"),
        "auc_macro": metrics["auc_macro"],
        "wing_mosq_recall": per_src.get("wingbeats", float("nan")),
        "humbug_mosq_recall": per_src.get("humbugdb", float("nan")),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", required=True,
                        help="experiment dir names under experiments/")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "benchmark.json")
    parser.add_argument("--latency-batch", type=int, default=128)
    parser.add_argument("--latency-iters", type=int, default=30)
    parser.add_argument("--input-len", type=int, default=1024)
    args = parser.parse_args(argv)

    rows = []
    print(f"{'exp':<32s} {'params':>8s} {'fp32 KB':>8s} {'int8 KB':>8s} "
          f"{'lat ms':>8s} {'acc':>7s} {'mosq F1':>7s} {'bg F1':>7s} {'ins F1':>7s} "
          f"{'Wing':>7s} {'HBdb':>7s} {'gap':>7s}")
    print("-" * 130)
    for exp_name in args.exp:
        exp_dir = ROOT / "experiments" / exp_name
        if not exp_dir.is_dir():
            print(f"skip {exp_name}: not a directory")
            continue
        model, info = _load_model_for_exp(exp_dir)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fp32_kb = _state_dict_kb(model)
        int8_kb = _int8_kb(n_params)

        x = torch.randn(args.latency_batch, 1, args.input_len)
        latency_ms = _measure_latency(model, x, n_iter=args.latency_iters)

        eval_metrics = _evaluate_acc_and_per_source(model, exp_dir)

        gap_pp = 100 * (eval_metrics["wing_mosq_recall"] - eval_metrics["humbug_mosq_recall"])

        row = {
            "exp": exp_name,
            "model": info["model_name"],
            "n_params": n_params,
            "fp32_kb": fp32_kb,
            "int8_kb": int8_kb,
            "latency_ms_per_batch": latency_ms,
            "latency_ms_per_sample": latency_ms / args.latency_batch,
            **eval_metrics,
            "domain_gap_pp": gap_pp,
        }
        rows.append(row)
        print(
            f"{exp_name:<32s} {n_params:>8,d} {fp32_kb:>8.1f} {int8_kb:>8.1f} "
            f"{latency_ms:>8.2f} "
            f"{eval_metrics['accuracy']:>7.3f} "
            f"{eval_metrics['mosq_f1']:>7.3f} "
            f"{eval_metrics['bg_f1']:>7.3f} "
            f"{eval_metrics['insect_f1']:>7.3f} "
            f"{eval_metrics['wing_mosq_recall']:>7.3f} "
            f"{eval_metrics['humbug_mosq_recall']:>7.3f} "
            f"{gap_pp:>6.1f}pp"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\n-- saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
