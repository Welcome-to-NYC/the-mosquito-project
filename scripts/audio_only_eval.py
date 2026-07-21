"""Cross-modality evaluation for the audio-only experiment.

The audio-only-trained model is evaluated on two test sets:

1. The audio-only test split (`data/processed_audio/test.npz`) — intrinsic
   performance on the modality it was trained on.
2. The mixed test split (`data/processed/test.npz`) — sliced by source so
   we can see how the audio-only model behaves when handed an *optical*
   Wingbeats signal. The expected outcome is "it doesn't know what to do"
   if the model truly relied on modality-invariant wingbeat features
   from the audio side; if it confidently predicts mosquito anyway, the
   "optical = mosquito" shortcut hypothesis from the mixed-training case
   is partially vindicated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate  # noqa: E402


def _load_model(exp_dir: Path, n_classes: int):
    import yaml
    from src.training.config import load_config
    from src.training.train import _build_model
    from src.utils.device import get_device

    tmp = exp_dir / "_aoe_cfg.yaml"
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


def _predict(model, device, X: np.ndarray) -> np.ndarray:
    bs = 256
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            chunk = torch.as_tensor(X[i : i + bs], dtype=torch.float32, device=device).unsqueeze(1)
            out.append(torch.softmax(model(chunk), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _eval_one_test(model, device, test_path: Path, label: str, classes_expected: list[str] | None = None):
    z = np.load(test_path, allow_pickle=True)
    X = z["X"]
    y = z["y"]
    classes = list(z["classes"])
    sources = z["source"]

    probs = _predict(model, device, X)
    y_pred = probs.argmax(axis=-1)
    metrics = evaluate(y, y_pred, y_score=probs, classes=classes)
    print(f"\n=== {label}: {test_path.name} ===")
    print(f"   shape {X.shape}, sources {sorted(set(sources))}")
    print(f"   accuracy {metrics['accuracy']:.4f}  macro_f1 {metrics['macro_f1']:.4f}  "
          f"auc_macro {metrics['auc_macro']:.4f}")
    for r in metrics["per_class"]:
        print(f"   {r['class']:<22s}  prec {r['precision']:.3f}  rec {r['recall']:.3f}  "
              f"f1 {r['f1']:.3f}  (n={r['support']})")

    # Per-source breakdown.
    mosq_idx = classes.index("mosquito")
    print("   per-source mosquito recall + prediction distribution:")
    for s in sorted(set(sources)):
        mask = sources == s
        sub_y = y[mask]
        sub_p = y_pred[mask]
        n = int(mask.sum())
        mosq_mask = mask & (y == mosq_idx)
        mosq_n = int(mosq_mask.sum())
        if mosq_n:
            rec = ((y_pred == mosq_idx) & mosq_mask).sum() / mosq_n
            print(f"     {s:<10s} n={n:>7d}  mosq support {mosq_n:>7d}  recall {rec:.4f}")
        else:
            print(f"     {s:<10s} n={n:>7d}  (no mosquito in this slice)")
        # What does the model predict when given samples from this source?
        pred_dist = np.bincount(sub_p, minlength=len(classes))
        pred_pct = pred_dist / max(pred_dist.sum(), 1)
        dist_str = ", ".join(f"{c}={pred_pct[i]*100:.1f}%" for i, c in enumerate(classes))
        print(f"                              pred dist: [{dist_str}]")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="exp_cnn_1d_audio_only")
    parser.add_argument("--audio-test", default="data/processed_audio/test.npz")
    parser.add_argument("--mixed-test", default="data/processed/test.npz")
    args = parser.parse_args(argv)

    exp_dir = ROOT / "experiments" / args.exp
    if not exp_dir.is_dir():
        raise SystemExit(f"missing {exp_dir}")

    # We need to know n_classes — read it from any test set we'll touch.
    z = np.load(ROOT / args.audio_test, allow_pickle=True)
    n_classes = len(list(z["classes"]))

    print(f"-- loading {args.exp}")
    model, device = _load_model(exp_dir, n_classes)

    _eval_one_test(model, device, ROOT / args.audio_test, "INTRINSIC (audio-only test)")
    _eval_one_test(model, device, ROOT / args.mixed_test, "CROSS-MODALITY (mixed test, incl. Wingbeats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
