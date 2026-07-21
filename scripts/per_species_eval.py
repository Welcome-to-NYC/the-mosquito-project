"""Per-species mosquito recall for the test set.

The test.npz carries a ``species`` column that the preprocess pipeline
populated from HumBugDB metadata and Wingbeats directory names. This
script breaks recall down by species so we can see, for the deployment
target (Hong Kong has Ae. albopictus, Ae. aegypti, Cu. quinquefasciatus
as the dominant mosquitoes), whether the model handles each one or whether
performance is dragged up by an easy species.
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


def _load_model(exp_dir: Path, n_classes: int):
    import yaml
    from src.training.config import load_config
    from src.training.train import _build_model
    from src.utils.device import get_device

    tmp = exp_dir / "_sp_cfg.yaml"
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
            out.append(model(chunk).argmax(dim=-1).cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="+", required=True)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "experiments" / "per_species.json")
    args = parser.parse_args(argv)

    z = np.load(ROOT / "data" / "processed" / "test.npz", allow_pickle=True)
    X = z["X"]
    y = z["y"]
    classes = list(z["classes"])
    species = z["species"]
    sources = z["source"]
    n_classes = len(classes)
    mosq_idx = classes.index("mosquito")

    # Unique mosquito species in the test set.
    mosq_mask = y == mosq_idx
    species_arr = np.array(species)
    unique_species = sorted(set(species_arr[mosq_mask].tolist()))
    print(f"-- mosquito species in test: {len(unique_species)}")
    for s in unique_species:
        n = int(((species_arr == s) & mosq_mask).sum()) if s else 0
        if not s:
            continue
        print(f"     {s:30s} {n:>7d} segments")

    results = []
    for name in args.exp:
        exp_dir = ROOT / "experiments" / name
        if not exp_dir.is_dir():
            print(f"skip {name}")
            continue
        print(f"\n=== {name} ===")
        model, device = _load_model(exp_dir, n_classes)
        y_pred = _predict(model, device, X)

        rows = []
        print(f"{'species':<35s} {'source':<10s} {'support':>8s} {'correct':>8s} {'recall':>8s}")
        for s in unique_species:
            if not s:
                continue
            mask = mosq_mask & (species_arr == s)
            if not mask.any():
                continue
            support = int(mask.sum())
            correct = int(((y_pred == mosq_idx) & mask).sum())
            recall = correct / support
            src_for_s = sources[mask]
            src_label = sorted(set(src_for_s.tolist()))[0] if len(src_for_s) else "?"
            rows.append({"species": s, "source": src_label, "support": support,
                         "correct": correct, "recall": recall})
            print(f"{s:<35s} {src_label:<10s} {support:>8d} {correct:>8d} {recall:>8.4f}")

        results.append({"exp": name, "per_species": rows})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n-- saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
