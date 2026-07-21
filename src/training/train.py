"""Generic training loop for any model under ``src.models``.

Wired to the project conventions:
* MPS-first device via :func:`src.utils.device.get_device`
* leakage-safe NPZ partitions from :mod:`src.data.preprocess_pipeline`
* :class:`src.data.dataset.WingbeatNpz` for batching
* :func:`src.evaluation.metrics.evaluate` for per-epoch metrics
* optional Wandb logging — silent when ``cfg.wandb.enabled=False``

Usage::

    python -m src.training.train --config configs/cnn1d.yaml
    python -m src.training.train --config configs/cnn1d.yaml --train-subset 200000
    python -m src.training.train --config configs/cnn1d.yaml --epochs 5

Outputs go to ``experiments/exp_<exp.name>/``:
  best.pt          — model weights at best val metric
  last.pt          — final epoch weights
  results.json     — full per-split metrics dict (train / val / test)
  summary.txt      — printable digest from ``metrics.summarize``
  history.json     — per-epoch (epoch, train_loss, val_loss, val_macro_f1, ...)
  config_used.yaml — exact resolved config (overrides applied)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from src.data.augment import WingbeatAugment
from src.data.dataset import WingbeatNpz, make_loader
from src.evaluation.metrics import evaluate, summarize
from src.training.config import FullConfig, TrainAugmentConfig, load_config
from src.utils.device import get_device, mps_memory_summary, to_device
from src.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# data loaders


def _maybe_subset(ds: WingbeatNpz, n: int | None) -> Subset | WingbeatNpz:
    if n is None or n >= len(ds):
        return ds
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=n, replace=False)
    return Subset(ds, sorted(indices.tolist()))


def _class_counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.int64)
    return counts


def _build_train_transform(aug: TrainAugmentConfig):
    """Translate the YAML aug section into a callable transform, or None."""
    if aug.type is None:
        return None
    if aug.type == "wingbeat":
        return WingbeatAugment(
            snr_db_range=(aug.snr_db_min, aug.snr_db_max),
            gain_range=(aug.gain_min, aug.gain_max),
            p_noise=aug.p_noise,
            p_gain=aug.p_gain,
            p_shift=aug.p_shift,
            max_shift_frac=aug.max_shift_frac,
            use_pink_noise=aug.use_pink_noise,
            seed=aug.seed,
        )
    raise ValueError(f"unknown train.augment.type: {aug.type!r}")


def _build_loaders(
    cfg: FullConfig,
    train_subset: int | None,
    val_subset: int | None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_transform = _build_train_transform(cfg.train.augment)
    train_ds = WingbeatNpz(ROOT / cfg.data.train_npz, transform=train_transform)
    val_ds = WingbeatNpz(ROOT / cfg.data.val_npz)   # eval stays clean
    test_ds = WingbeatNpz(ROOT / cfg.data.test_npz)  # eval stays clean
    classes = train_ds.classes

    train_view = _maybe_subset(train_ds, train_subset)
    val_view = _maybe_subset(val_ds, val_subset)

    if cfg.train.sampler.kind == "weighted":
        # WeightedRandomSampler weights are per-sample; using inverse class
        # frequency makes each minibatch roughly class-balanced regardless of
        # the global imbalance.
        if isinstance(train_view, Subset):
            y_train = train_ds.y[np.asarray(train_view.indices)]
        else:
            y_train = train_ds.y
        counts = _class_counts(y_train, len(classes)) + 1  # avoid div-by-zero
        per_class = 1.0 / counts.astype(np.float64)
        sample_weights = per_class[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(y_train),
            replacement=True,
        )
        train_loader = DataLoader(
            train_view,
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=False,
            persistent_workers=cfg.train.num_workers > 0,
        )
    else:
        train_loader = make_loader(
            train_view,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
        )

    val_loader = make_loader(val_view, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    test_loader = make_loader(test_ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)

    return train_loader, val_loader, test_loader, classes


# --------------------------------------------------------------------------- #
# model + optimizer


def _build_model(cfg: FullConfig, n_classes: int) -> nn.Module:
    """Pull the model class from ``src.models.<name>``.

    Resolution order:
      1. Explicit ``MODEL_CLASS`` attribute on the module (preferred —
         unambiguous when the file defines several nn.Module subclasses).
      2. PascalCase / SHOUTING_CASE / +CNN suffix variants of the name.
      3. Fallback: any nn.Module subclass alphabetically — only safe if the
         file defines exactly one.
    """
    module_name = f"src.models.{cfg.model.name}"
    mod = importlib.import_module(module_name)

    cls = getattr(mod, "MODEL_CLASS", None)
    if cls is None:
        pascal = cfg.model.name.replace("_", " ").title().replace(" ", "")
        candidates = [
            cfg.model.name.upper(),     # e.g. CNN_1D
            pascal,                     # e.g. Cnn1d
            f"{pascal}CNN",             # e.g. Cnn1dCNN
            f"{pascal}Net",
            f"{pascal}Model",
        ]
        for name in candidates:
            if hasattr(mod, name):
                cls = getattr(mod, name)
                break
    if cls is None:
        for name in sorted(dir(mod)):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                cls = obj
                break
    if cls is None:
        raise RuntimeError(f"no nn.Module class found in {module_name}")

    kwargs = dict(cfg.model.kwargs)
    kwargs.setdefault("n_classes", n_classes)
    return cls(**kwargs)


def _build_optimizer(cfg: FullConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9,
                               weight_decay=cfg.train.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.train.optimizer}")


def _build_scheduler(cfg: FullConfig, opt: torch.optim.Optimizer):
    s = cfg.train.scheduler
    if s.kind == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=s.factor, patience=s.patience, min_lr=s.min_lr,
        )
    return None


def _build_loss(cfg: FullConfig, y_train: np.ndarray, n_classes: int, device: torch.device) -> nn.Module:
    if cfg.train.class_weight == "balanced":
        # sklearn-style: N / (C * count_c)
        counts = _class_counts(y_train, n_classes) + 1
        weights = (len(y_train) / (n_classes * counts)).astype(np.float32)
        w = torch.as_tensor(weights, device=device, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
# epoch loops


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_n = 0
    correct = 0
    bar = tqdm(loader, desc="train", leave=False)
    for x, y in bar:
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        correct += (logits.argmax(dim=-1) == y).sum().item()
        bar.set_postfix(loss=f"{total_loss/total_n:.4f}", acc=f"{correct/total_n:.4f}")
    return {"loss": total_loss / max(total_n, 1), "acc": correct / max(total_n, 1)}


@torch.no_grad()
def _run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    classes: list[str],
) -> dict:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y_dev = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y_dev)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = prob.argmax(axis=-1)
        y_true.append(y.numpy())
        y_pred.append(pred)
        y_score.append(prob)
    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    y_score_arr = np.concatenate(y_score)
    metrics = evaluate(y_true_arr, y_pred_arr, y_score=y_score_arr, classes=classes)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics


# --------------------------------------------------------------------------- #
# main loop


def _is_better(curr: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return curr > best + min_delta
    return curr < best - min_delta


def _strip_metric_for_history(m: dict) -> dict:
    """Drop the heavy items so history.json stays small (omit confusion + per_class)."""
    return {
        k: v
        for k, v in m.items()
        if k in {"loss", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "auc_macro"}
    }


def train(
    cfg: FullConfig,
    train_subset: int | None = None,
    val_subset: int | None = None,
) -> dict:
    set_seed(cfg.train.seed)
    device = get_device()
    print(f"== device: {device}")

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = _build_loaders(cfg, train_subset, val_subset)
    n_classes = len(classes)
    print(f"== classes ({n_classes}): {classes}")
    print(f"== train batches: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

    model = _build_model(cfg, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"== model {cfg.model.name}: {n_params:,} params")

    # Build the loss using the underlying training-set labels (Subset-aware).
    base_train: WingbeatNpz = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    if isinstance(train_loader.dataset, Subset):
        y_train_for_loss = base_train.y[np.asarray(train_loader.dataset.indices)]
    else:
        y_train_for_loss = base_train.y
    loss_fn = _build_loss(cfg, y_train_for_loss, n_classes, device)

    opt = _build_optimizer(cfg, model)
    sched = _build_scheduler(cfg, opt)

    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb  # noqa: WPS433
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.exp.name,
                tags=cfg.wandb.tags,
                config=asdict(cfg),
                dir=str(out_dir),
            )
            wandb.watch(model, log="gradients", log_freq=200)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN  wandb disabled (init failed: {exc})")
            wandb_run = None

    best_metric: float | None = None
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.perf_counter()
        train_stats = _run_train_epoch(
            model, train_loader, loss_fn, opt, device, cfg.train.grad_clip_norm
        )
        val_stats = _run_eval_epoch(model, val_loader, loss_fn, device, classes)
        dt = time.perf_counter() - t0

        if sched is not None:
            sched.step(val_stats["macro_f1"])

        epoch_row = {
            "epoch": epoch,
            "elapsed_s": dt,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            **{f"val_{k}": v for k, v in _strip_metric_for_history(val_stats).items()},
        }
        history.append(epoch_row)

        print(
            f"epoch {epoch:3d}  "
            f"train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.4f}  | "
            f"val loss {val_stats['loss']:.4f} acc {val_stats['accuracy']:.4f} "
            f"macro_f1 {val_stats['macro_f1']:.4f} balanced_acc {val_stats['balanced_accuracy']:.4f}  "
            f"({dt:.1f}s)"
        )

        if wandb_run is not None:
            wandb_run.log(epoch_row, step=epoch)

        # Early-stopping bookkeeping
        es = cfg.train.early_stop
        # Strip "val_" prefix that history uses; metric should be 'macro_f1', etc.
        metric_key = es.metric.removeprefix("val_")
        target = val_stats.get(metric_key) if metric_key in val_stats else val_stats.get("macro_f1")
        if _is_better(target, best_metric, es.mode, es.min_delta):
            best_metric = target
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metric": target}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if es.enabled and bad_epochs >= es.patience:
                print(f"early stop at epoch {epoch} ({es.metric} hasn't improved in {bad_epochs} epochs)")
                break

    torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "last.pt")

    # Reload best for final test evaluation.
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_stats = _run_eval_epoch(model, test_loader, loss_fn, device, classes)

    print()
    print(summarize(test_stats))

    # ---- write artifacts
    results = {
        "model": cfg.model.name,
        "exp_name": cfg.exp.name,
        "best_val_metric": best_metric,
        "best_epoch": ckpt["epoch"],
        "n_params": n_params,
        "device": str(device),
        "test": _serializable_metrics(test_stats),
        "val_at_best": _serializable_metrics(_run_eval_epoch(model, val_loader, loss_fn, device, classes)),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "summary.txt").write_text(f"# {cfg.exp.name}\n\n## test\n{summarize(test_stats)}\n")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    mem = mps_memory_summary()
    if mem:
        print(f"-- MPS memory: current {mem['current_allocated']/1024**2:.1f} MB, "
              f"driver {mem['driver_allocated']/1024**2:.1f} MB")
    print(f"-- saved {out_dir}/")

    if wandb_run is not None:
        wandb_run.finish()

    return results


def _serializable_metrics(m: dict) -> dict:
    return {
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "auc_macro": m["auc_macro"],
        "per_class": m["per_class"],
        "confusion": m["confusion"].tolist() if hasattr(m["confusion"], "tolist") else m["confusion"],
        "classes": m["classes"],
        "loss": m.get("loss"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    parser.add_argument("--train-subset", type=int, default=None,
                        help="cap training rows (random subsample). Useful for smoke tests.")
    parser.add_argument("--val-subset", type=int, default=None, help="cap validation rows")
    parser.add_argument("--exp-name", default=None, help="override cfg.exp.name (so smoke tests don't overwrite real runs)")
    parser.add_argument("--no-wandb", action="store_true", help="force-disable wandb regardless of config")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.exp_name is not None:
        cfg.exp.name = args.exp_name
    if args.no_wandb:
        cfg.wandb.enabled = False

    train(cfg, train_subset=args.train_subset, val_subset=args.val_subset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
