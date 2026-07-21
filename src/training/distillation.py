"""Knowledge distillation trainer (W11).

A small student model is trained to match a frozen teacher's soft
logits + the ground-truth labels. The student is the deployment artefact
(quantized to INT8 for ESP32); the teacher is whichever model in
``experiments/`` we judge to be the best.

Loss::

    L = alpha * T^2 * KL(softmax(s / T) || softmax(t / T)) +
        (1 - alpha) * CE(s, y)

* Soft term lets the student learn from the teacher's full probability
  distribution (Hinton et al. 2015). The ``T^2`` factor keeps gradient
  magnitudes comparable to the hard term.
* Hard term anchors the student to the ground truth so it doesn't
  inherit the teacher's mistakes.

Usage::

    python -m src.training.distillation \\
        --student-config configs/cnn1d_tiny.yaml \\
        --teacher-exp exp_physics_informed_w6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate, summarize
from src.training.config import FullConfig, load_config
from src.training.train import (
    _build_loaders,
    _build_model,
    _build_optimizer,
    _build_scheduler,
    _is_better,
    _serializable_metrics,
    _strip_metric_for_history,
    ROOT,
)
from src.utils.device import get_device, mps_memory_summary
from src.utils.seed import set_seed


def _load_teacher(teacher_exp_dir: Path, n_classes: int, device: torch.device) -> nn.Module:
    """Reconstruct a teacher from its saved config + best.pt and put it in
    eval mode with grads disabled."""
    cfg_path = teacher_exp_dir / "config_used.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing teacher config_used.yaml at {cfg_path}")
    # The dump is already a FullConfig serialization; load it back through
    # the validator so any schema drift is caught.
    tmp = teacher_exp_dir / "_kd_teacher_cfg.yaml"
    tmp.write_text(cfg_path.read_text())
    try:
        cfg = load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    teacher = _build_model(cfg, n_classes).to(device)
    ckpt = torch.load(teacher_exp_dir / "best.pt", map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (loss, components) so the caller can log soft / hard
    contributions separately."""
    log_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    soft = F.kl_div(log_s, p_t, reduction="batchmean") * (temperature ** 2)
    hard = F.cross_entropy(student_logits, target)
    loss = alpha * soft + (1.0 - alpha) * hard
    return loss, {"soft": float(soft.detach()), "hard": float(hard.detach())}


def _train_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    alpha: float,
    grad_clip: float | None,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    total_loss = 0.0
    total_soft = 0.0
    total_hard = 0.0
    total_n = 0
    correct = 0
    from tqdm import tqdm  # local import to keep header light
    bar = tqdm(loader, desc="distill", leave=False)
    for x, y in bar:
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)

        with torch.no_grad():
            t_logits = teacher(x)

        opt.zero_grad(set_to_none=True)
        s_logits = student(x)
        loss, parts = _kd_loss(s_logits, t_logits, y, temperature, alpha)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_soft += parts["soft"] * bs
        total_hard += parts["hard"] * bs
        total_n += bs
        correct += (s_logits.argmax(dim=-1) == y).sum().item()
        bar.set_postfix(
            loss=f"{total_loss/total_n:.4f}",
            soft=f"{total_soft/total_n:.4f}",
            hard=f"{total_hard/total_n:.4f}",
            acc=f"{correct/total_n:.4f}",
        )
    return {
        "loss": total_loss / max(total_n, 1),
        "kd_soft": total_soft / max(total_n, 1),
        "kd_hard": total_hard / max(total_n, 1),
        "acc": correct / max(total_n, 1),
    }


@torch.no_grad()
def _eval_epoch(
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
) -> dict:
    student.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y_dev = y.to(device)
        logits = student(x)
        loss = ce(logits, y_dev)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(prob.argmax(axis=-1))
        y_score.append(prob)
    res = evaluate(np.concatenate(y_true), np.concatenate(y_pred), y_score=np.concatenate(y_score), classes=classes)
    res["loss"] = total_loss / max(total_n, 1)
    return res


def distill(
    student_cfg_path: str,
    teacher_exp: str,
    *,
    temperature: float = 4.0,
    alpha: float = 0.5,
    train_subset: int | None = None,
    val_subset: int | None = None,
    epochs_override: int | None = None,
    exp_name_override: str | None = None,
    no_wandb: bool = False,
) -> dict:
    cfg = load_config(student_cfg_path)
    if epochs_override is not None:
        cfg.train.epochs = epochs_override
    if exp_name_override is not None:
        cfg.exp.name = exp_name_override
    if no_wandb:
        cfg.wandb.enabled = False

    set_seed(cfg.train.seed)
    device = get_device()
    print(f"== device: {device}")

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = ROOT / "experiments" / teacher_exp
    if not teacher_dir.is_dir():
        raise SystemExit(f"missing teacher dir {teacher_dir}")

    train_loader, val_loader, test_loader, classes = _build_loaders(cfg, train_subset, val_subset)
    n_classes = len(classes)

    student = _build_model(cfg, n_classes).to(device)
    teacher = _load_teacher(teacher_dir, n_classes, device)
    n_s = sum(p.numel() for p in student.parameters() if p.requires_grad)
    # Teacher's requires_grad is off for KD, so count without that filter.
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"== student {cfg.model.name}: {n_s:,} params")
    print(f"== teacher {teacher.__class__.__name__}: {n_t:,} params (frozen)")
    print(f"== KD loss: alpha={alpha}, T={temperature}")

    opt = _build_optimizer(cfg, student)
    sched = _build_scheduler(cfg, opt)

    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb  # noqa: WPS433
            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.exp.name,
                tags=list(cfg.wandb.tags) + ["distillation"],
                config={**asdict(cfg), "kd_temperature": temperature, "kd_alpha": alpha,
                        "teacher_exp": teacher_exp},
                dir=str(out_dir),
            )
            wandb.watch(student, log="gradients", log_freq=200)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN  wandb disabled (init failed: {exc})")
            wandb_run = None

    best_metric: float | None = None
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.perf_counter()
        train_stats = _train_epoch(
            student, teacher, train_loader, opt, device,
            temperature=temperature, alpha=alpha, grad_clip=cfg.train.grad_clip_norm,
        )
        val_stats = _eval_epoch(student, val_loader, device, classes)
        dt = time.perf_counter() - t0

        if sched is not None:
            sched.step(val_stats["macro_f1"])

        epoch_row = {
            "epoch": epoch,
            "elapsed_s": dt,
            "train_loss": train_stats["loss"],
            "train_kd_soft": train_stats["kd_soft"],
            "train_kd_hard": train_stats["kd_hard"],
            "train_acc": train_stats["acc"],
            **{f"val_{k}": v for k, v in _strip_metric_for_history(val_stats).items()},
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:3d}  "
            f"train loss {train_stats['loss']:.4f} "
            f"(soft {train_stats['kd_soft']:.4f}, hard {train_stats['kd_hard']:.4f}) "
            f"acc {train_stats['acc']:.4f}  | "
            f"val acc {val_stats['accuracy']:.4f} macro_f1 {val_stats['macro_f1']:.4f}  ({dt:.1f}s)"
        )
        if wandb_run is not None:
            wandb_run.log(epoch_row, step=epoch)

        es = cfg.train.early_stop
        metric_key = es.metric.removeprefix("val_")
        target = val_stats.get(metric_key, val_stats.get("macro_f1"))
        if _is_better(target, best_metric, es.mode, es.min_delta):
            best_metric = target
            bad_epochs = 0
            torch.save({"model": student.state_dict(), "epoch": epoch, "metric": target},
                       out_dir / "best.pt")
        else:
            bad_epochs += 1
            if es.enabled and bad_epochs >= es.patience:
                print(f"early stop at epoch {epoch}")
                break

    torch.save({"model": student.state_dict(), "epoch": epoch}, out_dir / "last.pt")

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model"])
    test_stats = _eval_epoch(student, test_loader, device, classes)
    print()
    print(summarize(test_stats))

    results = {
        "model": cfg.model.name,
        "exp_name": cfg.exp.name,
        "teacher_exp": teacher_exp,
        "kd_temperature": temperature,
        "kd_alpha": alpha,
        "best_val_metric": best_metric,
        "best_epoch": ckpt["epoch"],
        "n_params_student": n_s,
        "n_params_teacher": n_t,
        "device": str(device),
        "test": _serializable_metrics(test_stats),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "summary.txt").write_text(
        f"# {cfg.exp.name} (KD from {teacher_exp})\n\n## test\n{summarize(test_stats)}\n"
    )
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


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-config", required=True)
    parser.add_argument("--teacher-exp", required=True,
                        help="experiment dir name under experiments/ for the teacher")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="weight on the soft-label term (1-alpha goes to hard CE)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    distill(
        student_cfg_path=args.student_config,
        teacher_exp=args.teacher_exp,
        temperature=args.temperature,
        alpha=args.alpha,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        epochs_override=args.epochs,
        exp_name_override=args.exp_name,
        no_wandb=args.no_wandb,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
