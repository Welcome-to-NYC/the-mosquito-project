"""YAML-backed training config used by every model from W3 onwards.

Three sections:

* ``data``  — paths to the NPZ partitions and how to interpret them
* ``model`` — class name + kwargs for the architecture under ``src.models``
* ``train`` — optimizer, scheduler, loss-weighting strategy, early stopping
* ``exp``   — experiment name, output dir, wandb toggle

Configs are loaded with :func:`load_config`, which returns a :class:`TrainConfig`
dataclass. Validation happens at load time — bad keys / missing fields fail
fast instead of crashing several minutes into a training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    train_npz: str = "data/processed/train.npz"
    val_npz: str = "data/processed/val.npz"
    test_npz: str = "data/processed/test.npz"


@dataclass
class ModelConfig:
    name: str = "cnn_1d"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainSchedulerConfig:
    kind: str = "reduce_on_plateau"          # 'reduce_on_plateau' | 'none'
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-6


@dataclass
class TrainEarlyStopConfig:
    enabled: bool = True
    metric: str = "val_macro_f1"             # which key in the validation dict
    mode: str = "max"                        # 'max' | 'min'
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class TrainSamplerConfig:
    kind: str = "weighted"                   # 'weighted' | 'shuffle'
    # WeightedRandomSampler: weight = 1 / class_count; one sample drawn per index
    # so every minibatch is roughly class-balanced. Strong default for the
    # ~600:1 imbalance in this project.


@dataclass
class TrainAugmentConfig:
    """On-the-fly augmentation applied only to the training set.

    ``type=None`` disables it. Only ``type='wingbeat'`` is wired up so far —
    it builds :class:`src.data.augment.WingbeatAugment` with the parameters
    below.
    """
    type: str | None = None
    p_noise: float = 0.5
    p_gain: float = 0.5
    p_shift: float = 0.3
    snr_db_min: float = 0.0
    snr_db_max: float = 20.0
    gain_min: float = 0.3
    gain_max: float = 1.5
    max_shift_frac: float = 0.1
    use_pink_noise: bool = False
    seed: int | None = None


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    grad_clip_norm: float | None = 1.0
    num_workers: int = 4
    sampler: TrainSamplerConfig = field(default_factory=TrainSamplerConfig)
    scheduler: TrainSchedulerConfig = field(default_factory=TrainSchedulerConfig)
    early_stop: TrainEarlyStopConfig = field(default_factory=TrainEarlyStopConfig)
    augment: TrainAugmentConfig = field(default_factory=TrainAugmentConfig)
    # Loss config: class_weight='balanced' computes weights as N / (C * count_c).
    # 'none' uses uniform weights (rely on the sampler instead).
    class_weight: str = "none"


@dataclass
class ExpConfig:
    name: str = "exp"
    output_dir: str = "experiments"
    save_best_only: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "lyssa-mosquito"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class FullConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def output_dir(self) -> Path:
        return ROOT / self.exp.output_dir / f"exp_{self.exp.name}"


_TOP_LEVEL_FIELDS = {"data", "model", "train", "exp", "wandb"}


def _build(klass, raw: dict | None):
    """Construct a dataclass from a possibly-incomplete dict, preferring nested
    dataclass defaults for sub-fields. Unknown keys raise; this catches typos
    that would otherwise silently use the default value.
    """
    if raw is None:
        return klass()

    field_types = {f.name: f.type for f in klass.__dataclass_fields__.values()}
    unknown = set(raw) - set(field_types)
    if unknown:
        raise ValueError(f"unknown keys in {klass.__name__}: {sorted(unknown)}")

    kwargs = {}
    for name, sub_raw in raw.items():
        ftype = field_types[name]
        # Recurse into nested dataclass-typed fields. We compare class identity
        # against the known nested dataclasses below.
        if isinstance(sub_raw, dict) and name in {"sampler", "scheduler", "early_stop", "augment"} and klass is TrainConfig:
            sub_klass = {
                "sampler": TrainSamplerConfig,
                "scheduler": TrainSchedulerConfig,
                "early_stop": TrainEarlyStopConfig,
                "augment": TrainAugmentConfig,
            }[name]
            kwargs[name] = _build(sub_klass, sub_raw)
        else:
            kwargs[name] = sub_raw
    return klass(**kwargs)


def load_config(path: str | Path) -> FullConfig:
    """Read ``path`` as YAML and validate it into a :class:`FullConfig`.

    Top-level keys must be a subset of {data, model, train, exp, wandb}.
    Anything else is rejected so the user notices typos before training
    silently runs with the default value.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    unknown = set(raw) - _TOP_LEVEL_FIELDS
    if unknown:
        raise ValueError(f"unknown top-level keys in {p}: {sorted(unknown)}")
    return FullConfig(
        data=_build(DataConfig, raw.get("data")),
        model=_build(ModelConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
        exp=_build(ExpConfig, raw.get("exp")),
        wandb=_build(WandbConfig, raw.get("wandb")),
    )
