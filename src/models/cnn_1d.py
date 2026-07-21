"""1D-CNN baseline (Fanioudakis 2018) for wingbeat classification.

Reference: Fanioudakis, E. & Potamitis, I. "Mosquito wingbeat analysis and
classification using deep learning" (EUSIPCO 2018). Three Conv1d -> BN ->
ReLU -> MaxPool blocks, adaptive pooling, then a small MLP head. Tiny enough
that the dynamic-INT8 quantized version fits inside ESP32 (~20-100 KB).

Default architecture for ``input_len=1024`` at 5 kHz:

  Conv1d(  1 ->  16, k=7, p=3) -> BN -> ReLU -> MaxPool(2)   # 1024 -> 512
  Conv1d( 16 ->  32, k=5, p=2) -> BN -> ReLU -> MaxPool(2)   #  512 -> 256
  Conv1d( 32 ->  64, k=3, p=1) -> BN -> ReLU -> MaxPool(2)   #  256 -> 128
  AdaptiveAvgPool1d(1) -> Flatten                            #     -> 64
  Dropout(0.3)
  Linear(64 -> 32) -> ReLU -> Dropout(0.3)
  Linear(32 -> n_classes)

Param count is ~12 K — small enough that 1D-CNN training on M1 Pro is
single-digit minutes per epoch even on the 1.6 M-segment training set.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d -> BN -> ReLU -> MaxPool. Padding=same so the time dim only
    changes through the explicit MaxPool — easier to reason about output
    shape than letting the conv silently shrink it."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool: int = 2) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric same-padding")
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class CNN1D(nn.Module):
    """Fanioudakis-style 1D-CNN. ``channels`` is the per-block output width;
    ``kernel_sizes`` is the per-block kernel size (length must match)."""

    def __init__(
        self,
        n_classes: int = 3,
        in_channels: int = 1,
        channels: Sequence[int] = (16, 32, 64),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        fc_hidden: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError(
                f"channels and kernel_sizes must align ({len(channels)} vs {len(kernel_sizes)})"
            )

        blocks: list[nn.Module] = []
        prev = in_channels
        for c, k in zip(channels, kernel_sizes):
            blocks.append(ConvBlock(prev, c, kernel_size=k))
            prev = c
        self.blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels[-1], fc_hidden)
        self.act = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, C, L) input, got {tuple(x.shape)}")
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)  # (B, C)
        x = self.dropout1(x)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
