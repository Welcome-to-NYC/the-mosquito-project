"""Physics-informed 1D CNN (W6 of the roadmap).

Three parallel branches encode prior knowledge about wingbeat acoustics:

1. **LearnableFFT** — a Conv1d whose weights are initialized as cos/sin
   pairs at frequencies inside the wingbeat band, with a Hann window. The
   forward output is the per-band magnitude — interpretable as a learned
   STFT row vector. Training is free to drift the filters off the
   Fourier basis if a better one exists.

2. **HarmonicAttention** — given the LearnableFFT magnitude (frequency ×
   time), estimate the fundamental f0 in the 200–1000 Hz band, then look
   up power at each harmonic ``k·f0`` (k = 1..5). Mosquito wingbeats have
   five strong harmonics; midges and fruit flies have two or three. So
   the harmonic profile is a discriminative feature.

3. **TemporalEnvelope** — full-wave-rectified signal → low-pass filter →
   small Conv1d → mean-pool. Captures the body-shadow time profile (entry
   → middle → exit) characteristic of an insect crossing the optical
   sensor beam.

The three branch outputs are concatenated and passed through a small MLP
classifier head. Total parameter count is comparable to the Fanioudakis
1D-CNN baseline (~12 K) so the ESP32 deployment story stays alive after
INT8 quantization.

Parameters worth knowing:

* ``n_filters``: number of frequency bands the LearnableFFT learns.
  Default 64. More bands = finer spectral resolution at training cost.
* ``kernel_size``: filter length. Default 129 (≈ 26 ms @ 5 kHz). Picks
  up to 1024-pt-FFT-equivalent resolution while staying causal-friendly
  for streaming inference.
* ``freq_range``: initial Fourier basis lower / upper bounds in Hz.
  Default (100, 1500) — covers the wingbeat band with margin.
* ``n_harmonics``: how many integer multiples of f0 to attend to.
  Default 5.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Branch 1: Learnable FFT


class LearnableFFT(nn.Module):
    """Conv1d initialized as a cos/sin Fourier basis. Output is magnitude.

    Weight layout: ``conv.weight`` has shape ``(2 * n_filters, 1, kernel_size)``.
    Even indices are cosine filters, odd indices are sine filters at the
    same frequency. After convolution we reshape to
    ``(B, n_filters, 2, T)`` and take ``sqrt(re^2 + im^2)`` so the filter
    interprets as a band-pass / instantaneous-amplitude pair.
    """

    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 129,
        sample_rate: int = 5000,
        freq_range: tuple[float, float] = (100.0, 1500.0),
        stride: int = 1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.freq_range = freq_range
        self.stride = stride
        self.eps = eps

        # 2 * n_filters filters: cos / sin pairs interleaved.
        # stride > 1 downsamples in time, which is the main speed lever — at
        # 5 kHz a stride of 4 keeps Nyquist at 625 Hz which is plenty above
        # the wingbeat band, and makes training per-epoch ~3-4× faster.
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=2 * n_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            bias=False,
        )
        self._init_fourier_basis()
        # Cache the initial center frequencies; useful for the harmonic-attention
        # branch and for visualization.
        self.register_buffer("init_freqs_hz", self._initial_freqs(), persistent=False)

    def _initial_freqs(self) -> torch.Tensor:
        f_lo, f_hi = self.freq_range
        return torch.linspace(f_lo, f_hi, self.n_filters)

    def _init_fourier_basis(self) -> None:
        freqs = self._initial_freqs()
        t = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size // 2)
        window = torch.hann_window(self.kernel_size, periodic=False)
        weights = torch.zeros(2 * self.n_filters, 1, self.kernel_size)
        for i, f in enumerate(freqs):
            angle = 2 * math.pi * float(f) / self.sample_rate * t
            weights[2 * i, 0] = window * torch.cos(angle)
            weights[2 * i + 1, 0] = window * torch.sin(angle)
        # Normalize so each filter has unit L2 norm; helps gradient scaling.
        weights = weights / (weights.norm(dim=-1, keepdim=True) + self.eps)
        with torch.no_grad():
            self.conv.weight.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"expected (B, 1, T) input, got {tuple(x.shape)}")
        out = self.conv(x)  # (B, 2F, T)
        b, _, t_ = out.shape
        out = out.view(b, self.n_filters, 2, t_)
        return torch.sqrt(out[:, :, 0, :] ** 2 + out[:, :, 1, :] ** 2 + self.eps)


# --------------------------------------------------------------------------- #
# Branch 2: Harmonic Attention


class HarmonicAttention(nn.Module):
    """Pool the LearnableFFT magnitude at ``k·f0`` for k = 1..n_harmonics.

    Operating on a fixed grid of ``n_filters`` frequency bands, we approximate
    "pick the bin closest to k·f0" by computing a soft selection: the
    nearest-bin index is computed argmax-style over a Gaussian-weighted
    similarity centered on each harmonic, and the spectral magnitude is
    pooled with those weights. This stays differentiable end-to-end while
    matching the conventional DSP harmonic-power feature.

    Returns a ``(B, n_harmonics)`` tensor of harmonic powers, plus the
    estimated f0 in Hz (as a side-channel for diagnostics).
    """

    def __init__(
        self,
        freqs_hz: torch.Tensor | Sequence[float],
        n_harmonics: int = 5,
        f0_band: tuple[float, float] = (200.0, 1000.0),
        bandwidth_hz: float = 25.0,
    ) -> None:
        super().__init__()
        freqs = torch.as_tensor(freqs_hz, dtype=torch.float32)
        if freqs.ndim != 1:
            raise ValueError("freqs_hz must be 1-D")
        self.n_harmonics = n_harmonics
        self.bandwidth = bandwidth_hz
        self.f0_band = f0_band
        # Precompute the fundamental-band mask so f0 search ignores out-of-band bins.
        in_band = (freqs >= f0_band[0]) & (freqs <= f0_band[1])
        self.register_buffer("freqs_hz", freqs)
        self.register_buffer("in_band_mask", in_band.float())

    def forward(self, magnitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            magnitude: ``(B, F, T)`` from :class:`LearnableFFT`.

        Returns:
            (harmonic_power: ``(B, n_harmonics)``, f0_hz: ``(B,)``).
        """
        # Pool time first so f0 is per-clip rather than per-frame.
        spec = magnitude.mean(dim=-1)            # (B, F)

        # f0 is the in-band frequency with maximum energy.
        in_band_spec = spec * self.in_band_mask  # zero out out-of-band bins
        f0_idx = in_band_spec.argmax(dim=-1)     # (B,)
        f0_hz = self.freqs_hz[f0_idx]            # (B,)

        # For each harmonic k, build a Gaussian weight over the F bins centered
        # at k * f0 with width ``bandwidth``. Pool the spectrum with that weight.
        device = magnitude.device
        freqs = self.freqs_hz.to(device)         # (F,)
        out = torch.zeros(magnitude.size(0), self.n_harmonics, device=device)
        for k in range(1, self.n_harmonics + 1):
            target = k * f0_hz                   # (B,)
            # Gaussian over bins, normalized to sum 1
            sq = (freqs.unsqueeze(0) - target.unsqueeze(1)) ** 2
            w = torch.exp(-sq / (2 * self.bandwidth ** 2))
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
            out[:, k - 1] = (spec * w).sum(dim=-1)
        return out, f0_hz


# --------------------------------------------------------------------------- #
# Branch 3: Temporal Envelope


class TemporalEnvelope(nn.Module):
    """Full-wave rectification + low-pass smoothing + small Conv1d encoder.

    The "body shadow" of an insect through the optical beam has a
    characteristic onset / sustain / offset time signature even when the
    fundamental frequency analysis says nothing — this branch picks it up
    independently of the spectral content.
    """

    def __init__(
        self,
        smooth_kernel: int = 21,
        out_channels: int = 32,
        feature_dim: int = 32,
    ) -> None:
        super().__init__()
        if smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel must be odd")
        # Fixed averaging filter for the low-pass step. Not learned.
        self.register_buffer(
            "smoother",
            torch.full((1, 1, smooth_kernel), 1.0 / smooth_kernel),
            persistent=False,
        )
        self.smooth_pad = smooth_kernel // 2
        # Two-block conv encoder over the envelope.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(out_channels, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"expected (B, 1, T) input, got {tuple(x.shape)}")
        env = torch.abs(x)
        env = F.conv1d(env, self.smoother, padding=self.smooth_pad)
        h = self.encoder(env)
        h = self.gap(h).squeeze(-1)
        return self.proj(h)


# --------------------------------------------------------------------------- #
# Fusion model


class PhysicsInformedCNN(nn.Module):
    """Three-branch fusion: LearnableFFT + Harmonic + Temporal.

    Each branch produces a feature vector; we concatenate and pass through
    a small MLP head. The spectral branch is reduced from
    ``(B, F, T) -> (B, F)`` via mean+max pooling along time, then projected
    to ``feature_dim`` so all three branches contribute roughly equal mass.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_filters: int = 64,
        kernel_size: int = 129,
        sample_rate: int = 5000,
        freq_range: tuple[float, float] = (100.0, 1500.0),
        f0_band: tuple[float, float] = (200.0, 1000.0),
        n_harmonics: int = 5,
        feature_dim: int = 32,
        dropout: float = 0.3,
        fft_stride: int = 4,
        enable_harmonic: bool = True,
        enable_temporal: bool = True,
        freeze_fft: bool = False,
    ) -> None:
        super().__init__()
        self.enable_harmonic = enable_harmonic
        self.enable_temporal = enable_temporal
        self.freeze_fft = freeze_fft

        self.fft = LearnableFFT(
            n_filters=n_filters,
            kernel_size=kernel_size,
            sample_rate=sample_rate,
            freq_range=freq_range,
            stride=fft_stride,
        )
        if freeze_fft:
            # Ablation: keep Fourier-basis init but disallow gradient updates.
            # Tests whether the *learnability* of the FFT matters or just the
            # informed initialization.
            for p in self.fft.conv.parameters():
                p.requires_grad_(False)

        if enable_harmonic:
            self.harmonic = HarmonicAttention(
                freqs_hz=self.fft.init_freqs_hz,
                n_harmonics=n_harmonics,
                f0_band=f0_band,
            )
            self.harm_proj = nn.Sequential(
                nn.Linear(n_harmonics, feature_dim),
                nn.ReLU(inplace=True),
            )
        if enable_temporal:
            self.temporal = TemporalEnvelope(out_channels=feature_dim, feature_dim=feature_dim)

        # Spectral pooling: mean+max along time -> 2 * n_filters -> feature_dim.
        # Always-on branch; the ablations only turn off the *other* two.
        self.spec_proj = nn.Sequential(
            nn.Linear(2 * n_filters, feature_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion + classifier head — fused dim depends on which branches are live.
        n_active = 1 + int(enable_harmonic) + int(enable_temporal)
        fused = feature_dim * n_active
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | dict:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"expected (B, 1, T) input, got {tuple(x.shape)}")

        magnitude = self.fft(x)                              # (B, F, T)
        spec_pool = torch.cat(
            [magnitude.mean(dim=-1), magnitude.amax(dim=-1)],
            dim=-1,
        )                                                    # (B, 2F)
        spec_feat = self.spec_proj(spec_pool)                # (B, D)

        feats = [spec_feat]
        harm_power = None
        f0_hz = None
        env_feat = None
        if self.enable_harmonic:
            harm_power, f0_hz = self.harmonic(magnitude)     # (B, K), (B,)
            feats.append(self.harm_proj(harm_power))
        if self.enable_temporal:
            env_feat = self.temporal(x)
            feats.append(env_feat)

        fused = torch.cat(feats, dim=-1)
        logits = self.head(fused)

        if return_aux:
            return {
                "logits": logits,
                "magnitude": magnitude,
                "spec_pool": spec_pool,
                "harmonic_power": harm_power,
                "f0_hz": f0_hz,
                "env_feat": env_feat,
            }
        return logits


# Explicit handle for src.training.train._build_model — disambiguates this
# module which defines several nn.Module subclasses.
MODEL_CLASS = PhysicsInformedCNN
