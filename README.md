# Audio Mosquito Detector on an ESP32

**91.6% accuracy, 9 KB model, running entirely on a $4 microcontroller — no cloud, no phone.**

Built for **LYSSA** (Lingnan–Yuanpei STEM Summer Academy), a Peking University × Hong
Kong Lingnan University business competition held in Hong Kong in July 2026. Teams
prototyped a hardware+software product against a real civic problem in one summer
sprint; ours targeted urban mosquito surveillance for Hong Kong's FEHD/CHP
(Food and Environmental Hygiene Department / Centre for Health Protection).

I was **team CTO** — I owned the ML model end to end (architecture, training,
compression) and the on-chip deployment (hand-written C++ inference, no ML runtime),
and co-owned the hardware/firmware integration. This repo contains that half of the
project: the model and everything needed to reproduce and run it on real silicon.

## The pitch, in one line

A 0.128-second sound clip → a frequency transform → a tiny CNN → "mosquito: yes/no."
Every step of that pipeline runs on the ESP32 itself.

| | |
|---|---|
| Accuracy | **91.6%** (91% recall, 93% precision, F1 0.92) |
| Model size | **9.2 KB** fp32 (2,443 parameters) — smaller than one second of MP3 |
| On-chip inference | **67 ms/sample** (~15 Hz) on an ESP32-CAM, no TFLite Micro |
| Deployment gap | **0** — 2,752 / 2,752 on-chip predictions match the PyTorch reference bit-for-bit (max logit diff 2×10⁻⁵) |
| Cost | ESP32-CAM + programmer board, ≈ $4 |

Full walkthrough: [`docs/presentation_en.md`](docs/presentation_en.md) /
[`docs/presentation_en.pdf`](docs/presentation_en.pdf). Technical deep-dive (Korean):
[`docs/model_summary.md`](docs/model_summary.md).

## How it works

1. **LearnableFFT** — a frequency transform initialized to a Fourier basis, then
   fine-tuned during training so the model can nudge which frequency bands it pays
   attention to. This is the single change that moved accuracy from 84% to 91% — more
   than making the model bigger did (that stalled at 88%).
2. **Tiny 1D-CNN** — 3 conv layers + 2 FC layers, distilled 13× down from a larger
   physics-informed teacher model (LearnableFFT + harmonic attention + temporal
   envelope branch) via knowledge distillation.
3. **BatchNorm folding + hand-coded C++ inference** — at 2,443 parameters the model
   is small enough that hand-written fixed-point-free C is simpler and more auditable
   than pulling in a full ML runtime. `scripts/export_student_for_esp32.py` folds BN
   into the preceding conv and emits a C header; the firmware runs the forward pass
   and self-checks every prediction against embedded PyTorch reference outputs.

See [`docs/images/model_architecture.png`](docs/images/model_architecture.png) for the
full diagram.

## Why this is an honest number, not a shortcut

A classifier can cheat by learning "which recording device" instead of "is there a
wingbeat." I deliberately added background audio recorded on the *same* device as the
positive mosquito recordings to the negative class — the model still rejects it 92% of
the time, so it's using the wingbeat signal itself, not a device fingerprint. The
[research ledger](experiments/RESULTS.md) documents this and several other domain-shortcut
traps that came up during development (e.g. one dataset being 79% of training data and
the model partly learning "is this that dataset" instead of "is this a mosquito") and
how each was diagnosed and closed.

## What's *not* solved yet (read before assuming this is production-ready)

- **Non-mosquito insect rejection is weak** — F1 0.087, ~5% precision. Public datasets
  of optical/close-mic non-mosquito insect wingbeats (midges, gnats) are extremely
  scarce, and it shows. This is the single biggest open problem, not a hidden one.
- **No live sensor pipeline yet.** Validation streams pre-recorded test data to the
  chip over USB serial (`scripts/eval_on_chip.py`, `firmware/wingbeat_stream/`); a real
  microphone/optical sensor → ADC → ring buffer path isn't wired in.
- **Training data skews lab-condition.** Field performance (wind, ambient noise, dust)
  is unverified beyond the SNR-robustness sweep in `experiments/RESULTS.md`.
- Full breakdown, including per-species recall and what's next, in
  [`docs/model_summary.md`](docs/model_summary.md) §4.4 and §7.

## Repo layout

```
src/
  data/          preprocessing, augmentation, dataset/split logic, UCR-format loader
  models/        1D-CNN baseline, LearnableFFT + physics-informed CNN
  training/      training loop, config schema, knowledge-distillation trainer
  evaluation/    metrics
  features/      spectral feature extraction (for the classical baseline)
  utils/         device (MPS) + seeding helpers
scripts/         training runs, ablations, ESP32 export + on-chip evaluation,
                 dataset fetch/inventory, all the plotting/reporting scripts
firmware/
  hello_world/       board bring-up sanity check
  wingbeat_stream/   USB-serial streaming protocol for on-chip validation
  wingbeat_lfft/     LearnableFFT stage standalone on-chip
  wingbeat_inference/  full model inference on-chip, self-checked against PyTorch
configs/         YAML training configs (baseline, augmented, physics-informed, ablations)
tests/           unit tests (data pipeline, model shapes, metrics, config)
experiments/
  RESULTS.md     full experiment ledger — every model trained, every result, every
                 dead end and why it was a dead end
docs/            presentation deck, technical summary, dataset-choice rationale, figures
```

`data/`, model checkpoints, and experiment artifacts (`.pt`, `.npz`, wandb logs) are
excluded — this is a snapshot of the code and the narrative, not the multi-GB dataset
cache. `configs/datasets.yaml` documents exactly what was used and where it comes from.

## Running it

```bash
./setup.sh                                   # creates .venv, installs deps, checks MPS
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/verify_mps.py
```

Training (example — physics-informed teacher):
```bash
python -m src.training.train --config configs/physics_informed.yaml
```

Export + flash the deployed student model:
```bash
python scripts/export_student_for_esp32.py       # writes firmware/wingbeat_inference/model_weights.h
arduino-cli compile --fqbn esp32:esp32:esp32cam firmware/wingbeat_inference
arduino-cli upload  --fqbn esp32:esp32:esp32cam --port <your-port> firmware/wingbeat_inference
```

## Team & scope note

LYSSA was a team competition. I was CTO, responsible for the ML model and the
firmware/on-chip deployment side of the demo — everything in this repo is that work.
Hardware assembly (sensor selection, board wiring) was a shared team effort; the wider
team also produced business-case, market-sizing, and forecasting material for the
pitch that lives outside this repo, some of it built by teammates on their own
detection approaches (e.g. an LSTM-based forecasting model, a separate CNN
architecture) as alternative/comparison angles — not included here, since it isn't
mine to publish. What's in this repo is scoped to what I personally designed, trained,
and deployed.

## Datasets used (all public)

- **HumBugDB** (Kiskin et al., NeurIPS 2021) — mosquito positives + matched background, real field recordings
- **InsectSound1000** (Branding et al., *Scientific Data* 2024) — non-mosquito Diptera negatives (hoverfly, gall midge, fungus gnat) — the 3 species whose wingbeat frequency actually overlaps the mosquito band
- **Wingbeats** (Potamitis 2018) and **UCR InsectWingbeat** (Chen 2014) — optical-sensor datasets used for a parallel optical-modality feasibility track (see `experiments/RESULTS.md`)

Full rationale for what was used and what was deliberately excluded (and why):
[`docs/data_choices_rationale.md`](docs/data_choices_rationale.md).
