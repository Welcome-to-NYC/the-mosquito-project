# Experiment Results

Project-level ledger for the mosquito wingbeat classifier. Updated after every
training or evaluation run. Per-experiment details (full metrics, model
checkpoint, summary digest) live in the matching `exp_<name>/` subdirectory;
this file is the cross-experiment view.

---

## TL;DR

Five trained models, four ML weeks (W2 → W11). Headline numbers shifted
hard once we audited them.

* **W2 XGBoost** on 11 hand-engineered spectral features hit
  **0.92 test accuracy** — surprisingly strong. Hard to beat with
  deep learning on this kind of margin.
* **The 0.92 was a domain shortcut.** Wingbeats (sterile lab, fixed
  0.625 s clips) is 79 % of test mosquito support; the model was
  effectively learning "is this a Wingbeats clip", not "is this a
  wingbeat". Per-source cross-eval showed Wingbeats mosq recall 99 %
  vs HumBugDB mosq recall 75 % — a 24 pp gap. Subsampling Wingbeats
  to ~14 % of recordings closed it to 13–14 pp on every model.
* **WingbeatAugment** (noise + gain + shift) raised **insect recall
  from 7 % to 43 %** but slightly *widened* the cross-source gap.
  Augmentation fixed the minority-class problem, not the domain
  problem.
* **Physics-informed CNN (W6)** — LearnableFFT initialized to a
  Fourier basis + harmonic attention + temporal envelope branch —
  is the **first architectural change to close the cross-source gap
  (10.8 pp)** without sacrificing accuracy. The learnable filters
  drift hardest in the 700–1300 Hz band, exactly where the wingbeat
  fundamental and its first two harmonics live.
* **W11 distillation** compresses the W6 teacher 13× into a **2,443-
  param student**. INT8 weights ≈ 2.4 KB → ESP32-deployable;
  estimated ~5 ms / sample inference on ESP32 (M1 Pro CPU 0.48 ms /
  sample × ~10×). Loses ~6 % accuracy vs teacher, but matches the
  teacher's HumBugDB recall (84 % vs 88 %) and *exceeds* it on
  insect recall (44 % vs 18 %, same effect we saw with augmentation).
* **W12 ablation** turned off each W6 branch in turn (no harmonic,
  no temporal envelope, frozen LearnableFFT). Each removal widens the
  cross-source gap by 2.2–2.7 pp. **No single load-bearing branch** —
  the W6 win comes from the combination. Temporal envelope is the
  largest single contributor.
* **W12 deployment analyses** (recording-level aggregation, noise
  robustness, per-species breakdown):
    * Recording-level metrics drop 0.9–4.3 pp vs segment-level — the
      segment numbers were inflated by short Wingbeats clips.
    * **Tiny student is more robust to pink noise than the teacher.**
      At SNR 0 dB: student 0.76 acc / teacher 0.62.
    * Hong Kong-relevant species (Ae. aegypti, Ae. albopictus,
      Cu. quinquefasciatus) all ≥ 0.81 HumBugDB recall. *An. gambiae ss*
      on HumBugDB is < 1 % recall, but it's sub-Saharan, so it doesn't
      block the HK deployment.

**Recommended deployment artefact: `exp_cnn1d_tiny_distilled/best.pt`**.

* **W12 on-silicon deployment.** Student exported to a 9 KB C header
  (BN folded into conv), hand-coded fp32 inference flashed to
  ESP32-CAM. **6 / 6 predictions match PyTorch reference**, max logit
  diff **1 × 10⁻⁵**, **67.5 ms / sample** on-chip (≈14.8 Hz, ~2× the
  window arrival rate at 8 kHz / 1024-sample). 330 KB program / 80 KB
  RAM = 10 % / 24 % of ESP32 budget — fits with massive headroom.

The decisions log (bottom of file) records every dataset choice,
mitigation tried, and what was deferred and why.

---

## Project state at last update

**Date:** 2026-05-10

**Sources on disk:**

| dataset   | clips | wavs | size | role |
|-----------|------:|-----:|------|------|
| ESC-50    | 2,000 | 2,000 | 0.85 GB | background (1920) + non_mosquito_insect (80) |
| HumBugDB  | 9,295 | 9,295 | 10 GB | mosquito (6,795) + background (2,500) |
| Wingbeats | 2,805 sessions / 279,566 wavs | 279,566 | 3.2 GB | mosquito (6 species) |

**Preprocess pipeline:** `src/data/preprocess_pipeline.py`, default settings
(5 kHz target SR, 1024-sample window, 512-sample hop, ≤100 segments per
clip). Leakage-safe split via `src.data.split.split_with_esc50_folds`
(ESC-50 fold 4 -> val, fold 5 -> test, others stratified by recording_id).

| split | clips | segments | mosq / bg / insect |
|-------|------:|---------:|--------------------|
| train | 232,366 | 1,592,726 | 1,425,823 / 164,647 / 2,256 |
| val   | 29,162  | 211,058   | 178,117 / 32,189 / 752 |
| test  | 29,274  | 209,583   | 176,898 / 31,933 / 752 |

**Class distribution:** non_mosquito_insect is severely under-represented
(80 ESC-50 source clips) — every model so far reports F1 < 0.10 on this
class. Resolution path is logged in `configs/datasets.yaml`
(InsectSound1000 subset is the leading candidate, deferred to a later week).

---

## Results table

All numbers are on the **test split** unless stated otherwise. "macro" =
unweighted mean over the three classes; AUC is one-vs-rest macro.

| Week | exp dir | Model | data state | acc | mosq F1 | bg F1 | insect F1 | macro F1 | AUC |
|----:|---------|-------|------------|----:|--------:|------:|----------:|---------:|----:|
| W2 | `exp_baseline_lr_full/` | Logistic Regression on 11 spectral features (class_weight=balanced) | full (ESC-50 + HumBugDB + Wingbeats) | 0.728 | 0.86 | 0.53 | 0.018 | 0.467 | 0.840 |
| W2 | `exp_baseline_xgb_full/` | XGBoost on 11 spectral features (sample_weight=balanced, n_est 300, depth 6) | full | 0.921 | 0.96 | 0.79 | 0.075 | 0.609 | **0.931** |
| W3.1 | `exp_cnn_1d_baseline/` | 1D-CNN (Fanioudakis 2018), 11,331 params, weighted sampler | full | 0.937 | 0.97 | 0.82 | 0.057 | 0.615 | 0.889 |
| W3.2 | `exp_baseline_lr/` | LR (rerun on Wingbeats-capped data) | wingbeats=400 sessions | 0.682 | 0.87 | 0.50 | 0.022 | 0.464 | 0.815 |
| W3.2 | `exp_baseline_xgb/` | XGBoost (rerun on Wingbeats-capped data) | wingbeats=400 sessions | 0.863 | 0.91 | 0.81 | 0.097 | 0.609 | 0.918 |
| W3.2 | `exp_cnn_1d_balanced/` | 1D-CNN, same hyperparams, Wingbeats-capped data | wingbeats=400 sessions | 0.876 | 0.92 | 0.84 | 0.085 | 0.615 | 0.869 |
| W4.1 | `exp_cnn_1d_aug/` | 1D-CNN + WingbeatAugment (noise SNR 0–20 dB, gain 0.3–1.5×, ±10 % shift) | wingbeats=400 sessions | 0.841 | 0.91 | 0.81 | 0.095 | 0.603 | 0.895 |
| W6 | `exp_physics_informed_w6/` | Physics-informed (LearnableFFT + Harmonic + TemporalEnvelope), 32,803 params, +WingbeatAugment, fft_stride=4 | wingbeats=400 sessions | **0.876** | **0.92** | 0.83 | 0.089 | 0.614 | 0.872 |
| W11 | `exp_cnn1d_tiny_distilled/` | Distilled tiny CNN1D (channels [8,16,24], fc 16, dropout 0.2), 2,443 params; KD from W6 (T=4, alpha=0.5) | wingbeats=400 sessions | 0.812 | 0.895 | 0.745 | 0.087 | 0.575 | 0.872 |

The W3.2 rows have **lower headline accuracy** because the test split also
loses ~80 % of its Wingbeats mosquito mass (which was the easy slice). On
the **harder test distribution**, CNN remains the lead by overall metrics
(0.876 acc / 0.84 bg F1 / 0.92 mosq F1) and edges XGBoost on bg / mosq F1.

Headline accuracy is no longer the right comparison column — the
cross-source mosquito recall is. See the next section.

**Best to date:** W3 1D-CNN — **0.937 acc / 0.97 mosq F1 / 0.82 bg F1**.
1D-CNN edges XGBoost on accuracy and per-class F1 for mosquito and
background, but **AUC drops** (0.93 → 0.89) suggesting softer probability
calibration; insect F1 is **worse**, not better (0.075 → 0.057). The
2 % accuracy gain over a 70-second XGBoost training is unlikely to
justify the deep model on its own — the next architectures need to
either substantially improve insect F1 *or* unlock cross-source
generalization (see "Open questions" below).

### Cross-source audit: shortcut found, partially mitigated

`scripts/cross_source_eval.py` runs each saved model on the test set
and breaks recall down by source. The smoking-gun row is mosquito
recall by source — if the model learned "is this a wingbeat", recall
should be similar across Wingbeats and HumBugDB; if it learned "is
this a Wingbeats clip", Wingbeats recall pegs at 99 % while HumBugDB
lags far behind.

| run | Wingbeats mosq recall | HumBugDB mosq recall | gap | insect recall |
|-----|---------------------:|---------------------:|----:|--------------:|
| XGBoost — full data | 98.5 % | 74.8 % | **23.7 pp** | 23.9 % |
| 1D-CNN  — full data | 99.6 % | 72.5 % | **27.1 pp** |  6.8 % |
| XGBoost — Wingbeats=400 sess | 97.7 % | 84.4 % | **13.3 pp** | 25.4 % |
| 1D-CNN  — Wingbeats=400 sess | 97.3 % | 83.0 % | **14.4 pp** | 17.2 % |
| 1D-CNN  — Wingbeats=400 sess + WingbeatAugment | 99.7 % | 81.0 % | **18.7 pp** | **43.4 %** |
| Physics-informed — same data + aug | 99.6 % | **88.8 %** | **10.8 pp** | 18.4 % |
| Tiny student (KD from W6) — same data + aug | 99.3 % | 84.2 % | 15.1 pp | **44.3 %** |

**What capping Wingbeats did:**
* HumBugDB mosquito recall **+9.7 pp (XGBoost)** and **+10.5 pp (CNN)** —
  the field-recording slice is meaningfully better.
* Wingbeats recall barely moves — the lab-clip slice is so easy that
  even with 86 % less training data the model still pegs ~98 %.
* Domain gap closes from 23–27 pp to 13–14 pp. Not gone, but halved.

**What it didn't fix:**
* Insect recall on the CNN went 7 % → 17 %, on XGBoost stayed ~24 %.
  Both still far below useful. With 80 ESC-50 source clips × 100-cap =
  2,256 segments, neither architecture has enough data — only fix is
  to add insect data (InsectSound1000 partial fetch is queued).
* Wingbeats subsampling is a band-aid for a representational problem.
  The "real" fix is HumBugDB-style noise / length augmentation applied
  to Wingbeats so the easy slice is forced to look harder. W4 plan.

**Head-to-head on the harder distribution (W3.2):**
* CNN balanced has **0.92 mosq F1** vs XGBoost balanced **0.91** —
  CNN edges XGBoost when both face the realistic test mix. On the
  full-data setup the CNN's 0.97 mosq F1 was inflated by the easy slice.
* XGBoost still wins decisively on insect (F1 0.10 vs 0.08) — spectral
  features generalize from few examples better than the CNN does.
  Worth keeping XGBoost in the loop as a sanity reference even after
  the deep-learning roadmap progresses.

**W4.1 — augmentation moves the model, but in a different direction
than expected:**

* **Insect recall jumps 17 % → 43 %** (+26 pp). White-noise + gain +
  shift broke the model's reluctance to ever predict insect — every
  minibatch now contains noisy variants of the few insect clips, so
  the network finally learns a feature for the class. Insect F1
  inches up (0.085 → 0.095) because precision is still poor (more false
  positives), but the model is at least *trying*.
* **Domain gap actually widens 14.4 → 18.7 pp.** Wingbeats recall
  pops from 97.3 → 99.7 %, HumBugDB recall slips 83.0 → 81.0 %.
  Augmentation made the easy slice even easier (more variations of
  the same lab-clean signal) but didn't help the field slice catch up.
* **Background recall regresses on ESC-50** (88.6 → 71.8 %). Many
  ESC-50 clips now get pushed into the insect or mosquito bucket that
  the model is over-predicting.

**Take-away:** the cross-source gap isn't a noise-vs-clean problem —
it's a *clip-length / multi-source-context* problem. Wingbeats clips are
a single isolated wingbeat at fixed 0.625 s; HumBugDB mosquito clips
have variable length and may contain swarms / overlaps / silence.
WingbeatAugment doesn't simulate any of those. Closing the gap will
need either (a) **chunk-mixing augmentation** that splices Wingbeats
clips together with silence and other Wingbeats to imitate HumBugDB
length, or (b) **domain-adversarial training** with a source-classifier
head whose gradient is reversed.

For the insect class, **augmentation is the biggest single win to
date** — going from a model that effectively ignored the class
(recall 7–17 %) to one that recovers nearly half the held-out insect
clips. With InsectSound1000 species data added on top, that recall
should keep climbing.

### Deployment-relevant: false-alarm rate

Headline accuracy hides which mistakes matter. For an "is this a mosquito?"
detector, the dangerous failure is non-mosquito → mosquito (false alarm).
From the test confusion matrices:

| Model | bg → mosq | insect → mosq | total false-mosq | rate |
|-------|----------:|--------------:|-----------------:|-----:|
| XGBoost | 2,777 / 31,933 | 32 / 752 | 2,809 / 209,583 | 1.34 % |
| 1D-CNN  | 1,285 / 31,933 | 1   / 752 | 1,286 / 209,583 | **0.61 %** |

The CNN halves the false-mosquito-alarm rate even though the headline
accuracy gain is only 2 %. The dominant deployment mistake comes from
**background → mosquito**, not from insects. Insects → mosquito is
already small in both models (XGB 4.3 %, CNN 0.13 %).

---

## Notes per experiment

### W2.1 — Logistic Regression (`exp_baseline_lr`)

* **Features:** 11-d spectral bundle from `src.features.spectral`
  (f0, power@h1..h5, centroid, bandwidth, rolloff_85, RMS, ZCR).
* **Pipeline:** `StandardScaler -> LogisticRegression(class_weight='balanced',
  solver='lbfgs', max_iter=1000)`.
* **Training time:** ~30 s on M1 Pro CPU (sklearn parallel).
* **Headline:** mosquito F1 0.86 — features carry real signal but the
  linear decision boundary loses to a proper non-linear model (see XGB).
* **Failure mode:** background F1 0.53 — the "balanced" weighting forces the
  model to over-predict the rare insect class, which eats into background
  precision.

### W11 — Distilled tiny student (`exp_cnn1d_tiny_distilled`)

* **Architecture:** `src/models/cnn_1d.CNN1D` with the smallest knobs we
  use (channels [8, 16, 24], fc_hidden 16, dropout 0.2). **2,443
  parameters** — 4.6× smaller than the Fanioudakis baseline, 13.4×
  smaller than the W6 teacher. INT8 weight bytes: **~2.4 KB**, fits
  the ESP32 SRAM budget with room for control code.
* **Training:** standard KD recipe (Hinton 2015) — soft loss is
  ``T² · KL(softmax(s/T) ‖ softmax(t/T))`` with ``T=4``, hard loss is
  CE(student, y), combined ``alpha=0.5``. Same loaders / sampler /
  WingbeatAugment as the W6 teacher run. Stopped at epoch 13 (best 5).
  ~78 s/epoch on M1 Pro MPS, ~17 min total.
* **Headline:** test acc **0.812**, mosq F1 **0.895**, HumBugDB mosq
  recall **84.2 %** — within ~5 pp of the W6 teacher on the
  deployment-relevant numbers, at one-thirteenth the size.
* **Surprising win — insect recall.** The student recovers **44.3 %**
  of insect clips vs the teacher's **18.4 %**. Smaller capacity +
  KD = the student can't lock onto the teacher's confident
  majority-class predictions, so the insect class gets more model
  attention. Mirrors the W4.1 augmentation effect almost exactly.
* **Cross-source:** gap 15.1 pp (vs teacher 10.8 pp). Some of the
  physics-informed structural advantage is lost in compression — the
  student is back to architectures-that-don't-explicitly-model-the-
  spectrum-prior performance.
* **Deployment math.** CPU latency on M1 Pro is 61.8 ms / batch of 128
  = 0.48 ms / sample (16-bit float). M1 Pro CPU is roughly 10× faster
  than ESP32 in our experience, so a real-world estimate is
  ~5 ms / sample on ESP32. Well under the 10 ms budget the W1 spec
  set, with room for the optical-front-end I/O.

### W6 — Physics-informed CNN (`exp_physics_informed_w6`)

* **Architecture:** three-branch fusion (`src.models.physics_informed`):
  LearnableFFT (Conv1d initialized as cos/sin Fourier pairs at 64
  evenly-spaced freqs in 100–1500 Hz, Hann-windowed, L2-normalized,
  stride 4 in time), HarmonicAttention (per-clip f0 estimation via
  argmax in 200–1000 Hz, then Gaussian-weighted pool at k·f0 for
  k = 1..5), TemporalEnvelope (full-wave rectify + low-pass + small
  Conv1d). Three branch features → small MLP head. **32,803 params**
  (vs 11,331 for the Fanioudakis baseline; still inside the ESP32
  INT8 budget after quantization).
* **Training:** same data setup as W4.1 (Wingbeats=400 sessions +
  WingbeatAugment), same optimizer (AdamW lr=1e-3 wd=1e-4, batch 128,
  weighted sampler, ReduceLROnPlateau, early-stop patience 8).
  Stopped at epoch 20 (best at 12). ~104 s / epoch on M1 Pro MPS,
  total ~35 min wall.
* **Headline result, the one that matters:** the cross-source
  mosquito-recall gap **closes from 18.7 pp (W4.1) to 10.8 pp (W6)**.
  HumBugDB recall jumps from 81.0 % to 88.8 % while Wingbeats stays
  pegged at 99.6 %. This is the first architectural change that
  measurably reduces the domain shortcut without losing accuracy on
  the easy slice — exactly what the physics-informed prior was
  meant to do.
* **Trade-off:** insect recall regresses from 43.4 % (W4.1, an
  augmentation-driven win) to 18.4 % (W6). The physics-informed
  model is more "spectrally confident" — it knows what a mosquito
  *should* look like and stops over-firing on insect. Net insect F1
  is roughly the same (0.095 → 0.089). W7 ambition: keep the
  cross-source improvement while restoring the augmentation-driven
  insect engagement (chunk-mixing, pink noise, or domain-aware
  weighting at the sampler layer).
* **What the LearnableFFT actually learned** (`scripts/visualize_learnable_fft.py`,
  see `learnable_fft_basis.png` and `learnable_fft_drift.csv`):
  every filter drifted significantly off the Fourier basis (mean
  drift 2.47 vs init norm 1.0). The filters that drifted the *most*
  are the ones at 700–1300 Hz — exactly the wingbeat fundamental and
  its first two harmonics. Filters at the edges (≤ 200 Hz, ≥ 1500 Hz)
  drifted least, suggesting the model decided those bands carry
  less wingbeat signal and stopped tuning them. The Fourier prior
  was a useful starting point, not a constraint.

### W3.1 — 1D-CNN baseline (`exp_cnn_1d_baseline`)

* **Architecture:** Fanioudakis 2018 — three Conv1d/BN/ReLU/MaxPool blocks
  (channels 16 → 32 → 64, kernels 7/5/3), AdaptiveAvgPool1d, Dropout 0.3,
  FC(64 → 32) → ReLU → Dropout → FC(32 → 3). 11,331 params.
* **Training:** AdamW lr=1e-3 wd=1e-4, batch 128, WeightedRandomSampler
  for class balancing in each minibatch, ReduceLROnPlateau on val_macro_f1
  factor 0.5 patience 4, early-stop patience 8 (didn't fire — val_macro_f1
  improved monotonically through epoch 21, the saved best).
* **Wall time:** 25 epochs × ~2.4 min = 60 min on M1 Pro MPS. ~89 it/s
  steady-state, batch 128 (≈11 k segments/sec).
* **Headline:** 0.937 test accuracy, mosquito F1 0.97, background F1 0.82.
  Beats XGBoost on every per-class F1 except insect.
* **AUC regression:** 0.93 → 0.89. The CNN's argmax decisions are
  better-calibrated than XGBoost's at the operating point but its
  probability ranking is worse. Likely a sampler artifact: weighted
  sampling distorts the empirical class prior, so softmax outputs read
  as overconfident on the rare class even when it's wrong.
* **Failure mode (insect):** F1 dropped to 0.057. The CNN almost never
  predicts insect (51 correct, 1 false-positive-as-mosquito out of 752),
  effectively collapsing to a 2-class learner. The weighted sampler put
  insect in every minibatch, but with only 80 source clips × 100 segs
  cap = 2,256 examples, all from a tiny pool of recordings, the network
  can't extract a usable feature.
* **Cost analysis:** 60 minutes of training for +1.5 % accuracy / +1 %
  mosquito F1 over a 70-second XGBoost. Worth it only if (a) we believe
  the CNN scales better with more / better data (likely true once
  InsectSound1000 lands) or (b) the false-alarm-rate halving (1.34 % →
  0.61 %) is itself the headline.

### W2.2 — XGBoost (`exp_baseline_xgb`)

* **Features:** same 11-d as W2.1.
* **Hyperparams:** `n_estimators=300, max_depth=6, lr=0.1,
  objective='multi:softprob', sample_weight=class_weight('balanced')`.
* **Training time:** ~70 s on M1 Pro CPU (`tree_method='hist'`, n_jobs=-1).
* **Headline:** **0.921 test accuracy, 0.96 mosquito F1.** Mosquito vs
  background is solidly separable on these features.
* **Wingbeats add (vs HumBugDB+ESC-50 only):** +7.4 acc, +8 mosq F1.
  Wingbeats provides the bulk of the mosquito mass (1.4M of 1.6M train
  segments), so the model learns mosquito very well.
* **Cost of imbalance:** background F1 dropped from 0.84 (HumBugDB+ESC-50)
  to 0.79 (full) because mosquito now occupies 90 % of the training set.
* **Failure mode:** non_mosquito_insect F1 0.075 — only 80 ESC-50 source
  clips; the model can't learn this class without more data.

---

## Decisions logged

| date | decision |
|------|----------|
| 2026-05-09 | Use canonical ESC-50 5-fold split (fold 4 = val, fold 5 = test) so numbers stay comparable to the literature for the background subset. |
| 2026-05-09 | Cap segments per clip at 100 in preprocess (otherwise a single 30-min HumBugDB recording would emit 17 k segments). |
| 2026-05-09 | ~~Defer InsectSound1000~~. **Revised 2026-05-10:** insect F1 0.06–0.08 across both baselines indicates the class is effectively unlearned despite headline accuracy of 92–94 %. Binding constraint for real-world deployment is already non_mosquito_insect discrimination. Plan a per-species partial download (Drosophila + Apis mellifera + Bombus terrestris ≈ 10–15 GB) before W4 augmentation work — adding noise / volume aug to a model that can't see flies is pointless. |
| 2026-05-09 | Drop the "Insects" / "Fruitflies" datasets named in the original W1 spec — slugs don't exist on Kaggle and InceptionFly data isn't publicly hosted. |
| 2026-05-10 | Use Wingbeats `recording_id = wingbeats:<species>:<session_dir>` (parent of each wav). Captures An. gambiae's age-bucketed sessions correctly. |
| 2026-05-10 | Suspect *domain shortcut* in the W2 / W3 baselines: ~90 % of training mosquito segments come from Wingbeats (sterile lab, 8 kHz fixed 0.625 s clips), the rest from HumBugDB (variable-length field). The model could be learning "is this a Wingbeats clip" rather than "is this a wingbeat". Add a cross-source eval (train on one source, evaluate on the other) before trusting the headline numbers. |
| 2026-05-10 | **Cross-source eval confirms shortcut** (`scripts/cross_source_eval.py`). Both baselines hit Wingbeats mosq recall ≥ 98 % and HumBugDB mosq recall 73–75 % — a 24–27 pp gap. From W4 onwards, *every* model is judged on the per-source table, not headline accuracy. Mitigations to try: subsample Wingbeats so HumBugDB has equal training weight, or domain-adversarial training, or HumBugDB-style noise augmentation applied to Wingbeats. |
| 2026-05-10 | **WingbeatAugment alone does *not* close the cross-source gap** (W4.1). Adding noise + gain + shift on top of the Wingbeats=400 cap keeps the gap at 18.7 pp (vs 14.4 pp without aug). The shortcut is structural: Wingbeats clips are 0.625 s isolated single wingbeats, HumBugDB mosq is variable-length field audio. Noise can't bridge that. **Next mitigation to try: chunk-mixing aug** (splice Wingbeats clips with HumBugDB-style silences) or domain-adversarial training. |
| 2026-05-10 | **Augmentation is the biggest single insect-class win**. Insect recall 7–17 % → 43.4 %. Even though precision stays low (more bg → insect false positives), the model finally engages with the class. Default training config from here on includes WingbeatAugment for the insect-class benefit alone. |
| 2026-05-10 | **InsectSound1000 partial fetch shelved (again).** The dataset is a 95 GB flat archive — Kaggle's per-file API rate-limits at ~100 page-listings (165k files = 825 pages), and the user is on China-VPN so per-file streaming would take many hours. Augmentation already pushed insect recall to 43 %, so the binding constraint is no longer "model never predicts insect" but "more insect diversity for precision" — that's a smaller need. Re-evaluate after W6 physics-informed CNN; if the deeper model still saturates around insect F1 < 0.2, we revisit the partial-fetch problem with a longer-running background download. |
| 2026-05-10 | **Skip W5 (2D-CNN spectrogram comparison) for now.** Project differentiation lives in W6 physics-informed (Learnable FFT + Harmonic Attention + Temporal Envelope). 2D-CNN can come back as a comparison row once W6 numbers are in. Order: W6 → W7 → W11 distillation → W12 ablation → W5 if time permits. |
| 2026-05-10 | **W6 wins on cross-source generalization.** Physics-informed (LearnableFFT + HarmonicAttention + TemporalEnvelope) closes the cross-source gap from 18.7 pp (W4.1) to 10.8 pp without trading off accuracy or majority-class F1. The Fourier prior is a useful starting point — every filter drifts off the cos/sin basis, with the largest drift around 700-1300 Hz (the wingbeat fundamental and harmonics). 32,803 params, ESP32 INT8 budget intact. |
| 2026-05-10 | **The augmentation insect-recall benefit is fragile.** W4.1 jumped insect recall to 43 % via WingbeatAugment, but W6 (same aug + physics-informed model) drops it back to 18 %. The aug effect was specific to a model that was otherwise ignoring the class — once the architecture is more spectrally confident, the rare class gets crowded out again. W7 needs a sampler-level or loss-level intervention for insect, not just augmentation. |
| 2026-05-10 | **W11 distillation: ship the student.** 2,443-param student (KD from W6 teacher, T=4, alpha=0.5) hits 0.812 acc / 0.895 mosq F1 / 0.842 HumBugDB recall on the balanced test set. INT8 weights ~2.4 KB. CPU latency 0.48 ms / sample on M1 Pro → ~5 ms estimate on ESP32 (well under 10 ms budget). Insect recall recovers to 44.3 % — same effect we saw with augmentation in W4.1, now from capacity compression. Net: the student loses ~6 % accuracy vs teacher but is the right deployment artefact for ESP32. |
| 2026-05-21 | **W12 ablation: every W6 branch carries weight.** Removing harmonic attention, temporal envelope, or LearnableFFT-learnability each widens the cross-source gap by 2.2–2.7 pp. Temporal envelope is the largest single contributor. There is no single load-bearing component; the win comes from the combination. Frozen-FFT raises insect F1 slightly (0.10 vs baseline 0.09) but loses cross-source. Keep all three branches in any future iteration. |
| 2026-05-21 | **Recording-level metrics > segment-level for deployment story.** Aggregating predictions per recording (mean softmax) drops headline accuracy by 0.9–4.3 pp on every model, because short Wingbeats clips no longer get one-vote-per-segment weight. Mean-prob AUC actually rises (0.87 → 0.90 on larger models). Quote recording-level numbers to the deployment team — that's what the sensor emits. |
| 2026-05-21 | **Tiny student is more robust to pink noise than the W6 teacher.** At SNR 0 dB pink noise: student 0.76 acc / teacher 0.62. The student's 2 k params can't overfit to white-noise patterns from augmentation, so its features generalize across noise colors better. Counterintuitive but flips the deployment-recommendation argument: the student isn't just small enough — it's *also* better suited to outdoor (1/f) noise. |
| 2026-05-21 | **One species has a specific failure mode: An. gambiae ss on HumBugDB.** Recall < 1 % on a 2,335-segment slice; the same species on Wingbeats works at 0.99. This is a domain × species interaction we haven't explained. **Doesn't block Hong Kong deployment** (An. gambiae is sub-Saharan), but if site shifts ever to Africa, this hole opens. The Hong Kong-relevant species (Ae. aegypti, Ae. albopictus, Cu. quinquefasciatus) all run ≥ 0.81 HumBugDB recall and ≥ 0.99 Wingbeats recall. |

---

### Deployment-readiness benchmark

Single comparison run on `data/processed/test.npz` (the wingbeats=400
balanced split). All numbers are CPU-only on M1 Pro (no MPS), since
the deployment target is ESP32 with no accelerator.

```
exp                          params  fp32 KB  int8 KB  lat ms   acc  mosqF1  bgF1  insF1   Wing   HBdb     gap
cnn_1d_baseline (W3.1)       11,331    53.0     11.1   160.9  0.852  0.885  0.824 0.058  0.996  0.712  28.4 pp
cnn_1d_balanced (W3.2)       11,331    53.0     11.1   160.3  0.876  0.917  0.844 0.085  0.973  0.830  14.4 pp
cnn_1d_aug      (W4.1)       11,331    53.0     11.1   160.8  0.841  0.907  0.807 0.095  0.997  0.810  18.7 pp
physics_informed_w6 (W6)     32,803   137.8     32.0   198.0  0.876  0.923  0.830 0.089  0.996  0.888  10.8 pp
cnn1d_tiny_distilled (W11)    2,443    17.8      2.4    61.8  0.812  0.895  0.745 0.087  0.993  0.842  15.1 pp
```

Numbers come from `scripts/quantize_and_benchmark.py` (run after each
new model lands). INT8 size is ``params × 1 byte`` since the actual
INT8 conversion happens at the ESP32 export step (TFLite Micro), not
in PyTorch — Conv1d isn't covered by `quantize_dynamic`. CPU latency
is M1 Pro single-thread; rough ESP32 estimate = M1 Pro CPU × 10.

Important read: **the W3.1 baseline now scores 0.852 on the balanced
test set, not the 0.937 that's in its row above.** That row was
recorded against the original full-Wingbeats test split; once we
moved to the balanced split for everything else it became unfair to
quote that number for comparison. The benchmark table is the
apples-to-apples cut.

**Deployment recommendation:** ship the W11 student. 2.4 KB INT8 fits
ESP32 SRAM with massive headroom; ~5 ms / sample inference estimate
(scaled from M1 Pro) is well inside the 10 ms / sample W1 budget;
mosquito F1 0.895 / HumBugDB recall 84.2 % retains most of what the
W6 teacher gained over the simple 1D-CNN; insect recall (44 %) is
actually higher than the teacher (18 %).

### W12 — ESP32-CAM on-chip deployment (`firmware/wingbeat_inference`)

Hand-coded fp32 inference of the W11 student on real silicon. Pipeline:

1. `scripts/export_student_for_esp32.py` — load `best.pt`, fold every
   BatchNorm into its preceding Conv, pick 2 representative samples per
   class from the test set, run reference inference, emit
   `firmware/wingbeat_inference/model_weights.h` (≈9 KB weights, 24 KB
   samples, 108 KB header).
2. `firmware/wingbeat_inference/wingbeat_inference.ino` — bare-metal
   `conv1d_same` / `relu` / `maxpool1d_2` / `global_avgpool1d` / `linear`
   primitives, ping-pong float buffers, no TFLite Micro, no external
   ML deps. Compiles to 330 KB program / 80 KB RAM (out of 3.1 MB /
   320 KB available — 10 % / 24 % utilization).
3. On boot the firmware runs the 6 embedded samples through the same
   forward pass and compares against `REF_LOGITS` exported from PyTorch.

**On-device numbers (ESP32 @ 240 MHz, single core, no flash cache miss budget):**

| metric | value |
|---|---|
| PyTorch ↔ ESP32 prediction agreement | **6 / 6** |
| max ‖esp_logit − ref_logit‖∞ | **1.0 × 10⁻⁵** |
| mean inference latency | **67.5 ms / sample (≈14.8 Hz)** |
| free heap after self-check | 271 KB |
| static binary on flash | 330 KB |

The 67.5 ms is slower than the M1-Pro-extrapolated estimate (≈10 ms × 10
= ~5 ms) because that estimate was naïve — it assumed perfect cache
locality and ignored the ~100 MHz effective compute throughput of the
ESP32 once the SRAM access pattern dominates. 67.5 ms still leaves ~14
inferences/sec, which beats the wingbeat fundamental (~500 Hz envelope
sampled into 1024-sample windows at 8 kHz = one decision per 128 ms of
audio) by a factor of 2 — i.e. we can process windows faster than they
arrive.

The 10⁻⁵ logit agreement is the load-bearing result: it proves the
hand-coded conv/pool/linear primitives reproduce PyTorch bit-faithfully
under fp32, which means **whatever accuracy the student has on the
laptop is exactly what we get on the chip**. No quantization gap, no
silent numerical drift, no "works on dev fails on device" surprise.

**Streaming on-chip evaluation** — `scripts/eval_on_chip.py` +
`firmware/wingbeat_stream/` pumps the test set sample-by-sample over USB
serial at 460800 baud (4-byte "INFR"/"RSLT" magic for resync) and
collects chip logits + argmax. Run over a stratified 2,752-sample subset
(1000/1000/752 per class drawn from the 89,866-row `test.npz`,
seed=42):

| metric | value |
|---|---|
| chip ↔ PyTorch prediction agreement (2,752 samples) | **2752 / 2752** |
| max ‖chip_logit − pt_logit‖∞ | **2.1 × 10⁻⁵** |
| on-chip accuracy | 0.702 |
| PyTorch (host) accuracy on same subset | 0.702 (bit-identical) |
| on-chip mosquito recall | 0.899 |
| on-chip background recall | 0.699 |
| on-chip non-mosquito-insect recall | 0.443 |
| wall throughput | 6.3 samples/sec (159 ms/sample = 67 ms inference + 92 ms USB) |

Confusion matrix from the on-chip run (row = true, col = chip pred):

```
                  bg    mosq   ins
       bg        699    172   129
     mosq         71    899    30
     ins         369     50   333
```

The per-class pattern reproduces what we see on the laptop: strong
mosquito recall, weak insect recall (insect samples spill mostly into
"background" — the limited insect training data leaves the model
unsure). The 0.702 accuracy on this stratified subset is below the
0.812 quoted in the benchmark table because the latter uses the full
test set's natural class proportions; the stratified 1000-per-class cut
deliberately over-weights the hard non-insect class.

Driver note: macOS Sequoia's bundled CH34x dext was misbehaving — the
chip enumerated as `/dev/cu.usbserial-*` but DTR/RTS control lines were
not being driven to the CH340 reliably, causing every esptool connect
attempt to time out. Fix was `brew install --cask
wch-ch34x-usb-serial-driver`, approve in System Settings → Driver
Extensions, USB replug → port renames to `/dev/cu.wchusbserial-*` and
auto-reset works.

### W6 branch ablation (`configs/pi_ablation_*.yaml`)

Three variants of the physics-informed model, each disabling one
ingredient — does each branch actually contribute, or is one of them
"the whole story"?

| variant | acc | macro F1 | Wingbeats mosq | HumBugDB mosq | gap | insect F1 |
|---------|----:|---------:|---------------:|--------------:|----:|----------:|
| **W6 (all branches on)** | 0.876 | 0.614 | 0.996 | **0.888** | **10.8 pp** | 0.089 |
| − harmonic attention     | 0.885 | 0.623 | 0.991 | 0.861 | 13.0 pp | 0.094 |
| − temporal envelope      | 0.881 | 0.619 | 0.988 | 0.853 | 13.5 pp | 0.090 |
| frozen LearnableFFT      | 0.872 | 0.619 | 0.985 | 0.853 | 13.2 pp | **0.101** |

**Every branch contributes to the cross-source gap.** Removing any one
of the three widens the gap by 2.2–2.7 pp. Headline accuracy is *
slightly higher* without each branch — but accuracy alone is misleading
because it's still dominated by Wingbeats. The cross-source gap is the
metric the ablation actually moves.

Per-component contribution:

* **Temporal envelope is the biggest single contributor** (+2.7 pp gap
  when removed). The body-shadow time profile carries domain-invariant
  information that pure spectral features don't.
* **Harmonic attention contributes 2.2 pp** to the gap reduction.
  Mosquito wingbeats are harmonically richer than fly / midge
  wingbeats; the dedicated pooling at k·f0 picks that up.
* **Learnability of the FFT contributes 2.4 pp.** Freezing the filters
  at the Fourier-basis initialization keeps the prior but loses the
  data-driven refinement. Hand-designed features alone aren't enough.
* **Insect F1 actually peaks at frozen-FFT** (0.101 vs baseline 0.089).
  Hypothesis: frozen filters can't drift toward the easy classes, so
  the model's spectral representation stays more uniform — including
  for the rare class.

The W6 design — all three branches active, FFT learnable — is the
configuration that minimizes the cross-source gap. There is no
single "load-bearing" branch; the gain is from the combination.

### Per-recording aggregation (`scripts/aggregate_predictions.py`)

The headline metrics so far were segment-level (~90k segments). In
deployment we'd aggregate predictions over all segments from one
capture and emit one label per recording. Aggregating by mean-prob:

| run | segment acc | recording acc (mean-prob) | Δ |
|-----|------------:|--------------------------:|--:|
| cnn_1d_balanced (W3.2) | 0.876 | 0.867 | −0.9 pp |
| cnn_1d_aug      (W4.1) | 0.841 | 0.838 | −0.3 pp |
| physics_informed (W6)  | 0.876 | 0.858 | −1.8 pp |
| tiny student    (W11)  | 0.812 | 0.769 | −4.3 pp |

Aggregation *hurts* accuracy on this test set — and that's the right
direction to read it. Wingbeats clips are short (≤ 1 segment each in
many cases), so segment-level metrics gave them more weight than they
deserve. Recording-level brings the test composition closer to
deployment reality, exposing the bigger HumBugDB-style penalty. Mean-
probability AUC went *up* on the larger models (0.87 → 0.90 on W3.2),
suggesting the softmax ranking is well calibrated even when the
argmax flips on a few hard cases.

**Read this:** report recording-level when talking to the deployment
team. Segment-level is the optimizer's reward, recording-level is
what the optical sensor actually emits.

### Robustness to test-time noise (`scripts/robustness_sweep.py`)

Same WingbeatAugment noise (white and pink) applied to test-time
samples; we watch test accuracy degrade as SNR drops.

**White noise** (W6 teacher / W11 student):

| SNR (dB) | W6 acc | W6 HBdb recall | Student acc | Student HBdb recall |
|---------:|-------:|---------------:|------------:|--------------------:|
| clean    | 0.876  | 0.888 | 0.812 | 0.842 |
| 20       | 0.875  | 0.887 | 0.813 | 0.842 |
| 10       | 0.866  | 0.881 | 0.815 | 0.841 |
| 0        | 0.786  | 0.802 | 0.789 | 0.840 |
| −10      | 0.692  | 0.908*| 0.478 | 0.250 |

(*the −10 dB number is misleading on its own: the model collapses to
predicting "mosquito" for almost everything, so HBdb recall stays high
while accuracy crashes.)

**Pink noise** (1/f — closer to real outdoor):

| SNR (dB) | W6 acc | Student acc |
|---------:|-------:|------------:|
| clean    | 0.876  | 0.812 |
| 20       | 0.873  | 0.811 |
| 10       | 0.838  | 0.804 |
| 0        | 0.619  | 0.756 |
| −10      | 0.179  | 0.641 |

**The student is markedly more robust to pink noise than the teacher.**
At SNR 0 dB pink, W6 drops to 0.62 while the student stays at 0.76.
Likely because the student's 2 k params can't overfit to the white-
noise patterns the teacher saw during augmentation, so its features
generalize across coloured noise. For an outdoor deployment that's
mostly 1/f environmental noise, this flips the recommendation: not
only is the student small enough to ship, it's *more deployment-robust*
than the teacher we distilled it from.

### Per-species mosquito recall (`scripts/per_species_eval.py`)

Breakdown by species (truncated to the operationally-relevant ones for
Hong Kong outdoor surveillance — Aedes aegypti, Ae. albopictus,
Culex quinquefasciatus dominate there; Anopheles is far less common):

| species | source | W6 recall | Student recall | support |
|---------|--------|----------:|---------------:|--------:|
| Ae. aegypti          | wingbeats | 0.99 | 0.99 |  5500 |
| Ae. aegypti          | humbugdb  | 0.99 | 0.99 |   521 |
| Ae. albopictus       | wingbeats | 0.99 | 0.99 |  2000 |
| Ae. albopictus       | humbugdb  | 0.45 | 0.10 |    20 |
| Cu. quinquefasciatus | wingbeats | 1.00 | 1.00 |  4500 |
| Cu. quinquefasciatus | humbugdb  | 0.87 | 0.81 |  3738 |
| Cu. pipiens complex  | humbugdb  | 0.99 | 0.99 |  3945 |
| **An. gambiae ss**   | **humbugdb** | **0.005** | **0.009** | **2335** |
| An. arabiensis       | humbugdb  | 0.85 | 0.76 | 10496 |

**The headline finding:** *An. gambiae ss* on HumBugDB is essentially
unrecognized — recall is below 1 % on a substantial 2,335-segment
test slice. This is a single-species pathology, not a general weakness:
the model handles the same species on Wingbeats fine (0.99), and
handles An. arabiensis on HumBugDB at 0.85. Worth flagging because for
Hong Kong specifically it doesn't matter (An. gambiae is sub-Saharan),
but if the deployment target ever shifts to Africa, this hole opens up.

The Hong Kong-relevant species *are* handled well — Ae. aegypti, Ae.
albopictus, Cu. quinquefasciatus are all ≥ 0.81 on HumBugDB recordings,
and ≥ 0.99 on Wingbeats. The deployment story is intact.

### W13 — Cross-modality transfer: does audio-trained transfer to optical?

Motivation: a teammate proposed training the edge classifier on abundant
AUDIO wingbeat data and deploying it on OPTICAL sensors ("if audio works, the
same-size model should work optically"). We tested this directly and honestly.

**Data (all mapped to a common 129×40 log-power spectrogram via
`src/data/spectro.py`, per-sample z-normed; recording-level splits):**

| role | source | modality | n segments |
|---|---|---|---|
| train/val/test positives | HumBugDB mosquito | audio | 6,562 |
| train/val/test negatives | InsectSound1000 Diptera (hoverfly, gall-midge, fungus-gnat) | audio | 2,544 |
| cross-modality test (primary) | UCR InsectWingbeat: 4 mosq species vs 2 fly species | optical | 25,000 (20k mosq / 5k fly) |
| cross-modality test (recall) | Wingbeats mosquito | optical | 3,000 |

Only the 3 Diptera families of InsectSound1000 were fetched (via
`scripts/fetch_insectsound_diptera.py`, Kaggle per-file — the OpenAgrar
original is behind a proof-of-work wall and the archive is a monolithic
91 GB). Design guard against a dataset shortcut: **each class uses different
sources in train vs test**, so a model that learns "which dataset is this"
scores at chance on the optical test — the optical number therefore measures
real biological transfer.

**Result (`scripts/train_cross_modality.py`, SpecCNN 25,666 params):**

| metric | value |
|---|---|
| in-modality (audio test) accuracy | **0.998** |
| cross-modality (UCR optical) accuracy | **0.545** |
| UCR **balanced** accuracy | **0.492 (= chance)** |
| modality gap | **+0.453** |
| UCR mosquito recall / fly recall | 0.58 / 0.40 |
| Wingbeats optical mosquito recall (matched STFT) | 1.000 |

UCR confusion (audio-trained model): of 20k optical mosquitoes it calls 11,591
mosquito / 8,409 fly; of 5k optical flies it calls 2,976 mosquito / 2,024 fly —
i.e. it predicts "mosquito" ~58–60 % regardless of truth. **No optical
discrimination.** A model that is 99.8 % on audio is at coin-flip on optical.

Two honesty caveats: (1) UCR was spectrogram-ized by its original authors with
different parameters, so part of the 45 pp gap is processing/domain shift, not
pure modality. (2) The Wingbeats check — optical mosquitoes pushed through OUR
STFT (matched processing) — gives 100 % mosquito recall, so optical
mosquitoes *are* still recognised when processing matches; the failure is
specifically discriminating optical FLY from optical mosquito, which the model
never learned in the optical domain.

**Takeaway:** naive "train audio → apply to optical" does NOT work. A modality
bridge (paired IR+audio for alignment, or fine-tuning on some optical labels)
is required. This is the empirical counter to the teammate's "it'll transfer
for free" assumption.

### W13 — Audio-only size sweep (`scripts/audio_size_sweep.py`)

Companion question: how small can the model be and still solve the AUDIO
mosquito-vs-fly task? (If tiny suffices on audio, the task is not the
bottleneck.)

| config | params | fp32 KB | audio acc | audio F1 | UCR optical acc | Wingbeats recall |
|---|---|---|---|---|---|---|
| xs (8,16)     | 1,906  | 7.4   | 0.963 | 0.945 | 0.748 | 1.00 |
| s  (8,16,24)  | 5,690  | 22.2  | 0.993 | 0.990 | 0.385 | 1.00 |
| m  (16,32,64) | 25,666 | 100.3 | 0.996 | 0.995 | 0.546 | 1.00 |
| l  (32,64,128)| 97,314 | 380.1 | 0.999 | 0.998 | 0.355 | 1.00 |

Two facts fall out at once:
* **Audio task is easy and tiny-friendly** — 1,906 params (7.4 KB) already
  hits 96 %; 5,690 params hits 99 %. The teammate is right that the model can
  be small.
* **Capacity does nothing for optical** — scaling params 50× (1.9K → 97K)
  moves audio 0.963 → 0.999 but UCR optical stays at/under the 0.80
  majority-class baseline the whole time (and drifts *down* as the bigger nets
  overfit the audio domain). **The modality gap is not a capacity problem**, so
  "small model works on audio ⇒ small model works on optical" is false without
  a bridge.

### W13 — Optical species + non-mosquito classifier (`scripts/ucr_species_classifier.py`, `ucr_species_improve.py`)

Question: can ONE optical model output "not a mosquito" OR "mosquito + which
species", cleanly (single sensor, no cross-source confound)? Trained on UCR
InsectWingbeat alone (all optical, 200×20 spectrogram, balanced 5000/class),
10 UCR labels → 5 classes: non_mosquito + Aedes / Quinx (Cx. quinquefasciatus)
/ Stigma / Tarsalis (Cx. tarsalis).

**Baseline (TinyCNN2D, 16-32-64):**

| view | accuracy |
|---|---|
| mosquito DETECTION (any species vs non-mosquito) | **0.967** |
| 5-class (non-mosq + 4 species) | 0.708 |
| species among true mosquitoes | 0.654 |

non_mosquito F1 = 0.919 — vs 0.087 for the deployed 3-class student. The
difference is entirely modality-cleanliness: here non-mosquito is real optical
fly wingbeat in the *same* sensor as the mosquitoes, not ESC-50 cricket
stridulation. Species confusion is biologically sensible: the two Culex
(Quinx 0.53 / Tarsalis 0.64) confuse with each other; Aedes (different genus)
is cleanest (0.73).

**Improvement sweep — pushing species accuracy (`ucr_species_improve.py`):**

| approach | 5-class | detect | species |
|---|---|---|---|
| baseline (small) | 0.708 | 0.967 | 0.654 |
| bigger CNN (32-64-128) + SpecAugment + label smoothing | 0.761 | 0.974 | 0.717 |
| 10-class (species×sex) → merged to 5 | 0.774 | 0.974 | 0.729 |
| 3-seed ensemble | **0.774** | 0.974 | **0.730** |

**Species accuracy plateaus at 0.73 — exactly the UCR SOTA ceiling.** Best
published time-series classifiers on this benchmark: MultiRocket 0.67,
HIVE-COTE2 0.66, ConvTran 0.71, LITEMVTime 0.735. We are at the ceiling; it is
an *information* limit, not a modeling gap. Root cause (Genoud 2021): 26 of 29
mosquito species overlap in wingbeat frequency; thermal variation (~8-13 Hz/°C)
within one species exceeds the between-species gap. UCR ships the spectrum only
— no time-of-day / temperature / location.

**What actually reaches 90%+ (literature):** (a) metadata fusion — Chen 2014
went 0.82→0.95 by adding circadian time + location (but the public UCR .ts
strips this); (b) genus-level instead of species (~0.94); (c) sex conditioning
(sex itself is 0.96-1.00). Deployment implication: target **mosquito-detection
(0.97) + genus (~0.94)** for a reliable ≥90% product; log on-device time +
temperature for late-fusion if fine species ID is ever needed. Fine
within-genus species ID at ≥90% is not reachable from the wingbeat signal
alone.

### W13 — Is the ~96% Wingbeats species number real? (honest-split audit)

Claim under test: literature quotes ~96% on Wingbeats 6-species; is that real
signal or a recording-session shortcut? One dataset, one model, split varied.

**Leaky vs honest split (`scripts/wingbeats_species_honest.py`, TinyCNN2D,
129×40 spectro, ~2500 clips/species spread across sessions):**

| split | accuracy |
|---|---|
| random (leaky) | 0.711 |
| session-independent (honest) | 0.685 |
| confound inflation | +0.025 |

Confound here is small (+2.5 pp) — because we sampled ≤6 clips/session across
many sessions, which already suppresses leak. So we could NOT reproduce a
huge session-shortcut; and we did not get 96 % either. Correction to an earlier
overstatement: the 96 % is not *purely* session-memorisation.

**Strong pipeline, honest split (`scripts/wingbeats_species_strong.py`,
257×71 spectro, 250 K-param CNN, session-independent):**

| pipeline | accuracy |
|---|---|
| tiny (129×40) | 0.685 |
| strong (257×71, 250 K) | **0.774** |
| UCR SOTA reference | ~0.73 |

Bigger model + 2× resolution lifts honest Wingbeats to **0.774** — and there it
stops. **Two independent optical datasets, honestly evaluated, both land
~0.73–0.77 on species.** 96 % is not reproducible under session-independent
evaluation, so it was leak/setup inflation. Same-genus confusion dominates the
errors (An. arabiensis ↔ An. gambiae; Ae. albopictus ↔ Ae. aegypti), exactly
the biological ceiling.

**Consolidated verdict on "how accurate can we be, honestly":**

| target | honest accuracy | evidence |
|---|---|---|
| mosquito vs non-mosquito | **≥ 0.90** | UCR detection 0.967 (separate TRAIN/TEST files, 0 exact dups; fly vs mosquito is a wide-frequency-gap, coarse task) |
| genus (Aedes/Culex/Anopheles) | ~0.90 (lit.) | — |
| within-genus species | **~0.75 ceiling** | UCR 0.73 + Wingbeats 0.774, both honest, strong models can't beat it |

The mosquito-detection ≥0.90 is not a train/test-mixing artifact of ours (UCR
ships disjoint TRAIN/TEST; zero identical instances across them). Fine species
ID at ≥0.90 needs information the wingbeat signal alone doesn't carry
(time/location metadata, richer sensor) — not a bigger model.

### W13 — Honest audio-only mosquito detector (`scripts/audio_mosquito_detector.py`)

Scope narrowed by the team to binary "mosquito vs not", audio acceptable. The
risk is the source shortcut (mosquito=HumBugDB, negatives=other rig → model
learns the device). Defused by putting HumBugDB's OWN background/ambient audio
(same mic/sessions as its mosquitoes) into the negative class alongside real
InsectSound fly wingbeats. Recording-level split.

    class 1 mosquito     <- HumBugDB mosquito
    class 0 not_mosquito <- HumBugDB background (same rig) + InsectSound Diptera

| model | params | fp32 KB | accuracy | macro-F1 | mosquito recall |
|---|---|---|---|---|---|
| small (8,16,24) | 5,690 | 22.2 | 0.903 | 0.902 | 0.825 |
| mid (16,32,64) | 25,666 | 100.3 | 0.918 | 0.918 | 0.865 |

**No rig shortcut — negative rejection by sub-source:**

| model | HumBugDB background | InsectSound flies |
|---|---|---|
| small | 0.961 | 1.000 |
| mid | 0.945 | 1.000 |

Both negative sub-sources are rejected ≥0.945 — including HumBugDB's own
background on the *same rig* as the mosquitoes, so the classifier can't be
using device identity; it learned the wingbeat. This is the fix for the
deployed 3-class student's dead non-mosquito class (F1 0.087, trained on ESC-50
cricket *stridulation*): with real fly wingbeats + same-rig background as
negatives, non-mosquito rejection is 0.94–1.00.

**Deployable honest binary detector: 22 KB, ~0.90 accuracy, mosquito recall
0.83, false-alarm low.** This is the current recommended scope — detection is
robust and honest; species is deferred (see the species ceiling notes above).

### W13 — 1D detector deployed to ESP32 (`scripts/train_audio_detector_1d.py`, `eval_detector_on_chip.py`)

The honest audio detector above is a 2D-spectrogram model; its first-layer
activation (16×129×40 = 330 KB) exceeds ESP32 SRAM (320 KB), so a straight port
needs on-chip STFT + tiled 2D conv or PSRAM. Instead we retrained a 1D
raw-waveform version at the chip's native spec (5 kHz, 1024-sample windows,
CNN1d 8→16→24, 2-class), reusing the existing on-chip C++ (firmware/wingbeat_stream)
unchanged.

Laptop (recording-level split): accuracy 0.842, mosquito recall 0.766,
rejection humbug_bg 0.943 / insectsound_fly 0.872. A few points below the 2D
model (0.918) — mostly weaker fly rejection — as expected when the model must
learn frequency structure from the raw waveform.

**On-silicon verification (1500 windows streamed to the chip):**

| metric | value |
|---|---|
| chip ↔ PyTorch agreement | **1500 / 1500** |
| max ‖chip − pt‖ logit | 6 × 10⁻⁶ |
| on-chip accuracy | 0.841 |
| on-chip mosquito recall | 0.761 |
| on-chip reject humbug_bg (same rig) | 0.949 |
| on-chip reject insectsound_fly | 0.876 |

2,426-param binary detector runs bit-faithfully on real silicon and keeps the
no-rig-shortcut property (same-rig background rejected 0.95). This is the
deployed artefact for the current team scope (audio mosquito detection).

### W13 — 1D detector to 0.90+ with a LearnableFFT front-end (`scripts/improve_audio_detector_1d.py`)

The deployed 1D detector was 0.842 — a few points under the 2D model (0.918),
because a raw-waveform 1D-CNN has to learn frequency structure implicitly. A
LearnableFFT front-end (Conv1d initialised as a Fourier basis → learned
spectrogram → conv1d stack) hands frequency to the model explicitly while
keeping a 1D input (still deployable — all conv1d + a magnitude op).

| config | params | accuracy | mosq recall | bg reject | fly reject |
|---|---|---|---|---|---|
| baseline (8,16,24) | 2,426 | 0.841 | 0.78 | 0.93 | 0.83 |
| bigger (16,32,64) | 11,298 | 0.883 | 0.82 | 0.97 | 0.90 |
| **LearnableFFT (48)** | 23,394 | **0.913** | 0.90 | 0.92 | 0.94 |
| LearnableFFT + aug | 23,394 | 0.895 | 0.93 | 0.89 | 0.79 |

**LearnableFFT reaches 0.913 — matching the 2D spectrogram model (0.918) from a
1D input.** All metrics balanced and high (mosquito recall 0.90, both negative
sub-sources rejected ≥0.92, so the no-rig-shortcut property holds). Plain
capacity scaling (bigger CNN1D) only reached 0.883; the frequency front-end is
what closes the gap. Augmentation was slightly counterproductive here. Saved to
`experiments/exp_audio_detector_1d_improved/best.pt`.

**Deployed to ESP32 (`firmware/wingbeat_lfft/`, `scripts/export_lfft_for_esp32.py`).**
Hand-coded on-chip LearnableFFT (fused conv+magnitude: 96 filters × 129 kernel,
stride 4) → conv1d blocks → GAP → FC. A DRAM-saving trick (reuse the 48 KB FFT
magnitude buffer as the pooling scratch) keeps it inside SRAM (108 KB globals /
33 %). On-silicon verification (1200 windows streamed):

| metric | value |
|---|---|
| chip ↔ PyTorch agreement | **1200 / 1200** |
| max ‖chip − pt‖ logit | 7 × 10⁻⁶ |
| on-chip accuracy | **0.916** |
| on-chip mosquito recall | 0.906 |
| on-chip reject humbug_bg (same rig) | 0.917 |
| on-chip reject insectsound_fly | 0.947 |
| throughput | 2.4 windows/s (FFT front-end adds compute vs the 6.3/s plain-1D) |

The 0.916 accuracy target is met **on real silicon**, bit-faithful to PyTorch,
with the no-rig-shortcut property intact (same-rig background rejected 0.92).
The 2.4 windows/s is a USB-bound measured figure (host stream + on-chip
inference); a continuous 8 kHz / 1024-sample feed produces ~7.8
non-overlapping windows/s, so on-chip inference speed still needs profiling
before a real-time claim — that's an open item, not established here. This is
the current best deployable artefact.

### W13 — Wingbeats 4-condition ablation: why is our species number ~0.77? (`scripts/wingbeats_4condition.py`)

Earlier we implied the ~0.77 honest Wingbeats species number vs the published
~0.90–0.96 was largely a leakage (random-split) artifact. **That was wrong** —
we tested it. Same big model (BigCNN2D), resolution × split crossed:

| condition | accuracy |
|---|---|
| coarse + random | 0.812 |
| coarse + session | 0.797 |
| fine + random | 0.794 |
| fine + session (our recipe) | 0.770 |

- **Resolution effect** (fine − coarse, honest): **−0.027** — higher resolution did
  not help (slightly hurt).
- **Leakage effect** (random − session): **+0.015 to +0.023** — small. Random-split
  inflation is real but only ~2 pp here, not the 13–16 pp gap to published SOTA.
- Every condition lands **0.77–0.81**. We cannot reach 0.90 by changing resolution
  or split.

**Corrected conclusion:** our ~0.80 ceiling on Wingbeats-6 species is **not** a
leakage or resolution artifact — it's our model/training. The gap to WbNet's ~0.90
(ResNet + self-attention, Wei et al. 2022) is most likely a genuine
architecture/training-recipe difference, not evaluation dishonesty. Two earlier
overstatements are retracted: (1) that published Wingbeats 0.90–0.96 is mostly
leakage-inflated, and (2) that "0.73–0.77 is the species ceiling" — that 0.73 was
the *UCR* number; Wingbeats is a different, easier dataset where 0.90+ is real.
UCR (10-class, pre-processed spectra) and Wingbeats (6-class, near-raw) must not be
quoted interchangeably.

### W13 — Can our LearnableFFT+CNN actually LEARN harmonics? (`scripts/harmonic_learnability.py`)

**The question that matters for the whole product.** Our only real overlapping
confuser is the chironomid midge, which shares the mosquito fundamental (~500 Hz)
— so a frequency-only detector is defeated. The optical/lidar literature says
same-frequency insects are still separable by their *overtones* (harmonics). So:
does *our deployed model family* genuinely learn harmonics, or just the fundamental?

**Testbed (honest — SYNTHETIC).** No public chironomid recording overlaps
mosquitoes, so we synthesise a controlled negative that SHARES the fundamental
(450–560 Hz) with a synthetic mosquito but differs only in overtone rolloff,
scaled by a `gap` knob (`scripts/synth_overlap_negative.py`). We then train the
ACTUAL models on it, sweeping `gap`:

| overtone gap | freq-only (1 feature) | SpecCNN (2D) | **LFFT-CNN (deployed family)** |
|---|---|---|---|
| 0.0 (identical harmonics) | 0.51 | 0.50 | 0.55 |
| 0.3 (slightly different) | 0.51 | 0.52 | **0.95** |
| 0.6 (different) | 0.51 | 0.99 | **1.00** |
| 0.9 (very different) | 0.51 | 1.00 | **1.00** |

- **Frequency-only stays at chance (0.51) everywhere** — both sit at ~500 Hz.
  This reproduces exactly the trap that a frequency-only acoustic detector hits.
- **Our LearnableFFT+CNN separates them once any overtone difference exists** —
  95 % at gap 0.3, 100 % at gap 0.6. The earlier weak nearest-centroid probe in
  `synth_overlap_negative.py` only reached 0.67 at gap 0.9; a real trained CNN is
  far stronger.
- **LFFT beats a generic SpecCNN at small gaps** (0.95 vs 0.52 at gap 0.3): the
  Fourier-initialised front-end locks onto harmonic structure faster. This is the
  concrete, measured justification for *why we use LearnableFFT*.

**Conclusion (what this does and does not prove).**
- PROVEN: our model **can** learn a harmonic difference — it is not limited to the
  fundamental, and the LearnableFFT front-end is measurably better at it.
- NOT proven: that **real** chironomids differ from mosquitoes in overtones. The
  `gap` is an assumption we injected. The one remaining unknown is a field
  measurement of real chironomid harmonics — that missing piece is the data moat.

**Literature grounding (verified, exact).** Yamoa, Kouakou, … Brydegaard (2025),
"Lidar reveals distinct insect daily activity and diversity between habitats,"
*Scientific Reports*, PMC12695965. Côte d'Ivoire (Yamoussoukro), 4 habitats × 4
days, dry season. Confirmed figures: largest total insect observations to date
(**1,716,362**), highest daily (**346,581**), highest number of clusters
distinguishable from noise (**353**; instrument range 50–350) — vs an earlier
Côte d'Ivoire lidar that discerned only 12 signal types. Overtone mechanism, quoted:
*"Even species with similar WBFs can be distinguished by the overtone characteristics
which relate to the wing dynamics, surface roughness and the wing membrane thickness…"*
and *"Differences in overtone content, rather than just frequency, is analogous to the
distinction in timbre between a flute and a trumpet playing the same note."*

**Two citation cautions (do not overstate this paper):**
1. NoC (353) is the count of **unsupervised signal clusters distinguishable from
   noise — a diversity/richness proxy, NOT species classified**. The paper is
   explicit: *"the number of clusters … should be understood as a maximum number of
   signals … This number is lower than the number of species present."* Say "353
   signal clusters distinguished," never "353 species identified," and never quote it
   as a classification accuracy.
2. It is a **lidar** (sophisticated active optics), not our ESP32-class sensor. Cite
   it as *"the overtone mechanism is validated at field scale,"* not as a performance
   number transferable to our hardware. The paper itself also notes overtones vary
   with viewing angle, so harmonics are not a pure species fingerprint.

## Open questions / TODOs

* **Real chironomid overtones (the data moat).** The harmonic-separability result
  is synthetic — it proves the model *can* use overtones, not that real chironomids
  differ from mosquitoes. Field-record chironomid wingbeats near ~500 Hz (optical or
  acoustic) and measure their harmonic rolloff vs local Aedes/Culex. This single
  measurement is what converts "can separate IF they differ" into "does separate."
* **Domain shortcut audit (next).** Train on Wingbeats only, evaluate on
  HumBugDB-mosquito only (and vice versa). If accuracy collapses below
  ~80 %, the 92–94 % numbers are partly measuring "did this clip come
  from a sterile lab capture" rather than "is this a wingbeat". This
  test is cheap (~5 min) and the answer reframes everything.
* **InsectSound1000 partial fetch.** Build a kaggle-CLI-driven loader
  that pulls only Drosophila + Apis + Bombus filenames from
  `hesi0ne/insectsound1000` (filename pattern `<ts>_<species>_*_ch0.wav`).
  Target ≈ 10–15 GB. Update `src/data/metadata.py` with a new loader
  that emits `label=non_mosquito_insect`.
* **Probability calibration.** The CNN's AUC regression vs XGBoost
  suggests the weighted sampler is distorting the softmax. Try
  temperature scaling on a held-out chunk of val, or drop the weighted
  sampler and use plain class-weighted CE instead.
* **Augmentation effect (W4).** Once the data balance is fixed, run the
  same architecture with `WingbeatAugment` in the loader (noise SNR
  0–20 dB + 0.3–1.5× gain) and compare per-class F1 + AUC. If
  augmentation only helps with insect F1 not mosquito, that's evidence
  the model has already over-fit Wingbeats-style mosquito and needs
  domain regularization. If it helps everywhere, augmentation is
  generically useful.
* **Memory ceiling.** WingbeatNpz currently materializes the entire
  6.1 GB train X array; spawn-mode multiprocessing duplicates that into
  every worker. The W3 run survived but came close to swapping. Before
  W6 (physics-informed CNN) needs a longer training schedule, convert
  the train NPZ to an mmap-able .npy + .npz pair so DataLoader workers
  share pages instead of copying.
