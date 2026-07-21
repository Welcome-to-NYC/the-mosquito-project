# Data choices — what we used and why

Record of which datasets went into the deployed **audio mosquito detector**, and
why. (For the full experiment ledger incl. optical work, see
`experiments/RESULTS.md`. For the non-mosquito data search, see
`experiments/optical_non_mosquito_data_search.md`.)

## The deployed detector uses only these (all audio)

| role | dataset | what | why this one |
|---|---|---|---|
| mosquito (positive) | **HumBugDB** (Kiskin et al., NeurIPS 2021) | 34 mosquito species, smartphone recordings, real outdoor/field conditions + its own background/ambient audio | Only large public corpus of **field-condition** mosquito audio with species labels. Real deployment noise, not sterile lab. |
| non-mosquito (negative) — flies | **InsectSound1000** (Branding et al., Sci. Data 2024) | 3 Diptera: hoverfly (*Episyrphus balteatus*), gall midge (*Aphidoletes aphidimyza*), fungus gnat (*Bradysia difformis*) | Real close-mic **wingbeat** audio (not stridulation). The 3 chosen are the ones whose wingbeat frequency overlaps the mosquito band — i.e. genuine *hard* negatives. |
| non-mosquito (negative) — background | **HumBugDB background/ambient** | non-mosquito audio from the *same* devices as the mosquitoes | Defuses the "device shortcut": if mosquito=HumBugDB and every negative came from a different rig, the model could learn the recording device instead of the wingbeat. Same-rig background forces it to learn the wingbeat. |

## Why NOT the other datasets we had

- **Wingbeats (optical, 6 mosquito species)** — optical, and **mosquito-only** (no
  negatives). Used only for the separate optical experiments; audio→optical does not
  transfer, so it is not mixed into the audio detector.
- **UCR InsectWingbeat (optical: 4 mosquito + Drosophila + Musca)** — optical
  pre-computed spectrograms, different modality/sensor. Used for the optical
  feasibility experiments only. Mixing it in would reintroduce a modality/source
  shortcut.
- **ESC-50 "insects" (crickets/cicadas)** — the *old* negative class. These are
  **stridulation** (leg-rubbing), not wingbeats, so a model trained on them never
  learns to reject a flying insect. Replacing ESC-50 with real fly wingbeats
  (InsectSound1000) is exactly what fixed the dead non-mosquito class (F1 0.087 →
  rejection 0.92–0.95).

## Why only 3 non-mosquito species (the honest gap)

This is **not** a shortcut we took — it is a limit of what exists publicly. A search
across audio and optical repositories (2026) found:

- The 3 InsectSound1000 Diptera we used are essentially the **only** public close-mic
  wingbeat recordings of insects whose frequency overlaps mosquitoes (~200–1000 Hz).
- InsectSound1000's other 9 species are below the band (bumblebee, stink bugs,
  ladybird, moth) → easy negatives, little added value. Borderline: greenhouse
  whitefly (~165–224 Hz, 2nd harmonic clips the band) and aphid (~120 Hz).
- Optically, only UCR's Musca + Drosophila add anything, and they are low-band
  (~190–250 Hz) and a different sensor.
- **The real hard confusers — Chironomidae (midges), Simuliidae (black flies),
  Ceratopogonidae (biting midges), Psychodidae, Tipulidae — have NO public audio or
  optical wingbeat data at all.** Papers report only frequency numbers, no reusable
  audio.

**Implication:** the non-mosquito diversity gap can only be closed by **recording it
ourselves** (chironomids swarm at dusk near water → thousands of samples in a few field
sessions). That missing data is the project's differentiator, not a weakness — it is
data no one else has.

## Split & honesty

- Recording-level train/val/test split (group by recording id) — no clip leaks across
  splits, so accuracy is not inflated by memorising a recording.
- Classes balanced ~1:1 (mosquito ≈ not-mosquito) + weighted sampling.
- Verified no device shortcut: same-rig HumBugDB background rejected 0.92 on-chip.
