# Ableton Sample Tagger — Project Journal

A running record of what we're building, the tools involved, and what we've learned.
Newest findings are appended to the **Log** at the bottom.

## Goal

Stop hunting for sounds one by one. Instead of "find a crash I like", say *"some crash
here"* and have Live drop several plausible candidates in place to audition.

Two parts:

1. **Sample tagger** (Python, runs outside Live): listens to every file in the sample
   library and records what kind of sound it is (kick, snare, cymbal, riser, ...).
2. **Live extension** (JavaScript, runs inside Live): right-click a time selection →
   "Try 5 crashes here" → places candidates on take lanes so you can comp between them.

## Layout

```
~/Dev/ableton-sdk/
├── JOURNAL.md                      ← this file
├── .nvmrc                          Node 24.16.0 for this folder
├── extensions-sdk-1.0.0-beta.1/    Ableton SDK zip (docs/, api/, examples/)
├── test-extension/                 hello-world extension (works)
└── sample-tagger/                  Python project (uv)
    ├── scripts/extract_video_audio.py
    ├── tagger/        audio, features, panns, index, rules, truth, CLI
    └── data/index.sqlite           analysis results (gitignored)
```

## Tools

| Tool | What it is | Why we use it |
|---|---|---|
| **Ableton Extensions SDK** (1.0.0-beta.1) | JavaScript/TypeScript API for scripting Live; extensions run on Node.js in a separate "Extension Host" | Placing clips, take lanes, right-click menu actions |
| **Node.js 24** via **nvm** | JavaScript runtime | Required by the SDK. Default stays on 22 (Paramify needs it); `nvm use` in this folder switches to 24 |
| **ffmpeg / ffprobe** | Audio/video swiss-army knife | Decoding every audio format; pulling audio out of videos |
| **uv** | Fast Python package/environment manager | Isolated Python env for the tagger (`uv run ...`) |
| **librosa** | Python audio analysis library | Our hand-built measurements (envelope, spectrum, onsets, key) |
| **PyTorch** | Neural network library | Runs the PANNs model |
| **PANNs** (Cnn14) | Pretrained audio classifier, trained on Google's AudioSet (2M YouTube clips, 527 sound classes) | Scores for "Snare drum", "Cymbal", "Speech", ... and a 2048-number *embedding* per file |
| **CLAP** *(deferred — see "Why not CLAP (yet)")* | Two networks (audio encoder + text encoder) trained so sounds and their descriptions land close together | Producer-specific categories (riser, impact) that AudioSet lacks |
| **scikit-learn** | Classic machine-learning library | Training small classifiers on our measurements |
| **SQLite** | Single-file database | The index the Live extension will read |

## Concepts

**Mel spectrogram** — a picture of sound: time on x, frequency on y (spaced like human
hearing), brightness = loudness. Neural audio models are image networks run on these.

**Embedding** — a list of numbers (2048 for PANNs) summarising "what a sound is".
Similar sounds get similar lists. Useful even when the model's named labels aren't.

**Our measurements ("features")** — numbers we compute directly from the audio:
duration, attack time, decay time, loudness slope (does it build up?), brightness slope,
frequency balance (sub / low / high energy ratios), noisiness (flatness),
tonal-vs-percussive ratio, number of hits, stereo width, estimated key.

**Classifier** — a function that takes a file's numbers (features and/or embedding) and
outputs a category.

**Training** — instead of writing the thresholds by hand ("if low_ratio > 0.3 and
decay < 0.5s then kick"), you show the classifier many examples whose answer is known,
and it adjusts its internal parameters until its outputs match those answers as well as
possible. The result is a set of learned thresholds/weights.

Two kinds we tried:

- **Logistic regression** — learns one weight per input number per category; a
  category's score is a weighted sum, and the highest score wins. Simple, fast, works
  best when the inputs are already very informative (like a good embedding).
- **Gradient-boosted decision trees** — builds hundreds of small if/then trees
  ("is decay_s < 0.4? is high_ratio > 0.3? ..."), each one correcting the mistakes of
  the ones before it. Very good at exactly our kind of data: a few dozen measurements
  with thresholds that interact. It's essentially the hand-written rules, but learned.

**MFCCs** (mel-frequency cepstral coefficients) — 13 numbers describing the overall
shape of the spectrum at a moment: roughly "tone colour". The classic feature for
telling instruments and drums apart. We keep their average over the whole sound, their
variation, and their average over just the first ~50 ms (the attack).

**Spectral contrast** — per frequency band, how far peaks stand above valleys. High =
tonal/pitched in that band, low = noisy. 7 numbers.

**Multi-tagging / candidate pools** — the classifier outputs a probability for every
category, not just a winner. A file joins *every* category it scores ≥ 0.2 in. So a
borderline open hi-hat sits in both the hi-hat and cymbal pools — deliberate, since
the goal is trialing interchangeable sounds. Lower threshold = more adventurous pools.

**Calibration** — whether a model's probabilities mean what they say: of all the times
it says "70%", is it right ~70% of the time? Overconfident models say 99% for
everything, which kills multi-tagging (nothing is ever borderline).

**Labels / ground truth** — the known answers used for training and scoring. We get
them for free from drum-machine file names (`TR-606_BD_01.wav` → BD = kick).

**Cross-validation** — to measure honestly, never test on data the classifier trained on.
Split the data into folds; train on all but one, test on the held-out one, rotate.

**Grouped cross-validation** — stricter: hold out *whole kits*. Samples from the same
drum machine sound alike, so testing on a kit the model has already seen inflates the
score. Grouped scores answer the real question: *how well does it do on a new pack?*

## How to run

```bash
cd ~/Dev/ableton-sdk/sample-tagger
uv run scripts/extract_video_audio.py          # new videos → Samples/from-videos/*.m4a
uv run python -m tagger scan                   # analyse new/changed samples (~1 file/s)
uv run python -m tagger scan --features-only   # recompute measurements only (after changing them)
uv run python -m tagger train                  # train + score the classifier (held-out kits)
uv run python -m tagger train --compare        #   ...also show the score with raw, uncleaned labels
uv run python -m tagger train --embedding      #   ...also feed it the PANNs embedding
uv run python -m tagger eval                   # score the old hand-written rules
uv run python -m tagger show --folder DRUMS -v # inspect per-file results
uv run python -m tagger stats                  # category counts
```

The extension:

```bash
cd ~/Dev/ableton-sdk/test-extension && nvm use && npm start
```

Live must be the **Beta** app, running, with **Settings → Extensions → Developer Mode** on.

---

## Log

### 2026-10-09 — Setup

- Live 12 Suite 12.4.6 installed; Extensions need the Live 12.4.x **Beta** (installed
  separately as *Ableton Live 12 Beta.app*) plus the SDK zip from Centercode.
- Installed Node 24.16.0 via nvm without changing the default (22).
- `nvm: command not found` in some terminals: nvm was only loaded in `~/.zprofile`
  (login shells). Added it to `~/.zshrc` (backup at `~/.zshrc.bak-nvm`).
- `npm start` → "Failed to connect to Live within 10s" until Live Beta was open **and**
  Developer Mode was enabled. Then hello-world worked.
- SDK capabilities confirmed from the API docs: `AudioTrack.createAudioClip({filePath,
  startTime})` places a file at an exact beat; `TakeLane.createAudioClip` exists (for
  auditioning candidates); right-click actions can target an arrangement time selection.
  No access to Live's browser — tags live in our own index.

### 2026-10-10 — Video audio extraction

- 222 videos (3.6 GB, ~139 min) in `User Library/Videos`; all have AAC audio.
- `scripts/extract_video_audio.py` copies the audio stream losslessly (no re-encode) to
  `Samples/from-videos/<same name>.m4a` → **116 MB**. Keeps file dates; writes
  `video-audio.json` mapping each audio file back to its video. Videos untouched.
- 93 files have random TikTok-style IDs as names — candidates for automatic renaming later.

### 2026-10-10 — First full scan + baseline

- Scanned 2,448 files in ~37 min (~1.1 files/s), 0 errors.
- 831 files have a label derivable from their name (mostly drum machines).
- **Hand-written rules: 3% correct.** Almost everything came out `unknown`.

**Why: PANNs doesn't recognise isolated one-shots.** True snares score ~0.000 for
"Snare drum"; same for hi-hats and kicks. AudioSet is 10-second clips of sound *in
context*; a lone 0.3 s hit followed by silence looks like nothing it was trained on.
Cymbals (longer) did better: median "Cymbal" score 0.23.

- Tried looping each one-shot into a 10 s pattern before PANNs: pick-the-best-of-6
  accuracy went 31% → 39%. Helps, not enough.

**Trained classifiers instead** (grouped cross-validation, held-out kits):

| Input | Accuracy |
|---|---|
| Hand rules | 3% |
| PANNs embedding, logistic regression | 41% |
| PANNs embedding + our measurements, logistic regression | 65% |
| **Our measurements only, boosted trees** | **74%** |

Per class (boosted trees): snare 93%, hihat 82%, tom 71%, cymbal 66%, perc 61%,
kick 60%, sub_bass 58% (only 12 examples).

Takeaways:
- Our own measurements carry most of the signal for drums; PANNs mainly helps for
  long, non-drum material (speech, music, ambience).
- Tuning moves from "hand-write thresholds" to "give the model better measurements and
  cleaner labels".
- Some name-derived labels are wrong (e.g. `Ride the Tide.mp3` → "cymbal" because of
  "ride"; TikTok IDs containing "lt"/"ch"). These both mis-teach and mis-score.

**Next:** automatic label cleaning → add MFCC / spectral-contrast features → try CLAP
embeddings → crash-vs-ride, risers, impacts.

### 2026-10-10 — Label cleaning + better measurements

**Label cleaning** (`truth.clean_labels`) — a name-derived label is trusted only if:
- the file sits in a folder named after a category (`KICKS`, `hh`, `808S`, ...), or
- it has a drum abbreviation (`BD`, `SD`, `CY`...) **and** its folder is a kit
  (≥50% of files there have one), or
- its name contains a full, unambiguous word (`snare`, `tambourine`, `kick`...) —
  safe outside kits, unlike `ch`/`lt`/`sn`. (`ride` excluded: "Ride the Tide".)
- and in all cases: not a machine-generated ID, and short enough to be a one-shot
  (≤6 s; risers ≤30 s).

Also taught it more kit spellings (`BDrum`, `SDrum`, `Bgo`, `CGa`, `China`...), which
recovered the RX5 kit. Result: 853 raw labels → 817 trusted.

**New measurements (feature version 2):** MFCCs (whole sound + attack), spectral
contrast, rolloff, zero-crossing rate, early decay slope. Re-computing just the
measurements: ~30 min for the library (PANNs not re-run).

**Results** (boosted trees, held-out kits):

| Step | Overall | Notes |
|---|---|---|
| v1 measurements, raw labels | 74% | previous result |
| v2 measurements, raw labels | 77% | MFCC etc. help |
| v2 measurements, cleaned labels | **79%** | snare 94%, tom 86%, hihat 83%, kick 73%, perc 69%, cymbal 67% |
| + PANNs embedding | 79% | cymbal ↑ to 73%, kick ↓ to 64% — no net gain; not used |

**Confident learning (suspect labels) — findings.** 106 files where the held-out model
was ≥70% sure of a different category than the label. Inspecting them, **most are not
mislabelled** — they are genuinely borderline sounds:
- open hi-hats ↔ short cymbals (most common: 18 + 6)
- congas/bongos (perc) ↔ toms/snares
- 808s ↔ kicks
- only a few are real label problems (AI-generated tambourine *patterns* labelled as one-shots)

So suspects are now **reported, not dropped** (`data/suspects.csv`). Dropping them
inflated the score to 91% but removed exactly the boundary examples the model needs.
**The honest number is 79%.**

**Insight: the remaining errors are mostly about the category list, not the model.**
- `sub_bass` scores 0%: its 10 examples come from only 2 packs, so held-out-kit testing
  can never see it in training; and an 808 is arguably a kind of kick.
- "cymbal" lumps crash, ride, china, splash; "hihat" lumps closed and open. The
  confusable pair is *open hat vs short cymbal*.
- For the actual use case ("some crash here"), what matters is long, washy, bright
  cymbals — a sharper category than "anything metallic".

Final model saved to `data/model.joblib` (trained on all 811 cleaned labels).

### 2026-10-10 — Category merges + multi-tagging

**Decision (Dave):** overlap between similar sounds is a *feature* — when trialing "some
crash", a short cymbal-ish open hat is a reasonable candidate and vice versa. So:
- files get **every** category above a threshold, not just the top one;
- **808s merged into kick**; **congas/bongos/timbales merged into tom**.

Merges alone: **79% → 86%** (kick 73% → 92%).

**Problem found: the model was overconfident.** When it said ≥95% it was right only 90%
of the time, and almost nothing was borderline (1.07 tags/file even at a 0.2 threshold).
Compared three variants, held-out kits:

| Model | Accuracy | Log-loss (lower = better probabilities) | Calibration |
|---|---|---|---|
| 300 deep trees (previous) | 86% | 0.88 | overconfident |
| **150 shallow, regularised trees** | **86%** | **0.54** | honest: "95%" → 95% right, "60-80%" → 70% right |
| shallow + sigmoid calibration | 84% | 0.61 | *under*confident — overcorrected |

→ Switched to the shallow, regularised model. Suspect count dropped 106 → 57 as a side
effect (fewer confident-but-wrong calls).

**Pools at threshold 0.2** (1.15 tags per file):

| Pool | Catches (recall) | Pool contents |
|---|---|---|
| cymbal | 78% | cymbal 83%, hihat 14%, perc 3% |
| hihat | 85% | hihat 73%, cymbal 17%, perc 8% |
| kick | 96% | kick 85%, tom 6%, snare 5% |
| perc | 80% | perc 55%, hihat 18%, cymbal 16% |
| snare | 94% | snare 86%, tom 9%, perc 3% |
| tom | 93% | tom 76%, kick 11%, snare 8% |

The "impurity" in each pool is mostly the intended neighbours (hats in the cymbal pool,
toms in the kick pool).

**Open issue: the model only knows drums.** It's trained on drum one-shots only, so for
a vocal chop or a pad it will still answer "which drum is this most like". Needs an
"is this a drum one-shot at all?" gate before tags are written for the whole library.

### 2026-10-11 — Clarifications: what the model actually uses, and why not CLAP (yet)

**Terminology.** "Kit" = a drum kit / sample pack folder (e.g. `Roland TR-606`). Not an
ML term — the ML term is *group*, as in grouped cross-validation (hold out whole groups).

**No neural network was trained.** PANNs was only *run* (inference, frozen weights).
What we trained are scikit-learn models: logistic regression (one experiment) and
gradient-boosted decision trees (the final model).

**The final model uses no PANNs data** — only our librosa measurements:

| Attempt | Input | Model | Score |
|---|---|---|---|
| Hand rules | PANNs class scores + measurements | if/then | 3% |
| Transfer learning ("linear probe") | PANNs embedding | logistic regression | 41% |
| Mixed | PANNs embedding + measurements | logistic regression | 65% |
| **Final** | **measurements only** | **boosted trees** | **86%** |
| Re-test | PANNs embedding + measurements | boosted trees | 79% vs 79% without — no gain, left out |

PANNs output stays in the index: likely useful for the "is this a drum at all?" gate
(it's good at speech / music / ambience) and for non-drum material.

**Why not CLAP (yet).** Deferred, not rejected.

- *What it is:* two networks trained together — an **audio encoder** (a CNN in some
  versions, e.g. Microsoft CLAP 2022 used Cnn14 = PANNs; a transformer, HTS-AT, in
  LAION-CLAP) and a **text encoder** (a language transformer). Trained on ~600k
  audio–caption pairs so a sound and its description end up close together.
  You can then score any sound against any phrase ("a crash cymbal") with no labelling.
- *Why it wasn't first:* Dave preferred a fixed-category classifier over free-text
  search (messy file names make text search less useful). CLAP *can* do fixed
  categories zero-shot, but the plan was PANNs first.
- *Why it stayed deferred:* PANNs (also trained on 10-second context clips) failed on
  isolated one-shots, which raised doubts about any off-the-shelf audio network there;
  meanwhile our own measurements + trees reached 86% on drums quickly.
- *Why it's still worth testing:* CLAP's training data includes Freesound and BBC
  sound-effects libraries — lots of isolated one-shots and FX — so it may handle them
  much better than PANNs. Untested.
- *Where it fits best:* the non-drum categories — **riser, impact, downlifter, sweep** —
  where file names give us almost no labels. Zero-shot CLAP scores (or its embedding fed
  to the trees) could supply those without manual labelling.
- *How we'd test it:* same as PANNs — add its embedding/scores as extra inputs, re-run
  `train`, and keep it only if the held-out-kit score improves.
