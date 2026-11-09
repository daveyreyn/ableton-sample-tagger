# Ableton Sample Tagger

Tags every sample in an Ableton Live library by what kind of sound it is, so a Live extension can answer "some crash here" with five candidates on take lanes instead of one file picked by hand.

Two parts:

- **`sample-tagger/`** (Python) listens to every file in the library, measures it, and trains a classifier on labels mined from drum-machine file names. On drum one-shots from kits the model never saw, it is right 86% of the time.
- **`test-extension/`** (TypeScript, Ableton Live 12 Extensions SDK) is the extension that will place the candidates. Right now it is the hello-world that proves the Extension Host connection; the right-click action comes next.

`JOURNAL.md` is the lab notebook: every experiment, its numbers, and the dead ends, in the order they happened.

## The problem

A sample library of a few thousand files is organised by where the files came from, not by what they sound like. Finding a crash means scrolling through packs and auditioning one file at a time. Extensions get no access to Live's browser or its tags, so the index lives outside Live in SQLite and the extension reads that.

## How the tagger works

1. **Decode** with ffmpeg into memory. The library is never written to.
2. **Measure** 65 numbers per file with librosa: attack and decay times, where the peak sits, loudness and brightness slopes (does it build up?), sub/low/high energy ratios, spectral flatness, centroid and rolloff, zero-crossing rate, tonal-vs-percussive ratio, onset count, stereo width, 13 MFCCs over the whole sound and again over the first 50 ms, and 7 spectral-contrast bands. Tonal material also gets an estimated key.
3. **Run PANNs** (Cnn14, trained on AudioSet) and keep its 527 class scores and 2048-number embedding, so later experiments don't need another 37-minute pass over the library.
4. **Mine labels from names.** Drum machines use standard abbreviations (`BD`, `SD`, `HH`, `CY`), which gives about 800 labelled files for free. A label is trusted only in context: the file sits in a folder named after a category, or it carries an abbreviation and its folder looks like a kit, or the name contains a full unambiguous word. Machine-generated IDs and files too long to be one-shots are rejected. `Ride the Tide.mp3` is not a cymbal.
5. **Train** scikit-learn gradient-boosted trees on the measurements. Scoring uses grouped cross-validation that holds out whole kits, because samples from one drum machine sound alike and a random split would flatter the result.
6. **Multi-tag.** The model outputs a probability per category, and a file joins every pool it scores at least 0.2 in. A borderline open hi-hat lands in both the hi-hat and cymbal pools on purpose: when trialing a crash, a washy open hat is a fair candidate.

## Results

Accuracy on held-out kits, drum one-shots labelled from file names:

| Approach | Accuracy |
|---|---|
| Hand-written rules over PANNs class scores | 3% |
| PANNs embedding, logistic regression | 41% |
| PANNs embedding + measurements, logistic regression | 65% |
| Measurements only, boosted trees | 74% |
| + MFCCs, spectral contrast, cleaned labels | 79% |
| + 808s merged into kick, congas and bongos into tom | **86%** |

Candidate pools at threshold 0.2, which is what the extension will draw from:

| Pool | Catches of the true category | Pool contents |
|---|---|---|
| kick | 96% | kick 85%, tom 6%, snare 5% |
| snare | 94% | snare 86%, tom 9%, perc 3% |
| tom | 93% | tom 76%, kick 11%, snare 8% |
| hihat | 85% | hihat 73%, cymbal 17%, perc 8% |
| perc | 80% | perc 55%, hihat 18%, cymbal 16% |
| cymbal | 78% | cymbal 83%, hihat 14%, perc 3% |

The impurity in each pool is mostly the intended neighbours: hats in the cymbal pool, toms in the kick pool.

## What I learned

- **PANNs can't see isolated one-shots.** A true snare scores about 0.000 for "Snare drum". AudioSet is ten seconds of sound in context, and a lone 0.3 s hit followed by silence looks like nothing it was trained on. Looping each hit into a 10 s pattern before inference moved a pick-the-best-of-6 test from 31% to 39%, which is not enough.
- **Hand-built measurements beat the embedding for drums.** Adding the 2048-number embedding to the trees changed nothing overall (cymbal up, kick down), so the final model doesn't use it. PANNs output stays in the index for the material it is good at: speech, music, ambience.
- **Calibration matters once you multi-tag.** 300 deep trees and 150 shallow regularised trees both scored 86%, but the deep ones said "95%" and were right 90% of the time, and almost nothing came out borderline. The shallow model's log-loss is 0.54 against 0.88, and when it says 70% it is right about 70% of the time. That is what makes the pools work.
- **Most "mislabelled" files were labelled right.** Confident learning flagged 106 files where the held-out model disagreed with the name. Listening to them, most were genuine boundary cases: open hats that sound like short cymbals, congas that sound like toms. Dropping them pushed the score to 91% and removed exactly the examples the model needs, so they are written to a CSV for review instead.

## Running it

The tagger needs Python 3.12+, [uv](https://docs.astral.sh/uv/) and ffmpeg. The PANNs checkpoint (about 300 MB) downloads to `~/panns_data` on first run.

```bash
cd sample-tagger
uv run python -m tagger scan                   # analyse new or changed files (~1 file/s)
uv run python -m tagger scan --features-only   # recompute measurements only, after changing them
uv run python -m tagger train                  # train and score on held-out kits
uv run python -m tagger train --compare        # also score with raw, uncleaned labels
uv run python -m tagger train --embedding      # also feed the PANNs embedding to the trees
uv run python -m tagger show --folder DRUMS -v # per-file results
uv run python -m tagger stats                  # category counts
uv run scripts/extract_video_audio.py          # copy the audio track out of every video, losslessly
```

`scan` reads `~/Music/Ableton/User Library/Samples` by default; pass folders to scan something else.

The extension needs Node 24 (see `.nvmrc`), Ableton Live 12 Beta with Developer Mode on in Settings, and the Extensions SDK. The SDK licence doesn't allow redistributing it, so put the `@ableton-extensions/sdk` and `@ableton-extensions/cli` tarballs from Ableton's download into `test-extension/vendor/` and set `EXTENSION_HOST_PATH` in `test-extension/.env`.

```bash
cd test-extension && nvm use && npm install && npm start
```

## Layout

```
├── README.md
├── JOURNAL.md               lab notebook
├── sample-tagger/
│   ├── tagger/
│   │   ├── audio.py         ffmpeg decode, read-only
│   │   ├── features.py      the measurements (bump VERSION when they change)
│   │   ├── panns.py         Cnn14 inference, windowed for long files
│   │   ├── index.py         SQLite: raw measurements and model outputs per file
│   │   ├── truth.py         labels from names, with the context filters
│   │   ├── model.py         boosted trees, grouped CV, pools, suspects
│   │   ├── rules.py         the original hand-written rules, kept for `eval`
│   │   └── __main__.py      CLI
│   ├── scripts/extract_video_audio.py
│   └── data/                index.sqlite, model.joblib, suspects.csv (gitignored)
└── test-extension/          Live extension; SDK tarballs go in vendor/ (gitignored)
```

## Next

- A gate for "is this a drum one-shot at all?" before tags are written for the whole library, probably from the PANNs scores, which are good at speech, music and ambience.
- CLAP for risers, impacts and sweeps, where file names give almost no labels. Its training data includes Freesound, so it may handle isolated sounds better than PANNs did.
- The extension: right-click a time selection, pick a pool, place N candidates on take lanes to comp between.
