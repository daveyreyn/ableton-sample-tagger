"""sample-tagger CLI.

  uv run python -m tagger scan [FOLDER ...] [--limit N]   analyse new/changed files
  uv run python -m tagger show [--folder TEXT] [--cat NAME] [--verbose]
  uv run python -m tagger stats [--folder TEXT]
  uv run python -m tagger eval [--folder TEXT] [--misses CAT]   score rules against names
  uv run python -m tagger train [--embedding] [--compare]   train + score the classifier
"""

import argparse
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

from . import audio, features, index, model, panns, rules, truth

SAMPLES = Path.home() / "Music/Ableton/User Library/Samples"


def iter_audio(folders: list[Path]):
    for folder in folders:
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in audio.AUDIO_EXTS:
                yield f


def cmd_scan(a) -> None:
    db = index.connect()
    if a.features_only:
        todo = [Path(r["path"]) for r in db.execute(
            "SELECT path, features FROM files WHERE features IS NOT NULL")
            if index.features_version(r["features"]) != features.VERSION]
    else:
        todo = [f for f in iter_audio(a.folders or [SAMPLES]) if not index.is_current(db, f)]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} files to analyse", flush=True)
    model = None if a.features_only else panns.Panns()
    t0 = time.time()
    for i, f in enumerate(todo, 1):
        try:
            feats = features.measure(audio.load(f, features.SR))
            if model is None:
                index.save_features(db, f, feats)
            else:
                scores, emb = model(audio.load(f, panns.SR, channels=1)[0])
                index.save(db, f, feats, scores, emb)
        except Exception as e:  # keep going; record the failure
            if model is not None:
                index.save(db, f, None, None, None, error=repr(e))
            print(f"  error: {f.name}: {e!r}", file=sys.stderr, flush=True)
        if i % 100 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(todo)}  ({rate:.1f} files/s)", flush=True)


def classified(folder: str | None):
    db = index.connect()
    q = "SELECT * FROM files WHERE features IS NOT NULL"
    args: tuple = ()
    if folder:
        q += " AND path LIKE ?"
        args = (f"%{folder}%",)
    for row in db.execute(q + " ORDER BY path", args):
        feats, scores = index.load_row(row)
        cats = rules.classify(feats, partial(panns.score, scores))
        yield Path(row["path"]), feats, scores, cats


def cmd_show(a) -> None:
    for path, feats, scores, cats in classified(a.folder):
        if a.cat and a.cat not in [c for c, _ in cats]:
            continue
        rel = path.relative_to(SAMPLES) if path.is_relative_to(SAMPLES) else path
        label = ", ".join(f"{c} {s:.2f}" for c, s in cats[:3])
        print(f"{label:<34} {rel}")
        if a.verbose:
            tops = "  ".join(f"{n} {s:.2f}" for n, s in panns.top(scores, 4))
            keys = ("duration_s", "attack_s", "decay_s", "peak_pos", "loudness_slope_db_s",
                    "low_ratio", "high_ratio", "percussive_ratio", "onsets", "key")
            fs = "  ".join(f"{k}={feats[k]}" for k in keys if k in feats)
            print(f"{'':<34}   panns: {tops}\n{'':<34}   {fs}")


def cmd_stats(a) -> None:
    counts = Counter(cats[0][0] for *_, cats in classified(a.folder))
    for cat, n in counts.most_common():
        print(f"{n:>6}  {cat}")


def cmd_eval(a) -> None:
    """Score primary categories against categories implied by file/folder names."""
    rows = [(path, truth.true_category(path), cats)
            for path, _, _, cats in classified(a.folder)]
    rows = [r for r in rows if r[1]]
    confusion: dict[str, Counter] = {}
    for _, want, cats in rows:
        confusion.setdefault(want, Counter())[cats[0][0]] += 1
    total_ok = 0
    print(f"{'true':<10} {'n':>4} {'right':>6}   most common guesses")
    for want, got in sorted(confusion.items()):
        n, ok = sum(got.values()), got[want]
        total_ok += ok
        guesses = ", ".join(f"{c} {k}" for c, k in got.most_common(4))
        print(f"{want:<10} {n:>4} {ok / n:>6.0%}   {guesses}")
    print(f"\noverall: {total_ok}/{len(rows)} = {total_ok / max(len(rows), 1):.0%}"
          f"  (files with a name-derived label)")
    if a.misses:
        print(f"\nmisses for {a.misses}:")
        for path, want, cats in rows:
            if want == a.misses and cats[0][0] != want:
                print(f"  {cats[0][0]:<10} {path.relative_to(SAMPLES)}")


def cmd_train(a) -> None:
    db = index.connect()
    stale = sum(index.features_version(r[0]) != features.VERSION for r in
                db.execute("SELECT features FROM files WHERE features IS NOT NULL"))
    if stale:
        print(f"note: {stale} files still have old measurements (run scan --features-only)")

    if a.compare:
        raw = model.load_dataset(db, SAMPLES, a.embedding, raw_labels=True)
        probs, classes = model.held_out_probs(raw)
        print("\n[1] raw name labels, held-out kits:")
        print(model.report(raw.y, classes[probs.argmax(axis=1)]))

    ds = model.load_dataset(db, SAMPLES, a.embedding)
    reasons = Counter(ds.reasons)
    print(f"\n[2] cleaned labels ({len(ds.y)} files: "
          + ", ".join(f"{n} via {r}" for r, n in reasons.items()) + "), held-out kits:")
    probs, classes = model.held_out_probs(ds)
    pred = classes[probs.argmax(axis=1)]
    print(model.report(ds.y, pred))
    print("\n    candidate pools (multi-tag):")
    print(model.pool_report(ds.y, probs, classes))

    suspects = model.find_suspects(ds, probs, classes)
    model.write_suspects(ds, suspects, probs, classes, SAMPLES)
    print(f"\n[3] {suspects.sum()} suspect labels (confident disagreements, for review)"
          f" -> {model.SUSPECTS_PATH}")
    # Suspects are reported, not dropped: most are borderline sounds (open hat vs
    # cymbal, conga vs tom) whose labels are right and that the model needs to see.

    clf = model.new_classifier().fit(ds.X, ds.y)
    model.save(clf, a.embedding)
    print(f"\nsaved final model (trained on {len(ds.y)} files) -> {model.MODEL_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(prog="tagger")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("scan")
    s.add_argument("folders", nargs="*", type=Path)
    s.add_argument("--limit", type=int)
    s.add_argument("--features-only", action="store_true",
                   help="recompute measurements for indexed files; skip PANNs")
    s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("show")
    s.add_argument("--folder")
    s.add_argument("--cat")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_show)
    s = sub.add_parser("stats")
    s.add_argument("--folder")
    s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("eval")
    s.add_argument("--folder")
    s.add_argument("--misses", metavar="CAT")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("train")
    s.add_argument("--embedding", action="store_true", help="also use the PANNs embedding")
    s.add_argument("--compare", action="store_true", help="also score with raw labels")
    s.set_defaults(fn=cmd_train)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
