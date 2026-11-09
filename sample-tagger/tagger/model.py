"""Learned classifier: measurements (+ optionally PANNs embedding) -> category.

Training labels come from file names (see truth.py), cleaned in two steps:
  1. context rules (truth.clean_labels)
  2. confident learning: files where held-out predictions confidently disagree
     with their label are flagged as suspects and left out of the final model.
"""

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import truth
from .index import DB_PATH

MODEL_PATH = DB_PATH.parent / "model.joblib"
SUSPECTS_PATH = DB_PATH.parent / "suspects.csv"

SCALARS = ["duration_s", "sounding_s", "attack_s", "decay_s", "peak_pos",
           "loudness_slope_db_s", "brightness_slope_hz_s", "early_decay_db_s",
           "centroid_hz", "rolloff_hz", "sub_ratio", "low_ratio", "high_ratio",
           "flatness", "zcr", "percussive_ratio", "onsets", "onset_rate", "stereo_width"]
VECTORS = {"mfcc_mean": 13, "mfcc_std": 13, "mfcc_attack": 13, "contrast": 7}
TAG_THRESHOLD = 0.2         # a file joins every category pool it scores at least this in
MIN_EXAMPLES = 10           # categories with fewer labelled files are left out
SUSPECT_CONFIDENCE = 0.7    # held-out prediction at least this sure...
SUSPECT_LABEL_PROB = 0.15   # ...and the given label at most this likely


def feature_names(embedding: bool = False) -> list[str]:
    names = SCALARS + [f"{k}_{i}" for k, n in VECTORS.items() for i in range(n)]
    return names + ([f"panns_emb_{i}" for i in range(2048)] if embedding else [])


def vectorize(f: dict, emb: np.ndarray | None = None) -> np.ndarray:
    """Missing values become NaN, which the boosted trees handle natively."""
    vals = [float(f[k]) if f.get(k) is not None else np.nan for k in SCALARS]
    for k, n in VECTORS.items():
        vals += f.get(k) or [np.nan] * n
    x = np.array(vals, dtype=np.float32)
    return np.concatenate([x, emb]) if emb is not None else x


def new_classifier() -> HistGradientBoostingClassifier:
    # Shallow, regularised trees: same accuracy as deeper ones, but the probabilities
    # are honest (when it says 70%, it's right ~70% of the time). That matters because
    # files get every tag above TAG_THRESHOLD, so borderline sounds land in several pools.
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
        min_samples_leaf=20, class_weight="balanced", random_state=0)


@dataclass
class Dataset:
    paths: list[Path]
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    reasons: list[str]


def load_dataset(db, samples_root: Path, embedding: bool = False,
                 raw_labels: bool = False) -> Dataset:
    rows = {}
    for r in db.execute("SELECT * FROM files WHERE features IS NOT NULL"):
        f = json.loads(r["features"])
        if f.get("silent"):
            continue
        emb = (np.frombuffer(r["panns_embedding"], np.float16).astype(np.float32)
               if embedding else None)
        rows[Path(r["path"])] = (f, emb)

    if raw_labels:
        labels = {p: (c, "raw name") for p in rows if (c := truth.true_category(p))}
    else:
        labels = truth.clean_labels({p: f["duration_s"] for p, (f, _) in rows.items()})
    counts = Counter(c for c, _ in labels.values())
    keep = sorted(p for p, (c, _) in labels.items() if counts[c] >= MIN_EXAMPLES)

    return Dataset(
        paths=keep,
        X=np.stack([vectorize(*rows[p]) for p in keep]),
        y=np.array([labels[p][0] for p in keep]),
        groups=np.array([truth.kit_group(p, samples_root) for p in keep]),
        reasons=[labels[p][1] for p in keep],
    )


def held_out_probs(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Predictions for every file from a model that never saw that file's kit."""
    n_splits = min(8, len(set(ds.groups)))
    probs = cross_val_predict(new_classifier(), ds.X, ds.y, groups=ds.groups,
                              cv=GroupKFold(n_splits=n_splits), method="predict_proba")
    classes = np.array(sorted(set(ds.y)))
    return probs, classes


def report(y: np.ndarray, pred: np.ndarray) -> str:
    lines = [f"  overall {np.mean(pred == y):.0%}  (n={len(y)})"]
    for c in sorted(set(y)):
        m = y == c
        wrong = Counter(pred[m & (pred != c)]).most_common(2)
        conf = ", ".join(f"{k} {v}" for k, v in wrong)
        lines.append(f"  {c:<9} {np.mean(pred[m] == c):>4.0%}  n={m.sum():<4} "
                     f"{'mistaken for: ' + conf if conf else ''}")
    return "\n".join(lines)


def pool_report(y: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> str:
    """Multi-tag view: each category's pool = files scoring >= TAG_THRESHOLD."""
    tagged = probs >= TAG_THRESHOLD
    lines = [f"  threshold {TAG_THRESHOLD}: {tagged.sum(axis=1).mean():.2f} tags per file"]
    for i, c in enumerate(classes):
        pool = y[tagged[:, i]]
        comp = ", ".join(f"{k} {v / len(pool):.0%}" for k, v in Counter(pool).most_common(3))
        lines.append(f"  {c:<9} catches {tagged[y == c, i].mean():>4.0%} of true {c}s; "
                     f"pool of {len(pool)} is {comp}")
    return "\n".join(lines)


def find_suspects(ds: Dataset, probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    pred_p = probs.max(axis=1)
    label_p = probs[np.arange(len(ds.y)), [idx[c] for c in ds.y]]
    pred = classes[probs.argmax(axis=1)]
    return (pred != ds.y) & (pred_p >= SUSPECT_CONFIDENCE) & (label_p <= SUSPECT_LABEL_PROB)


def write_suspects(ds: Dataset, mask: np.ndarray, probs: np.ndarray, classes: np.ndarray,
                   samples_root: Path) -> None:
    pred = classes[probs.argmax(axis=1)]
    with SUSPECTS_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "model_says", "confidence", "label_reason", "file"])
        for i in np.where(mask)[0]:
            w.writerow([ds.y[i], pred[i], f"{probs[i].max():.2f}", ds.reasons[i],
                        ds.paths[i].relative_to(samples_root)])


def save(clf, embedding: bool) -> None:
    joblib.dump({"model": clf, "embedding": embedding,
                 "features": feature_names(embedding)}, MODEL_PATH)
