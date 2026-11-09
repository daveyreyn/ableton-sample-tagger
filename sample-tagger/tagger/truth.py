"""Best-guess "true" category from folder and file names, for training and scoring.

The tagger itself never looks at names at prediction time; names only supply
training labels. Drum-machine kits use standard abbreviations (BD, SD, HH, CY...),
which makes them good ground truth — but only in the right context, so labels
are filtered by `clean_labels` before use.
"""

import re
from collections import defaultdict
from pathlib import Path

TOKENS = {
    # 808s count as kicks; congas/bongos/timbales count as toms (by design: these are
    # interchangeable when trialing sounds, and the overlap is handled by multi-tagging)
    "kick": {"bd", "kick", "kicks", "kik", "bassdrum", "bdrum", "kd", "808", "808s"},
    "snare": {"sd", "snr", "snare", "snares", "sn", "clap", "cp", "clp", "rim", "rimshot",
              "sdrum"},
    "hihat": {"hh", "hihat", "hat", "hats", "ch", "oh"},
    "cymbal": {"cy", "cym", "cymbal", "crash", "ride", "china", "splash"},
    "tom": {"tom", "toms", "ht", "mt", "lt", "drltom", "conga", "bongo", "bgo", "cga",
            "tmbl", "timbl", "timbale"},
    "perc": {"perc", "percs", "cow", "cowbell", "cwbl", "cbl", "agogo", "shaker", "shk",
             "tmb", "tamb", "tambourine", "cab", "cabasa", "clave", "ago"},
    "riser": {"riser", "uplifter"},
}
_LOOKUP = {tok: cat for cat, toks in TOKENS.items() for tok in toks}

# Full words are unambiguous enough to trust outside kit folders
# (unlike "ch", "lt", "sn"...). "ride" is excluded: "Ride the Tide".
WORDS = {"kick", "snare", "clap", "rimshot", "hihat", "hat", "cymbal", "crash", "splash",
         "tom", "toms", "tambourine", "shaker", "cowbell", "conga", "bongo", "808",
         "riser", "uplifter"}

# A folder counts as a "kit" when at least this share of its files have a
# drum abbreviation in their name.
KIT_SHARE = 0.5
# Name-derived labels on files longer than this are not trusted (one-shots are short).
MAX_DURATION_S = {"riser": 30.0, "kick": 8.0}
DEFAULT_MAX_DURATION_S = 6.0
# Machine-generated names (TikTok IDs, long hashes) contain accidental "abbreviations".
_RANDOM_ID = re.compile(r"[a-z0-9]{16,}", re.I)


def tokens(name: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    out = []
    for part in parts:
        out.append(part.lower())
        out.append(part.lower().rstrip("0123456789"))  # "BDrum1" -> "bdrum"
        # split camelCase and letter/digit runs: "HiHat2" -> hi, hat, 2; "BD08" -> bd, 08
        out += [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", part)]
    return [t for t in out if t]


def _name_label(name: str) -> str | None:
    """Category implied by a name, or None if absent/ambiguous."""
    cats = {_LOOKUP[t] for t in tokens(name) if t in _LOOKUP}
    return cats.pop() if len(cats) == 1 else None


def _word_label(name: str) -> str | None:
    cats = {_LOOKUP[t] for t in tokens(name) if t in WORDS}
    return cats.pop() if len(cats) == 1 else None


def _folder_label(path: Path) -> str | None:
    """Folders named exactly after a category (KICKS, SNARES, hh, 808S...)."""
    for folder in (path.parent.name, path.parent.parent.name):
        if folder.lower() in _LOOKUP:
            return _LOOKUP[folder.lower()]
    return None


def true_category(path: Path) -> str | None:
    """Raw, unfiltered label: file name first, then folders."""
    for name in (path.stem, path.parent.name, path.parent.parent.name):
        cats = {_LOOKUP[t] for t in tokens(name) if t in _LOOKUP}
        if len(cats) == 1:
            return cats.pop()
        if len(cats) > 1:
            return None
    return None


def clean_labels(durations: dict[Path, float]) -> dict[Path, tuple[str, str]]:
    """Context-filtered labels: {path: (category, reason)} for trusted files only.

    `durations` maps every analysed file to its length in seconds.
    """
    by_folder: dict[Path, list[Path]] = defaultdict(list)
    for p in durations:
        by_folder[p.parent].append(p)
    kit_share = {folder: sum(_name_label(p.stem) is not None for p in files) / len(files)
                 for folder, files in by_folder.items()}

    out = {}
    for p, dur in durations.items():
        name_cat = None if _RANDOM_ID.search(p.stem) else _name_label(p.stem)
        folder_cat = _folder_label(p)
        if folder_cat and name_cat and name_cat != folder_cat:
            continue  # folder and name disagree
        if folder_cat:
            cat, reason = folder_cat, "category folder"
        elif name_cat and kit_share[p.parent] >= KIT_SHARE:
            cat, reason = name_cat, "kit abbreviation"
        elif name_cat and name_cat == _word_label(p.stem):
            cat, reason = name_cat, "descriptive name"
        else:
            continue
        if dur > MAX_DURATION_S.get(cat, DEFAULT_MAX_DURATION_S):
            continue
        out[p] = (cat, reason)
    return out


def kit_group(path: Path, samples_root: Path) -> str:
    """Which kit/pack a file belongs to, for held-out-kit cross-validation."""
    rel = path.relative_to(samples_root).parts
    depth = 3 if len(rel) > 3 and rel[1] == "Drum_Machines" else 2
    return "/".join(rel[:depth][:len(rel) - 1]) or rel[0]
