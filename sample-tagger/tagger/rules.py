"""Turn measurements + model scores into categories.  <- this is the file we tune.

Each rule gets `f` (features dict) and `p(...)` (highest PANNs score among the
given AudioSet class names) and returns a confidence 0..1. A file gets every
category scoring >= THRESHOLD; the best one is its primary category.
"""

from collections.abc import Callable

THRESHOLD = 0.3

Rule = Callable[[dict, Callable[..., float]], float]


def one_shot(f: dict) -> bool:
    return f.get("onsets", 0) <= 2 and f.get("sounding_s", 0) < 6


def loop(f: dict) -> bool:
    return f.get("onsets", 0) >= 4 and f.get("onset_rate", 0) >= 1.0


def rising(f: dict) -> bool:
    return (f.get("loudness_slope_db_s", 0) > 2 and f.get("peak_pos", 0) > 0.6
            and f.get("sounding_s", 0) >= 1.0)


RULES: dict[str, Rule] = {
    "kick": lambda f, p: p("Bass drum") * (one_shot(f) and f.get("low_ratio", 0) > 0.3),
    "snare": lambda f, p: max(p("Snare drum"), p("Rimshot"), p("Clapping") * 0.8) * one_shot(f),
    "hihat": lambda f, p: p("Hi-hat") * one_shot(f) * (f.get("high_ratio", 0) > 0.2),
    "cymbal": lambda f, p: max(p("Cymbal"), p("Smash, crash") * 0.7)
        * (f.get("sounding_s", 0) > 0.8) * (not rising(f)) * (f.get("attack_s", 1) < 0.1),
    "tom": lambda f, p: p("Drum") * one_shot(f) * (0.1 < f.get("low_ratio", 0) < 0.6)
        * (f.get("percussive_ratio", 0) > 0.5) * 0.7,
    "perc": lambda f, p: max(p("Percussion", "Tambourine", "Cowbell", "Mallet percussion"), 0)
        * one_shot(f),
    "sub_bass": lambda f, p: (f.get("sub_ratio", 0) > 0.4) * (f.get("percussive_ratio", 1) < 0.5)
        * 0.8,
    "bass": lambda f, p: p("Bass guitar", "Double bass") * (f.get("low_ratio", 0) > 0.3),
    "riser": lambda f, p: rising(f) * max(0.5, p("Whoosh, swoosh, swish", "Noise", "Synthesizer")),
    "impact": lambda f, p: p("Boom", "Explosion", "Bang", "Thump, thud", "Slam")
        * (f.get("attack_s", 1) < 0.05),
    "drum_loop": lambda f, p: p("Drum kit", "Drum machine", "Drum", "Drum and bass") * loop(f),
    "vocal": lambda f, p: p("Singing", "Male singing", "Female singing", "Vocal music",
                            "Synthetic singing"),
    "speech": lambda f, p: p("Speech", "Male speech, man speaking",
                             "Female speech, woman speaking", "Speech synthesizer"),
    "synth": lambda f, p: p("Synthesizer", "Electronic organ") * (f.get("percussive_ratio", 1) < 0.5),
    "keys": lambda f, p: p("Piano", "Electric piano"),
    "guitar": lambda f, p: p("Guitar", "Electric guitar", "Acoustic guitar"),
    "noise": lambda f, p: p("Noise", "White noise", "Pink noise", "Static")
        * (f.get("flatness", 0) > 0.1),
    "fx": lambda f, p: p("Whoosh, swoosh, swish", "Beep, bleep", "Siren", "Scratch",
                         "Sine wave") * 0.8,
}


def classify(f: dict, p: Callable[..., float]) -> list[tuple[str, float]]:
    if f.get("silent"):
        return [("silent", 1.0)]
    out = [(name, round(float(rule(f, p)), 3)) for name, rule in RULES.items()]
    out = sorted((c for c in out if c[1] >= THRESHOLD), key=lambda c: -c[1])
    return out or [("unknown", 0.0)]
