"""PANNs Cnn14 AudioSet tagger.

Returns all 527 AudioSet class scores plus the 2048-d embedding, so rules can
be re-tuned later without re-running the model.
"""

from pathlib import Path

import numpy as np
import torch
from panns_inference import AudioTagging
from panns_inference.config import labels as LABELS

SR = 32000
WINDOW_S = 10.0
HOP_S = 5.0
MIN_S = 1.0
CHECKPOINT = Path.home() / "panns_data/Cnn14_mAP=0.431.pth"


class Panns:
    def __init__(self) -> None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AudioTagging(checkpoint_path=str(CHECKPOINT), device=device)

    def __call__(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """mono: float32 at 32 kHz. Returns (scores[527], embedding[2048])."""
        win, hop, min_len = int(WINDOW_S * SR), int(HOP_S * SR), int(MIN_S * SR)
        if len(mono) < min_len:
            mono = np.pad(mono, (0, min_len - len(mono)))
        if len(mono) <= win:
            chunks = [mono]
        else:
            starts = range(0, len(mono) - win + hop, hop)
            chunks = [mono[s:s + win] for s in starts]
            chunks[-1] = np.pad(chunks[-1], (0, win - len(chunks[-1])))
        scores, embs = [], []
        for i in range(0, len(chunks), 16):
            batch = np.stack(chunks[i:i + 16]) if len(chunks) > 1 else chunks[0][None]
            s, e = self.model.inference(batch)
            scores.append(s)
            embs.append(e)
        # A sound is "present" if any window detects it; embedding is the average.
        return np.concatenate(scores).max(axis=0), np.concatenate(embs).mean(axis=0)


def top(scores: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    idx = np.argsort(scores)[::-1][:n]
    return [(LABELS[i], round(float(scores[i]), 3)) for i in idx]


def score(scores: np.ndarray, *names: str) -> float:
    """Highest score among the named AudioSet classes."""
    return max(float(scores[LABELS.index(n)]) for n in names)
