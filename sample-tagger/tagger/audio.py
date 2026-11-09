"""Read-only audio decoding via ffmpeg.

Files are decoded into memory; nothing is ever written back to the library.
"""

import subprocess
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".mp3", ".m4a", ".flac", ".ogg"}


def load(path: Path, sr: int, channels: int | None = None) -> np.ndarray:
    """Decode to float32 of shape (channels, samples) at `sr`.

    channels=None keeps the file's own channel count.
    """
    if channels is None:
        channels = probe_channels(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T.copy()


def probe_channels(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return max(1, min(2, int(out or 1)))
