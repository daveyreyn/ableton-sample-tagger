"""Hand-built measurements: the things the neural models don't know about.

All values are plain floats/strings so they can be stored as JSON.
"""

import warnings

import librosa
import numpy as np

# Bump when measurements change; `scan --features-only` recomputes older rows.
VERSION = 2

SR = 22050
HOP = 256
N_MFCC = 13
SILENCE_DB = -60.0

warnings.filterwarnings("ignore", message="n_fft=.* is too large")

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-9))


def stereo_width(stereo: np.ndarray) -> float:
    """Side/mid energy ratio: 0 = mono, ~1 = very wide or out of phase."""
    if stereo.shape[0] < 2:
        return 0.0
    mid = (stereo[0] + stereo[1]) / 2
    side = (stereo[0] - stereo[1]) / 2
    return float(np.sqrt(np.mean(side**2) / max(np.mean(mid**2), 1e-12)))


def estimate_key(y: np.ndarray) -> tuple[str | None, float]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP * 2).mean(axis=1)
    if chroma.sum() <= 0:
        return None, 0.0
    best, best_r = None, -1.0
    for i in range(12):
        for profile, mode in ((_MAJOR, "maj"), (_MINOR, "min")):
            r = np.corrcoef(chroma, np.roll(profile, i))[0, 1]
            if r > best_r:
                best, best_r = f"{_NOTES[i]} {mode}", r
    return best, float(best_r)


def measure(stereo: np.ndarray) -> dict:
    y = stereo.mean(axis=0)
    duration = len(y) / SR
    f: dict = {"v": VERSION, "duration_s": round(duration, 3), "channels": int(stereo.shape[0]),
               "stereo_width": round(stereo_width(stereo), 3)}
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    f["peak_db"] = round(float(_db(np.array(peak))), 1)
    if peak <= 1e-6 or len(y) < HOP * 4:
        f["silent"] = True
        return f

    # Loudness envelope, relative to the loudest point
    rms = librosa.feature.rms(y=y, frame_length=HOP * 4, hop_length=HOP)[0]
    env = _db(rms) - _db(np.array(rms.max()))
    t = librosa.frames_to_time(np.arange(len(env)), sr=SR, hop_length=HOP)
    sounding = np.where(env > -40)[0]
    start, end = sounding[0], sounding[-1]
    peak_i = int(np.argmax(env))
    f["attack_s"] = round(float(t[peak_i] - t[start]), 3)
    f["decay_s"] = round(float(t[end] - t[peak_i]), 3)
    f["sounding_s"] = round(float(t[end] - t[start]), 3)
    f["peak_pos"] = round(float((t[peak_i] - t[start]) / max(t[end] - t[start], 1e-3)), 3)

    # Does it build up? Slope of loudness and brightness over the sounding part.
    seg = slice(start, end + 1)
    cent = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0]
    if end - start > 8:
        ts = t[seg] - t[start]
        f["loudness_slope_db_s"] = round(float(np.polyfit(ts, env[seg], 1)[0]), 2)
        f["brightness_slope_hz_s"] = round(float(np.polyfit(ts, cent[seg], 1)[0]), 1)

    # Spectral shape, weighted toward the loud frames
    w = rms[seg] / rms[seg].sum()
    f["centroid_hz"] = round(float(np.sum(cent[seg] * w)), 1)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = S.sum(axis=1)
    total = band.sum()
    f["sub_ratio"] = round(float(band[freqs < 60].sum() / total), 3)
    f["low_ratio"] = round(float(band[freqs < 200].sum() / total), 3)
    f["high_ratio"] = round(float(band[freqs > 4000].sum() / total), 3)
    f["flatness"] = round(float(np.median(librosa.feature.spectral_flatness(S=S)[0][seg])), 4)

    # Tonal vs percussive energy
    harm, perc = librosa.decompose.hpss(S)
    f["percussive_ratio"] = round(float(perc.sum() / max(harm.sum() + perc.sum(), 1e-12)), 3)

    # Count only clear hits: strong onsets that are within 20 dB of the peak
    # (pad with silence first, or a hit at sample 0 isn't seen as an onset)
    pad = HOP * 16
    frames = librosa.onset.onset_detect(y=np.pad(y, (pad, 0)), sr=SR, hop_length=HOP,
                                        delta=0.3, wait=8) - 16
    onsets = [i for i in frames if env[min(max(i, 0) + 2, len(env) - 1)] > -20]
    f["onsets"] = int(len(onsets))
    f["onset_rate"] = round(len(onsets) / max(duration, 1e-3), 2)

    # Timbre: MFCCs summarise the overall spectral shape ("tone colour") of each
    # frame. Averaged over the whole sound and, separately, over the first ~50 ms,
    # since the attack is what most distinguishes drums.
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(
        librosa.feature.melspectrogram(S=S, sr=SR)), n_mfcc=N_MFCC)
    att = slice(start, min(start + 5, end + 1))
    f["mfcc_mean"] = [round(float(v), 2) for v in mfcc[:, seg].mean(axis=1)]
    f["mfcc_std"] = [round(float(v), 2) for v in mfcc[:, seg].std(axis=1)]
    f["mfcc_attack"] = [round(float(v), 2) for v in mfcc[:, att].mean(axis=1)]
    # Spectral contrast: peak-vs-valley per frequency band (tonal vs noisy, per band)
    contrast = librosa.feature.spectral_contrast(S=np.sqrt(S), sr=SR)
    f["contrast"] = [round(float(v), 2) for v in contrast[:, seg].mean(axis=1)]
    f["rolloff_hz"] = round(float(np.median(
        librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0][seg])), 1)
    f["zcr"] = round(float(np.mean(librosa.feature.zero_crossing_rate(
        y, frame_length=HOP * 4, hop_length=HOP)[0][seg])), 4)
    # How fast does it die away? Loudness slope in the 200 ms after the peak.
    tail = env[peak_i:peak_i + int(0.2 * SR / HOP)]
    if len(tail) > 3:
        f["early_decay_db_s"] = round(float(np.polyfit(
            np.arange(len(tail)) * HOP / SR, tail, 1)[0]), 1)

    if f["percussive_ratio"] < 0.6:
        key, conf = estimate_key(y)
        f["key"], f["key_conf"] = key, round(conf, 3)
    return f
