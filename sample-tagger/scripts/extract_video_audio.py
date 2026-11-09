"""Extract the audio track from every video in the Videos folder.

The audio stream is copied as-is (no re-encode) into an .m4a next to the
other samples, keeping the video's name and modification date. Videos are
never modified or deleted. Re-running only processes new videos.

A manifest (video-audio.json) records which audio file came from which
video, so names can be improved later without losing the link.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

USER_LIBRARY = Path.home() / "Music/Ableton/User Library"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}


def audio_codec(video: Path) -> str | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    ).stdout.strip()
    return out or None


def extract(video: Path, dest: Path, codec: str) -> None:
    # AAC can be copied losslessly into .m4a; anything else becomes WAV.
    args = ["-c:a", "copy"] if codec == "aac" else ["-c:a", "pcm_s16le"]
    tmp = dest.with_name(dest.stem + ".partial" + dest.suffix)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-map", "0:a:0",
         *args, str(tmp)],
        check=True,
    )
    tmp.rename(dest)
    st = video.stat()
    os.utime(dest, (st.st_atime, st.st_mtime))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--videos", type=Path, default=USER_LIBRARY / "Videos")
    p.add_argument("--out", type=Path, default=USER_LIBRARY / "Samples/from-videos")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out / "video-audio.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    done_sources = {e["source"] for e in manifest.values()}

    videos = sorted(f for f in a.videos.iterdir()
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
    made = skipped = silent = 0
    for video in videos:
        if str(video) in done_sources:
            skipped += 1
            continue
        codec = audio_codec(video)
        if codec is None:
            print(f"no audio: {video.name}")
            silent += 1
            continue
        dest = a.out / (video.stem + (".m4a" if codec == "aac" else ".wav"))
        n = 2
        while dest.exists():
            dest = dest.with_name(f"{video.stem} ({n}){dest.suffix}")
            n += 1
        print(f"{video.name} -> {dest.name}")
        if a.dry_run:
            continue
        extract(video, dest, codec)
        manifest[dest.name] = {"source": str(video), "original_name": video.stem}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        made += 1

    print(f"\nextracted {made}, already done {skipped}, no audio {silent}")


if __name__ == "__main__":
    main()
