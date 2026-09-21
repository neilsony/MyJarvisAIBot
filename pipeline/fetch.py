"""Download Numbers on the Board episodes as audio.

Audio is mastered at 24 kHz mono WAV. That is higher than diarization needs
(pyannote resamples to 16 kHz internally) but it is what the TTS engines want:
Piper trains at 22.05 kHz, NeuTTS and Chatterbox at 24 kHz. Downsampling later
is free; bandwidth you never downloaded is gone for good, and re-fetching 500
episodes to fix it is a bad afternoon.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "EpisodeRef",
    "NOTB_CHANNEL",
    "MASTER_SAMPLE_RATE",
    "list_episodes",
    "download_audio",
    "fetch_reference_clip",
]

NOTB_CHANNEL = "https://www.youtube.com/@numbersontheboard/videos"
MASTER_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class EpisodeRef:
    """One episode, before download."""

    video_id: str
    title: str
    url: str
    duration_seconds: float | None

    @property
    def audio_filename(self) -> str:
        return f"{self.video_id}.wav"


def _require_tool(name: str) -> None:
    from shutil import which

    if which(name) is None:
        raise RuntimeError(f"`{name}` is not installed. See SETUP.md step 1.")


def list_episodes(channel_url: str = NOTB_CHANNEL, limit: int | None = None) -> list[EpisodeRef]:
    """List episodes newest-first, without downloading them.

    Uses a flat extract, so this is one cheap request rather than one per video.
    """
    import yt_dlp  # imported lazily: heavy, and only Phase 0 needs it

    opts: dict[str, object] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
    }
    if limit is not None:
        opts["playlistend"] = limit

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    entries = info.get("entries") or [] if isinstance(info, dict) else []
    episodes: list[EpisodeRef] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        episodes.append(
            EpisodeRef(
                video_id=str(entry["id"]),
                title=str(entry.get("title") or "untitled"),
                url=str(entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}"),
                duration_seconds=(
                    float(entry["duration"]) if entry.get("duration") is not None else None
                ),
            )
        )
    return episodes


def download_audio(episode: EpisodeRef, out_dir: Path, *, overwrite: bool = False) -> Path:
    """Download one episode to `out_dir` as 24 kHz mono WAV. Returns the path.

    Idempotent: an existing file is reused unless `overwrite` is set, so a run
    interrupted at episode 14 of 20 resumes instead of restarting.
    """
    _require_tool("yt-dlp")
    _require_tool("ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / episode.audio_filename
    if target.exists() and not overwrite:
        return target

    # The CLI rather than the Python API: the postprocessor chain for
    # format-and-resample is far more legible as flags, and yt-dlp's CLI is the
    # interface its own docs are written against.
    subprocess.run(
        [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", f"ffmpeg:-ac 1 -ar {MASTER_SAMPLE_RATE}",
            "--output", str(out_dir / f"{episode.video_id}.%(ext)s"),
            episode.url,
        ],
        check=True,
    )

    if not target.exists():
        raise RuntimeError(f"yt-dlp reported success but {target} is missing")
    return target


def fetch_reference_clip(url: str, out_dir: Path, *, overwrite: bool = False) -> Path:
    """Download one arbitrary clip as reference audio — not an NOTB episode.

    For building Darrick's voice fingerprint from a source *outside* the show:
    a solo clip where he's the only speaker, so there's no risk of a diarizer
    confusing him with a similar-sounding co-host. Any yt-dlp-supported site
    works, not just YouTube. Lands in its own directory, separate from the
    show episodes in `data/audio/`, since it's for a different purpose.
    """
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    if not isinstance(info, dict) or not info.get("id"):
        raise RuntimeError(f"Could not read video info from {url}")

    episode = EpisodeRef(
        video_id=str(info["id"]),
        title=str(info.get("title") or "untitled"),
        url=url,
        duration_seconds=float(info["duration"]) if info.get("duration") is not None else None,
    )
    return download_audio(episode, out_dir, overwrite=overwrite)
