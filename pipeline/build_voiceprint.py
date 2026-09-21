"""Cut a clean, single-speaker training set out of diarized episodes.

This is the step whose output you actually clone from, so it is deliberately
picky. Everything it emits should be Darrick, alone, uninterrupted, at full
bandwidth. Quantity is not the constraint — 30-60 minutes is plenty for every
engine in the bake-off, and there are 500 episodes to draw from. Quality is the
constraint, so the filters throw away far more than they keep, on purpose.

Two artifacts come out:

  clips/       numbered WAVs, the Piper fine-tuning set
  reference.wav  the single longest clean turn, for zero-shot engines
                 (NeuTTS Air, Chatterbox) that clone from one sample
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.segments import Turn, clean_turns, take_until_duration, total_duration

__all__ = ["VoiceprintDataset", "DEFAULTS", "build_voiceprint"]


@dataclass(frozen=True)
class VoiceprintBuildSettings:
    """Filter thresholds. Loosen `min_duration` only if you can't reach target."""

    # Long turns carry prosody, not just phonemes. Below ~4s you get words
    # without delivery, which is the opposite of what a character voice needs.
    min_duration: float = 4.0
    # Diarizer boundaries are fuzzy and tend to catch the tail of whoever spoke
    # before. Trimming a beat off each end costs little and removes real bleed.
    pad: float = 0.25
    target_minutes: float = 45.0


DEFAULTS = VoiceprintBuildSettings()


@dataclass(frozen=True)
class VoiceprintDataset:
    """What `build_voiceprint` produced, for reporting and sanity-checking."""

    clips_dir: Path
    reference_clip: Path | None
    clip_count: int
    total_seconds: float
    source_episodes: int

    @property
    def total_minutes(self) -> float:
        return self.total_seconds / 60.0


def _read_audio(path: Path) -> tuple[object, int]:
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return data, int(sample_rate)


def build_voiceprint(
    episodes: dict[Path, tuple[list[Turn], str]],
    out_dir: Path,
    *,
    settings: VoiceprintBuildSettings = DEFAULTS,
) -> VoiceprintDataset:
    """Extract clean solo audio for the identified speaker in each episode.

    `episodes` maps an audio file to its diarized turns and the diarization
    label confirmed to be Darrick in *that* file — labels are per-file, so the
    caller must have run identification first.

    Turns are pooled across episodes and the longest are taken until the target
    duration is met, so a couple of great monologues beat a hundred fragments.
    """
    import soundfile as sf

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Pool candidates across every episode before selecting, so selection is
    # global — otherwise each episode contributes its own mediocre best.
    pooled: list[tuple[Path, Turn]] = []
    for audio_path, (turns, speaker) in episodes.items():
        kept = clean_turns(
            turns,
            speaker=speaker,
            min_duration=settings.min_duration,
            pad=settings.pad,
        )
        pooled.extend((audio_path, turn) for turn in kept)

    chosen = take_until_duration(
        [turn for _, turn in pooled],
        target_seconds=settings.target_minutes * 60.0,
    )
    chosen_set = {(t.start, t.end, t.speaker) for t in chosen}
    selected = [(p, t) for p, t in pooled if (t.start, t.end, t.speaker) in chosen_set]

    manifest: list[dict[str, object]] = []
    longest: tuple[Path, Turn] | None = None

    # Read each episode's audio once, not once per clip — these are ~250MB files.
    for audio_path in sorted({p for p, _ in selected}):
        data, sample_rate = _read_audio(audio_path)
        for index, (_, turn) in enumerate(
            sorted(((p, t) for p, t in selected if p == audio_path), key=lambda pair: pair[1])
        ):
            start_frame = int(turn.start * sample_rate)
            end_frame = int(turn.end * sample_rate)
            clip_name = f"{audio_path.stem}_{index:04d}.wav"
            sf.write(str(clips_dir / clip_name), data[start_frame:end_frame], sample_rate)  # type: ignore[index]
            manifest.append(
                {
                    "clip": clip_name,
                    "source": audio_path.name,
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.duration,
                }
            )
            if longest is None or turn.duration > longest[1].duration:
                longest = (audio_path, turn)

    reference_path: Path | None = None
    if longest is not None:
        # Zero-shot engines clone from one sample; give them the best one we have.
        data, sample_rate = _read_audio(longest[0])
        reference_path = out_dir / "reference.wav"
        sf.write(
            str(reference_path),
            data[int(longest[1].start * sample_rate) : int(longest[1].end * sample_rate)],  # type: ignore[index]
            sample_rate,
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return VoiceprintDataset(
        clips_dir=clips_dir,
        reference_clip=reference_path,
        clip_count=len(manifest),
        total_seconds=total_duration([t for _, t in selected]),
        source_episodes=len({p for p, _ in selected}),
    )
