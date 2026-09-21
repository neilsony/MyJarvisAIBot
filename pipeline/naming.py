"""Resolve diarization labels into real host names.

Diarization labels (SPEAKER_00..03) are per-episode and meaningless. Two
artifacts turn them into names, in layers of decreasing trust:

  `data/speaker_names.json`  — label-level. Which label is Kenny, Mike, or
  Pierre in this episode, hand-confirmed by ear. One name per label.

  `data/darrick_turns/<ep>.json` — time-level. Darrick often shares a label
  with Pierre (the diarizer merges them), so a label-level name cannot
  separate them. This file lists the exact windows the per-turn verifier
  scored as Darrick; those windows override whatever the label said.

Names apply FIRST (they cover whole labels), windows SECOND (they are the
more precise evidence and must win). A segment claimed by neither keeps its
raw diarization label — an honest SPEAKER_02 beats a guessed name.

Like everything in this pipeline, files that don't exist are simply skipped:
naming is opt-in per episode, and episodes without evidence stay unlabeled.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.transcript import AttributedSegment

__all__ = [
    "DARRICK_NAME",
    "apply_darrick_windows",
    "apply_names",
    "load_names",
    "load_windows",
]

DARRICK_NAME = "DARRICK"


def load_names(path: Path) -> dict[str, dict[str, str]]:
    """Read `speaker_names.json`: {episode: {label: name}}. Missing file = {}."""
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must map episode -> {{label: name}}")
    return {
        str(episode): {str(label): str(name) for label, name in mapping.items()}
        for episode, mapping in raw.items()
    }


def load_windows(path: Path) -> list[tuple[float, float]]:
    """Read a `darrick_turns/<episode>.json` window list. Missing file = [].

    Accepts both `[start, end]` pairs and full Turn records
    (`{"speaker": ..., "start": ..., "end": ...}`) — the latter is what
    `export-darrick-turns` writes, sharing the turn-cache format so every
    pipeline artifact reads the same way. Only start/end matter here: the
    speaker is implied by the directory.
    """
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    windows: list[tuple[float, float]] = []
    for entry in raw:
        if isinstance(entry, dict):
            try:
                start, end = float(entry["start"]), float(entry["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed window in {path}: {entry!r}") from exc
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            start, end = float(entry[0]), float(entry[1])
        else:
            raise ValueError(f"malformed window in {path}: {entry!r}")
        if end <= start:
            raise ValueError(f"window end must be after start in {path}: {entry!r}")
        windows.append((start, end))
    return sorted(windows)


def apply_names(
    segments: list[AttributedSegment],
    names: dict[str, str],
) -> list[AttributedSegment]:
    """Replace each label with its real name where the map has one.

    Labels the map doesn't cover keep their raw form — a missing entry is a
    gap to report, not a licence to invent a name.
    """
    return [
        s if s.speaker not in names
        else AttributedSegment(
            text=s.text, speaker=names[s.speaker], start=s.start, end=s.end
        )
        for s in segments
    ]


def _darrick_coverage(
    start: float, end: float, windows: list[tuple[float, float]]
) -> float:
    """Fraction of [start, end] covered by Darrick windows. 0 for empty spans."""
    duration = end - start
    if duration <= 0:
        return 0.0
    covered = sum(
        max(0.0, min(end, w_end) - max(start, w_start)) for w_start, w_end in windows
    )
    return min(covered / duration, 1.0)


def apply_darrick_windows(
    segments: list[AttributedSegment],
    windows: list[tuple[float, float]],
    *,
    majority: float = 0.5,
) -> list[AttributedSegment]:
    """Re-attribute segments whose time is mostly inside Darrick windows.

    Majority rule, mirroring `attribute`'s most-overlap principle: a segment
    that Darrick holds for more than `majority` of its span is his; one that
    merely touches a window edge (the crosstalk-tail failure the verifier
    showed at 51:36 in oocok) keeps whoever the label says.
    """
    return [
        s if _darrick_coverage(s.start, s.end, windows) <= majority
        else AttributedSegment(
            text=s.text, speaker=DARRICK_NAME, start=s.start, end=s.end
        )
        for s in segments
    ]
