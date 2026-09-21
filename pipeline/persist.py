"""Cache diarization output to disk.

Diarizing a 90-minute episode is minutes of compute. Everything downstream —
speaker identification, voiceprint extraction, Canon chunking — re-reads those
turns repeatedly while you iterate on thresholds. Caching turns that from a
recurring cost into a one-off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.segments import Turn

__all__ = ["load_turns", "save_turns", "turns_from_records", "turns_to_records"]

TurnRecord = dict[str, Any]


def turns_to_records(turns: list[Turn]) -> list[TurnRecord]:
    """Turns as plain JSON-safe dicts."""
    return [{"speaker": t.speaker, "start": t.start, "end": t.end} for t in turns]


def turns_from_records(records: list[TurnRecord]) -> list[Turn]:
    """Rebuild turns from `turns_to_records` output.

    Malformed records raise rather than being skipped: a partially-read cache
    would silently shrink the corpus, which is much harder to notice than a crash.
    """
    turns: list[Turn] = []
    for record in records:
        try:
            speaker = record["speaker"]
            start = float(record["start"])
            end = float(record["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed turn record: {record!r}") from exc
        turns.append(Turn(speaker=str(speaker), start=start, end=end))
    return turns


def save_turns(path: Path, turns: list[Turn]) -> None:
    """Write turns to `path` as JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(turns_to_records(turns), indent=2), encoding="utf-8")


def load_turns(path: Path) -> list[Turn] | None:
    """Read cached turns, or None when there is no cache yet."""
    if not path.is_file():
        return None
    return turns_from_records(json.loads(path.read_text(encoding="utf-8")))
