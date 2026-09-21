"""Turn-level segment logic for the corpus pipeline.

Diarization gives us "who spoke when" as a soup of overlapping turns across four
hosts. This module turns that soup into the two things downstream needs:

  - Canon:      every turn, attributed, for RAG over what was said.
  - Voiceprint: only turns where Darrick speaks *alone and uninterrupted*,
                because a clone trained on crosstalk learns the crosstalk.

Everything here is pure and deterministic. The audio I/O and the model
inference live in the thin wrappers that call this.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "Turn",
    "clean_turns",
    "merge_adjacent",
    "overlap_fraction",
    "overlaps",
    "take_until_duration",
    "total_duration",
]


@dataclass(frozen=True, order=True)
class Turn:
    """A contiguous stretch of one speaker talking, in seconds from file start.

    `speaker` is a diarization label (``SPEAKER_01``), not a name — mapping
    labels to humans is `pipeline.speakers`' job.
    """

    speaker: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"end must be after start (got {self.start} -> {self.end})")

    @property
    def duration(self) -> float:
        return self.end - self.start


def overlaps(a: Turn, b: Turn) -> bool:
    """True when two turns share any wall-clock time.

    Turns that merely touch (one ends exactly where the next begins) do not
    overlap — that's the normal case between consecutive speakers, not crosstalk.
    """
    return a.start < b.end and b.start < a.end


def overlap_fraction(turn: Turn, others: list[Turn]) -> float:
    """How much of `turn` is covered by someone else, as a 0-1 fraction.

    `clean_turns` is all-or-nothing — any crosstalk at all disqualifies a
    turn, which is right for training data but throws away the distinction
    between "one person says 'yeah' for half a second at the edge" and "two
    people talk over each other the whole time." When nothing is perfectly
    clean, that distinction is exactly what's worth ranking by.
    """
    if turn.duration <= 0:
        return 0.0
    covered = sum(
        max(0.0, min(turn.end, other.end) - max(turn.start, other.start))
        for other in others
        if other.speaker != turn.speaker
    )
    return min(covered / turn.duration, 1.0)


def merge_adjacent(turns: list[Turn], max_gap: float) -> list[Turn]:
    """Join same-speaker turns separated by a gap of at most `max_gap` seconds.

    Diarizers fragment a single sentence across a breath. Merging first means
    `clean_turns` sees whole thoughts rather than clauses, which materially
    increases how much usable training audio survives the duration filter.
    """
    if not turns:
        return []

    merged: list[Turn] = []
    for turn in sorted(turns):
        prev = merged[-1] if merged else None
        if prev is not None and prev.speaker == turn.speaker and turn.start - prev.end <= max_gap:
            merged[-1] = replace(prev, end=max(prev.end, turn.end))
        else:
            merged.append(turn)
    return merged


def clean_turns(
    turns: list[Turn],
    speaker: str,
    min_duration: float,
    pad: float,
) -> list[Turn]:
    """Select turns where `speaker` talks alone, long enough to be worth training on.

    Three filters, in order:

    1. Drop turns from anyone else.
    2. Drop turns that any *other* speaker overlaps — that's crosstalk, and it
       poisons a voice clone with a second person's timbre.
    3. Trim `pad` seconds off each end (the diarizer's boundaries are fuzzy and
       tend to catch the tail of the previous speaker), then keep what still
       runs at least `min_duration`.
    """
    others = [x for x in turns if x.speaker != speaker]

    kept: list[Turn] = []
    for turn in sorted(x for x in turns if x.speaker == speaker):
        if any(overlaps(turn, other) for other in others):
            continue
        start, end = turn.start + pad, turn.end - pad
        if end - start >= min_duration:
            kept.append(replace(turn, start=start, end=end))
    return kept


def take_until_duration(turns: list[Turn], target_seconds: float) -> list[Turn]:
    """Take the fewest, longest turns that reach `target_seconds` of audio.

    Longest-first because a voice clone learns more from one 30-second monologue
    than from ten 3-second interjections — longer turns carry prosody and pacing,
    not just phonemes. Returns chronological order so the output is reproducible
    and easy to eyeball against the source.
    """
    selected: list[Turn] = []
    accumulated = 0.0
    for turn in sorted(turns, key=lambda x: x.duration, reverse=True):
        if accumulated >= target_seconds:
            break
        selected.append(turn)
        accumulated += turn.duration
    return sorted(selected)


def total_duration(turns: list[Turn]) -> float:
    """Total speech time across `turns`, in seconds. Does not deduplicate overlap."""
    return sum(x.duration for x in turns)
