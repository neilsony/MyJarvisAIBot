"""Join ASR text to diarized speakers, then chunk it for retrieval.

Whisper hears words but not who spoke them. pyannote knows who spoke but not
what they said. Canon needs both — "Darrick said X" is the whole point, and
attributing Kenny's take to Darrick is worse than losing it, because nothing
downstream can tell it went wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.segments import Turn

__all__ = [
    "AttributedSegment",
    "Chunk",
    "TranscriptSegment",
    "UNKNOWN_SPEAKER",
    "attribute",
    "chunk_transcript",
    "parse_chunk_lines",
]

UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass(frozen=True)
class TranscriptSegment:
    """A stretch of recognised speech, straight from ASR. No speaker yet."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AttributedSegment:
    """A transcript segment with a speaker attached."""

    text: str
    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit: several consecutive segments, with provenance.

    `start`/`end` are kept so a retrieved take can be traced back to the exact
    moment in the episode — useful when the bot cites something and you want to
    check it actually said that.
    """

    text: str
    speakers: tuple[str, ...]
    episode: str
    start: float
    end: float


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def attribute(
    segments: list[TranscriptSegment],
    turns: list[Turn],
) -> list[AttributedSegment]:
    """Assign each ASR segment the speaker who overlaps it most.

    ASR and diarization draw their boundaries independently, so a segment
    routinely straddles a speaker change. Most-overlap is the right call: the
    person who said the bulk of the words owns the line.

    A segment overlapping nothing becomes `UNKNOWN` rather than being dropped —
    it still belongs in the Canon, just unattributed. Ties break on speaker name
    so repeated runs over identical input agree.
    """
    attributed: list[AttributedSegment] = []
    for segment in segments:
        best_speaker = UNKNOWN_SPEAKER
        best_overlap = 0.0
        for turn in sorted(turns, key=lambda t: t.speaker):
            overlap = _overlap(segment.start, segment.end, turn.start, turn.end)
            if overlap > best_overlap:
                best_speaker, best_overlap = turn.speaker, overlap
        attributed.append(
            AttributedSegment(
                text=segment.text,
                speaker=best_speaker,
                start=segment.start,
                end=segment.end,
            )
        )
    return attributed


def chunk_transcript(
    segments: list[AttributedSegment],
    episode: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Group segments into overlapping, speaker-labelled retrieval chunks.

    Speaker names are written into the chunk text rather than kept only as
    metadata, because the model reads the text — "DARRICK: ..." is what lets it
    tell his takes from Kenny's when the chunk lands in the prompt.

    Chunks overlap so a thought split across a boundary is still retrievable
    whole from one side or the other.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be less than max_chars")
    if not segments:
        return []

    chunks: list[Chunk] = []
    current: list[AttributedSegment] = []
    current_len = 0

    def flush() -> None:
        if not current:
            return
        chunks.append(
            Chunk(
                text="\n".join(f"{s.speaker}: {s.text}" for s in current),
                speakers=tuple(dict.fromkeys(s.speaker for s in current)),
                episode=episode,
                start=current[0].start,
                end=current[-1].end,
            )
        )

    for segment in segments:
        line_len = len(segment.speaker) + len(segment.text) + 2
        if current and current_len + line_len > max_chars:
            flush()
            # Carry the tail forward so a thought spanning the seam survives.
            carried: list[AttributedSegment] = []
            carried_len = 0
            for prior in reversed(current):
                prior_len = len(prior.speaker) + len(prior.text) + 2
                if carried_len + prior_len > overlap_chars:
                    break
                carried.insert(0, prior)
                carried_len += prior_len
            current, current_len = carried, carried_len
        current.append(segment)
        current_len += line_len

    flush()
    return chunks


def parse_chunk_lines(text: str) -> list[tuple[str, str]]:
    """Reverse `chunk_transcript`'s `"SPEAKER: text"` join back into pairs.

    For analysis over already-ingested Canon, where only the joined chunk text
    is stored — not the original segments. Splits on the *first* `": "` per
    line, since a speaker label is a bare identifier (`DARRICK`, `SPEAKER_00`)
    that never contains a colon itself, while the dialogue after it might.
    """
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        speaker, sep, rest = line.partition(": ")
        if sep:
            pairs.append((speaker, rest))
    return pairs
