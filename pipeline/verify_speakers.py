"""Per-turn speaker verification — which diarization labels actually contain Darrick?

`identify` averages every turn of a label into one centroid and compares that
to the reference. That works when labels are pure, and silently fails when
they are not: the diarizer sometimes *merges* two hosts into one label (and
sometimes splits one host across two), and an averaged centroid then sits
between the two real voices, matching neither well or — worse — matching the
wrong one plausibly. Neil's ear caught exactly this: of three sampled turns
from one label, one was Darrick and two were not.

The fix is to score each turn individually and look at the *distribution*
inside a label, not its average:

  - a pure label is one tight cluster of scores;
  - a mixed label is two clusters — Darrick's turns high, the co-host's low.

Two design points keep this cheap and trustworthy:

  - The reference comes from the SOLO videos, never from show episodes —
    the solo pipeline has no 4-way separation problem, so its clips are
    ground truth by construction.
  - Only a slice from the MIDDLE of the episode is scored. Darrick talks
    less in cold opens; the middle is where every host is warm and the
    conversation is flowing. A slice bounds the model work, keeping this
    a minutes-per-episode check rather than a hours-long pass.

A short slice also bounds the ear-check: the report prints the highest- and
lowest-scoring turns of each candidate label, so a human confirms a label
with two listens instead of twenty.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.identify import MIN_EMBED_DURATION, MIN_SCORE
from pipeline.segments import Turn
from pipeline.speakers import cosine_similarity

__all__ = [
    "LabelReport",
    "TurnScore",
    "ear_check_turns",
    "label_reports",
    "middle_slice",
    "rank_labels",
    "score_turns",
    "select_slice_turns",
]


# Turns below this embed badly — too few phonemes to characterise a voice.
# Same floor as `identify`; one constant for the whole pipeline.
MIN_DURATION = MIN_EMBED_DURATION

# A label whose score distribution splits like this is two people, not one.
# The gap must clear the noise floor of turn-to-turn variation (intonation,
# mic distance) while staying well inside the MIN_SCORE..1.0 band where
# "plausibly the same person" lives.
BIMODAL_GAP = 0.20


@dataclass(frozen=True)
class TurnScore:
    """One turn's cosine similarity to the Darrick reference."""

    speaker: str
    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class LabelReport:
    """The score distribution of one diarization label, versus the reference.

    `high_median`/`low_median` split at MIN_SCORE: a pure label has everything
    on one side; a merged label shows two clusters with a real gap between
    them, and the gap — not the label average — is the finding.
    """

    label: str
    turn_count: int
    above_threshold: int
    median: float
    high_median: float | None
    low_median: float | None

    @property
    def above_fraction(self) -> float:
        if self.turn_count == 0:
            return 0.0
        return self.above_threshold / self.turn_count

    @property
    def bimodal(self) -> bool:
        """Two clusters with a real gap between them — this label is mixed."""
        if self.high_median is None or self.low_median is None:
            return False
        return (self.high_median - self.low_median) >= BIMODAL_GAP


def middle_slice(duration: float, minutes: float) -> tuple[float, float]:
    """A `minutes`-long window centered in audio of `duration` seconds.

    Centered, not from the top: cold opens and sponsor reads are monologues
    that over-represent whoever hosts the intro. The middle of the episode is
    where the conversation — and every voice in it — is actually going.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    span = minutes * 60.0
    if span >= duration:
        return 0.0, duration
    start = (duration - span) / 2.0
    return start, start + span


def select_slice_turns(
    turns: list[Turn],
    slice_start: float,
    slice_end: float,
    *,
    min_duration: float = MIN_DURATION,
) -> list[Turn]:
    """Turns worth scoring: inside the slice, long enough to embed.

    A turn merely *overlapping* the slice boundary still counts — the slice
    bounds the work, it does not need to amputate boundary turns mid-word.
    """
    return sorted(
        turn
        for turn in turns
        if turn.duration >= min_duration
        and turn.start < slice_end
        and turn.end > slice_start
    )


def score_turns(
    embed_turn: Callable[[Turn], np.ndarray],
    turns: Sequence[Turn],
    reference: np.ndarray,
) -> list[TurnScore]:
    """Score every turn individually against the reference. No averaging.

    Averaging is what hid the mixed-label problem in the first place; the
    whole point of this module is to see the turns the centroid washes out.
    """
    return [
        TurnScore(
            speaker=turn.speaker,
            start=turn.start,
            end=turn.end,
            score=cosine_similarity(embed_turn(turn), reference),
        )
        for turn in turns
    ]


def label_reports(
    scores: Sequence[TurnScore],
    *,
    min_score: float = MIN_SCORE,
) -> list[LabelReport]:
    """One distribution report per label, best candidate first.

    Ties break on label so repeated runs over the same audio agree — the same
    determinism rule `rank_speakers` follows.
    """
    by_label: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        by_label[s.speaker].append(s.score)

    reports: list[LabelReport] = []
    for label, values in by_label.items():
        above = [v for v in values if v >= min_score]
        below = [v for v in values if v < min_score]
        reports.append(
            LabelReport(
                label=label,
                turn_count=len(values),
                above_threshold=len(above),
                median=float(np.median(values)),
                high_median=float(np.median(above)) if above else None,
                low_median=float(np.median(below)) if below else None,
            )
        )
    return sorted(reports, key=lambda r: (-r.above_fraction, -r.median, r.label))


def rank_labels(reports: Sequence[LabelReport]) -> LabelReport | None:
    """The single best candidate label, or None when nothing looks like Darrick.

    "Best" means most turns above threshold. A label with three strong turns
    beats one with ten weak ones — Darrick talks a lot when he talks.
    """
    if not reports or reports[0].above_threshold == 0:
        return None
    return reports[0]


def ear_check_turns(
    scores: Sequence[TurnScore],
    label: str,
    *,
    per_side: int = 3,
) -> tuple[list[TurnScore], list[TurnScore]]:
    """The highest- and lowest-scoring turns of `label`, for a human to check.

    If the label is pure, top and bottom all sound like Darrick. If it is
    merged, the bottom turns are the co-host — that contrast is the whole
    diagnosis, and it takes two listens.
    """
    label_scores = [s for s in scores if s.speaker == label]
    ranked = sorted(label_scores, key=lambda s: (-s.score, s.start))
    top = ranked[:per_side]
    bottom = ranked[-per_side:] if len(ranked) > per_side else []
    return top, bottom


def format_report(
    reports: Sequence[LabelReport],
    scores: Sequence[TurnScore],
    *,
    min_score: float = MIN_SCORE,
) -> str:
    """Human-readable verification report for one episode."""
    lines: list[str] = [f"  (threshold {min_score:.2f})"]
    best = rank_labels(reports)
    for report in reports:
        marker = " <-- best candidate" if best is not None and report.label == best.label else ""
        lines.append(
            f"{report.label}: {report.turn_count} turns, "
            f"{report.above_threshold} above threshold "
            f"({report.above_fraction:.0%}), median {report.median:.3f}{marker}"
        )
        if report.bimodal:
            assert report.high_median is not None and report.low_median is not None
            lines.append(
                f"    BIMODAL — high cluster {report.high_median:.3f}, "
                f"low cluster {report.low_median:.3f}: likely TWO speakers "
                f"merged into this label"
            )

    if best is not None:
        top, bottom = ear_check_turns(scores, best.label)
        if top:
            lines.append(f"\n  Ear-check {best.label} — top-scoring turns (expect Darrick):")
            for s in top:
                lines.append(f"    {s.start:8.1f} - {s.end:8.1f}  score {s.score:.3f}")
        if bottom:
            lines.append("  ...and bottom-scoring turns (expect NOT Darrick if bimodal):")
            for s in bottom:
                lines.append(f"    {s.start:8.1f} - {s.end:8.1f}  score {s.score:.3f}")
        lines.append("\n  Listen to these, then decide. Labels are per-episode — a label")
        lines.append("  confirmed here says nothing about other episodes.")
    else:
        lines.append("\n  No label scored above threshold. Darrick may be absent from")
        lines.append("  this slice, or merged so deeply his turns never score alone.")
        lines.append("  Try a wider slice before concluding he is missing.")
    return "\n".join(lines)


def embed_turn_crop(inference: Any, audio_path: str) -> Callable[[Turn], np.ndarray]:
    """Build a turn-embedding function from a pyannote Inference.

    Lives here so the CLI passes one closure and the pure logic above stays
    testable without pyannote installed.
    """
    from pyannote.core import Segment

    def embed(turn: Turn) -> np.ndarray:
        emb = inference.crop(audio_path, Segment(turn.start, turn.end))
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    return embed
