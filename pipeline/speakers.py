"""Map diarization labels to actual humans.

Diarization tells us there were four voices and calls them SPEAKER_00..03. It
has no idea which one is Darrick, and the labels are not stable across episodes
— SPEAKER_01 in one file is a different person in the next.

The fix: hand-label Darrick once, average his speaker embeddings into a
reference centroid, then match by cosine similarity in every other episode.

The design rule here is that **ambiguity is reported, never resolved.** Four
hosts who are close friends have similar recording chains and overlapping vocal
ranges. If the top two candidates score within a whisker of each other, the
honest answer is "I don't know" — because the cost of guessing wrong is 60
minutes of training audio from the wrong person, discovered only by ear, weeks
later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SpeakerMatch",
    "centroid",
    "cosine_similarity",
    "identify_speaker",
    "rank_speakers",
]

Embedding = np.ndarray


@dataclass(frozen=True)
class SpeakerMatch:
    """A resolved diarization label, with the evidence that resolved it.

    `runner_up` is carried so a human can sanity-check a close call rather than
    discovering it downstream.
    """

    label: str
    score: float
    runner_up: str | None
    runner_up_score: float | None

    @property
    def margin(self) -> float:
        """How far clear of the next-best candidate this match sits.

        With a single candidate there is nothing to be confused with, so the
        score itself is the margin.
        """
        if self.runner_up_score is None:
            return self.score
        return self.score - self.runner_up_score


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """Cosine similarity in [-1, 1]. Magnitude-independent, so recording level
    and mic gain don't affect the comparison — only vocal character does."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cannot compare a zero-magnitude embedding")
    return float(np.dot(a, b) / (norm_a * norm_b))


def centroid(embeddings: Sequence[Embedding]) -> Embedding:
    """The unit-length mean of `embeddings` — one vector standing for a speaker.

    Averaging across many turns washes out what was said and leaves who said it.
    """
    if not embeddings:
        raise ValueError("need at least one embedding to build a centroid")

    dims = {e.shape for e in embeddings}
    if len(dims) > 1:
        raise ValueError(f"all embeddings must share the same dimension, got {sorted(dims)}")

    mean = np.mean(np.stack(embeddings), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        raise ValueError("embeddings average to a zero-magnitude vector; no centroid exists")
    return np.asarray(mean / norm, dtype=np.float32)


def rank_speakers(
    candidates: Mapping[str, Embedding],
    reference: Embedding,
) -> list[tuple[str, float]]:
    """Every candidate label scored against `reference`, best first.

    Ties break on label so repeated runs over the same audio agree.
    """
    scored = [(label, cosine_similarity(emb, reference)) for label, emb in candidates.items()]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


def identify_speaker(
    candidates: Mapping[str, Embedding],
    reference: Embedding,
    min_score: float,
    min_margin: float,
) -> SpeakerMatch | None:
    """Which candidate is the reference speaker — or None if it isn't clear.

    Returns None in two distinct failure modes, both of which mean "skip this
    episode and tell the human":

      - **Absent**: the best score is below `min_score`. Darrick probably isn't
        in this episode at all (guest host, solo show, clip compilation).
      - **Ambiguous**: two candidates are within `min_margin` of each other.
        He may well be one of them, but picking is a coin flip.
    """
    ranked = rank_speakers(candidates, reference)
    if not ranked:
        return None

    label, score = ranked[0]
    runner_up, runner_up_score = ranked[1] if len(ranked) > 1 else (None, None)

    match = SpeakerMatch(
        label=label,
        score=score,
        runner_up=runner_up,
        runner_up_score=runner_up_score,
    )
    if score < min_score or match.margin < min_margin:
        return None
    return match
