"""Work out which diarization label is Darrick.

Diarization labels (SPEAKER_00..03) are per-file and meaningless across
episodes. The bridge is a voice embedding: hand-label Darrick once, average his
embeddings into a reference centroid, then match every other episode against it.

Thresholds are deliberately conservative. An episode we skip costs nothing —
there are 500 of them. An episode we mislabel poisons the training set.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.segments import Turn, total_duration
from pipeline.speakers import SpeakerMatch, centroid, identify_speaker

__all__ = [
    "EMBEDDING_MODEL",
    "MIN_MARGIN",
    "MIN_SCORE",
    "build_reference",
    "embed_speakers",
    "find_speaker",
]

EMBEDDING_MODEL = "pyannote/embedding"

# Cosine similarity floor for "this is plausibly the same person at all".
MIN_SCORE = 0.55
# Required gap to the runner-up. Four close friends on similar mics score
# closer together than you'd expect; below this, refuse to guess.
MIN_MARGIN = 0.10

# Short turns embed badly — too few phonemes to characterise a voice.
MIN_EMBED_DURATION = 3.0


def _inference(hf_token: str) -> Any:
    try:
        from pyannote.audio import Inference, Model
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "pyannote is not installed. Run: pip install -e '.[pipeline]' — see SETUP.md step 2."
        ) from exc

    model = Model.from_pretrained(EMBEDDING_MODEL, token=hf_token)
    if model is None:
        raise RuntimeError(
            f"Could not load {EMBEDDING_MODEL}. Accept its terms on HuggingFace "
            f"while signed in — SETUP.md step 3."
        )
    return Inference(model, window="whole")


def embed_speakers(
    audio_path: Path,
    turns: list[Turn],
    hf_token: str,
    *,
    min_duration: float = MIN_EMBED_DURATION,
) -> dict[str, np.ndarray]:
    """One averaged voice embedding per diarization label in this episode.

    Turns shorter than `min_duration` are skipped — they carry too little signal
    and dragging them in blurs the centroid toward whoever else is on the mic.
    """
    from pyannote.core import Segment

    inference = _inference(hf_token)

    per_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    for turn in turns:
        if turn.duration < min_duration:
            continue
        embedding = inference.crop(str(audio_path), Segment(turn.start, turn.end))
        per_speaker[turn.speaker].append(np.asarray(embedding, dtype=np.float32).reshape(-1))

    return {label: centroid(embs) for label, embs in per_speaker.items() if embs}


def build_reference(clip_paths: list[Path], hf_token: str) -> np.ndarray:
    """Build Darrick's reference centroid from hand-picked solo clips.

    These clips are the ground truth for the entire pipeline. Pick stretches
    where he is unmistakably talking alone — SETUP.md step 5 walks through it.
    """
    if not clip_paths:
        raise ValueError("need at least one reference clip to identify a speaker")

    inference = _inference(hf_token)
    embeddings = [
        np.asarray(inference(str(path)), dtype=np.float32).reshape(-1) for path in clip_paths
    ]
    return centroid(embeddings)


def find_speaker(
    audio_path: Path,
    turns: list[Turn],
    reference: np.ndarray,
    hf_token: str,
    *,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
) -> SpeakerMatch | None:
    """Which label in this episode is the reference speaker, or None if unclear.

    None means "skip this episode and tell the human" — see `identify_speaker`
    for the two distinct reasons that happens.
    """
    candidates = embed_speakers(audio_path, turns, hf_token)
    if not candidates:
        return None
    return identify_speaker(
        candidates, reference=reference, min_score=min_score, min_margin=min_margin
    )


def speaking_time(turns: list[Turn], speaker: str) -> float:
    """Total seconds `speaker` holds the floor. A sanity check: Darrick should
    be roughly a quarter of a four-host show, so 2% or 60% both mean something
    went wrong."""
    return total_duration([t for t in turns if t.speaker == speaker])
