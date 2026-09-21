"""Speaker diarization — "who spoke when" — via pyannote.

Every NOTB episode has four hosts talking over each other. Diarization is the
step that makes any of it usable: without it there is no way to separate
Darrick's voice from Kenny's, and a clone trained on the mix learns the mix.

pyannote's models are gated on HuggingFace. You must accept their terms once,
per model, while signed in — SETUP.md step 3. The error you get otherwise is an
unhelpful 401, so this module checks and says so plainly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.persist import load_turns, save_turns
from pipeline.segments import Turn, merge_adjacent

__all__ = ["DIARIZATION_MODEL", "diarize", "turns_from_annotation"]

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# NOTB has four regular hosts. Pinning the count materially improves accuracy
# over letting the model guess, and guessing tends to over-split one speaker
# across two labels when someone's mic level shifts.
NOTB_SPEAKER_COUNT = 4

# Diarizers fragment a sentence across a breath. Stitching sub-second gaps back
# together is what makes turns long enough to survive the voiceprint filter.
DEFAULT_MERGE_GAP = 0.4


def turns_from_annotation(annotation: Any) -> list[Turn]:
    """Convert a pyannote Annotation into our own Turn list.

    Isolating this keeps pyannote's types at the edge of the system — everything
    downstream works on plain Turns that are trivial to construct in a test.
    """
    turns: list[Turn] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        if segment.end > segment.start:  # pyannote occasionally emits zero-width spans
            turns.append(
                Turn(speaker=str(label), start=float(segment.start), end=float(segment.end))
            )
    return sorted(turns)


def diarize(
    audio_path: Path,
    hf_token: str,
    *,
    cache_path: Path | None = None,
    num_speakers: int | None = NOTB_SPEAKER_COUNT,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> list[Turn]:
    """Diarize one episode into merged speaker turns.

    Results are cached to `cache_path` when given — diarization is minutes of
    compute per episode and everything downstream re-reads it while you tune
    thresholds.
    """
    if cache_path is not None:
        cached = load_turns(cache_path)
        if cached is not None:
            return cached

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "pyannote is not installed. Run: pip install -e '.[pipeline]' — see SETUP.md step 2."
        ) from exc

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
    if pipeline is None:
        raise RuntimeError(
            f"Could not load {DIARIZATION_MODEL}. This almost always means the gated "
            f"model terms haven't been accepted on your HuggingFace account, or HF_TOKEN "
            f"is wrong. See SETUP.md step 3."
        )

    # Typed as Any deliberately: pyannote's own signature is a union covering
    # its streaming mode, and this module's job is to keep those types at the
    # edge of the system rather than leak them into everything downstream.
    output: Any = pipeline(str(audio_path), num_speakers=num_speakers)

    # pyannote 4 wraps the result. Take `speaker_diarization`, never
    # `exclusive_speaker_diarization` — the exclusive variant has already
    # stripped overlapping speech, which would leave `clean_turns` with no
    # crosstalk left to detect and quietly pass contaminated audio into the
    # voiceprint. The overlaps are the signal, not noise, at this stage.
    turns = merge_adjacent(
        turns_from_annotation(output.speaker_diarization), max_gap=merge_gap
    )

    if cache_path is not None:
        save_turns(cache_path, turns)
    return turns
