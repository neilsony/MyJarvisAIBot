"""Speaker diarization — "who spoke when" — via pyannote.

Every NOTB episode has four hosts talking over each other. Diarization is the
step that makes any of it usable: without it there is no way to separate
Darrick's voice from Kenny's, and a clone trained on the mix learns the mix.

pyannote's models are gated on HuggingFace. You must accept their terms once,
per model, while signed in — SETUP.md step 3. The error you get otherwise is an
unhelpful 401, so this module checks and says so plainly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.persist import load_turns, save_turns
from pipeline.segments import Turn, merge_adjacent
from pipeline.verify_speakers import LabelReport, save_label_reports, verify_labels

__all__ = ["DIARIZATION_MODEL", "diarize", "turns_from_annotation"]

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# NOTB has four regular hosts, but pinning the count to exactly 4 gave the
# clusterer no room to fail safely: when two hosts blended acoustically it
# fused them into one label and split someone else instead (the Darrick-
# trapped-in-Pierre failure — see verify_speakers.py). Over-segmenting by one
# gives a blended pair somewhere to go besides a real host's label. The extra
# label costs nothing downstream: it maps to no host name in speaker_names.json
# and stays unnamed, or merges trivially once identified.
NOTB_SPEAKER_COUNT = 5

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
    verify_embed: Callable[[Turn], np.ndarray] | None = None,
    verify_reference: np.ndarray | None = None,
    conf_cache_path: Path | None = None,
) -> list[Turn]:
    """Diarize one episode into merged speaker turns.

    Results are cached to `cache_path` when given — diarization is minutes of
    compute per episode and everything downstream re-reads it while you tune
    thresholds.

    When `verify_embed` and `verify_reference` are both given, every turn is
    also scored against the reference voice right here — while the audio is
    already loaded — and the per-label distributions are cached to
    `conf_cache_path` (see `verify_speakers.verify_labels`). This is what lets
    a merged label (see `NOTB_SPEAKER_COUNT`) get flagged the moment it's
    created rather than discovered later by a separate `verify-speakers` run.
    The turns cache itself is never touched by this — it stays immutable
    evidence; confidence is a parallel, later-computed opinion about it.
    """
    turns = load_turns(cache_path) if cache_path is not None else None
    if turns is None:
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "pyannote is not installed. Run: pip install -e '.[pipeline]' "
                "— see SETUP.md step 2."
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

    # Verification is cheap relative to diarization (per-turn embedding, not
    # full clustering) and useful even for turns that were loaded from cache,
    # so it isn't gated on whether diarization actually ran above — only on
    # whether its own cache already exists.
    if (
        verify_embed is not None
        and verify_reference is not None
        and (conf_cache_path is None or not conf_cache_path.is_file())
    ):
        reports = verify_labels(verify_embed, turns, verify_reference)
        if conf_cache_path is not None:
            save_label_reports(conf_cache_path, reports)

    return turns


def bimodal_labels(reports: list[LabelReport]) -> list[LabelReport]:
    """Labels whose confidence distribution looks like two speakers merged.

    A thin, named filter over `verify_labels` output so callers (the `diarize`
    CLI command) can flag these immediately instead of re-deriving the check.
    """
    return [r for r in reports if r.bimodal]
