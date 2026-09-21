"""Speech-to-text over episode audio, via faster-whisper.

Runs locally: the corpus is ~800 hours and paying a hosted ASR service per
minute for material that only needs transcribing once would be a real bill.
`small.en` is the sweet spot — noticeably better than `base` on player names,
several times faster than `medium`, and int8 quantised it runs on CPU.

Note this is a *separate* pass from diarization. Whisper hears the words but not
who said them; pyannote knows who spoke but not what. `pipeline.transcript`
joins the two.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.transcript import TranscriptSegment

__all__ = ["DEFAULT_MODEL", "transcribe"]

DEFAULT_MODEL = "small.en"


def transcribe(
    audio_path: Path,
    *,
    model_size: str = DEFAULT_MODEL,
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[TranscriptSegment]:
    """Transcribe one episode into timestamped segments.

    Timestamps are what make the join to diarization possible, so segment-level
    output is required — a flat transcript would be useless here.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -e '.[pipeline]'"
        ) from exc

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)

    return [
        TranscriptSegment(text=s.text.strip(), start=float(s.start), end=float(s.end))
        for s in segments
        if s.text.strip()
    ]
