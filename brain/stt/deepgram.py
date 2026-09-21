"""Speech-to-text via Deepgram.

Push-to-talk means the whole utterance exists before anything is sent, so this
uses the batch endpoint rather than the streaming websocket: same model, same
accuracy, far less lifecycle to get wrong. Swapping to true streaming later is
a change inside this module — the loop above it just asks for a transcript.
"""

from __future__ import annotations

import io
import wave
from typing import Any

__all__ = ["DeepgramSTT", "DEFAULT_MODEL", "pcm16_to_wav"]

# Nova-3 is Deepgram's current general model. Player names and team nicknames
# are exactly the proper-noun case cloud STT handles better than local models.
DEFAULT_MODEL = "nova-3"


class DeepgramSTT:
    """Transcribe raw PCM audio through Deepgram."""

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL) -> None:
        from deepgram import DeepgramClient

        self._client = DeepgramClient(api_key=api_key)
        self._model = model

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        """Return the transcript of 16-bit mono PCM audio, or "" if silent."""
        if not pcm16:
            return ""

        # Sent as a WAV rather than raw PCM: the SDK's file endpoint has no
        # sample-rate parameter, so the audio has to describe its own format.
        # A header also makes a mismatch impossible rather than merely unlikely.
        response: Any = self._client.listen.v1.media.transcribe_file(
            request=pcm16_to_wav(pcm16, sample_rate),
            model=self._model,
            smart_format=True,
            punctuate=True,
        )
        return _first_transcript(response)


def pcm16_to_wav(pcm16: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM in a WAV container, in memory."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16)
    return buffer.getvalue()


def _first_transcript(response: Any) -> str:
    """Pull the best transcript out of a Deepgram response, tolerantly.

    The response is a typed object, but the shape differs across SDK versions
    and a missing alternative is normal for silence. Anything unexpected reads
    as "nothing was said" rather than raising — a dropped word is recoverable
    in conversation, a crashed voice loop is not.
    """
    try:
        channels = response.results.channels
        if not channels:
            return ""
        alternatives = channels[0].alternatives
        if not alternatives:
            return ""
        return str(alternatives[0].transcript or "").strip()
    except AttributeError:
        return ""
