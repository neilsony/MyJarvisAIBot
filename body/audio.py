"""Microphone capture and speaker playback.

The first real piece of the Body — everything here runs wherever the hardware
is, which today is the Mac and later the Pi. Deliberately thin: capture raw
PCM, play a WAV, nothing clever. The Brain doesn't know or care what recorded
the audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["RECORD_SAMPLE_RATE", "MicRecorder", "play_wav"]

# Deepgram accepts 16 kHz happily and it's a quarter the bytes of 48 kHz over
# the wire. The voiceprint needs full bandwidth; a transcript does not.
RECORD_SAMPLE_RATE = 16_000


class MicRecorder:
    """Records from the default mic between `start()` and `stop()`. 16-bit PCM.

    Push-to-talk: something says "start", something says "stop", and this
    hands back the bytes in between. Today that's Enter or the button on the
    page; in Phase 3 it's the hardware button. The seam is the same.
    """

    def __init__(self, sample_rate: int = RECORD_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[bytes] = []
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd

        self._chunks = []

        def callback(indata, _frames, _time, status) -> None:  # type: ignore[no-untyped-def]
            if status:
                print(f"  (audio warning: {status})")
            self._chunks.append(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16", callback=callback
        )
        self._stream.start()

    def stop(self) -> bytes:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return b"".join(self._chunks)


def play_wav(path: Path) -> None:
    """Play a WAV file through the default output device, blocking until done."""
    import sounddevice as sd
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32")
    sd.play(data, sample_rate)
    sd.wait()
