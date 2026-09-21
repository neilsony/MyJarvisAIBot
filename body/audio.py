"""Microphone capture and speaker playback.

The first real piece of the Body — everything here runs wherever the hardware
is, which today is the Mac and later the Pi. Deliberately thin: capture raw
PCM, play a WAV, nothing clever. The Brain doesn't know or care what recorded
the audio.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["RECORD_SAMPLE_RATE", "record_until_enter", "play_wav"]

# Deepgram accepts 16 kHz happily and it's a quarter the bytes of 48 kHz over
# the wire. The voiceprint needs full bandwidth; a transcript does not.
RECORD_SAMPLE_RATE = 16_000


def record_until_enter(sample_rate: int = RECORD_SAMPLE_RATE) -> bytes:
    """Record from the default mic until Enter is pressed. Returns 16-bit PCM.

    Push-to-talk, keyboard edition: the hardware button comes in Phase 3, and
    the seam is the same either way — something says "start", something says
    "stop", and this hands back the bytes in between.
    """
    import sounddevice as sd

    chunks: list[bytes] = []

    def callback(indata, _frames, _time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            print(f"  (audio warning: {status})")
        chunks.append(bytes(indata))

    with sd.RawInputStream(
        samplerate=sample_rate, channels=1, dtype="int16", callback=callback
    ):
        input()

    return b"".join(chunks)


def play_wav(path: Path) -> None:
    """Play a WAV file through the default output device, blocking until done."""
    import sounddevice as sd
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32")
    sd.play(data, sample_rate)
    sd.wait()
