"""The TTS swap point.

Every engine in the bake-off hides behind this one protocol, so choosing a
different one later is a constructor change rather than a rewrite — and tests
can run against a fake with no model, no GPU, and no 30-second load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

__all__ = ["TTSBackend", "FakeTTS"]


class TTSBackend(Protocol):
    """Turn text into a spoken WAV file."""

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio this backend produces."""
        ...

    def synthesize(self, text: str, out_path: Path) -> Path:
        """Render `text` to `out_path` as a WAV. Returns the path written."""
        ...

    def close(self) -> None:
        """Release whatever the backend is holding (subprocess, model, port)."""
        ...


class FakeTTS:
    """A backend that writes silence, for tests and for running the loop
    end-to-end without loading a model."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self._sample_rate = sample_rate
        self.calls: list[str] = []

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str, out_path: Path) -> Path:
        import wave

        self.calls.append(text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # A quarter second of silence — long enough to be a real playable file.
        with wave.open(str(out_path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self._sample_rate)
            f.writeframes(b"\x00\x00" * (self._sample_rate // 4))
        return out_path

    def close(self) -> None:
        return None
