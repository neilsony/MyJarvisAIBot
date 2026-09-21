"""Audio slicing helpers shared by the pipeline steps.

Episode masters are ~250MB, so the rule everywhere is: read a file once, cut
many clips from the array in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.segments import Turn

__all__ = ["read_audio", "write_turn"]


def read_audio(path: Path) -> tuple[Any, int]:
    """Read a WAV as (samples, sample_rate)."""
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return data, int(sample_rate)


def write_turn(data: Any, sample_rate: int, turn: Turn, out_path: Path) -> Path:
    """Write the slice of `data` covered by `turn` to `out_path`."""
    import soundfile as sf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = int(turn.start * sample_rate)
    end = int(turn.end * sample_rate)
    sf.write(str(out_path), data[start:end], sample_rate)
    return out_path
