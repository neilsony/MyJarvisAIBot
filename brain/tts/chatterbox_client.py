"""Client for the Chatterbox daemon — the main-venv half of the TTS split.

Connects to an already-running daemon if one exists; spawns a new, detached
one (survives after this process exits) if not. Either way the ~30s model
load and reference-conditioning cost happen at most once per machine session,
not once per `voice_loop` run — that's the whole point of the daemon over the
subprocess-per-run version this replaced.

See `chatterbox_server.py` for the protocol and why the engine lives in a
separate interpreter at all.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from brain.config import REPO_ROOT

__all__ = ["ChatterboxTTS", "TTS_VENV_PYTHON", "DEFAULT_REFERENCE", "SOCKET_PATH"]

TTS_VENV_PYTHON = REPO_ROOT / ".venv-tts" / "bin" / "python3"
SERVER_SCRIPT = Path(__file__).parent / "chatterbox_server.py"
DEFAULT_REFERENCE = REPO_ROOT / "voices" / "dmills" / "reference.wav"
# Not imported from chatterbox_server.py: that module does `import librosa` at
# top level, a real dependency in .venv-tts but only an accidental transitive
# one here (pulled in by pyannote). A duplicated constant is more robust than
# an import that happens to work today.
SOCKET_PATH = Path("/tmp/dmills-tts.sock")

# The daemon's own first run pays the full model-load cost; give it real room.
STARTUP_TIMEOUT_SECONDS = 180.0


class ChatterboxTTS:
    """Chatterbox-Turbo behind the `TTSBackend` protocol."""

    def __init__(
        self,
        reference: Path = DEFAULT_REFERENCE,
        *,
        venv_python: Path = TTS_VENV_PYTHON,
        socket_path: Path = SOCKET_PATH,
    ) -> None:
        if not venv_python.is_file():
            raise RuntimeError(
                f"No TTS interpreter at {venv_python}. Chatterbox needs its own venv "
                f"(torch 2.6 vs the main venv's 2.8+). Create it with:\n"
                f"  python3 -m venv .venv-tts && .venv-tts/bin/pip install "
                f"chatterbox-tts 'setuptools<81'"
            )
        if not reference.is_file():
            raise RuntimeError(
                f"No reference clip at {reference}. Build one with: "
                f"python -m pipeline voiceprint-solo"
            )

        self._socket_path = socket_path
        sock = self._connect() or self._spawn_and_connect(reference, venv_python)

        self._sock = sock
        self._rfile = sock.makefile("r")
        self._wfile = sock.makefile("w")

        handshake = self._read_reply()
        if not handshake.get("ok"):
            self.close()
            raise RuntimeError(f"TTS daemon failed to start: {handshake.get('error')}")
        self._sample_rate = int(str(handshake.get("sample_rate", 24_000)))

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _connect(self) -> socket.socket | None:
        """Try an existing daemon. Returns None for "not running," never raises."""
        if not self._socket_path.exists():
            return None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(self._socket_path))
            return sock
        except ConnectionRefusedError:
            # A socket file with nothing listening — a prior daemon crashed
            # without cleaning up. Treat it the same as "not there."
            sock.close()
            return None

    def _spawn_and_connect(self, reference: Path, venv_python: Path) -> socket.socket:
        subprocess.Popen(
            [str(venv_python), str(SERVER_SCRIPT), str(reference)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(REPO_ROOT),
            # Detached from our process group: this daemon must outlive
            # whatever short-lived script spawned it, which is the entire
            # reason it exists instead of the old per-run subprocess.
            start_new_session=True,
        )

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            sock = self._connect()
            if sock is not None:
                return sock
            time.sleep(0.5)

        raise RuntimeError(
            f"TTS daemon did not come up within {STARTUP_TIMEOUT_SECONDS:.0f}s. "
            f"Check for errors: {TTS_VENV_PYTHON} {SERVER_SCRIPT} 2>&1 | head -30"
        )

    def _read_reply(self) -> dict[str, Any]:
        line = self._rfile.readline()
        if not line:
            raise RuntimeError("TTS daemon connection closed")
        parsed: dict[str, Any] = json.loads(line)
        return parsed

    def synthesize(self, text: str, out_path: Path) -> Path:
        self._wfile.write(json.dumps({"text": text, "out": str(out_path)}) + "\n")
        self._wfile.flush()

        reply = self._read_reply()
        if not reply.get("ok"):
            raise RuntimeError(f"TTS failed: {reply.get('error')}")
        return Path(str(reply["path"]))

    def close(self) -> None:
        """Disconnect this client. The daemon itself keeps running for the
        next caller — that persistence is the point. To actually stop it:
        `pkill -f brain.tts.chatterbox_server`."""
        self._rfile.close()
        self._wfile.close()
        self._sock.close()

    def __enter__(self) -> ChatterboxTTS:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def main() -> int:
    """Smoke-test the backend: `python -m brain.tts.chatterbox_client "some text"`."""
    text = sys.argv[1] if len(sys.argv) > 1 else "Testing, one two, this is the voice."
    out = REPO_ROOT / "voices" / "dmills" / "bakeoff" / "tts_smoke_test.wav"

    print("Connecting to TTS daemon (starts it if not already running)...")
    with ChatterboxTTS() as tts:
        print(f"Ready at {tts.sample_rate} Hz. Synthesizing...")
        path = tts.synthesize(text, out)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
