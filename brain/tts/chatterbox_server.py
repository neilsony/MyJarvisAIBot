"""Chatterbox-Turbo, as a persistent background daemon.

**This file runs under `.venv-tts`, not the main venv.** That isn't a style
choice: chatterbox-tts pins `torch==2.6.0` while pyannote (and so the rest of
this project) needs `torch>=2.8`. One process cannot hold both, so the engine
gets its own interpreter and the two talk over a Unix socket.

A daemon, not a subprocess tied to one run: the client used to spawn this
fresh per `voice_loop` invocation, paying the ~30s model load and the
reference-conditioning cost every single time. This version starts once,
serves every client that connects afterward — across as many separate
`voice_loop` runs as you want — until you kill it by hand.

Protocol, over a Unix domain socket at `SOCKET_PATH` — newline-delimited JSON,
one request per line, same shape whether it's the first message on a fresh
connection or the hundredth:

    (on connect)                <- {"ok": true, "ready": true, "sample_rate": 24000}
    {"text": "say this", "out": "/tmp/reply.wav"}   ->   {"ok": true, "path": "..."}
                                                          {"ok": false, "error": "..."}

Single-threaded on purpose: this is a personal desk bot with one user talking
at a time, and a shared PyTorch model across concurrent requests is a real
correctness risk for no actual benefit here.

To stop it: `pkill -f brain.tts.chatterbox_server`.
"""

from __future__ import annotations

import json
import socketserver
import sys
from pathlib import Path
from typing import Any

# librosa.resample() hands back float64 in this version while the model's
# weights are float32, which crashes inside the mel filterbank. Patch it here,
# before chatterbox is imported, rather than editing site-packages.
import librosa  # noqa: E402
import numpy as np  # noqa: E402

_orig_resample = librosa.resample


def _resample_f32(*args: Any, **kwargs: Any) -> Any:
    return _orig_resample(*args, **kwargs).astype(np.float32)


librosa.resample = _resample_f32

SOCKET_PATH = Path("/tmp/dmills-tts.sock")
# Duplicated from chatterbox_client.py rather than imported: this module runs
# under .venv-tts, where the `brain` package (and its own dependencies) was
# never installed. Keep it in sync by hand if it ever moves.
DEFAULT_REFERENCE = (
    Path(__file__).resolve().parent.parent.parent / "voices" / "dmills" / "reference.wav"
)


class _Handler(socketserver.StreamRequestHandler):
    server: TTSServer

    def _reply(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload) + "\n").encode())
        self.wfile.flush()

    def handle(self) -> None:
        import soundfile as sf

        model = self.server.model
        self._reply({"ok": True, "ready": True, "sample_rate": int(model.sr)})

        for raw in self.rfile:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                text = str(request["text"]).strip()
                out_path = Path(request["out"])
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # No audio_prompt_path: the reference was conditioned once at
                # startup (see main()), and re-passing it here would redo that
                # work on every line — exactly the per-utterance cost this
                # daemon exists to avoid paying twice, let alone every run.
                wav = model.generate(text)
                # soundfile rather than torchaudio.save: newer torchaudio
                # routes saving through torchcodec, not installed here and not
                # worth adding for a one-line file write.
                sf.write(str(out_path), wav.squeeze(0).cpu().numpy(), int(model.sr))
                self._reply({"ok": True, "path": str(out_path)})
            except Exception as exc:  # a bad line shouldn't kill a warm daemon
                self._reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class TTSServer(socketserver.UnixStreamServer):
    def __init__(self, address: str, model: Any) -> None:
        self.model = model
        super().__init__(address, _Handler)


def main() -> int:
    reference = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFERENCE
    if not reference.is_file():
        print(f"reference clip not found: {reference}", file=sys.stderr)
        return 1

    if SOCKET_PATH.exists():
        # A stale socket file from a crashed prior daemon — nothing is
        # actually listening, so it's safe to clear before rebinding.
        SOCKET_PATH.unlink()

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading model (first time only, ~30s)...", file=sys.stderr)
    # CPU, not MPS: the tokenizer's mel filterbank uses float64, which Apple's
    # MPS backend cannot represent at all.
    model = ChatterboxTurboTTS.from_pretrained(device="cpu")
    model.prepare_conditionals(str(reference))
    print(f"Ready. Listening on {SOCKET_PATH}", file=sys.stderr)

    server = TTSServer(str(SOCKET_PATH), model)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        SOCKET_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
