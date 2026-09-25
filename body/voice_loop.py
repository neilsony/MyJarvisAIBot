"""Talk to DmillsGPT out loud — the Phase 2 milestone, push-to-talk edition.

    python -m body.voice_loop            # opens the page at http://127.0.0.1:8765
    python -m body.voice_loop --no-ui    # terminal only

Press Enter (or the Talk button on the page, or space there) to start
recording, and again to send. The transcript goes to the same agent
`brain.cli` uses, and the reply comes back in his voice. The page shows the
Body state — listening, thinking, speaking — around a big picture of Darrick.

Deliberately simple for a first version: no barge-in, no sentence-level
streaming, no wake word. Those are additive once the basic loop is proven —
and this is the loop that proves the three pieces (ears, brain, voice) can
actually hold a conversation together.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import threading
from contextlib import AsyncExitStack
from pathlib import Path

from body.audio import RECORD_SAMPLE_RATE, MicRecorder, play_wav
from body.session import Controls, Press, converse
from brain.agent import Jarvis
from brain.config import Settings
from brain.store.db import Store
from brain.stt.deepgram import DeepgramSTT
from brain.tts.chatterbox_client import ChatterboxTTS

BANNER = """\
DmillsGPT — voice mode
  model: {model}
  canon: {canon_chunks} chunks   memories: {memories}
"""

QUIT_WORDS = {"q", "quit", "exit"}


class VoicePipeline:
    """The real ears, brain and voice behind `converse`.

    The blocking pieces (Deepgram, the TTS socket, playback) run in threads:
    the page's server shares this event loop and has to stay responsive.
    """

    def __init__(self, stt: DeepgramSTT, jarvis: Jarvis, tts: ChatterboxTTS, scratch: Path) -> None:
        self._stt = stt
        self._jarvis = jarvis
        self._tts = tts
        self._scratch = scratch
        self._turn = 0

    async def transcribe(self, audio: bytes) -> str:
        return await asyncio.to_thread(self._stt.transcribe, audio, RECORD_SAMPLE_RATE)

    async def reply(self, said: str) -> str:
        return await self._jarvis.ask(said)

    async def synthesize(self, reply: str) -> Path:
        self._turn += 1
        stem = self._scratch / f"turn_{self._turn:03d}"
        # The text beside the audio: a garbled reply can then be traced to
        # the exact words and replayed through the TTS on its own.
        stem.with_suffix(".txt").write_text(reply, encoding="utf-8")
        return await asyncio.to_thread(self._tts.synthesize, reply, stem.with_suffix(".wav"))

    async def play(self, wav: Path) -> None:
        # Ducked only around playback, not synthesis: no reason to hold the
        # music down through seconds of silence.
        async with self._jarvis.speaking():
            await asyncio.to_thread(play_wav, wav)


def read_terminal(controls: Controls, loop: asyncio.AbstractEventLoop) -> None:
    """Turn terminal lines into presses. Runs on its own thread: stdin blocks."""
    for line in sys.stdin:
        press = Press.QUIT if line.strip().lower() in QUIT_WORDS else Press.TOGGLE
        loop.call_soon_threadsafe(controls.press, press)
    loop.call_soon_threadsafe(controls.press, Press.QUIT)


async def conversation(ui: bool) -> int:
    settings = Settings.load()
    store = Store(settings.data_dir / "jarvis.db")

    stt = DeepgramSTT(settings.require("deepgram_api_key"))

    print("Starting the voice engine (first load takes ~30s)...")
    tts = ChatterboxTTS()
    print(BANNER.format(model=settings.model, **store.stats()))

    scratch = Path(tempfile.mkdtemp(prefix="dmills-voice-"))
    controls = Controls()

    try:
        async with AsyncExitStack() as stack:
            jarvis = await stack.enter_async_context(Jarvis(settings, store))
            if ui:
                from body.ui.server import running_ui

                url = await stack.enter_async_context(
                    running_ui(controls, settings.data_dir / "ui")
                )
                print(f"  UI: {url}\n")

            threading.Thread(
                target=read_terminal,
                args=(controls, asyncio.get_running_loop()),
                daemon=True,
            ).start()
            await converse(controls, MicRecorder(), VoicePipeline(stt, jarvis, tts, scratch))
            return 0
    finally:
        tts.close()


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m body.voice_loop")
    parser.add_argument("--no-ui", action="store_true", help="skip the web page; terminal only")
    args = parser.parse_args()
    try:
        return asyncio.run(conversation(ui=not args.no_ui))
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
