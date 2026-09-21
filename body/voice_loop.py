"""Talk to DmillsGPT out loud — the Phase 2 milestone, push-to-talk edition.

    python -m body.voice_loop

Press Enter to start recording, Enter again to stop. The transcript goes to
the same agent `brain.cli` uses, and the reply comes back in his voice.

Deliberately simple for a first version: no barge-in, no sentence-level
streaming, no wake word. Those are additive once the basic loop is proven —
and this is the loop that proves the three pieces (ears, brain, voice) can
actually hold a conversation together.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from body.audio import RECORD_SAMPLE_RATE, play_wav, record_until_enter
from brain.agent import Jarvis
from brain.config import Settings
from brain.store.db import Store
from brain.stt.deepgram import DeepgramSTT
from brain.tts.chatterbox_client import ChatterboxTTS

BANNER = """\
DmillsGPT — voice mode
  model: {model}
  canon: {canon_chunks} chunks   memories: {memories}

  Enter to start talking, Enter again to send.
  Type 'q' then Enter to quit.
"""


async def conversation() -> int:
    settings = Settings.load()
    store = Store(settings.data_dir / "jarvis.db")

    stt = DeepgramSTT(settings.require("deepgram_api_key"))

    print("Starting the voice engine (first load takes ~30s)...")
    tts = ChatterboxTTS()
    print(BANNER.format(model=settings.model, **store.stats()))

    scratch = Path(tempfile.mkdtemp(prefix="dmills-voice-"))
    turn = 0

    try:
        async with Jarvis(settings, store) as jarvis:
            while True:
                try:
                    command = input("[Enter to talk] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if command in {"q", "quit", "exit"}:
                    return 0

                print("  listening... (Enter to stop)")
                audio = record_until_enter()
                if not audio:
                    print("  nothing recorded\n")
                    continue

                print("  transcribing...")
                said = stt.transcribe(audio, RECORD_SAMPLE_RATE)
                if not said:
                    print("  didn't catch that\n")
                    continue
                print(f"  you > {said}")

                reply = await jarvis.ask(said)
                if not reply:
                    print("  (no reply)\n")
                    continue
                print(f"  mills > {reply}")

                print("  speaking...")
                turn += 1
                wav = tts.synthesize(reply, scratch / f"turn_{turn:03d}.wav")
                play_wav(wav)
                print()
    finally:
        tts.close()


def main() -> int:
    try:
        return asyncio.run(conversation())
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
