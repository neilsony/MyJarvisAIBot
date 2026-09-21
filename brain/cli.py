"""Talk to DmillsGPT from the terminal — no microphone, no hardware.

This is the dev loop for Phase 1. The persona, retrieval, tools and memory are
all live here; only the audio is missing. Tune `persona/register.md`, restart,
talk to it again.

    python -m brain.cli
"""

from __future__ import annotations

import asyncio
import sys

from brain.agent import Jarvis
from brain.config import Settings
from brain.store.db import Store

BANNER = """\
DmillsGPT — text mode
  model: {model}
  canon: {canon_chunks} chunks   memories: {memories}
  ctrl-c or 'exit' to quit
"""


async def conversation() -> int:
    settings = Settings.load()
    store = Store(settings.data_dir / "jarvis.db")
    stats = store.stats()

    print(BANNER.format(model=settings.model, **stats))
    if stats["canon_chunks"] == 0:
        print("No Canon ingested yet — it'll answer in character but has nothing")
        print("from the show to draw on. See SETUP.md.\n")

    async with Jarvis(settings, store) as jarvis:
        while True:
            try:
                message = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not message:
                continue
            if message.lower() in {"exit", "quit"}:
                return 0
            try:
                print(f"\nmills > {await jarvis.ask(message)}\n")
            except Exception as exc:
                print(f"\n[error] {exc}\n", file=sys.stderr)


def main() -> int:
    try:
        return asyncio.run(conversation())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
