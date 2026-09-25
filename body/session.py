"""The Body's conversation loop, as a small state machine.

Two things can press the talk button: Enter in the terminal and the button on
the web page (body/ui). Both feed one queue, so whichever comes first wins and
either can stop what the other started. The loop owns the Body state (see
CONTEXT.md) and announces every change to whoever is listening — the page, and
later the Arduino.

Presses are only accepted while idle or listening. One that lands while he is
thinking or speaking is dropped rather than queued: otherwise a stray tap would
silently open the mic the moment he finished. Quit always gets through.

Kept free of audio, network and model code so it can be tested with fakes:
the real work arrives through `Recorder` and `Pipeline`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class BodyState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Press(StrEnum):
    TOGGLE = "toggle"
    QUIT = "quit"


@dataclass(frozen=True)
class Status:
    state: BodyState
    # Why the last turn ended early ("didn't catch that"), shown briefly.
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"state": self.state.value, "note": self.note}


Listener = Callable[[Status], None]


class Controls:
    """The shared talk button and the current Body state."""

    def __init__(self) -> None:
        self._presses: asyncio.Queue[Press] = asyncio.Queue()
        self._listeners: list[Listener] = []
        self.status = Status(BodyState.IDLE)

    def press(self, press: Press) -> bool:
        """Queue a press. Returns False if it was dropped (he's busy)."""
        if press is Press.TOGGLE and self.status.state not in (
            BodyState.IDLE,
            BodyState.LISTENING,
        ):
            return False
        self._presses.put_nowait(press)
        return True

    async def next_press(self) -> Press:
        return await self._presses.get()

    def discard_pending(self) -> bool:
        """Drop toggles left over from a double tap. True if a quit was among them."""
        quit_pending = False
        while not self._presses.empty():
            quit_pending |= self._presses.get_nowait() is Press.QUIT
        return quit_pending

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Call `listener` on every state change. Returns an unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def set(self, state: BodyState, note: str = "") -> None:
        self.status = Status(state, note)
        for listener in list(self._listeners):
            listener(self.status)


class Recorder(Protocol):
    def start(self) -> None: ...
    def stop(self) -> bytes: ...


class Pipeline(Protocol):
    """One turn's work, step by step, so the state can change between steps."""

    async def transcribe(self, audio: bytes) -> str: ...
    async def reply(self, said: str) -> str: ...
    async def synthesize(self, reply: str) -> Path: ...
    async def play(self, wav: Path) -> None: ...


async def converse(
    controls: Controls,
    recorder: Recorder,
    pipeline: Pipeline,
    log: Callable[[str], None] = print,
) -> None:
    """Run turns until a quit press."""
    while True:
        log("[Enter or the button to talk, 'q' to quit]")
        if await controls.next_press() is Press.QUIT:
            return

        recorder.start()
        controls.set(BodyState.LISTENING)
        log("  listening... (Enter or the button to stop)")
        stop = await controls.next_press()
        audio = recorder.stop()
        if stop is Press.QUIT:
            return

        controls.set(BodyState.THINKING)
        note = await _turn(audio, pipeline, controls, log)
        controls.set(BodyState.IDLE, note)
        if controls.discard_pending():
            return
        log("")


async def _turn(
    audio: bytes, pipeline: Pipeline, controls: Controls, log: Callable[[str], None]
) -> str:
    """Run one turn. Returns a note if it ended early, else ''."""
    if not audio:
        log("  nothing recorded")
        return "nothing recorded"

    log("  transcribing...")
    said = await pipeline.transcribe(audio)
    if not said:
        log("  didn't catch that")
        return "didn't catch that"
    log(f"  you > {said}")

    reply = await pipeline.reply(said)
    if not reply:
        log("  (no reply)")
        return "no reply"
    log(f"  mills > {reply}")

    wav = await pipeline.synthesize(reply)
    controls.set(BodyState.SPEAKING)
    log("  speaking...")
    await pipeline.play(wav)
    return ""
