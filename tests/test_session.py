"""Tests for the Body's conversation loop and its shared talk button.

No audio, no models: a fake recorder and pipeline stand in, and a script of
presses drives the loop. The pipeline fires scripted presses at chosen steps
to check what happens when someone taps while he's busy.
"""

import asyncio
from pathlib import Path

from body.session import BodyState, Controls, Press, Status, converse


class FakeRecorder:
    def __init__(self, audio: bytes = b"pcm"):
        self.audio = audio
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1
        return self.audio


class FakePipeline:
    def __init__(self, controls, said="what's up", reply="Not much.", during=None):
        self.controls = controls
        self.said = said
        self.reply_text = reply
        # step name -> presses fired while that step runs
        self.during = during or {}
        self.played: list[Path] = []

    def _fire(self, step):
        for press in self.during.get(step, []):
            self.controls.press(press)

    async def transcribe(self, audio):
        self._fire("transcribe")
        return self.said

    async def reply(self, said):
        self._fire("reply")
        return self.reply_text

    async def synthesize(self, reply):
        return Path("reply.wav")

    async def play(self, wav):
        self._fire("play")
        self.played.append(wav)


def run(presses, recorder=None, **pipeline_kwargs):
    """Feed presses, run the loop to completion, return (states, notes, pipeline)."""

    async def go():
        controls = Controls()
        seen: list[Status] = []
        controls.subscribe(seen.append)
        pipeline = FakePipeline(controls, **pipeline_kwargs)
        for press in presses:
            controls.press(press)
        await asyncio.wait_for(
            converse(controls, recorder or FakeRecorder(), pipeline, log=lambda _: None),
            timeout=2,
        )
        return seen, pipeline

    seen, pipeline = asyncio.run(go())
    return [s.state for s in seen], [s.note for s in seen], pipeline


class TestTurn:
    def test_full_turn_goes_through_every_state(self):
        states, _, pipeline = run([Press.TOGGLE, Press.TOGGLE, Press.QUIT])
        assert states == [
            BodyState.LISTENING,
            BodyState.THINKING,
            BodyState.SPEAKING,
            BodyState.IDLE,
        ]
        assert pipeline.played == [Path("reply.wav")]

    def test_quit_while_idle_ends_without_recording(self):
        recorder = FakeRecorder()
        states, _, _ = run([Press.QUIT], recorder=recorder)
        assert states == []
        assert recorder.started == 0

    def test_quit_while_listening_closes_the_mic(self):
        recorder = FakeRecorder()
        states, _, pipeline = run([Press.TOGGLE, Press.QUIT], recorder=recorder)
        assert recorder.stopped == 1
        assert states == [BodyState.LISTENING]
        assert pipeline.played == []


class TestEndsEarly:
    def test_nothing_recorded(self):
        states, notes, pipeline = run(
            [Press.TOGGLE, Press.TOGGLE, Press.QUIT], recorder=FakeRecorder(b"")
        )
        assert states[-1] is BodyState.IDLE
        assert notes[-1] == "nothing recorded"
        assert BodyState.SPEAKING not in states

    def test_didnt_catch_that(self):
        states, notes, _ = run([Press.TOGGLE, Press.TOGGLE, Press.QUIT], said="")
        assert notes[-1] == "didn't catch that"
        assert BodyState.SPEAKING not in states

    def test_no_reply(self):
        _, notes, pipeline = run([Press.TOGGLE, Press.TOGGLE, Press.QUIT], reply="")
        assert notes[-1] == "no reply"
        assert pipeline.played == []


class TestBusy:
    def test_tap_while_thinking_is_dropped(self):
        # Without the drop, this tap would reopen the mic right after he spoke.
        recorder = FakeRecorder()
        run(
            [Press.TOGGLE, Press.TOGGLE, Press.QUIT],
            recorder=recorder,
            during={"reply": [Press.TOGGLE]},
        )
        assert recorder.started == 1

    def test_tap_while_speaking_is_dropped(self):
        recorder = FakeRecorder()
        run(
            [Press.TOGGLE, Press.TOGGLE, Press.QUIT],
            recorder=recorder,
            during={"play": [Press.TOGGLE]},
        )
        assert recorder.started == 1

    def test_quit_while_busy_still_quits(self):
        recorder = FakeRecorder()
        run([Press.TOGGLE, Press.TOGGLE], recorder=recorder, during={"play": [Press.QUIT]})
        assert recorder.started == 1

    def test_double_tap_to_send_does_not_reopen_the_mic(self):
        # Both taps land while listening; the second must not start a turn.
        recorder = FakeRecorder()

        async def go():
            controls = Controls()
            pipeline = FakePipeline(controls)
            controls.press(Press.TOGGLE)
            task = asyncio.create_task(
                converse(controls, recorder, pipeline, log=lambda _: None)
            )
            await asyncio.sleep(0)
            assert controls.status.state is BodyState.LISTENING
            controls.press(Press.TOGGLE)
            controls.press(Press.TOGGLE)
            while controls.status.state is not BodyState.IDLE:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            controls.press(Press.QUIT)
            await asyncio.wait_for(task, timeout=2)

        asyncio.run(go())
        assert recorder.started == 1


class TestControls:
    def test_press_accepted_only_when_idle_or_listening(self):
        async def go():
            controls = Controls()
            results = {}
            for state in BodyState:
                controls.set(state)
                results[state] = controls.press(Press.TOGGLE)
            return results

        assert asyncio.run(go()) == {
            BodyState.IDLE: True,
            BodyState.LISTENING: True,
            BodyState.THINKING: False,
            BodyState.SPEAKING: False,
        }

    def test_unsubscribe_stops_updates(self):
        controls = Controls()
        seen: list[Status] = []
        unsubscribe = controls.subscribe(seen.append)
        controls.set(BodyState.LISTENING)
        unsubscribe()
        controls.set(BodyState.THINKING)
        assert [s.state for s in seen] == [BodyState.LISTENING]

    def test_status_wire_format(self):
        assert Status(BodyState.IDLE, "didn't catch that").to_dict() == {
            "state": "idle",
            "note": "didn't catch that",
        }
