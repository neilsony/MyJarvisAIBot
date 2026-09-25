"""Tests for the TTS backend seam.

The engines themselves can't be tested here — they need a model, a second
venv, and 30 seconds of load. What *is* testable is the contract every engine
has to satisfy, and that the fake genuinely satisfies it, since the fake is
what everything above the seam gets tested against.
"""

import wave

from brain.tts.base import FakeTTS, TTSBackend


class TestFakeTTS:
    def test_satisfies_the_protocol(self):
        # Not decoration: if the fake drifts from the protocol, every test
        # written against it is testing something the real engine won't do.
        backend: TTSBackend = FakeTTS()
        assert backend.sample_rate == 24_000

    def test_writes_a_real_playable_wav(self, tmp_path):
        out = FakeTTS().synthesize("hello", tmp_path / "out.wav")
        with wave.open(str(out), "rb") as f:
            assert f.getnchannels() == 1
            assert f.getframerate() == 24_000
            assert f.getnframes() > 0

    def test_returns_the_path_it_wrote(self, tmp_path):
        target = tmp_path / "reply.wav"
        assert FakeTTS().synthesize("hi", target) == target

    def test_creates_missing_parent_directories(self, tmp_path):
        out = FakeTTS().synthesize("hi", tmp_path / "nested" / "deep" / "out.wav")
        assert out.is_file()

    def test_records_what_it_was_asked_to_say(self, tmp_path):
        tts = FakeTTS()
        tts.synthesize("first", tmp_path / "a.wav")
        tts.synthesize("second", tmp_path / "b.wav")
        assert tts.calls == ["first", "second"]

    def test_honours_a_custom_sample_rate(self, tmp_path):
        out = FakeTTS(sample_rate=16_000).synthesize("hi", tmp_path / "out.wav")
        with wave.open(str(out), "rb") as f:
            assert f.getframerate() == 16_000

    def test_close_is_safe_to_call(self):
        FakeTTS().close()

