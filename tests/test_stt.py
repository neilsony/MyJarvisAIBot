"""Tests for Deepgram response parsing.

The network call isn't tested — the parsing is, because that's where a
malformed or empty response turns into either a sensible "" or a crashed
voice loop mid-conversation.
"""

import io
import wave
from types import SimpleNamespace

from brain.stt.deepgram import _first_transcript, pcm16_to_wav


def response(transcript: str):
    alt = SimpleNamespace(transcript=transcript)
    channel = SimpleNamespace(alternatives=[alt])
    return SimpleNamespace(results=SimpleNamespace(channels=[channel]))


class TestFirstTranscript:
    def test_extracts_the_transcript(self):
        assert _first_transcript(response("what did the guys say")) == "what did the guys say"

    def test_strips_surrounding_whitespace(self):
        assert _first_transcript(response("  hello  ")) == "hello"

    def test_empty_transcript_is_empty_string(self):
        assert _first_transcript(response("")) == ""

    def test_no_channels_is_empty_string(self):
        empty = SimpleNamespace(results=SimpleNamespace(channels=[]))
        assert _first_transcript(empty) == ""

    def test_no_alternatives_is_empty_string(self):
        channel = SimpleNamespace(alternatives=[])
        resp = SimpleNamespace(results=SimpleNamespace(channels=[channel]))
        assert _first_transcript(resp) == ""

    def test_null_transcript_is_empty_string(self):
        assert _first_transcript(response(None)) == ""

    def test_unexpected_shape_does_not_raise(self):
        # Silence, a schema change, an error payload — all should read as
        # "nothing was said" rather than taking down the loop.
        assert _first_transcript(SimpleNamespace()) == ""
        assert _first_transcript(SimpleNamespace(results=SimpleNamespace())) == ""


class TestPcm16ToWav:
    """The SDK's file endpoint has no sample-rate parameter, so the WAV header
    is the only thing telling Deepgram how to interpret the bytes. If this is
    wrong, transcripts come back garbled rather than failing loudly."""

    def test_produces_a_readable_wav(self):
        wav_bytes = pcm16_to_wav(b"\x00\x00" * 1000, 16_000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as f:
            assert f.getframerate() == 16_000
            assert f.getnchannels() == 1
            assert f.getsampwidth() == 2

    def test_preserves_the_samples(self):
        pcm = b"\x01\x02" * 500
        wav_bytes = pcm16_to_wav(pcm, 16_000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as f:
            assert f.readframes(f.getnframes()) == pcm

    def test_frame_count_matches_the_input(self):
        wav_bytes = pcm16_to_wav(b"\x00\x00" * 800, 16_000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as f:
            assert f.getnframes() == 800

    def test_honours_a_different_sample_rate(self):
        wav_bytes = pcm16_to_wav(b"\x00\x00" * 10, 24_000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as f:
            assert f.getframerate() == 24_000

    def test_empty_pcm_still_produces_a_valid_header(self):
        with wave.open(io.BytesIO(pcm16_to_wav(b"", 16_000)), "rb") as f:
            assert f.getnframes() == 0
