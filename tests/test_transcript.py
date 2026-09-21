"""Tests for turning raw ASR output into speaker-attributed, retrievable chunks.

Whisper hears words but not who said them; pyannote knows who spoke but not what
they said. This is the join, and a wrong join attributes Kenny's takes to Darrick
in the Canon — which is worse than missing them, because it's invisible.
"""

import pytest

from pipeline.segments import Turn
from pipeline.transcript import (
    AttributedSegment,
    TranscriptSegment,
    attribute,
    chunk_transcript,
)


def seg(text: str, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, end=end)


class TestAttribute:
    def test_segment_inside_a_turn_gets_that_speaker(self):
        got = attribute([seg("hey", 1.0, 2.0)], [Turn("D", 0.0, 5.0)])
        assert got == [AttributedSegment(text="hey", speaker="D", start=1.0, end=2.0)]

    def test_segment_spanning_two_turns_goes_to_the_bigger_overlap(self):
        # 0-3 overlaps K by 1s and D by 2s -> D
        got = attribute([seg("mine", 0.0, 3.0)], [Turn("K", 0.0, 1.0), Turn("D", 1.0, 3.0)])
        assert got[0].speaker == "D"

    def test_segment_with_no_overlapping_turn_is_unknown(self):
        got = attribute([seg("?", 50.0, 51.0)], [Turn("D", 0.0, 5.0)])
        assert got[0].speaker == "UNKNOWN"

    def test_no_turns_at_all_is_unknown(self):
        assert attribute([seg("x", 0.0, 1.0)], [])[0].speaker == "UNKNOWN"

    def test_empty_segments(self):
        assert attribute([], [Turn("D", 0.0, 5.0)]) == []

    def test_preserves_input_order(self):
        got = attribute(
            [seg("a", 0.0, 1.0), seg("b", 2.0, 3.0)],
            [Turn("D", 0.0, 5.0)],
        )
        assert [x.text for x in got] == ["a", "b"]

    def test_ties_break_deterministically(self):
        # Exactly 1s of overlap each; must not vary run to run.
        turns = [Turn("B", 2.0, 3.0), Turn("A", 1.0, 2.0)]
        first = attribute([seg("x", 1.0, 3.0)], turns)[0].speaker
        assert first == attribute([seg("x", 1.0, 3.0)], list(reversed(turns)))[0].speaker


class TestChunkTranscript:
    def _segs(self, n: int, words: str = "word") -> list[AttributedSegment]:
        return [
            AttributedSegment(text=words, speaker="D", start=float(i), end=float(i + 1))
            for i in range(n)
        ]

    def test_short_transcript_is_one_chunk(self):
        chunks = chunk_transcript(self._segs(3), episode="ep1", max_chars=500, overlap_chars=0)
        assert len(chunks) == 1
        assert chunks[0].episode == "ep1"

    def test_splits_when_over_max_chars(self):
        chunks = chunk_transcript(self._segs(50), episode="ep1", max_chars=100, overlap_chars=0)
        assert len(chunks) > 1
        assert all(len(c.text) <= 150 for c in chunks)

    def test_chunk_carries_its_speakers(self):
        segs = [
            AttributedSegment(text="a", speaker="D", start=0.0, end=1.0),
            AttributedSegment(text="b", speaker="K", start=1.0, end=2.0),
        ]
        chunks = chunk_transcript(segs, episode="ep1", max_chars=500, overlap_chars=0)
        assert set(chunks[0].speakers) == {"D", "K"}

    def test_chunk_spans_its_segments_in_time(self):
        chunks = chunk_transcript(self._segs(4), episode="ep1", max_chars=500, overlap_chars=0)
        assert chunks[0].start == 0.0
        assert chunks[0].end == 4.0

    def test_overlap_repeats_tail_into_next_chunk(self):
        chunks = chunk_transcript(self._segs(40), episode="ep1", max_chars=100, overlap_chars=40)
        assert len(chunks) > 1
        # Consecutive chunks should share time, not butt up exactly.
        assert chunks[1].start < chunks[0].end

    def test_speaker_labels_appear_in_the_text(self):
        segs = [AttributedSegment(text="hoops", speaker="DARRICK", start=0.0, end=1.0)]
        chunks = chunk_transcript(segs, episode="ep1", max_chars=500, overlap_chars=0)
        assert "DARRICK" in chunks[0].text

    def test_empty_input(self):
        assert chunk_transcript([], episode="ep1", max_chars=500, overlap_chars=0) == []

    def test_overlap_must_be_smaller_than_max(self):
        with pytest.raises(ValueError, match="overlap_chars must be less than max_chars"):
            chunk_transcript(self._segs(5), episode="ep1", max_chars=50, overlap_chars=50)
