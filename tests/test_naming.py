"""Tests for label->name resolution and Darrick window overrides.

The override layer is what separates Darrick from a co-host the diarizer
merged him with, so its edge cases — partial coverage, window edges,
crosstalk tails — are tested hard.
"""

import json

import pytest

from pipeline.naming import (
    DARRICK_NAME,
    apply_darrick_windows,
    apply_names,
    load_names,
    load_windows,
)
from pipeline.transcript import AttributedSegment


def seg(speaker: str, start: float, end: float) -> AttributedSegment:
    return AttributedSegment(text="hello", speaker=speaker, start=start, end=end)


class TestLoadNames:
    def test_missing_file_is_empty(self, tmp_path):
        assert load_names(tmp_path / "nope.json") == {}

    def test_reads_episode_maps(self, tmp_path):
        path = tmp_path / "names.json"
        path.write_text(json.dumps({"ep1": {"SPEAKER_00": "KENNY"}}))
        assert load_names(path) == {"ep1": {"SPEAKER_00": "KENNY"}}

    def test_rejects_non_dict(self, tmp_path):
        path = tmp_path / "names.json"
        path.write_text(json.dumps(["not", "a", "map"]))
        with pytest.raises(ValueError, match="must map episode"):
            load_names(path)


class TestLoadWindows:
    def test_missing_file_is_empty(self, tmp_path):
        assert load_windows(tmp_path / "nope.json") == []

    def test_reads_sorted_pairs(self, tmp_path):
        path = tmp_path / "ep.json"
        path.write_text(json.dumps([[100.0, 110.0], [50.0, 60.0]]))
        assert load_windows(path) == [(50.0, 60.0), (100.0, 110.0)]

    def test_reads_turn_records(self, tmp_path):
        # export-darrick-turns writes the shared turn-cache format.
        path = tmp_path / "ep.json"
        path.write_text(json.dumps([
            {"speaker": "DARRICK", "start": 100.0, "end": 110.0},
            {"speaker": "DARRICK", "start": 50.0, "end": 60.0},
        ]))
        assert load_windows(path) == [(50.0, 60.0), (100.0, 110.0)]

    def test_rejects_bad_shape(self, tmp_path):
        path = tmp_path / "ep.json"
        path.write_text(json.dumps([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError, match="malformed window"):
            load_windows(path)

    def test_rejects_inverted_window(self, tmp_path):
        path = tmp_path / "ep.json"
        path.write_text(json.dumps([[10.0, 5.0]]))
        with pytest.raises(ValueError, match="end must be after start"):
            load_windows(path)


class TestApplyNames:
    def test_replaces_mapped_labels(self):
        out = apply_names([seg("SPEAKER_00", 0, 5)], {"SPEAKER_00": "KENNY"})
        assert out[0].speaker == "KENNY"

    def test_keeps_unmapped_labels(self):
        out = apply_names([seg("SPEAKER_02", 0, 5)], {"SPEAKER_00": "KENNY"})
        assert out[0].speaker == "SPEAKER_02"

    def test_empty_map_is_identity(self):
        segments = [seg("SPEAKER_00", 0, 5)]
        assert apply_names(segments, {}) == segments


class TestApplyDarrickWindows:
    def test_majority_inside_window_is_darrick(self):
        # 4 of 5 seconds covered: clear majority.
        out = apply_darrick_windows([seg("PIERRE", 100.0, 105.0)], [(99.0, 104.0)])
        assert out[0].speaker == DARRICK_NAME

    def test_minority_touch_keeps_label(self):
        # 0.5 of 5 seconds covered — the crosstalk-tail case.
        out = apply_darrick_windows([seg("PIERRE", 100.0, 105.0)], [(104.5, 107.0)])
        assert out[0].speaker == "PIERRE"

    def test_exactly_half_stays_with_label(self):
        # Majority means strictly more than half.
        out = apply_darrick_windows([seg("PIERRE", 100.0, 104.0)], [(102.0, 106.0)])
        assert out[0].speaker == "PIERRE"

    def test_disjoint_window_does_nothing(self):
        segments = [seg("PIERRE", 100.0, 105.0)]
        assert apply_darrick_windows(segments, [(200.0, 210.0)]) == segments

    def test_no_windows_is_identity(self):
        segments = [seg("PIERRE", 100.0, 105.0)]
        assert apply_darrick_windows(segments, []) == segments

    def test_already_darrick_stays_darrick(self):
        out = apply_darrick_windows([seg(DARRICK_NAME, 100.0, 105.0)], [(99.0, 104.0)])
        assert out[0].speaker == DARRICK_NAME

    def test_multiple_windows_sum_coverage(self):
        # Two windows each covering 3s of a 5s segment: 6/5 capped at 100%.
        out = apply_darrick_windows(
            [seg("PIERRE", 100.0, 105.0)], [(98.0, 103.0), (103.0, 106.0)]
        )
        assert out[0].speaker == DARRICK_NAME

    def test_mixed_segments_resolved_independently(self):
        segments = [
            seg("PIERRE", 100.0, 105.0),   # mostly inside -> DARRICK
            seg("PIERRE", 106.0, 110.0),   # outside -> stays
            seg("MIKE", 109.0, 114.0),     # tail touches -> stays
        ]
        out = apply_darrick_windows(segments, [(99.0, 104.0)])
        assert [s.speaker for s in out] == [DARRICK_NAME, "PIERRE", "MIKE"]
