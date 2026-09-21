"""Round-trip tests for cached diarization results.

Diarizing one episode takes minutes. The cache means a crash at episode 14 of
20 doesn't redo the first 13 — so its round-trip has to be exact.
"""

import json

import pytest

from pipeline.persist import load_turns, save_turns, turns_from_records, turns_to_records
from pipeline.segments import Turn


class TestRecordRoundTrip:
    def test_round_trips_exactly(self):
        turns = [Turn("SPEAKER_00", 0.0, 1.5), Turn("SPEAKER_01", 1.5, 4.25)]
        assert turns_from_records(turns_to_records(turns)) == turns

    def test_records_are_json_serialisable(self):
        records = turns_to_records([Turn("SPEAKER_00", 0.0, 1.5)])
        assert json.loads(json.dumps(records)) == records

    def test_empty_round_trips(self):
        assert turns_from_records(turns_to_records([])) == []

    def test_rejects_record_missing_a_field(self):
        with pytest.raises(ValueError, match="malformed turn record"):
            turns_from_records([{"speaker": "SPEAKER_00", "start": 0.0}])

    def test_rejects_non_numeric_bounds(self):
        with pytest.raises(ValueError, match="malformed turn record"):
            turns_from_records([{"speaker": "S", "start": "nope", "end": 1.0}])

    def test_propagates_invalid_spans(self):
        with pytest.raises(ValueError, match="end must be after start"):
            turns_from_records([{"speaker": "S", "start": 5.0, "end": 1.0}])


class TestFileRoundTrip:
    def test_save_then_load(self, tmp_path):
        turns = [Turn("SPEAKER_00", 0.0, 1.5), Turn("SPEAKER_01", 2.0, 3.0)]
        path = tmp_path / "ep.turns.json"
        save_turns(path, turns)
        assert load_turns(path) == turns

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "ep.turns.json"
        save_turns(path, [Turn("S", 0.0, 1.0)])
        assert path.is_file()

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_turns(tmp_path / "absent.json") is None
