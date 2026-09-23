"""Tests for diarize()'s caching and inline verification wiring.

pyannote itself is never exercised here — these tests only need to prove
that (a) a turns cache hit skips the pipeline entirely, and (b) the optional
verification pass runs against whatever turns come back (cached or fresh)
and writes/skips its own cache correctly.
"""

import numpy as np

from pipeline.diarize import bimodal_labels, diarize
from pipeline.persist import save_turns
from pipeline.segments import Turn
from pipeline.verify_speakers import LabelReport, load_label_reports, save_label_reports


def unit(dim: int, value: float) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[int(value) % dim] = 1.0
    return vec


class TestCaching:
    def test_cache_hit_never_touches_pyannote(self, tmp_path, monkeypatch):
        cache = tmp_path / "ep.json"
        cached_turns = [Turn("SPEAKER_00", 0.0, 5.0)]
        save_turns(cache, cached_turns)

        def boom(*_a, **_k):
            raise AssertionError("pyannote should not be touched on a cache hit")

        monkeypatch.setattr("pyannote.audio.Pipeline.from_pretrained", boom, raising=False)

        turns = diarize(tmp_path / "ep.wav", "fake-token", cache_path=cache)
        assert turns == cached_turns


class TestInlineVerification:
    def test_writes_confidence_cache_for_cached_turns(self, tmp_path):
        turns = [Turn("SPEAKER_00", 0.0, 5.0), Turn("SPEAKER_01", 10.0, 15.0)]
        cache = tmp_path / "ep.json"
        save_turns(cache, turns)
        conf_cache = tmp_path / "ep.conf.json"

        def embed(turn):
            return unit(4, 0) if turn.speaker == "SPEAKER_00" else unit(4, 1)

        diarize(
            tmp_path / "ep.wav",
            "fake-token",
            cache_path=cache,
            verify_embed=embed,
            verify_reference=unit(4, 0),
            conf_cache_path=conf_cache,
        )

        reports = load_label_reports(conf_cache)
        assert reports is not None
        assert {r.label for r in reports} == {"SPEAKER_00", "SPEAKER_01"}

    def test_skips_verification_when_confidence_cache_already_exists(self, tmp_path):
        turns = [Turn("SPEAKER_00", 0.0, 5.0)]
        cache = tmp_path / "ep.json"
        save_turns(cache, turns)
        conf_cache = tmp_path / "ep.conf.json"
        sentinel = [LabelReport("SPEAKER_00", 99, 99, 0.99, 0.99, None)]
        save_label_reports(conf_cache, sentinel)

        def boom(_turn):
            raise AssertionError("verification should not re-run once cached")

        diarize(
            tmp_path / "ep.wav",
            "fake-token",
            cache_path=cache,
            verify_embed=boom,
            verify_reference=unit(4, 0),
            conf_cache_path=conf_cache,
        )

        assert load_label_reports(conf_cache) == sentinel

    def test_no_confidence_cache_without_reference(self, tmp_path):
        turns = [Turn("SPEAKER_00", 0.0, 5.0)]
        cache = tmp_path / "ep.json"
        save_turns(cache, turns)
        conf_cache = tmp_path / "ep.conf.json"

        diarize(tmp_path / "ep.wav", "fake-token", cache_path=cache)

        assert not conf_cache.is_file()


class TestBimodalLabels:
    def test_filters_to_only_bimodal(self):
        pure = LabelReport("A", 5, 5, 0.9, 0.9, None)
        merged = LabelReport("B", 6, 3, 0.5, 0.8, 0.2)
        assert bimodal_labels([pure, merged]) == [merged]

    def test_empty_when_none_bimodal(self):
        pure = LabelReport("A", 5, 5, 0.9, 0.9, None)
        assert bimodal_labels([pure]) == []
