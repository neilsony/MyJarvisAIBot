"""Tests for mapping diarization labels to actual humans.

The stakes: pick the wrong SPEAKER_0N and you train a voice clone on Kenny while
believing it's Darrick, and you don't find out until you hear it. So ambiguity
must be reported, never silently resolved.
"""

import numpy as np
import pytest

from pipeline.speakers import (
    SpeakerMatch,
    centroid,
    cosine_similarity,
    identify_speaker,
    rank_speakers,
)


def vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        assert cosine_similarity(vec(1, 2, 3), vec(1, 2, 3)) == pytest.approx(1.0)

    def test_scale_does_not_matter(self):
        assert cosine_similarity(vec(1, 0), vec(7, 0)) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        assert cosine_similarity(vec(1, 0), vec(0, 1)) == pytest.approx(0.0)

    def test_opposite_is_negative_one(self):
        assert cosine_similarity(vec(1, 0), vec(-1, 0)) == pytest.approx(-1.0)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero-magnitude"):
            cosine_similarity(vec(0, 0), vec(1, 0))


class TestCentroid:
    def test_single_embedding_is_itself_normalised(self):
        got = centroid([vec(3, 4)])
        assert np.allclose(got, vec(0.6, 0.8))

    def test_averages_then_normalises(self):
        got = centroid([vec(1, 0), vec(0, 1)])
        assert np.allclose(got, vec(2**-0.5, 2**-0.5))

    def test_is_unit_length(self):
        got = centroid([vec(1, 2, 3), vec(4, 5, 6)])
        assert np.linalg.norm(got) == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one embedding"):
            centroid([])

    def test_cancelling_embeddings_raise(self):
        # Opposing vectors average to zero — there is no meaningful centroid,
        # and silently returning garbage would mislabel a speaker.
        with pytest.raises(ValueError, match="zero-magnitude"):
            centroid([vec(1, 0), vec(-1, 0)])

    def test_mismatched_dimensions_raise(self):
        with pytest.raises(ValueError, match="same dimension"):
            centroid([vec(1, 0), vec(1, 0, 0)])


class TestRankSpeakers:
    def test_orders_by_similarity_descending(self):
        candidates = {"S0": vec(0, 1), "S1": vec(1, 0), "S2": vec(1, 1)}
        got = rank_speakers(candidates, reference=vec(1, 0))
        assert [label for label, _ in got] == ["S1", "S2", "S0"]

    def test_empty_candidates(self):
        assert rank_speakers({}, reference=vec(1, 0)) == []


class TestIdentifySpeaker:
    def test_picks_the_clear_winner(self):
        candidates = {"S0": vec(1, 0), "S1": vec(0, 1)}
        got = identify_speaker(candidates, reference=vec(1, 0), min_score=0.5, min_margin=0.1)
        assert got is not None
        assert got.label == "S0"
        assert got.score == pytest.approx(1.0)

    def test_returns_none_when_best_is_below_min_score(self):
        candidates = {"S0": vec(0, 1)}  # orthogonal -> 0.0
        got = identify_speaker(candidates, reference=vec(1, 0), min_score=0.5, min_margin=0.0)
        assert got is None

    def test_returns_none_when_top_two_are_too_close(self):
        # Both near-identical to the reference: which one is Darrick is a coin flip.
        candidates = {"S0": vec(1.0, 0.01), "S1": vec(1.0, 0.02)}
        got = identify_speaker(candidates, reference=vec(1, 0), min_score=0.5, min_margin=0.1)
        assert got is None

    def test_single_candidate_needs_no_margin(self):
        got = identify_speaker(
            {"S0": vec(1, 0)}, reference=vec(1, 0), min_score=0.5, min_margin=0.9
        )
        assert got is not None
        assert got.label == "S0"

    def test_reports_runner_up_for_diagnostics(self):
        candidates = {"S0": vec(1, 0), "S1": vec(1, 1)}
        got = identify_speaker(candidates, reference=vec(1, 0), min_score=0.5, min_margin=0.1)
        assert got is not None
        assert got.runner_up == "S1"
        assert got.margin == pytest.approx(1.0 - 2**-0.5)

    def test_empty_candidates_returns_none(self):
        assert identify_speaker({}, reference=vec(1, 0), min_score=0.5, min_margin=0.1) is None


class TestSpeakerMatch:
    def test_margin_with_no_runner_up_is_the_score_itself(self):
        m = SpeakerMatch(label="S0", score=0.9, runner_up=None, runner_up_score=None)
        assert m.margin == pytest.approx(0.9)
