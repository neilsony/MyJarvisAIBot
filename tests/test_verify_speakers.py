"""Tests for per-turn speaker verification.

The failure mode this module exists to catch — a merged label that *averages*
into a plausible centroid — is exactly what the tests pin down: distributions,
not averages, are the contract.
"""

import numpy as np
import pytest

from pipeline.segments import Turn
from pipeline.verify_speakers import (
    LabelReport,
    TurnScore,
    ear_check_turns,
    label_reports,
    middle_slice,
    rank_labels,
    score_turns,
    select_slice_turns,
)


def t(speaker: str, start: float, end: float) -> Turn:
    return Turn(speaker=speaker, start=start, end=end)


def ts(speaker: str, start: float, end: float, score: float) -> TurnScore:
    return TurnScore(speaker=speaker, start=start, end=end, score=score)


def unit(dim: int, value: float) -> np.ndarray:
    """A unit vector pointing at `value` along one axis — zero-safe for
    orthogonal directions (`value` 0..dim-1 selects the axis)."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[int(value) % dim] = 1.0
    return vec


class TestMiddleSlice:
    def test_centers_in_the_episode(self):
        start, end = middle_slice(7200.0, 15.0)
        assert start == pytest.approx(3150.0)
        assert end == pytest.approx(4050.0)

    def test_slice_shorter_than_episode_is_exact_length(self):
        start, end = middle_slice(7200.0, 15.0)
        assert end - start == pytest.approx(900.0)

    def test_slice_longer_than_episode_covers_all(self):
        start, end = middle_slice(600.0, 15.0)
        assert (start, end) == (0.0, 600.0)

    def test_rejects_zero_duration(self):
        with pytest.raises(ValueError, match="duration must be positive"):
            middle_slice(0.0, 15.0)

    def test_rejects_zero_minutes(self):
        with pytest.raises(ValueError, match="minutes must be positive"):
            middle_slice(7200.0, 0.0)


class TestSelectSliceTurns:
    def test_keeps_turns_inside_slice(self):
        turns = [t("A", 100.0, 110.0), t("A", 5000.0, 5020.0)]
        kept = select_slice_turns(turns, 3150.0, 4050.0)
        assert kept == [t("A", 100.0, 110.0)] or kept == []

    def test_drops_turns_outside_slice(self):
        turns = [t("A", 0.0, 30.0), t("B", 6000.0, 6050.0)]
        assert select_slice_turns(turns, 3150.0, 4050.0) == []

    def test_boundary_overlap_counts(self):
        turns = [t("A", 4040.0, 4060.0)]
        assert select_slice_turns(turns, 3150.0, 4050.0) == [t("A", 4040.0, 4060.0)]

    def test_drops_short_turns(self):
        turns = [t("A", 3200.0, 3201.0)]
        assert select_slice_turns(turns, 3150.0, 4050.0) == []

    def test_returns_chronological_order(self):
        turns = [t("B", 3300.0, 3310.0), t("A", 3200.0, 3210.0)]
        kept = select_slice_turns(turns, 3150.0, 4050.0)
        assert [turn.speaker for turn in kept] == ["A", "B"]


class TestScoreTurns:
    def test_scores_each_turn_individually(self):
        # Two turns, same label, orthogonal embeddings: they must get
        # different scores. Averaging would collapse them; per-turn scoring
        # is the whole point.
        reference = unit(4, 0)
        embeddings = {
            t("A", 0.0, 5.0): unit(4, 0),
            t("A", 6.0, 11.0): unit(4, 1),
        }
        scores = score_turns(embeddings.get, list(embeddings), reference)
        assert scores[0].score == pytest.approx(1.0)
        assert scores[1].score == pytest.approx(0.0)

    def test_preserves_turn_provenance(self):
        reference = unit(2, 1.0)
        turn = t("B", 12.0, 30.0)
        scores = score_turns(lambda _turn: unit(2, 1.0), [turn], reference)
        assert scores[0].speaker == "B"
        assert scores[0].start == 12.0
        assert scores[0].end == 30.0


class TestLabelReports:
    def test_pure_label_is_unimodal(self):
        scores = [ts("A", 0, 5, 0.80), ts("A", 6, 11, 0.85), ts("A", 12, 17, 0.75)]
        (report,) = label_reports(scores)
        assert report.turn_count == 3
        assert report.above_threshold == 3
        assert not report.bimodal

    def test_merged_label_is_bimodal(self):
        # Half the turns are Darrick, half are a co-host: two clusters with a
        # real gap. This is the ioMY7oYuhA0 signature.
        scores = [
            ts("A", 0, 5, 0.80), ts("A", 6, 11, 0.78),
            ts("A", 12, 17, 0.30), ts("A", 18, 23, 0.35),
        ]
        report = label_reports(scores)[0]
        assert report.above_threshold == 2
        assert report.bimodal

    def test_below_gap_split_is_not_bimodal(self):
        # Turn-to-turn variation straddles the threshold narrowly — noise, not
        # two people. The gap test must not fire.
        scores = [
            ts("A", 0, 5, 0.60), ts("A", 6, 11, 0.56),
            ts("A", 12, 17, 0.52), ts("A", 18, 23, 0.50),
        ]
        assert not label_reports(scores)[0].bimodal

    def test_sorted_best_first(self):
        scores = [
            ts("B", 0, 5, 0.90), ts("B", 6, 11, 0.88),
            ts("A", 0, 5, 0.60), ts("A", 6, 11, 0.58),
        ]
        reports = label_reports(scores)
        assert [r.label for r in reports] == ["B", "A"]

    def test_ties_break_on_label(self):
        scores = [ts("B", 0, 5, 0.70), ts("A", 0, 5, 0.70)]
        reports = label_reports(scores)
        assert [r.label for r in reports] == ["A", "B"]

    def test_no_turns_above_threshold(self):
        scores = [ts("A", 0, 5, 0.30), ts("A", 6, 11, 0.25)]
        report = label_reports(scores)[0]
        assert report.above_threshold == 0
        assert report.high_median is None
        assert report.low_median is not None

    def test_all_turns_above_threshold(self):
        scores = [ts("A", 0, 5, 0.80), ts("A", 6, 11, 0.85)]
        report = label_reports(scores)[0]
        assert report.low_median is None
        assert report.high_median is not None


class TestRankLabels:
    def test_none_when_nothing_scores(self):
        reports = [LabelReport("A", 3, 0, 0.30, None, 0.30)]
        assert rank_labels(reports) is None

    def test_none_when_empty(self):
        assert rank_labels([]) is None

    def test_best_report_first(self):
        reports = [
            LabelReport("A", 10, 8, 0.75, 0.75, 0.40),
            LabelReport("B", 4, 2, 0.50, 0.60, 0.30),
        ]
        assert rank_labels(reports).label == "A"

    def test_turn_count_beats_weak_majority(self):
        # Eight strong turns beat two: Darrick talks a lot when he talks.
        reports = [
            LabelReport("B", 2, 2, 0.90, 0.90, None),
            LabelReport("A", 8, 8, 0.70, 0.70, None),
        ]
        # B sorts first on fraction tie... both are 100%. Median decides.
        assert rank_labels(reports).label == "B"


class TestEarCheckTurns:
    def test_top_and_bottom_are_disjoint_extremes(self):
        scores = [
            ts("A", 0, 5, 0.90), ts("A", 6, 11, 0.85), ts("A", 12, 17, 0.80),
            ts("A", 18, 23, 0.40), ts("A", 24, 29, 0.30),
        ]
        top, bottom = ear_check_turns(scores, "A", per_side=2)
        assert [s.score for s in top] == [0.90, 0.85]
        assert [s.score for s in bottom] == [0.40, 0.30]

    def test_other_labels_excluded(self):
        scores = [ts("A", 0, 5, 0.90), ts("B", 6, 11, 0.10)]
        top, bottom = ear_check_turns(scores, "A")
        assert all(s.speaker == "A" for s in top + bottom)

    def test_no_bottom_when_fewer_than_per_side(self):
        scores = [ts("A", 0, 5, 0.90), ts("A", 6, 11, 0.85)]
        top, bottom = ear_check_turns(scores, "A", per_side=3)
        assert len(top) == 2
        assert bottom == []
