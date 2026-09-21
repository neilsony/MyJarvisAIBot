"""Tests for turn-level segment logic — the part that decides which audio is
clean enough to clone a voice from. Getting this wrong means training on 60
minutes of the wrong person, so it's tested hard."""

import pytest

from pipeline.segments import (
    Turn,
    clean_turns,
    merge_adjacent,
    overlap_fraction,
    overlaps,
    take_until_duration,
    total_duration,
)


def t(speaker: str, start: float, end: float) -> Turn:
    return Turn(speaker=speaker, start=start, end=end)


class TestTurn:
    def test_duration(self):
        assert t("A", 1.0, 4.5).duration == pytest.approx(3.5)

    def test_rejects_inverted_span(self):
        with pytest.raises(ValueError, match="end must be after start"):
            Turn(speaker="A", start=5.0, end=5.0)


class TestOverlaps:
    def test_disjoint(self):
        assert not overlaps(t("A", 0, 1), t("B", 2, 3))

    def test_touching_is_not_overlap(self):
        assert not overlaps(t("A", 0, 1), t("B", 1, 2))

    def test_partial(self):
        assert overlaps(t("A", 0, 2), t("B", 1, 3))

    def test_contained(self):
        assert overlaps(t("A", 0, 10), t("B", 4, 5))

    def test_is_symmetric(self):
        a, b = t("A", 0, 2), t("B", 1, 3)
        assert overlaps(a, b) == overlaps(b, a)


class TestOverlapFraction:
    def test_fully_alone_is_zero(self):
        turn = t("A", 0, 10)
        assert overlap_fraction(turn, [t("B", 20, 25)]) == 0.0

    def test_no_other_turns_is_zero(self):
        assert overlap_fraction(t("A", 0, 10), []) == 0.0

    def test_fully_covered_is_one(self):
        turn = t("A", 0, 10)
        assert overlap_fraction(turn, [t("B", 0, 10)]) == pytest.approx(1.0)

    def test_partial_overlap_is_the_right_fraction(self):
        # A brief interjection at the edge — 2 of 10 seconds covered.
        turn = t("A", 0, 10)
        assert overlap_fraction(turn, [t("B", 8, 12)]) == pytest.approx(0.2)

    def test_distinguishes_a_brief_interjection_from_a_shouting_match(self):
        # This is the whole point: two turns can both "technically overlap"
        # while being very different listens.
        mostly_clean = t("A", 0, 10)
        heavily_crowded = t("A", 100, 110)
        interjection = [t("B", 9, 10)]
        shouting_match = [t("B", 100, 108), t("C", 102, 110)]
        assert overlap_fraction(mostly_clean, interjection) < overlap_fraction(
            heavily_crowded, shouting_match
        )

    def test_multiple_others_sum_their_coverage(self):
        turn = t("A", 0, 10)
        others = [t("B", 0, 3), t("C", 5, 8)]
        assert overlap_fraction(turn, others) == pytest.approx(0.6)

    def test_overlapping_others_do_not_double_count_past_one(self):
        # Two other speakers both covering the same span shouldn't push the
        # fraction past 1.0 — it's a fraction of this turn, not a sum of theirs.
        turn = t("A", 0, 10)
        others = [t("B", 0, 10), t("C", 0, 10)]
        assert overlap_fraction(turn, others) == pytest.approx(1.0)

    def test_ignores_other_turns_from_the_same_speaker(self):
        # A duplicate/re-diarized turn for the same speaker isn't crosstalk.
        turn = t("A", 0, 10)
        assert overlap_fraction(turn, [t("A", 5, 15)]) == 0.0

    def test_touching_is_zero_overlap(self):
        turn = t("A", 0, 10)
        assert overlap_fraction(turn, [t("B", 10, 15)]) == 0.0


class TestMergeAdjacent:
    def test_empty(self):
        assert merge_adjacent([], max_gap=0.5) == []

    def test_merges_same_speaker_within_gap(self):
        got = merge_adjacent([t("A", 0, 2), t("A", 2.3, 4)], max_gap=0.5)
        assert got == [t("A", 0, 4)]

    def test_does_not_merge_beyond_gap(self):
        turns = [t("A", 0, 2), t("A", 3.0, 4)]
        assert merge_adjacent(turns, max_gap=0.5) == turns

    def test_does_not_merge_different_speakers(self):
        turns = [t("A", 0, 2), t("B", 2.1, 4)]
        assert merge_adjacent(turns, max_gap=0.5) == turns

    def test_merges_a_run_of_three(self):
        got = merge_adjacent([t("A", 0, 1), t("A", 1.2, 2), t("A", 2.1, 3)], max_gap=0.5)
        assert got == [t("A", 0, 3)]

    def test_sorts_unordered_input(self):
        got = merge_adjacent([t("A", 5, 6), t("A", 0, 1)], max_gap=0.5)
        assert [x.start for x in got] == [0, 5]


class TestCleanTurns:
    def test_keeps_isolated_long_turn(self):
        turns = [t("A", 0, 10), t("B", 20, 30)]
        got = clean_turns(turns, speaker="A", min_duration=4.0, pad=0.0)
        assert got == [t("A", 0, 10)]

    def test_drops_turn_overlapped_by_another_speaker(self):
        turns = [t("A", 0, 10), t("B", 5, 6)]
        assert clean_turns(turns, speaker="A", min_duration=4.0, pad=0.0) == []

    def test_ignores_overlap_from_same_speaker(self):
        turns = [t("A", 0, 10), t("A", 5, 6)]
        got = clean_turns(turns, speaker="A", min_duration=4.0, pad=0.0)
        assert t("A", 0, 10) in got

    def test_drops_turn_below_min_duration(self):
        assert clean_turns([t("A", 0, 3)], speaker="A", min_duration=4.0, pad=0.0) == []

    def test_padding_trims_both_ends(self):
        got = clean_turns([t("A", 0, 10)], speaker="A", min_duration=4.0, pad=0.5)
        assert got == [t("A", 0.5, 9.5)]

    def test_padding_can_push_a_turn_below_minimum(self):
        # 5s turn, 0.5s trimmed each end -> 4.0s, which is not > 4.0 min
        assert clean_turns([t("A", 0, 5)], speaker="A", min_duration=4.5, pad=0.5) == []

    def test_returns_only_requested_speaker(self):
        turns = [t("A", 0, 10), t("B", 20, 30)]
        got = clean_turns(turns, speaker="B", min_duration=4.0, pad=0.0)
        assert [x.speaker for x in got] == ["B"]

    def test_empty_input(self):
        assert clean_turns([], speaker="A", min_duration=4.0, pad=0.0) == []


class TestTakeUntilDuration:
    def test_prefers_longest_turns_first(self):
        turns = [t("A", 0, 2), t("A", 10, 20), t("A", 30, 35)]
        got = take_until_duration(turns, target_seconds=14.0)
        # 10s + 5s = 15s clears the target; the 2s turn is not needed
        assert sorted(x.duration for x in got) == [5.0, 10.0]

    def test_returns_chronological_order(self):
        turns = [t("A", 30, 40), t("A", 0, 5)]
        got = take_until_duration(turns, target_seconds=100.0)
        assert [x.start for x in got] == [0, 30]

    def test_returns_all_when_target_unreachable(self):
        turns = [t("A", 0, 5), t("A", 10, 15)]
        assert len(take_until_duration(turns, target_seconds=999.0)) == 2

    def test_empty(self):
        assert take_until_duration([], target_seconds=10.0) == []

    def test_zero_target_takes_nothing(self):
        assert take_until_duration([t("A", 0, 5)], target_seconds=0.0) == []


class TestTotalDuration:
    def test_sums(self):
        assert total_duration([t("A", 0, 2), t("A", 10, 13)]) == pytest.approx(5.0)

    def test_empty_is_zero(self):
        assert total_duration([]) == 0.0
