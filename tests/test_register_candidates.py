"""Tests for register-candidate extraction.

The whole point of this tool is to ground candidates in real, countable text
rather than an LLM's guess at "what he sounds like" — these tests check the
counting and ranking logic actually does that, not that it produces plausible-
looking output.
"""

from pipeline.register_candidates import candidate_phrases, format_report, turn_openers
from pipeline.transcript import parse_chunk_lines


class TestParseChunkLines:
    def test_splits_speaker_and_text(self):
        assert parse_chunk_lines("DARRICK: wemby is him") == [("DARRICK", "wemby is him")]

    def test_multiple_lines(self):
        text = "DARRICK: line one\nKENNY: line two"
        assert parse_chunk_lines(text) == [("DARRICK", "line one"), ("KENNY", "line two")]

    def test_only_splits_on_the_first_colon_space(self):
        # Dialogue itself might contain ": " — the speaker label never does.
        text = "DARRICK: the score was 3: 2 at half"
        assert parse_chunk_lines(text) == [("DARRICK", "the score was 3: 2 at half")]

    def test_a_line_without_a_speaker_prefix_is_skipped(self):
        assert parse_chunk_lines("not attributed text") == []

    def test_empty_text_yields_nothing(self):
        assert parse_chunk_lines("") == []


class TestCandidatePhrases:
    def test_a_phrase_only_the_speaker_says_ranks_highest(self):
        lines = [("DARRICK", "listen man listen man listen man")] * 3 + [
            ("KENNY", "totally different words entirely")
        ] * 3
        got = candidate_phrases(lines, "DARRICK", min_count=1)
        assert got[0].phrase == "listen man"

    def test_a_phrase_everyone_says_equally_is_not_top_ranked(self):
        # Generic filler both hosts use at the same rate isn't a Darrick trait.
        shared = [("DARRICK", "you know what I mean")] * 5 + [
            ("KENNY", "you know what I mean")
        ] * 5
        distinctive = [("DARRICK", "cook him up cook him up cook him up")] * 5
        got = candidate_phrases(shared + distinctive, "DARRICK", min_count=1)
        assert got[0].phrase != "you know what"

    def test_below_min_count_is_excluded(self):
        lines = [("DARRICK", "rare phrase here")] * 2
        assert candidate_phrases(lines, "DARRICK", min_count=3) == []

    def test_at_min_count_is_included(self):
        lines = [("DARRICK", "rare phrase here")] * 3
        got = candidate_phrases(lines, "DARRICK", min_count=3, ngram_sizes=(3,))
        assert any(c.phrase == "rare phrase here" for c in got)

    def test_no_lines_from_speaker_yields_nothing(self):
        lines = [("KENNY", "some words here")] * 5
        assert candidate_phrases(lines, "DARRICK") == []

    def test_rates_are_case_insensitive(self):
        lines = [("DARRICK", "Cook Him Up")] * 3 + [("DARRICK", "cook him up")] * 3
        got = candidate_phrases(lines, "DARRICK", min_count=1, ngram_sizes=(3,))
        matches = [c for c in got if c.phrase == "cook him up"]
        assert len(matches) == 1
        assert matches[0].speaker_count == 6

    def test_example_is_a_real_line_containing_the_phrase(self):
        lines = [("DARRICK", "cook him up right there")] * 3
        got = candidate_phrases(lines, "DARRICK", min_count=1, ngram_sizes=(3,))
        phrase = next(c for c in got if c.phrase == "cook him up")
        assert phrase.example == "cook him up right there"

    def test_top_n_limits_results(self):
        lines = [("DARRICK", f"unique phrase number {i} here") for i in range(20) for _ in range(3)]
        got = candidate_phrases(lines, "DARRICK", min_count=1, top_n=5)
        assert len(got) == 5


class TestTurnOpeners:
    def test_ranks_by_frequency(self):
        lines = [("DARRICK", "nah that's crazy talk")] * 4 + [
            ("DARRICK", "listen to what happened")
        ] * 2
        got = turn_openers(lines, "DARRICK", word_count=1, min_count=1)
        assert got[0] == ("nah", 4)

    def test_ignores_other_speakers(self):
        lines = [("KENNY", "nah that's not right")] * 5
        assert turn_openers(lines, "DARRICK", word_count=1, min_count=1) == []

    def test_below_min_count_excluded(self):
        lines = [("DARRICK", "rare opener here")] * 2
        assert turn_openers(lines, "DARRICK", word_count=1, min_count=3) == []

    def test_short_lines_below_word_count_are_skipped(self):
        lines = [("DARRICK", "no")] * 5
        assert turn_openers(lines, "DARRICK", word_count=3, min_count=1) == []


class TestFormatReport:
    def test_includes_the_speaker_name(self):
        assert "DARRICK" in format_report("DARRICK", [], [])

    def test_disclaims_auto_writing_to_register_md(self):
        # The report's whole job is to hand off to a human, not to imply this
        # is going to get auto-applied anywhere.
        report = format_report("DARRICK", [], [])
        assert "register.md" in report
        assert "nothing here is written to" in report.lower()

    def test_empty_input_says_so_rather_than_looking_broken(self):
        report = format_report("DARRICK", [], [])
        assert "ingest more episodes" in report.lower()

    def test_renders_a_real_candidate(self):
        from pipeline.register_candidates import PhraseCandidate

        candidate = PhraseCandidate(
            phrase="cook him up",
            speaker_count=5,
            speaker_per_1000_words=4.2,
            others_per_1000_words=0.1,
            example="he's gonna cook him up tonight",
        )
        report = format_report("DARRICK", [candidate], [])
        assert "cook him up" in report
        assert "cook him up tonight" in report
