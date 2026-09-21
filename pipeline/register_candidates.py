"""Surface real speech-pattern candidates for a human to fold into register.md.

"Trained on his personality" isn't really a thing here — GLM isn't fine-tuned,
so the only honest path to a sharper register is a human listening to real
episodes and writing down what they actually notice (see MASTER-PLAN.md).
This module narrows *what to listen for*: phrases and turn-openers Darrick
says disproportionately more than the other three hosts, ranked from real
transcript text, never invented.

**This never writes register.md.** It produces a report; a person reads it,
checks the examples in context, and edits the register file by hand. That
split is deliberate — an automatic writer would eventually fold in a
transcription error or a one-off remark as if it were a real pattern, with no
human catching it before it shapes every future reply.

Deliberately conservative in the same spirit as `pipeline.speakers`: a phrase
both Darrick and the other hosts say constantly is just how the show talks,
not a Darrick trait. Only the *gap* between his rate and theirs is a signal
worth a human's attention, and a low occurrence count is noise, not a pattern.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "MIN_COUNT",
    "PhraseCandidate",
    "candidate_phrases",
    "format_report",
    "turn_openers",
]

_WORD = re.compile(r"[a-z']+")

# Below this, a phrase's count is too small to trust as a real pattern rather
# than a lucky repetition or a transcription quirk.
MIN_COUNT = 3


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


@dataclass(frozen=True)
class PhraseCandidate:
    """One candidate phrase, with enough context for a human to judge it.

    Rates are per 1,000 words so episodes of different lengths compare fairly
    — a raw count is dominated by whoever simply talked more.
    """

    phrase: str
    speaker_count: int
    speaker_per_1000_words: float
    others_per_1000_words: float
    example: str


def candidate_phrases(
    lines: list[tuple[str, str]],
    speaker: str,
    *,
    ngram_sizes: tuple[int, ...] = (2, 3, 4),
    min_count: int = MIN_COUNT,
    top_n: int = 40,
) -> list[PhraseCandidate]:
    """Phrases `speaker` uses disproportionately more than everyone else.

    `lines` is (speaker, text) pairs — e.g. from
    `pipeline.transcript.parse_chunk_lines` over every ingested Canon chunk.

    Ranked by the gap between the speaker's per-1,000-word rate and everyone
    else's, not raw count — so generic filler that everyone says constantly
    doesn't crowd out something rarer but genuinely distinctive.
    """
    speaker_words = 0
    others_words = 0
    speaker_ngrams: Counter[str] = Counter()
    others_ngrams: Counter[str] = Counter()
    examples: dict[str, str] = {}

    for line_speaker, text in lines:
        words = _words(text)
        is_speaker = line_speaker == speaker
        if is_speaker:
            speaker_words += len(words)
        else:
            others_words += len(words)

        for n in ngram_sizes:
            for phrase in _ngrams(words, n):
                if is_speaker:
                    speaker_ngrams[phrase] += 1
                    examples.setdefault(phrase, text.strip())
                else:
                    others_ngrams[phrase] += 1

    candidates = []
    for phrase, count in speaker_ngrams.items():
        if count < min_count:
            continue
        speaker_rate = (count / speaker_words * 1000) if speaker_words else 0.0
        others_rate = (others_ngrams.get(phrase, 0) / others_words * 1000) if others_words else 0.0
        candidates.append(
            PhraseCandidate(
                phrase=phrase,
                speaker_count=count,
                speaker_per_1000_words=speaker_rate,
                others_per_1000_words=others_rate,
                example=examples[phrase],
            )
        )

    candidates.sort(key=lambda c: c.speaker_per_1000_words - c.others_per_1000_words, reverse=True)
    return candidates[:top_n]


def turn_openers(
    lines: list[tuple[str, str]],
    speaker: str,
    *,
    word_count: int = 3,
    min_count: int = MIN_COUNT,
    top_n: int = 20,
) -> list[tuple[str, int]]:
    """The first few words of `speaker`'s lines, ranked by frequency.

    Comedic timing and reactions tend to live at the front of a turn — "nah,"
    "listen," "see this is" — this is a cheap, direct way to surface them
    without the distinctiveness math `candidate_phrases` needs.
    """
    openers: Counter[str] = Counter()
    for line_speaker, text in lines:
        if line_speaker != speaker:
            continue
        words = _words(text)
        if len(words) >= word_count:
            openers[" ".join(words[:word_count])] += 1
    return [(phrase, count) for phrase, count in openers.most_common(top_n) if count >= min_count]


def format_report(
    speaker: str,
    phrases: list[PhraseCandidate],
    openers: list[tuple[str, int]],
) -> str:
    """Render both analyses as a markdown report for a human to read.

    Never touches register.md — this is the entire handoff to the human.
    """
    lines = [
        f"# Register candidates for {speaker}",
        "",
        "Generated from real transcript. Nothing here is written to "
        "`register.md` automatically — read the examples in context, and "
        "only add what actually sounds like him when you check it against "
        "the real episode.",
        "",
        "## Distinctive phrases",
        "",
        "Ranked by how much more often he says this than the other hosts "
        "combined, not by raw frequency.",
        "",
    ]

    if not phrases:
        lines.append("*Nothing cleared the minimum count yet — ingest more episodes.*")
    else:
        for c in phrases:
            lines.append(
                f"- **\"{c.phrase}\"** — {c.speaker_count}x, "
                f"{c.speaker_per_1000_words:.2f}/1000 words vs "
                f"{c.others_per_1000_words:.2f}/1000 for everyone else\n"
                f"  > {c.example}"
            )

    lines += ["", "## Common turn-openers", ""]

    if not openers:
        lines.append("*Nothing cleared the minimum count yet — ingest more episodes.*")
    else:
        for phrase, count in openers:
            lines.append(f"- **\"{phrase}\"** — {count}x")

    return "\n".join(lines) + "\n"
