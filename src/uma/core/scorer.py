"""Sentence segmentation and cross-encoder relevance scoring.

This module implements the first two stages of the Uma pipeline:

    Context -> Sentence segmentation -> Cross-encoder -> per-sentence score

Segmentation is a dependency-free regex splitter tuned to survive the kind
of text that shows up in financial / business RAG context: decimal numbers
("$391.0 billion", "12.5%"), abbreviations ("Inc.", "U.S.", "e.g."), and
initials, none of which should be treated as sentence boundaries.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Abbreviations that are commonly followed by a period but do NOT end a
# sentence. Matched case-insensitively against the token immediately before
# the period.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc",
    "ltd", "co", "corp", "fig", "no", "vol", "approx", "e.g", "i.e",
    "u.s", "u.k", "u.n", "a.m", "p.m", "vs", "vol",
}

# Matches a sentence-ending punctuation mark followed by whitespace and a
# capital letter / quote / digit (start of next sentence), while requiring
# a lookbehind that isn't part of a decimal number.
_SPLIT_RE = re.compile(
    r"""
    (?<!\d)          # not preceded by a digit (avoid splitting "3. 5")
    (?<=[.!?])       # split right after ., !, or ?
    (?<!\b[A-Z]\.)   # not a single-capital-letter initial like "A."
    \s+
    (?=[A-Z"'“(0-9])
    """,
    re.VERBOSE,
)

_WORD_BEFORE_PERIOD_RE = re.compile(r"(\b[A-Za-z\.]+)\.\s*$")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into a list of trimmed, non-empty sentences.

    This is intentionally simple and deterministic (no ML, no network, no
    NLTK data download) so it behaves the same in every environment.
    """
    if not text or not text.strip():
        return []

    normalized = re.sub(r"\s+", " ", text.strip())

    # First pass: naive split on the regex.
    raw_parts = _SPLIT_RE.split(normalized)

    sentences: list[str] = []
    buffer = ""
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{buffer} {part}".strip() if buffer else part
        last_word_match = _WORD_BEFORE_PERIOD_RE.search(candidate)
        if last_word_match and last_word_match.group(1).lower().strip(".") in _ABBREVIATIONS:
            # Likely a false split on an abbreviation; keep accumulating.
            buffer = candidate
            continue
        sentences.append(candidate)
        buffer = ""

    if buffer:
        sentences.append(buffer)

    return [s for s in sentences if s]


@dataclass(frozen=True)
class ScoredSentence:
    """A single sentence with its cross-encoder relevance score.

    ``score`` is a sigmoid applied to the raw cross-encoder logit, squashed
    to [0, 1] purely for a bounded, consistent scale to threshold against —
    NOT a calibrated probability. ms-marco-MiniLM-L-6-v2 is documented as a
    ranking model (pass/passage pairs get a raw score, sorted highest to
    lowest); its model card makes no claim that those scores, sigmoid or
    otherwise, correspond to "probability of relevance." Treat ``score`` as
    a normalized relevance score, and ``threshold`` as an operating point on
    it — 0.5 is simply Uma's default operating threshold, not "50% likely
    relevant." See :mod:`uma.core.calibrate` for finding a threshold
    empirically rather than assuming 0.5 is right for a given workload.
    """

    index: int
    text: str
    score: float
    raw_score: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score_sentences(
    query: str,
    sentences: list[str],
    *,
    batch_size: int = 32,
) -> list[ScoredSentence]:
    """Score each sentence's relevance to ``query`` using the local cross-encoder.

    Batched for throughput; order of the input list is preserved in the
    output.
    """
    if not sentences:
        return []

    from uma.core.model import get_cross_encoder

    model = get_cross_encoder()
    pairs = [(query, sentence) for sentence in sentences]
    raw_scores = model.predict(pairs, batch_size=batch_size)

    return [
        ScoredSentence(
            index=i,
            text=sentence,
            score=_sigmoid(float(raw)),
            raw_score=float(raw),
        )
        for i, (sentence, raw) in enumerate(zip(sentences, raw_scores))
    ]
