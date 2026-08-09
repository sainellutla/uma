"""Token counting utilities.

Uma uses tiktoken's ``cl100k_base`` encoding as a consistent, dependency-light
token counter for measuring context size and reduction. This is the same
encoding family used by GPT-4o-class models, so counts are directly
meaningful for OpenAI-compatible providers and a reasonable proxy for others.

This module intentionally exposes a single small surface (``count_tokens`` /
``truncate_to_tokens``) so the rest of Uma never has to know which tokenizer
library is in use underneath.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text``."""
    if not text:
        return 0
    return len(_encoding().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` so it contains at most ``max_tokens`` tokens.

    Truncation happens on token boundaries, not characters, so the result is
    always <= max_tokens as measured by :func:`count_tokens`.
    """
    if max_tokens <= 0:
        return ""
    enc = _encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
