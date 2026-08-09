"""Local cross-encoder model loading.

The cross-encoder is the only thing that decides relevance in Uma — there is
no LLM involved in filtering. The model is loaded once per process and
reused for every scoring call, since loading is the expensive part (model
weights + tokenizer initialization).
"""

from __future__ import annotations

import os
import threading

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_lock = threading.Lock()
_model = None
_model_name: str | None = None


def get_model_name() -> str:
    return os.environ.get("UMA_CROSS_ENCODER_MODEL", DEFAULT_MODEL_NAME)


def get_cross_encoder():
    """Return the process-wide cross-encoder instance, loading it if needed.

    Thread-safe, idempotent. Subsequent calls with the same configured model
    name return the cached instance instantly.
    """
    global _model, _model_name

    name = get_model_name()
    if _model is not None and _model_name == name:
        return _model

    with _lock:
        if _model is not None and _model_name == name:
            return _model

        # Imported lazily so importing uma.core.model doesn't require torch
        # to be installed unless a cross-encoder is actually requested (e.g.
        # this keeps `uma --help` and unit tests of unrelated modules fast).
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(name)
        _model_name = name
        return _model


def reset_model_cache() -> None:
    """Drop the cached model. Mainly useful for tests."""
    global _model, _model_name
    with _lock:
        _model = None
        _model_name = None
