"""Crop must be bit-exact.  Adapted from check_crop.py, routed through compat.

A wrong crop does not raise -- it silently corrupts decoder state and the
output diverges after roughly fifty tokens.  So we do not test that crop
reports the right length; we re-feed the same tokens and compare logits.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import compat

MODEL_PATH = "models/draft-0.5b"
PROBE = (
    "The capital of France is Paris. The capital of Japan is Tokyo. "
    "The capital of Egypt is Cairo. The capital of Peru is"
)
ATOL = 1e-3

_loaded: tuple | None = None


def _load() -> tuple:
    """Load the draft model and probe ids once per process."""
    global _loaded
    if _loaded is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = compat.load_model(MODEL_PATH, quantize=False, device="cuda")
        ids = tok(PROBE, return_tensors="pt").input_ids.to("cuda")
        _loaded = (model, ids)
    return _loaded


@pytest.fixture(scope="module")
def fixture():
    return _load()


def _feed(model, cache, chunk):
    """One forward pass over `chunk`, appending to `cache`.  Returns logits."""
    return model(input_ids=chunk, past_key_values=cache, use_cache=True).logits


def _reference(model, ids, keep: int):
    """Logits for ids[:, keep:] given a clean cache holding exactly ids[:, :keep]."""
    cache = compat.make_cache(model)
    _feed(model, cache, ids[:, :keep])
    return _feed(model, cache, ids[:, keep:])


def _run_shrink(model, ids, keep: int) -> float:
    """Grow the cache to full length, crop back to `keep`, re-feed, diff logits."""
    ref = _reference(model, ids, keep)

    cache = compat.make_cache(model)
    _feed(model, cache, ids[:, :keep])
    _feed(model, cache, ids[:, keep:])
    assert compat.cache_length(cache) == ids.shape[1]

    compat.crop_cache(cache, keep)
    assert compat.cache_length(cache) == keep

    again = _feed(model, cache, ids[:, keep:])
    return (ref - again).abs().max().item()


def _run_noop(model, ids, keep: int, requested: int) -> float:
    """crop_cache must not touch a cache already at or below `requested`."""
    ref = _reference(model, ids, keep)

    cache = compat.make_cache(model)
    _feed(model, cache, ids[:, :keep])

    # Prove crop() is never called on the no-op path.
    def _forbidden(*a, **k):
        raise AssertionError("crop() called on a no-op crop_cache request")

    cache.crop = _forbidden
    compat.crop_cache(cache, requested)
    del cache.crop

    assert compat.cache_length(cache) == keep

    again = _feed(model, cache, ids[:, keep:])
    return (ref - again).abs().max().item()


def _report(name: str, diff: float) -> None:
    verdict = "PASS" if diff <= ATOL else "FAIL"
    print(f"[{verdict}] {name}: max logit diff = {diff}")


@torch.inference_mode()
def test_crop_to_middle(fixture):
    """Case 1: the original check_crop.py case.  Previously gave exactly 0.0."""
    model, ids = fixture
    diff = _run_shrink(model, ids, ids.shape[1] - 10)
    _report("crop to middle", diff)
    assert diff <= ATOL


@torch.inference_mode()
def test_crop_to_one_token(fixture):
    """Case 2: crop all the way down to a single token."""
    model, ids = fixture
    diff = _run_shrink(model, ids, 1)
    _report("crop to 1 token", diff)
    assert diff <= ATOL


@torch.inference_mode()
def test_crop_to_current_length(fixture):
    """Case 3: crop to exactly the current length -- no-op, must not error."""
    model, ids = fixture
    keep = ids.shape[1] - 10
    diff = _run_noop(model, ids, keep, requested=keep)
    _report("crop to current length (no-op)", diff)
    assert diff <= ATOL


@torch.inference_mode()
def test_crop_beyond_current_length(fixture):
    """Case 4: crop to more than the current length -- no-op, must not error."""
    model, ids = fixture
    keep = ids.shape[1] - 10
    diff = _run_noop(model, ids, keep, requested=keep + 50)
    _report("crop beyond current length (no-op)", diff)
    assert diff <= ATOL


if __name__ == "__main__":
    model, ids = _load()
    print("describe():", compat.describe())
    print("probe tokens:", ids.shape[1])
    results = []
    with torch.inference_mode():
        for name, fn in [
            ("crop to middle", lambda: _run_shrink(model, ids, ids.shape[1] - 10)),
            ("crop to 1 token", lambda: _run_shrink(model, ids, 1)),
            ("crop to current length (no-op)",
             lambda: _run_noop(model, ids, ids.shape[1] - 10, ids.shape[1] - 10)),
            ("crop beyond current length (no-op)",
             lambda: _run_noop(model, ids, ids.shape[1] - 10, ids.shape[1] + 40)),
        ]:
            d = fn()
            _report(name, d)
            results.append(d <= ATOL)
    print("VERDICT:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
