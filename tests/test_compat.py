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


MAX_CACHE = 256


@torch.inference_mode()
def test_static_cache_matches_dynamic(fixture):
    """A StaticCache must produce the same logits as a DynamicCache."""
    model, ids = fixture
    ref = _reference(model, ids, ids.shape[1] - 10)

    static = compat.make_static_cache(model, MAX_CACHE)
    keep = ids.shape[1] - 10
    model(input_ids=ids[:, :keep], past_key_values=static, use_cache=True)
    got = model(input_ids=ids[:, keep:], past_key_values=static,
                use_cache=True).logits

    diff = (ref - got).abs().max().item()
    _report("static cache vs dynamic", diff)
    assert diff <= ATOL


@torch.inference_mode()
def test_static_cache_rollback_is_bit_exact(fixture):
    """Rolling a StaticCache back must equal a real crop, exactly.

    StaticLayer.update writes at cumulative_length and ignores cache_position,
    so a rollback that failed to move that counter would write to the wrong slot
    and corrupt attention silently. This is the graphed decoder's foundation.
    """
    model, ids = fixture
    keep = ids.shape[1] - 10
    junk = torch.tensor([[100, 200, 300, 400, 500]], device=ids.device)

    ref = _reference(model, ids, keep)

    static = compat.make_static_cache(model, MAX_CACHE)
    model(input_ids=ids[:, :keep], past_key_values=static, use_cache=True)
    assert compat.cache_length(static) == keep
    model(input_ids=junk, past_key_values=static, use_cache=True)
    assert compat.cache_length(static) == keep + junk.shape[1]

    compat.crop_cache(static, keep)
    assert compat.cache_length(static) == keep

    got = model(input_ids=ids[:, keep:], past_key_values=static,
                use_cache=True).logits
    diff = (ref - got).abs().max().item()
    _report("static rollback after junk", diff)
    assert diff == 0.0, "rollback must be bit-exact, not merely close"


@torch.inference_mode()
def test_static_cache_is_not_croppable(fixture):
    """Guard the assumption the rollback path depends on."""
    model, _ = fixture
    static = compat.make_static_cache(model, MAX_CACHE)
    assert getattr(static, "is_croppable", False) is False
    print("[PASS] StaticCache.is_croppable is False, so crop_cache must roll back")


def _decode_static_eager(model, ids, steps: int) -> list[int]:
    cache = compat.make_static_cache(model, MAX_CACHE)
    model(input_ids=ids, past_key_values=cache, use_cache=True)
    cur, out = ids[:, -1:].clone(), []
    for _ in range(steps):
        logits = model(input_ids=cur, past_key_values=cache,
                       use_cache=True).logits
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out.append(int(nxt.item()))
        cur = nxt
    return out


def _decode_dynamic_eager(model, ids, steps: int) -> list[int]:
    cache = compat.make_cache(model)
    seq, cached, out = ids, 0, []
    for _ in range(steps):
        logits = model(input_ids=seq[:, cached:], past_key_values=cache,
                       use_cache=True).logits
        cached = seq.shape[1]
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        out.append(int(nxt.item()))
    return out


@torch.inference_mode()
def test_graphed_forward_matches_static_eager(fixture):
    """The graph must reproduce the eager computation of the SAME cache type.

    This is the graph's own correctness property, and it must be exact. It is
    deliberately measured against static-eager rather than dynamic-eager: a
    StaticCache and a DynamicCache are not numerically identical over a long
    decode (see test_static_vs_dynamic_numerical_drift), so comparing across
    both cache type and execution mode at once would conflate two effects.
    """
    model, ids = fixture
    steps = 64

    ref = _decode_static_eager(model, ids, steps)

    static = compat.make_static_cache(model, MAX_CACHE)
    model(input_ids=ids, past_key_values=static, use_cache=True)
    graph = compat.GraphedForward(model, static, width=1)
    static.reset()
    model(input_ids=ids, past_key_values=static, use_cache=True)

    tok_buf = ids[:, -1:].clone()
    got = []
    for _ in range(steps):
        logits = graph.replay(tok_buf)
        got.append(int(logits[0, -1, :].argmax().item()))
        tok_buf.fill_(got[-1])

    match = got == ref
    first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), -1)
    print(f"[{'PASS' if match else 'FAIL'}] graphed vs static-eager: {steps} "
          f"tokens, identical={match}"
          + ("" if match else f", diverges at {first}"))
    assert match


@torch.inference_mode()
def test_static_vs_dynamic_numerical_drift(fixture):
    """Characterise, do not hide, the StaticCache/DynamicCache difference.

    A single forward matches at 0.0, but attention over a fully-allocated static
    cache reduces in a different order than over a snug dynamic one, and that
    drift compounds. It can eventually flip a token. This test records the
    per-forward logit drift so a regression in it would be visible; it is NOT a
    losslessness gate. Losslessness is compared WITHIN one cache type.
    """
    model, ids = fixture
    steps = 64

    dyn_toks = _decode_dynamic_eager(model, ids, steps)
    st_toks = _decode_static_eager(model, ids, steps)

    first = next((i for i, (a, b) in enumerate(zip(dyn_toks, st_toks))
                  if a != b), -1)

    # Per-forward drift on a shared prefix, before any divergence can compound.
    prefix = ids
    dyn = compat.make_cache(model)
    a = model(input_ids=prefix, past_key_values=dyn, use_cache=True
              ).logits[0, -1, :].float()
    st = compat.make_static_cache(model, MAX_CACHE)
    b = model(input_ids=prefix, past_key_values=st, use_cache=True
              ).logits[0, -1, :].float()
    drift = (a - b).abs().max().item()

    print(f"[INFO] static vs dynamic over {steps} tokens: "
          f"{'identical' if first == -1 else f'first token difference at {first}'}")
    print(f"[INFO] single-forward max logit drift on the prompt: {drift}")
    assert drift <= ATOL, (
        f"per-forward drift {drift} exceeds {ATOL}; the two cache types should "
        "agree closely on a single forward even if long decodes diverge"
    )


@torch.inference_mode()
def test_graphed_forward_rejects_wrong_width(fixture):
    """A graph is valid only at its captured width."""
    model, ids = fixture
    static = compat.make_static_cache(model, MAX_CACHE)
    model(input_ids=ids, past_key_values=static, use_cache=True)
    graph = compat.GraphedForward(model, static, width=1)
    with pytest.raises(ValueError):
        graph.replay(ids[:, -3:])
    print("[PASS] replay rejects a token width the graph was not captured at")


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
