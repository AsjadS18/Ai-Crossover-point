"""Graphed speculative decoding: mechanics, then losslessness.

Losslessness is compared WITHIN the graphed regime -- graphed spec against the
graphed baseline. Comparing against the eager DynamicCache baseline would show
false mismatches from StaticCache/DynamicCache numerical drift, which
tests/test_compat.py characterises separately.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive import AdaptiveGamma
from src.baseline import GraphedGreedyDecoder
from src.models import chat_ids, load_pair
from src.specdec import GraphedSpecDecoder

MAX_CACHE = 512
PROMPT = "Write a Python function that reverses a list in place."
PROMPTS = [
    "Write a Python function that reverses a list in place.",
    "Explain what a hash map is and when you would use one.",
    "Return a JSON object describing a book: title, author, year, tags.",
    "A train leaves at 14:05 and takes 2 hours 50 minutes. When does it arrive?",
]

_pair: tuple | None = None


def _load() -> tuple:
    global _pair
    if _pair is None:
        _pair = load_pair()
    return _pair


@pytest.fixture(scope="module")
def pair():
    return _load()


def test_self_draft_acceptance_is_exactly_one(pair):
    """The graphed diagnostic: the target drafting for itself accepts all.

    If the verify window offset is wrong under graph replay, this fails and
    nothing downstream is trustworthy.
    """
    tokenizer, target, _ = pair
    ids = chat_ids(tokenizer, PROMPT)
    gamma = 4
    dec = GraphedSpecDecoder(target, target, gamma=gamma,
                             max_cache_len=MAX_CACHE)
    _, stats = dec.generate(ids, max_new_tokens=5 * (gamma + 1), eos_id=None)
    print(f"[self-draft] acceptance={stats['acceptance_rate']} "
          f"tokens_per_round={stats['tokens_per_round']} "
          f"rounds={stats['rounds']}")
    assert stats["acceptance_rate"] == 1.0
    assert stats["tokens_per_round"] == gamma + 1


@pytest.mark.parametrize("gamma", [1, 2, 3, 5, 8])
def test_lossless_against_graphed_baseline(pair, gamma):
    """Graphed spec output must equal graphed baseline output."""
    tokenizer, target, draft = pair
    base = GraphedGreedyDecoder(target, max_cache_len=MAX_CACHE)
    spec = GraphedSpecDecoder(target, draft, gamma=gamma,
                              max_cache_len=MAX_CACHE)

    mismatches = []
    for prompt in PROMPTS:
        ids = chat_ids(tokenizer, prompt)
        b_seq, _ = base.generate(ids, max_new_tokens=64,
                                 eos_id=tokenizer.eos_token_id)
        s_seq, stats = spec.generate(ids, max_new_tokens=64,
                                     eos_id=tokenizer.eos_token_id)
        b_text = tokenizer.decode(b_seq[0, ids.shape[1]:],
                                  skip_special_tokens=True)
        s_text = tokenizer.decode(s_seq[0, ids.shape[1]:],
                                  skip_special_tokens=True)
        if b_text != s_text:
            first = next((i for i, (x, y) in enumerate(zip(b_text, s_text))
                          if x != y), min(len(b_text), len(s_text)))
            mismatches.append((prompt[:40], first))
        print(f"  gamma={gamma} {'OK ' if b_text == s_text else 'DIFF'} "
              f"acc={stats['acceptance_rate']:.3f} "
              f"t/round={stats['tokens_per_round']:.2f} "
              f"{stats['tok_per_s']:.1f} tok/s")

    assert not mismatches, f"gamma={gamma} mismatches: {mismatches}"


def test_gamma_zero_matches_graphed_baseline(pair):
    """gamma=0 is a plain target step and must equal the graphed baseline."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)
    base = GraphedGreedyDecoder(target, max_cache_len=MAX_CACHE)
    spec = GraphedSpecDecoder(target, draft, gamma=0, max_cache_len=MAX_CACHE)

    b_seq, _ = base.generate(ids, max_new_tokens=48,
                             eos_id=tokenizer.eos_token_id)
    s_seq, stats = spec.generate(ids, max_new_tokens=48,
                                 eos_id=tokenizer.eos_token_id)
    b = tokenizer.decode(b_seq[0, ids.shape[1]:], skip_special_tokens=True)
    s = tokenizer.decode(s_seq[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"[gamma=0] proposed={stats['proposed']} identical={b == s}")
    assert stats["proposed"] == 0
    assert s == b


def test_graphed_spec_is_faster_than_graphed_baseline(pair):
    """Speculation must pay against the GRAPHED baseline, not the eager one."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPTS[2])      # structured, high acceptance
    base = GraphedGreedyDecoder(target, max_cache_len=MAX_CACHE)
    base.generate(ids, max_new_tokens=8, eos_id=None)
    _, b_stats = base.generate(ids, max_new_tokens=64, eos_id=None)

    best = (0, 0.0)
    for gamma in (1, 2, 3, 5):
        spec = GraphedSpecDecoder(target, draft, gamma=gamma,
                                  max_cache_len=MAX_CACHE)
        _, s_stats = spec.generate(ids, max_new_tokens=64, eos_id=None)
        ratio = s_stats["tok_per_s"] / b_stats["tok_per_s"]
        print(f"  gamma={gamma}: {s_stats['tok_per_s']:.1f} tok/s "
              f"({ratio:.2f}x graphed baseline, acc="
              f"{s_stats['acceptance_rate']:.3f})")
        if ratio > best[1]:
            best = (gamma, ratio)
    print(f"[INFO] graphed baseline {b_stats['tok_per_s']:.1f} tok/s; "
          f"best gamma={best[0]} at {best[1]:.2f}x")
    assert best[1] > 1.0, (
        f"no gamma beat the graphed baseline (best {best[1]:.2f}x); "
        "speculation does not pay in the graphed regime"
    )


def test_trace_requires_tokenizer(pair):
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)
    spec = GraphedSpecDecoder(target, draft, gamma=2, max_cache_len=MAX_CACHE)
    with pytest.raises(ValueError):
        spec.generate(ids, max_new_tokens=8, trace=[])


def test_trace_from_graphed_decoder(pair):
    """The graphed path must produce the same trace shape as the eager one."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)
    spec = GraphedSpecDecoder(target, draft, gamma=4, max_cache_len=MAX_CACHE)
    trace: list = []
    _, stats = spec.generate(ids, max_new_tokens=48,
                             eos_id=tokenizer.eos_token_id,
                             trace=trace, tokenizer=tokenizer)

    assert len(trace) == stats["rounds"]
    counts: dict[str, int] = {}
    for rec in trace:
        assert rec["accepted"] <= rec["gamma"]
        assert rec["rate"] == rec["accepted"] / rec["gamma"]
        for tok in rec["tokens"]:
            assert tok["origin"] in ("accept", "correct", "bonus")
            counts[tok["origin"]] = counts.get(tok["origin"], 0) + 1
            if tok["origin"] == "bonus":
                assert tok["draft_guessed"] is None
            else:
                assert isinstance(tok["draft_guessed"], str)
    print(f"[trace] rounds={len(trace)} origins={counts}")
    assert sum(counts.values()) == stats["tokens"]


def test_adaptive_drives_the_graphed_decoder(pair):
    """The controller must actually steer gamma and receive every round."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPTS[3])        # reasoning, lower acceptance
    controller = AdaptiveGamma(r=0.248, gmax=8, cooldown=2)
    spec = GraphedSpecDecoder(target, draft, gamma=controller.gamma,
                              max_cache_len=MAX_CACHE,
                              capture_gammas=(1, 2, 3, 4, 6, 8))
    _, stats = spec.generate(ids, max_new_tokens=64,
                             eos_id=tokenizer.eos_token_id,
                             adaptive=controller)

    gammas_used = [row[2] for row in controller.history]
    print(f"[adaptive] rounds={stats['rounds']} "
          f"history={len(controller.history)} "
          f"alpha={controller.alpha:.3f} gamma path={gammas_used}")
    assert len(controller.history) == stats["rounds"], (
        "the controller must be updated once per round")
    assert stats["rounds"] > 0
    assert all(0 <= g <= 8 for g in gammas_used)


def test_adaptive_output_is_still_lossless(pair):
    """Changing gamma mid-generation must not change the output."""
    tokenizer, target, draft = pair
    base = GraphedGreedyDecoder(target, max_cache_len=MAX_CACHE)
    controller = AdaptiveGamma(r=0.248, gmax=6, cooldown=1)
    spec = GraphedSpecDecoder(target, draft, gamma=controller.gamma,
                              max_cache_len=MAX_CACHE,
                              capture_gammas=(1, 2, 3, 4, 5, 6))

    for prompt in PROMPTS[:3]:
        ids = chat_ids(tokenizer, prompt)
        b_seq, _ = base.generate(ids, max_new_tokens=64,
                                 eos_id=tokenizer.eos_token_id)
        s_seq, stats = spec.generate(ids, max_new_tokens=64,
                                     eos_id=tokenizer.eos_token_id,
                                     adaptive=controller)
        b = tokenizer.decode(b_seq[0, ids.shape[1]:], skip_special_tokens=True)
        s = tokenizer.decode(s_seq[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"  adaptive {'OK ' if b == s else 'DIFF'} "
              f"rounds={stats['rounds']} "
              f"acc={stats['acceptance_rate']:.3f} "
              f"{stats['tok_per_s']:.1f} tok/s")
        assert s == b, f"adaptive output diverged on {prompt[:40]!r}"


def test_uncaptured_gamma_falls_back_to_eager(pair):
    """A gamma with no captured graph must still be correct, just slower."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)
    base = GraphedGreedyDecoder(target, max_cache_len=MAX_CACHE)
    b_seq, _ = base.generate(ids, max_new_tokens=48,
                             eos_id=tokenizer.eos_token_id)

    # Capture only gamma=2, then run at gamma=7: verify has no graph for width 8.
    spec = GraphedSpecDecoder(target, draft, gamma=2, max_cache_len=MAX_CACHE)
    assert 8 not in spec.target_graphs
    spec.gamma = 7
    s_seq, stats = spec.generate(ids, max_new_tokens=48,
                                 eos_id=tokenizer.eos_token_id)
    b = tokenizer.decode(b_seq[0, ids.shape[1]:], skip_special_tokens=True)
    s = tokenizer.decode(s_seq[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"[fallback] gamma=7 with no width-8 graph: identical={b == s}, "
          f"{stats['tok_per_s']:.1f} tok/s")
    assert s == b


def test_rejects_cache_overflow(pair):
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)
    spec = GraphedSpecDecoder(target, draft, gamma=3, max_cache_len=64)
    with pytest.raises(ValueError):
        spec.generate(ids, max_new_tokens=256, eos_id=None)
    print("[PASS] cache overflow raises")


if __name__ == "__main__":
    tokenizer, target, draft = _load()
    ids = chat_ids(tokenizer, PROMPT)
    dec = GraphedSpecDecoder(target, target, gamma=4, max_cache_len=MAX_CACHE)
    _, stats = dec.generate(ids, max_new_tokens=25, eos_id=None)
    ok = stats["acceptance_rate"] == 1.0 and stats["tokens_per_round"] == 5
    print("self-draft:", stats)
    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
