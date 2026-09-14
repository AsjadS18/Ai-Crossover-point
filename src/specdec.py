"""Speculative decoding: the draft proposes, the target decides.

Every token in the output is one the target model itself chose, so the output
is byte-identical to greedy decoding from the target alone.  Draft quality
affects speed only, never correctness.

THE OFFSET IS COMPUTED, NEVER HARDCODED.  On the first round both caches are
empty and the verification window sits at a different offset than on later
rounds.  A hardcoded offset produces plausible-looking text that diverges
after roughly fifty tokens.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from src import compat
from src.baseline import forward_new


@torch.inference_mode()
def spec_generate(
    target: Any,
    draft: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    gamma: int = 5,
    eos_id: int | None = None,
    adaptive: Any | None = None,
    trace: list | None = None,
    tokenizer: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Speculatively decode from `input_ids` using `draft` to propose tokens.

    gamma is the number of tokens drafted per round; gamma == 0 means a plain
    single-token target step with no drafting.  If `adaptive` is given, its
    .gamma is read at the start of every round and .update(accepted, gamma) is
    called at the end.  If `trace` is a list, one record per round is appended
    to it, which requires `tokenizer`.

    Returns (sequence, stats).  sequence is [1, prompt + generated].  stats:
        tokens            int   generated tokens, excluding the prompt
        seconds           float wall clock for the generation loop
        tok_per_s         float tokens / seconds
        rounds            int   draft-verify rounds executed
        proposed          int   draft tokens offered across all rounds
        accepted          int   draft tokens the target agreed with
        acceptance_rate   float accepted / proposed
        tokens_per_round  float tokens / rounds
    """
    if trace is not None and tokenizer is None:
        raise ValueError("trace requires tokenizer= in order to record token text")

    target_cache = compat.make_cache(target)
    draft_cache = compat.make_cache(draft)

    seq = input_ids
    prompt_len = seq.shape[1]
    tgt_len = 0

    rounds = 0
    proposed = 0
    accepted_total = 0
    stop = False

    torch.cuda.synchronize()
    started = time.perf_counter()

    while seq.shape[1] - prompt_len < max_new_tokens and not stop:
        g = int(adaptive.gamma) if adaptive is not None else gamma
        L = seq.shape[1]

        # 1. DRAFT -- greedy, one token at a time, on the draft's own cache.
        draft_toks: list[int] = []
        for _ in range(g):
            d_len = compat.cache_length(draft_cache)
            d_logits = forward_new(draft, draft_cache, seq, d_len)
            d_tok = d_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, d_tok], dim=1)
            draft_toks.append(int(d_tok.item()))

        # 2. VERIFY -- one target forward over everything it has not seen.
        off = (L - 1) - tgt_len
        t_logits = forward_new(target, target_cache, seq, tgt_len)
        tgt_len = seq.shape[1]
        window = t_logits[:, off : off + g + 1, :]

        # 3. ACCEPT -- longest prefix the target agrees with.
        picks = window[0].argmax(dim=-1)
        n = 0
        while n < g and int(picks[n].item()) == draft_toks[n]:
            n += 1

        # 4. EMIT -- correction if n < g, free bonus token if n == g.
        next_tok = picks[n].view(1, 1)

        # 5. REBUILD
        seq = torch.cat([seq[:, : L + n], next_tok], dim=1)

        # 6. CROP -- both caches back to the sequence we actually kept.
        compat.crop_cache(target_cache, L + n)
        tgt_len = compat.cache_length(target_cache)
        compat.crop_cache(draft_cache, min(compat.cache_length(draft_cache), L + n))

        rounds += 1
        proposed += g
        accepted_total += n

        # Trim to the token budget, then look for EOS among this round's output.
        produced = seq.shape[1] - prompt_len
        if produced > max_new_tokens:
            seq = seq[:, : prompt_len + max_new_tokens]
            stop = True

        new_tokens = seq[0, L:].tolist()
        if eos_id is not None and eos_id in new_tokens:
            cut = new_tokens.index(eos_id) + 1
            seq = seq[:, : L + cut]
            new_tokens = new_tokens[:cut]
            stop = True

        if trace is not None:
            records = []
            for i, tok_id in enumerate(new_tokens):
                if i < n:
                    origin, guess = "accept", draft_toks[i]
                elif n < g:
                    origin, guess = "correct", draft_toks[n]
                else:
                    origin, guess = "bonus", None
                records.append({
                    "text": tokenizer.decode([tok_id]),
                    "origin": origin,
                    "draft_guessed": (
                        tokenizer.decode([guess]) if guess is not None else None
                    ),
                })
            trace.append({
                "round": rounds,
                "gamma": g,
                "accepted": n,
                "rate": (n / g) if g > 0 else 0.0,
                "tokens": records,
            })

        if adaptive is not None:
            adaptive.update(n, g)

    torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    tokens = seq.shape[1] - prompt_len
    stats = {
        "tokens": tokens,
        "seconds": seconds,
        "tok_per_s": tokens / seconds if seconds > 0 else 0.0,
        "rounds": rounds,
        "proposed": proposed,
        "accepted": accepted_total,
        "acceptance_rate": (accepted_total / proposed) if proposed > 0 else 0.0,
        "tokens_per_round": (tokens / rounds) if rounds > 0 else 0.0,
    }
    return seq, stats


if __name__ == "__main__":
    from src.models import chat_ids, load_pair

    tok, target, draft = load_pair()
    ids = chat_ids(tok, "Write a Python function that reverses a list in place.")

    trace: list = []
    seq, stats = spec_generate(
        target, draft, ids, max_new_tokens=96, gamma=5,
        eos_id=tok.eos_token_id, trace=trace, tokenizer=tok,
    )
    print(tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True))
    print("\nstats:", stats)
    print("first round trace:", trace[0])
