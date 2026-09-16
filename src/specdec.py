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


def trace_tokens(
    new_tokens: list[int],
    draft_toks: list[int],
    accepted: int,
    gamma: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Label one round's emitted tokens with where each came from.

    ``new_tokens`` are the ids the round actually kept: the ``accepted`` draft
    tokens followed by the target's own token.  Origins:
        accept   the draft guessed it and the target agreed
        correct  the target overrode the draft's guess
        bonus    the target's free token after a fully accepted round

    A pure function so it can be tested without a GPU.  Deliberately duplicated
    rather than refactored out of spec_generate, which is already proven.
    """
    records: list[dict[str, Any]] = []
    for i, tok_id in enumerate(new_tokens):
        if i < accepted:
            origin, guess = "accept", draft_toks[i]
        elif accepted < gamma:
            origin, guess = "correct", draft_toks[accepted]
        else:
            origin, guess = "bonus", None
        records.append({
            "text": tokenizer.decode([tok_id]),
            "origin": origin,
            "draft_guessed": (
                tokenizer.decode([guess]) if guess is not None else None
            ),
        })
    return records


class GraphedSpecDecoder:
    """Speculative decoding with CUDA-graph-replayed forwards.

    Same algorithm as spec_generate, same offset arithmetic, same rollback --
    only the forward passes change.  Which widths are worth capturing falls out
    of the algorithm rather than being guessed:

      VERIFY: after a round, tgt_len = L + n, and the next round has
        L' = L + n + 1, so the verify feeds (L' + gamma) - (L' - 1) = gamma + 1
        tokens.  Constant from round 2 onward, so ONE target graph per gamma.
      DRAFT: each round's first draft feed is 1 token, or 2 when the previous
        round accepted all gamma (the last drafted token was never fed).
      Round 1 feeds the whole prompt to both models, a width that varies per
        prompt, so it runs eagerly.

    Anything without a captured graph falls back to eager, so correctness never
    depends on a width having been anticipated.

    Rollback goes through compat.crop_cache, which dispatches to the static
    cumulative_length path.  A StaticCache cannot be cropped.
    """

    def __init__(
        self,
        target: Any,
        draft: Any,
        gamma: int,
        max_cache_len: int = 512,
        device: str = "cuda",
        capture_prompt_len: int = 32,
        capture_gammas: tuple[int, ...] | None = None,
    ) -> None:
        """Build the decoder and capture its graphs.

        ``capture_gammas`` pre-captures verify graphs for more than one gamma,
        which is what an adaptive controller needs: it changes gamma between
        rounds, and each gamma implies a different verify width. Any gamma
        without a captured graph still works, just eagerly, so the controller is
        never restricted to the captured set -- only slower outside it.

        Each extra gamma costs one captured target graph, so keep the set small.
        """
        self.target = target
        self.draft = draft
        self.gamma = gamma
        self.device = device
        self.max_cache_len = max_cache_len
        self.target_cache = compat.make_static_cache(target, max_cache_len)
        self.draft_cache = compat.make_static_cache(draft, max_cache_len)
        self.target_graphs: dict[int, compat.GraphedForward] = {}
        self.draft_graphs: dict[int, compat.GraphedForward] = {}

        wanted = set(capture_gammas or ())
        if gamma > 0:
            wanted.add(gamma)
        widths = tuple(sorted(g + 1 for g in wanted if g > 0))
        if widths:
            self._capture(capture_prompt_len, widths)

    def _capture(self, prompt_len: int, target_widths: tuple[int, ...]) -> None:
        """Capture the graphs on a throwaway prefill.

        Never captured inside a real generation: the warmup passes write into
        the cache, and a run starting from that residue differs from one
        starting clean (Phase 9b bug 2).
        """
        dummy = torch.zeros(1, prompt_len, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            for cache, model, graphs, widths in (
                (self.target_cache, self.target, self.target_graphs,
                 target_widths),
                (self.draft_cache, self.draft, self.draft_graphs, (1, 2)),
            ):
                for width in widths:
                    cache.reset()
                    model(input_ids=dummy, past_key_values=cache, use_cache=True)
                    graphs[width] = compat.GraphedForward(
                        model, cache, width=width, device=self.device)
                cache.reset()

    def _forward(
        self,
        model: Any,
        cache: Any,
        graphs: dict[int, compat.GraphedForward],
        seq: torch.Tensor,
        cached_len: int,
    ) -> torch.Tensor:
        """Forward over seq[:, cached_len:], replayed if a graph fits the width."""
        width = seq.shape[1] - cached_len
        graph = graphs.get(width)
        if graph is not None:
            return graph.replay(seq[:, cached_len:], start_position=cached_len)
        return model(
            input_ids=seq[:, cached_len:], past_key_values=cache, use_cache=True
        ).logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_id: int | None = None,
        adaptive: Any | None = None,
        trace: list | None = None,
        tokenizer: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Speculatively decode.  Same contract and stats as spec_generate.

        ``adaptive`` is read at the start of every round and updated at the end,
        exactly as in spec_generate.  A gamma it picks without a captured graph
        falls back to eager verification for that round.
        """
        if trace is not None and tokenizer is None:
            raise ValueError("trace requires tokenizer= in order to record text")

        prompt_len = input_ids.shape[1]
        headroom = max(self.gamma, getattr(adaptive, "gmax", 0)) + 1
        if prompt_len + max_new_tokens + headroom > self.max_cache_len:
            raise ValueError(
                f"prompt {prompt_len} + {max_new_tokens} new + gamma headroom "
                f"exceeds max_cache_len {self.max_cache_len}"
            )

        self.target_cache.reset()
        self.draft_cache.reset()

        seq = input_ids
        tgt_len = 0
        drf_len = 0
        rounds = proposed = accepted_total = 0
        stop = False

        torch.cuda.synchronize()
        started = time.perf_counter()

        while seq.shape[1] - prompt_len < max_new_tokens and not stop:
            g = int(adaptive.gamma) if adaptive is not None else self.gamma
            L = seq.shape[1]

            draft_toks: list[int] = []
            for _ in range(g):
                # drf_len is tracked in Python rather than read back from the
                # cache: reading a StaticCache's length costs a GPU sync.
                d_logits = self._forward(
                    self.draft, self.draft_cache, self.draft_graphs, seq, drf_len)
                drf_len = seq.shape[1]
                d_tok = d_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                seq = torch.cat([seq, d_tok], dim=1)
                draft_toks.append(int(d_tok.item()))

            off = (L - 1) - tgt_len
            t_logits = self._forward(
                self.target, self.target_cache, self.target_graphs, seq, tgt_len)
            tgt_len = seq.shape[1]
            window = t_logits[:, off : off + g + 1, :]

            picks = window[0].argmax(dim=-1)
            n = 0
            while n < g and int(picks[n].item()) == draft_toks[n]:
                n += 1
            next_tok = picks[n].view(1, 1)
            seq = torch.cat([seq[:, : L + n], next_tok], dim=1)

            compat.crop_cache(self.target_cache, L + n)
            tgt_len = L + n
            drf_len = min(drf_len, L + n)
            compat.crop_cache(self.draft_cache, drf_len)

            rounds += 1
            proposed += g
            accepted_total += n

            if seq.shape[1] - prompt_len > max_new_tokens:
                seq = seq[:, : prompt_len + max_new_tokens]
                stop = True
            new_tokens = seq[0, L:].tolist()
            if eos_id is not None and eos_id in new_tokens:
                cut = new_tokens.index(eos_id) + 1
                seq = seq[:, : L + cut]
                # Truncate the trace view too, so it records only tokens kept.
                new_tokens = new_tokens[:cut]
                stop = True

            if trace is not None:
                trace.append({
                    "round": rounds,
                    "gamma": g,
                    "accepted": n,
                    "rate": (n / g) if g > 0 else 0.0,
                    "tokens": trace_tokens(
                        new_tokens, draft_toks, n, g, tokenizer),
                })

            if adaptive is not None:
                adaptive.update(n, g)

        torch.cuda.synchronize()
        seconds = time.perf_counter() - started

        tokens = seq.shape[1] - prompt_len
        return seq, {
            "tokens": tokens,
            "seconds": seconds,
            "tok_per_s": tokens / seconds if seconds > 0 else 0.0,
            "rounds": rounds,
            "proposed": proposed,
            "accepted": accepted_total,
            "acceptance_rate": (accepted_total / proposed) if proposed else 0.0,
            "tokens_per_round": (tokens / rounds) if rounds else 0.0,
            "graphed": True,
        }


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
