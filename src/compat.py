"""Compatibility firewall around unstable transformers APIs.

This is the ONLY module in this project permitted to call ``cache.crop()``
or ``from_pretrained()``.  It imports nothing from this project, by design:
when transformers changes a cache or loading API, exactly one file changes.

See CLAUDE.md Section 4 for the verified facts this module encodes.
"""

from __future__ import annotations

from typing import Any

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DynamicCache,
    StaticCache,
)

# Which code path each wrapper actually selected at runtime.  Reported by
# describe() so every results file records how the caches were built.
_cache_ctor_path: str = "not-yet-called"
_dtype_kwarg_path: str = "not-yet-called"
_rollback_path: str = "not-yet-called"


def make_cache(model: Any) -> DynamicCache:
    """Build a fresh, empty KV cache for ``model``.

    Returns a DynamicCache holding 0 tokens.  Prefers the config-aware
    constructor and falls back to the no-argument form if a future version
    drops the ``config`` keyword.
    """
    global _cache_ctor_path
    try:
        cache = DynamicCache(config=model.config)
        _cache_ctor_path = "DynamicCache(config=model.config)"
    except TypeError:
        cache = DynamicCache()
        _cache_ctor_path = "DynamicCache()"
    return cache


def cache_length(cache: DynamicCache) -> int:
    """Return the number of tokens currently stored in ``cache`` (tokens)."""
    return cache.get_seq_length()


def make_static_cache(model: Any, max_cache_len: int) -> StaticCache:
    """Build a pre-allocated, CUDA-graph-capturable KV cache for ``model``.

    Holds 0 tokens but reserves ``max_cache_len`` slots at fixed addresses,
    which is what lets a forward pass be captured once and replayed.  Generation
    must never exceed ``max_cache_len`` tokens in total.

    A StaticCache is NOT croppable; crop_cache() rolls it back instead.
    """
    return StaticCache(config=model.config, max_cache_len=max_cache_len)


def _rollback_static(cache: StaticCache, max_length: int) -> None:
    """Make a StaticCache believe it holds exactly ``max_length`` tokens.

    StaticLayer.update() writes at ``cumulative_length`` and IGNORES the
    cache_position kwarg, so discarding rejected tokens means setting that
    counter back.  It is a tensor at a pinned address (transformers marks it
    static precisely so cudagraphs work), so it must be mutated in place.

    The stale slots beyond ``max_length`` are deliberately left as they are:
    they sit past the causal horizon of every later query and are masked out.
    Zeroing them would not help anyway -- a zero key still contributes to the
    softmax, so zeroing is not masking.  Verified bit-exact against the
    DynamicCache crop path.
    """
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        counter = getattr(layer, "cumulative_length", None)
        if counter is None:
            raise RuntimeError(
                f"{type(layer).__name__} has no cumulative_length; cannot roll "
                "back a static cache on this transformers version"
            )
        if torch.is_tensor(counter):
            counter.fill_(max_length)
        else:
            layer.cumulative_length = max_length


def crop_cache(cache: Any, max_length: int) -> Any:
    """Truncate ``cache`` in place to exactly ``max_length`` tokens.

    Dispatches on what the cache supports, so callers never need to know which
    kind they hold:
      - croppable (DynamicCache) -> cache.crop() with the negative convention
      - static (StaticCache)     -> roll cumulative_length back in place

    A no-op when the cache is already at or below ``max_length``.  Returns the
    same cache object.

    Raises RuntimeError if the cache supports neither route, or if the resulting
    length is not exactly ``max_length``.
    """
    global _rollback_path

    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")

    current = cache_length(cache)
    if current <= max_length:
        return cache

    if getattr(cache, "is_croppable", False):
        # NEGATIVE argument = number of tokens to REMOVE.  Positive values take
        # a deprecated legacy path (absolute length) and are removed in
        # transformers 5.18.  This is the only crop() call site in the project.
        cache.crop(-(current - max_length))
        _rollback_path = "DynamicCache.crop(negative)"
    elif isinstance(cache, StaticCache) or hasattr(cache, "layers"):
        _rollback_static(cache, max_length)
        _rollback_path = "StaticCache.cumulative_length rollback"
    else:
        raise RuntimeError(
            f"cache of type {type(cache).__name__} is neither croppable nor "
            "static; truncating it would silently corrupt decoder state"
        )

    after = cache_length(cache)
    if after != max_length:
        raise RuntimeError(
            f"truncation produced length {after}, expected {max_length} "
            f"(was {current}); cache API behaviour may have changed"
        )
    return cache


def load_model(path: str, quantize: bool = False, device: str = "cuda") -> Any:
    """Load a causal LM from ``path`` in eval mode on ``device``.

    quantize=True  -> 4-bit NF4 via bitsandbytes, float16 compute (~5 GB for 7B)
    quantize=False -> plain float16 (~1.2 GB for 0.5B)

    Returns the model, already ``.eval()``-ed.
    """
    global _dtype_kwarg_path

    kwargs: dict[str, Any] = {"device_map": device}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, **kwargs
        )
        _dtype_kwarg_path = "dtype="
    except TypeError:
        # Forward compatibility only: transformers v5 removed torch_dtype=.
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, **kwargs
        )
        _dtype_kwarg_path = "torch_dtype="

    model.eval()
    return model


class GraphedForward:
    """One CUDA-graph-captured forward pass of fixed token width.

    On this machine an eager forward is host-bound: ~940 kernel launches at
    ~12.9 us each for ~5 ms of GPU work.  A captured graph replays the whole
    sequence with a single launch, which measured 7.3x faster for the 0.5B and
    2.6x for the 4-bit 7B, with token-identical output.

    The width is fixed at capture: 1 for a decode step, gamma+1 for a
    speculative verify.  Capture requires the cache to be static, since the
    graph bakes in tensor addresses.

    IMPORTANT: capturing advances nothing, but the warmup passes do write to the
    cache.  reset_and_prefill() must be used to put the cache into a known state
    before the first real replay.
    """

    def __init__(
        self,
        model: Any,
        cache: StaticCache,
        width: int,
        device: str = "cuda",
        warmup: int = 3,
    ) -> None:
        self.model = model
        self.cache = cache
        self.width = width
        self.device = device
        self.tokens = torch.zeros(1, width, dtype=torch.long, device=device)
        # cache_position MUST be an explicit static buffer. Left implicit, part
        # of the mask construction is baked in at capture and the decode
        # diverges after ~35 tokens; passed explicitly and updated in place, 256
        # tokens stay identical.
        self.positions = torch.zeros(width, dtype=torch.long, device=device)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._out: Any = None
        self._capture(warmup)

    def _step(self) -> Any:
        return self.model(
            input_ids=self.tokens,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.positions,
        )

    def _capture(self, warmup: int) -> None:
        """Warm up on a side stream, then record the kernel sequence."""
        before = cache_length(self.cache)
        self.positions.copy_(
            torch.arange(before, before + self.width, device=self.device)
        )
        with torch.inference_mode():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._out = self._step()
            torch.cuda.synchronize()
        self._graph = graph
        # The warmup wrote `warmup * width` tokens; undo that bookkeeping so the
        # caller sees the length it started with.
        crop_cache(self.cache, before)

    def replay(
        self, token_ids: torch.Tensor, start_position: int | None = None
    ) -> torch.Tensor:
        """Run the captured pass over ``token_ids`` of shape [1, width].

        ``start_position`` is the absolute cache position of the first token.
        It defaults to the cache's current length, which is what a plain
        sequential decode wants.

        Returns the logits tensor, shape [1, width, vocab].  The tensor is
        REUSED on every replay, so clone it if it must outlive the next call.
        """
        if token_ids.shape[-1] != self.width:
            raise ValueError(
                f"graph was captured at width {self.width}, got "
                f"{token_ids.shape[-1]}; capture a separate graph per width"
            )
        if start_position is None:
            start_position = cache_length(self.cache)
        self.tokens.copy_(token_ids)
        self.positions.copy_(
            torch.arange(
                start_position, start_position + self.width, device=self.device
            )
        )
        self._graph.replay()
        return self._out.logits


def describe() -> dict[str, Any]:
    """Return environment provenance for embedding in results metadata."""
    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cache_ctor_path": _cache_ctor_path,
        "dtype_kwarg_path": _dtype_kwarg_path,
        "rollback_path": _rollback_path,
    }


if __name__ == "__main__":
    m = load_model("models/draft-0.5b", quantize=False)
    c = make_cache(m)
    print("describe():", describe())
    print("empty cache length:", cache_length(c))
    print("is_croppable:", getattr(c, "is_croppable", "ATTR MISSING"))
