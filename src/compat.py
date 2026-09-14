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
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache

# Which code path each wrapper actually selected at runtime.  Reported by
# describe() so every results file records how the caches were built.
_cache_ctor_path: str = "not-yet-called"
_dtype_kwarg_path: str = "not-yet-called"


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


def crop_cache(cache: DynamicCache, max_length: int) -> DynamicCache:
    """Truncate ``cache`` in place to exactly ``max_length`` tokens.

    A no-op when the cache is already at or below ``max_length`` -- crop() is
    not called at all in that case.  Returns the same cache object.

    Raises RuntimeError if the cache cannot be cropped losslessly, or if the
    resulting length is not exactly ``max_length``.
    """
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")

    current = cache_length(cache)
    if current <= max_length:
        return cache

    if not getattr(cache, "is_croppable", False):
        raise RuntimeError(
            f"cache of type {type(cache).__name__} is not croppable; "
            "cropping it would silently corrupt decoder state"
        )

    # NEGATIVE argument = number of tokens to REMOVE.  Positive values take a
    # deprecated legacy path (absolute length) and are removed in transformers
    # 5.18.  This is the only crop() call site in the project.
    cache.crop(-(current - max_length))

    after = cache_length(cache)
    if after != max_length:
        raise RuntimeError(
            f"crop produced length {after}, expected {max_length} "
            f"(was {current}); crop sign convention may have changed"
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


def describe() -> dict[str, Any]:
    """Return environment provenance for embedding in results metadata."""
    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cache_ctor_path": _cache_ctor_path,
        "dtype_kwarg_path": _dtype_kwarg_path,
    }


if __name__ == "__main__":
    m = load_model("models/draft-0.5b", quantize=False)
    c = make_cache(m)
    print("describe():", describe())
    print("empty cache length:", cache_length(c))
    print("is_croppable:", getattr(c, "is_croppable", "ATTR MISSING"))
