"""FastAPI + SSE front end for the decoder.

Models load once at startup.  /stream and /replay emit an IDENTICAL event
schema, which is what lets the frontend be built and demoed against recorded
traces while the engine is still being worked on.

Event schema, one `round` event per speculative round:

    {"type": "round", "round": int, "gamma": int, "accepted": int,
     "rate": float,
     "tokens": [{"text": str, "origin": "accept"|"correct"|"bonus",
                 "draft_guessed": str|None}, ...]}

Bracketed by:

    {"type": "start", "mode": str, "gamma": int, "prompt_tokens": int,
     "regime": "eager"|"graphed", ...}
    {"type": "done", "stats": {...}}
    {"type": "error", "message": str}

Run it:
    cd /d E:\\Crossover_point && conda activate crossover && ^
        python -m uvicorn src.server:app --port 8000

The generation loop is synchronous and CPU/GPU-bound, so it runs in a worker
thread while the endpoint polls its trace list.  That keeps spec_generate
untouched rather than rewriting proven code into a generator.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from src import compat
from src.adaptive import AdaptiveGamma
from src.baseline import baseline_generate
from src.models import DRAFT_PATH, TARGET_PATH, chat_ids, load_pair
from src.specdec import spec_generate

WEB_INDEX = "web/index.html"
DEFAULT_MAX_NEW_TOKENS = 128
POLL_SECONDS = 0.02

# The only global mutable state in the project, as Section 10 allows.
_models: dict[str, Any] = {}

# One GPU, one set of model singletons, and each generation owns the models' KV
# caches for its duration. Two concurrent /stream requests would interleave
# writes into the same cache and corrupt BOTH outputs, silently. Generations are
# therefore serialised: a second request waits rather than racing.
_generation_lock = threading.Lock()


def _load_once() -> tuple:
    """Load the model pair on first use.  Returns (tokenizer, target, draft)."""
    if "pair" not in _models:
        _models["pair"] = load_pair()
    return _models["pair"]


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    """Load the models before serving.

    FastAPI 0.141 deprecates @app.on_event, so this uses the lifespan API.
    Set CROSSOVER_LAZY_LOAD=1 to skip the load, which makes /health and /config
    testable without occupying ~6 GB of VRAM.
    """
    if os.environ.get("CROSSOVER_LAZY_LOAD") != "1":
        _load_once()
    yield
    _models.clear()


app = FastAPI(title="The Crossover Point", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    """Liveness plus whether the models are resident.  Cheap, for curl."""
    return JSONResponse({
        "status": "ok",
        "models_loaded": "pair" in _models,
        "target_path": TARGET_PATH,
        "draft_path": DRAFT_PATH,
    })


@app.get("/config")
def config() -> JSONResponse:
    """Environment provenance, the same dict embedded in every results file."""
    return JSONResponse({
        "compat": compat.describe(),
        "default_max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "measured": {
            "r_eager": 0.64,
            "r_graphed": 0.248,
            "baseline_tok_per_s_eager": 21.5,
            "baseline_tok_per_s_graphed": 38.9,
        },
    })


@app.get("/")
def index() -> Any:
    if not os.path.exists(WEB_INDEX):
        raise HTTPException(status_code=404, detail=f"{WEB_INDEX} not found")
    return FileResponse(WEB_INDEX)


def _run_generation(
    mode: str,
    prompt: str,
    gamma: int,
    max_new_tokens: int,
    trace: list,
    result: dict,
) -> None:
    """Generate in a worker thread, appending to `trace` as rounds complete.

    Holds _generation_lock for the whole generation: the models' caches are
    shared state and two generations at once would corrupt each other.
    """
    tokenizer, target, draft = _load_once()
    try:
        with _generation_lock:
            ids = chat_ids(tokenizer, prompt)
            result["prompt_tokens"] = ids.shape[1]
            if mode == "baseline":
                seq, stats = baseline_generate(
                    target, ids, max_new_tokens=max_new_tokens,
                    eos_id=tokenizer.eos_token_id)
            else:
                controller = (
                    AdaptiveGamma(r=0.64, gmax=12)
                    if mode == "adaptive" else None
                )
                seq, stats = spec_generate(
                    target, draft, ids, max_new_tokens=max_new_tokens,
                    gamma=gamma, eos_id=tokenizer.eos_token_id,
                    adaptive=controller, trace=trace, tokenizer=tokenizer)
                if controller is not None:
                    stats["controller_history"] = controller.history[-50:]
            result["stats"] = stats
            result["text"] = tokenizer.decode(
                seq[0, ids.shape[1]:], skip_special_tokens=True)
    except Exception as exc:                      # surfaced to the client
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished"] = True


@app.get("/stream")
async def stream(
    prompt: str = Query(..., min_length=1),
    mode: str = Query("spec", pattern="^(spec|baseline|adaptive)$"),
    gamma: int = Query(5, ge=0, le=16),
    max_new_tokens: int = Query(DEFAULT_MAX_NEW_TOKENS, ge=1, le=256),
) -> EventSourceResponse:
    """Stream one generation, one SSE event per speculative round."""
    trace: list = []
    result: dict[str, Any] = {}

    worker = threading.Thread(
        target=_run_generation,
        args=(mode, prompt, gamma, max_new_tokens, trace, result),
        daemon=True,
    )

    async def events() -> Any:
        worker.start()
        # Wait for the prompt to be tokenized so `start` can report its length.
        while "prompt_tokens" not in result and not result.get("finished"):
            await asyncio.sleep(POLL_SECONDS)

        yield {"event": "message", "data": json.dumps({
            "type": "start", "mode": mode, "gamma": gamma,
            "regime": "eager",
            "prompt_tokens": result.get("prompt_tokens"),
            "max_new_tokens": max_new_tokens,
        })}

        sent = 0
        while True:
            while sent < len(trace):
                record = dict(trace[sent])
                record["type"] = "round"
                sent += 1
                yield {"event": "message", "data": json.dumps(record)}
            if result.get("finished") and sent >= len(trace):
                break
            await asyncio.sleep(POLL_SECONDS)

        if "error" in result:
            yield {"event": "message", "data": json.dumps(
                {"type": "error", "message": result["error"]})}
        else:
            yield {"event": "message", "data": json.dumps({
                "type": "done", "stats": result.get("stats", {}),
                "text": result.get("text", ""),
            })}

    return EventSourceResponse(events())


@app.get("/replay")
async def replay(
    file: str = Query("results/mixed_trace.json"),
    speed: float = Query(1.0, gt=0.0, le=100.0),
) -> EventSourceResponse:
    """Replay a recorded trace with the SAME event schema as /stream.

    Lets the frontend be built against recorded data, and means a live demo
    cannot fail on a misbehaving GPU -- which is fine as long as you say so.
    """
    # Only serve traces from inside results/, so this cannot read arbitrary
    # files off the disk.
    safe_root = os.path.abspath("results")
    path = os.path.abspath(file)
    if not path.startswith(safe_root + os.sep) or not os.path.exists(path):
        raise HTTPException(
            status_code=400,
            detail=f"file must be an existing path under results/, got {file}")

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    rounds = payload.get("trace", payload if isinstance(payload, list) else [])

    async def events() -> Any:
        yield {"event": "message", "data": json.dumps({
            "type": "start", "mode": payload.get("mode", "replay"),
            "gamma": payload.get("gamma", 0),
            "regime": payload.get("regime", "recorded"),
            "prompt_tokens": payload.get("prompt_tokens"),
            "replay_of": os.path.basename(path),
        })}
        for record in rounds:
            out = dict(record)
            out["type"] = "round"
            yield {"event": "message", "data": json.dumps(out)}
            delay = record.get("seconds", 0.05) / speed
            await asyncio.sleep(min(max(delay, 0.0), 2.0))
        yield {"event": "message", "data": json.dumps({
            "type": "done", "stats": payload.get("stats", {}),
            "text": payload.get("text", ""),
        })}

    return EventSourceResponse(events())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
