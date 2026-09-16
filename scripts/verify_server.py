"""End-to-end check of every server endpoint, both regimes, real models.

Uses FastAPI's TestClient, so no separate server process or port is needed.
Loads the models once (~40 s, ~6 GB VRAM).

    python scripts/verify_server.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient

from src.server import app

PROMPT = "Write a Python function that reverses a list in place."
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    if not ok:
        failures.append(name)


def stream_events(client: TestClient, url: str) -> list[dict]:
    events = []
    with client.stream("GET", url) as resp:
        if resp.status_code != 200:
            return [{"type": "http", "status": resp.status_code}]
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


def summarise(events: list[dict]) -> tuple[dict, list[dict], dict]:
    start = next((e for e in events if e.get("type") == "start"), {})
    rounds = [e for e in events if e.get("type") == "round"]
    done = next((e for e in events if e.get("type") == "done"), {})
    return start, rounds, done


with TestClient(app) as client:
    r = client.get("/health").json()
    check("/health models loaded", r.get("models_loaded") is True, str(r))

    r = client.get("/config").json()
    check("/config provenance filled",
          r["compat"]["dtype_kwarg_path"] != "not-yet-called",
          f"default_regime={r.get('default_regime')} "
          f"default_gamma={r.get('default_gamma')}")

    r = client.get("/")
    check("/ serves the UI", r.status_code == 200 and "Crossover" in r.text)

    texts: dict[str, str] = {}
    for label, url in (
        ("graphed baseline",
         f"/stream?prompt={PROMPT}&mode=baseline&regime=graphed&max_new_tokens=64"),
        ("graphed spec gamma=2",
         f"/stream?prompt={PROMPT}&mode=spec&gamma=2&regime=graphed&max_new_tokens=64"),
        ("graphed adaptive",
         f"/stream?prompt={PROMPT}&mode=adaptive&regime=graphed&max_new_tokens=64"),
        ("eager baseline",
         f"/stream?prompt={PROMPT}&mode=baseline&regime=eager&max_new_tokens=64"),
        ("eager spec gamma=2",
         f"/stream?prompt={PROMPT}&mode=spec&gamma=2&regime=eager&max_new_tokens=64"),
    ):
        t0 = time.perf_counter()
        start, rounds, done = summarise(stream_events(client, url))
        took = time.perf_counter() - t0
        stats = done.get("stats", {})
        texts[label] = done.get("text", "")
        ok = bool(start) and bool(done) and start.get("regime") in label
        if "spec" in label or "adaptive" in label:
            origins_ok = all(t["origin"] in ("accept", "correct", "bonus")
                             for rd in rounds for t in rd["tokens"])
            ok = ok and len(rounds) > 0 and origins_ok
        gammas = sorted({rd["gamma"] for rd in rounds})
        check(f"/stream {label}", ok,
              f"{len(rounds)} rounds, {stats.get('tokens')} tokens, "
              f"{stats.get('tok_per_s', 0):.1f} tok/s, gammas={gammas}, "
              f"{took:.1f}s wall")

    check("graphed spec output == graphed baseline output",
          texts["graphed spec gamma=2"] == texts["graphed baseline"])
    check("graphed adaptive output == graphed baseline output",
          texts["graphed adaptive"] == texts["graphed baseline"])
    check("eager spec output == eager baseline output",
          texts["eager spec gamma=2"] == texts["eager baseline"])

    start, rounds, done = summarise(stream_events(
        client, "/replay?file=results/mixed_trace.json&speed=100"))
    check("/replay mixed_trace.json", bool(start) and len(rounds) > 0 and bool(done),
          f"{len(rounds)} rounds replayed")

    for url, expect in (("/stream?prompt=hi&mode=bogus", 422),
                        ("/stream?prompt=hi&regime=bogus", 422),
                        ("/stream?prompt=hi&gamma=99", 422),
                        ("/replay?file=../CLAUDE.md", 400),
                        ("/replay?file=results/nope.json", 400)):
        code = client.get(url).status_code
        check(f"reject {url}", code == expect, f"got {code}")

print()
print(f"VERDICT: {'PASS' if not failures else 'FAIL'} "
      f"({len(failures)} failures{': ' + ', '.join(failures) if failures else ''})")
sys.exit(1 if failures else 0)
