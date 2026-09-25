"""NAIVE BASELINE: same mission, same tools, classic append-everything message history.

Every observation (full page text!) is appended to `history` and the whole history is sent
on every LLM call. Tokens per step are logged to the store under agent="naive" so the
dashboard can draw Lethe (flat) vs naive (climbing). It stops after --steps steps or when a
single call would exceed --max-tokens (which is the point).

    python -m baseline.naive_agent --steps 24
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

from lethe.config import settings
from lethe.llm.base import LLMProvider, count_tokens, make_llm
from lethe.models import LLMResult
from lethe.runtime.events import EventLog
from lethe.runtime.mission import load_mission, load_units
from lethe.store.base import make_store
from lethe.tools.nimble import MockNimble, make_nimble
from lethe.agents.scout import _truncate_tokens

# Even a naive agent truncates pages; this keeps step 1 under the provider's request cap so the
# climb is visible for several steps before the wall (Groq free tier refuses > ~8k tokens/request).
PAGE_TOKENS = int(os.environ.get("NAIVE_PAGE_TOKENS", "350"))

NAIVE_SYSTEM = ("You are a pricing agent. Use the whole conversation so far to keep track of listings and prices. "
                "Reply with JSON {\"thought\": \"...\", \"recommendation\": <number>, \"next_unit\": \"...\"}")


class UnboundedLLM:
    """Wraps a provider but bypasses the per-call budget (that's what makes it naive)."""

    def __init__(self, inner: LLMProvider):
        self.inner = inner

    async def complete(self, system: str, messages: list[dict[str, str]], *, meta: dict[str, Any]) -> LLMResult:
        # flatten the ever-growing history into one user turn (works for every provider)
        user = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        t0 = time.perf_counter()
        res = await self.inner._complete(system, user, role="naive", meta=meta, max_tokens=300, temperature=0.2, json_mode=True)
        res.latency_ms = res.latency_ms or int((time.perf_counter() - t0) * 1000)
        res.input_tokens = res.input_tokens or (count_tokens(system) + count_tokens(user) + 8)
        res.output_tokens = res.output_tokens or count_tokens(res.text)
        return res


async def main(steps: int, max_tokens: int) -> None:
    mission = load_mission(settings.mission_file)
    units = list(load_units(mission, settings.units_limit).values())
    store = make_store(settings)
    log = EventLog(store, run_id=f"naive-{settings.run_id}", agent="naive")
    llm = UnboundedLLM(make_llm(settings, purpose="chief"))
    nimble = make_nimble(settings)
    history: list[dict[str, str]] = []
    print(f"naive baseline: {steps} steps over {len(units)} units, appending everything to one history. "
          f"Each step = live Nimble fetch + one LLM call on the whole history (rate-limited, so steps take 10-60s).")
    await log.emit("naive", "started", payload={"steps": steps, "units": [u.unit_id for u in units]})
    for step in range(steps):
        unit = units[step % len(units)]
        print(f"step {step + 1}/{steps}: {unit.label} ... history so far: {sum(count_tokens(m['content']) for m in history)} tokens", flush=True)
        MockNimble.set_cycle(step // len(units))
        query = f"{unit.year} {unit.make} {unit.model} {unit.trim} for sale"
        try:
            print("   fetching live pages via Nimble (10-30s) ...", flush=True)
            results = await nimble.search(query, count=4)
            pages = []
            for r in results[:3]:
                page = r.content or (await nimble.extract(r.url)).text
                pages.append(f"URL: {r.url}\n{_truncate_tokens(page, PAGE_TOKENS)}")
            print(f"   {len(pages)} pages appended; calling the model with the whole history (may wait for rate limit) ...", flush=True)
        except Exception as exc:
            await log.error("naive", "nimble", exc, unit_id=unit.unit_id)
            pages = []
        # append everything, forever
        history.append({"role": "user", "content": f"Step {step + 1}. Unit {unit.label} (ours ${unit.our_price:,.0f}). Observations:\n" + "\n\n".join(pages)})
        prompt_tokens = count_tokens(NAIVE_SYSTEM) + sum(count_tokens(m["content"]) for m in history)
        if prompt_tokens > max_tokens:
            await log.emit("naive", "context_overflow", unit_id=unit.unit_id, payload={"step": step + 1, "prompt_tokens": prompt_tokens, "limit": max_tokens})
            break
        try:
            res = await llm.complete(NAIVE_SYSTEM, history, meta={"payload": {"suggested": unit.our_price, "next_unit": unit.unit_id}})
        except Exception as exc:  # e.g. provider refuses the oversized prompt: that *is* the wall
            # record the refused attempt so the chart shows where the climb ended
            await log.emit("naive", "llm_call", unit_id=unit.unit_id, payload={"role": "naive", "step": step + 1, "refused": True},
                           input_tokens=prompt_tokens, output_tokens=0, latency_ms=0, model="refused")
            await log.emit("naive", "context_overflow", unit_id=unit.unit_id,
                           payload={"step": step + 1, "prompt_tokens": prompt_tokens, "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"stopped at step {step + 1}: the provider refused a {prompt_tokens}-token prompt ({type(exc).__name__}). That's the wall.")
            break
        await log.emit("naive", "llm_call", unit_id=unit.unit_id, payload={"role": "naive", "step": step + 1, "history_messages": len(history)},
                       input_tokens=res.input_tokens, output_tokens=res.output_tokens, latency_ms=res.latency_ms, model=res.model)
        history.append({"role": "assistant", "content": res.text})
        print(f"   done: prompt was {res.input_tokens} tokens", flush=True)
        await store.flush()
    await log.emit("naive", "finished", payload={"steps": len([m for m in history if m["role"] == "assistant"]),
                                                  "final_prompt_tokens": count_tokens(NAIVE_SYSTEM) + sum(count_tokens(m["content"]) for m in history)})
    await store.close()
    await nimble.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=120_000, help="stop when a single prompt would exceed this")
    a = ap.parse_args()
    asyncio.run(main(a.steps, a.max_tokens))
