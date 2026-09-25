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
import time
from typing import Any

from lethe.config import settings
from lethe.llm.base import LLMProvider, count_tokens, make_llm
from lethe.models import LLMResult
from lethe.runtime.events import EventLog
from lethe.runtime.mission import load_mission, load_units
from lethe.store.base import make_store
from lethe.tools.nimble import MockNimble, make_nimble

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
    await log.emit("naive", "started", payload={"steps": steps, "units": [u.unit_id for u in units]})
    for step in range(steps):
        unit = units[step % len(units)]
        MockNimble.set_cycle(step // len(units))
        query = f"{unit.year} {unit.make} {unit.model} {unit.trim} for sale"
        try:
            results = await nimble.search(query, count=4)
            pages = []
            for r in results[:3]:
                page = r.content or (await nimble.extract(r.url)).text
                pages.append(f"URL: {r.url}\n{page}")
        except Exception as exc:
            await log.error("naive", "nimble", exc, unit_id=unit.unit_id)
            pages = []
        # append everything, forever
        history.append({"role": "user", "content": f"Step {step + 1}. Unit {unit.label} (ours ${unit.our_price:,.0f}). Observations:\n" + "\n\n".join(pages)})
        prompt_tokens = count_tokens(NAIVE_SYSTEM) + sum(count_tokens(m["content"]) for m in history)
        if prompt_tokens > max_tokens:
            await log.emit("naive", "context_overflow", unit_id=unit.unit_id, payload={"step": step + 1, "prompt_tokens": prompt_tokens, "limit": max_tokens})
            break
        res = await llm.complete(NAIVE_SYSTEM, history, meta={"payload": {"suggested": unit.our_price, "next_unit": unit.unit_id}})
        await log.emit("naive", "llm_call", unit_id=unit.unit_id, payload={"role": "naive", "step": step + 1, "history_messages": len(history)},
                       input_tokens=res.input_tokens, output_tokens=res.output_tokens, latency_ms=res.latency_ms, model=res.model)
        history.append({"role": "assistant", "content": res.text})
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
