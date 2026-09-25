"""Event logger (hard rule #6: everything is an event) + budget-aware LLM call helper."""
from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError
from rich.console import Console

from ..llm.base import LLMProvider, TokenBudgetExceeded
from ..models import Event, LLMResult, extract_json
from ..store.base import StateStore

console = Console()


class EventLog:
    def __init__(self, store: StateStore, run_id: str, agent: str = "lethe", quiet: bool = False):
        self.store = store
        self.run_id = run_id
        self.agent = agent
        self.quiet = quiet
        self.max_input_tokens = 0
        self.n_llm_calls = 0

    async def emit(self, component: str, event_type: str, *, unit_id: str = "", payload: dict[str, Any] | None = None,
                   input_tokens: int = 0, output_tokens: int = 0, latency_ms: int = 0, model: str = "") -> None:
        ev = Event(run_id=self.run_id, agent=self.agent, component=component, event_type=event_type, unit_id=unit_id,
                   payload=payload or {}, input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=latency_ms, model=model)
        await self.store.append_events([ev])
        if not self.quiet:
            self._print(ev)

    def _print(self, ev: Event) -> None:
        colors = {"chief": "cyan", "scout": "green", "librarian": "magenta", "auditor": "yellow", "gc": "red", "system": "white", "naive": "grey50"}
        c = colors.get(ev.component, "white")
        extra = f" [dim]{ev.input_tokens}in/{ev.output_tokens}out {ev.latency_ms}ms[/dim]" if ev.event_type == "llm_call" else ""
        p = json.dumps(ev.payload, default=str)
        if len(p) > 160:
            p = p[:157] + "..."
        console.print(f"[dim]{ev.ts[11:19]}[/dim] [{c}]{ev.component:<9}[/{c}] {ev.event_type:<18} {ev.unit_id:<4}{extra} [dim]{p}[/dim]")

    async def error(self, component: str, where: str, exc: BaseException, *, unit_id: str = "") -> None:
        await self.emit(component, "error", unit_id=unit_id, payload={"where": where, "error": f"{type(exc).__name__}: {exc}"[:400]})

    async def llm_json(self, llm: LLMProvider, component: str, role: str, system: str, user: str, *,
                       meta: dict[str, Any] | None = None, unit_id: str = "", max_tokens: int = 900,
                       retries: int = 2, validator: type | None = None) -> Any:
        """Call the LLM, log the call with real token counts, parse JSON (retry on failure)."""
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                res: LLMResult = await llm.complete(system, user, role=role, meta=meta, max_tokens=max_tokens)
            except TokenBudgetExceeded as exc:
                await self.emit(component, "budget_exceeded", unit_id=unit_id, payload={"role": role, "error": str(exc)})
                raise
            self.n_llm_calls += 1
            self.max_input_tokens = max(self.max_input_tokens, res.input_tokens)
            await self.emit(component, "llm_call", unit_id=unit_id,
                            payload={"role": role, "attempt": attempt, "provider": res.provider, "chars_out": len(res.text)},
                            input_tokens=res.input_tokens, output_tokens=res.output_tokens, latency_ms=res.latency_ms, model=res.model)
            try:
                data = extract_json(res.text)
                if validator is not None:
                    data = validator.model_validate(data)
                return data
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                last_exc = exc
                await self.emit(component, "parse_retry", unit_id=unit_id, payload={"role": role, "attempt": attempt, "error": str(exc)[:200]})
                user = user + "\n\nYour previous reply was not valid JSON for the schema. Reply with ONLY the JSON object."
        raise RuntimeError(f"{role}: LLM output invalid after {retries + 1} attempts: {last_exc}")
