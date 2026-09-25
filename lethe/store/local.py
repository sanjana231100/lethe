"""Local JSONL store: the MOCK=1 stand-in for RawTree.

Files: <state_dir>/events.jsonl, <state_dir>/facts.jsonl (append-only, like the datasources).
Every `query(pipe)` here re-implements the matching RawTree query in Python so the dashboard
and the resume path behave identically offline.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..models import Event, Fact, Ledger, now, parse_ts
from .base import StateStore


class LocalStore(StateStore):
    name = "local-jsonl"

    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.facts_path = self.dir / "facts.jsonl"

    # ------------------------------------------------------------------ writes
    async def append_events(self, events: list[Event]) -> None:
        with self.events_path.open("a") as f:
            for e in events:
                f.write(json.dumps(e.to_row(), default=str) + "\n")

    async def append_facts(self, facts: list[Fact]) -> None:
        with self.facts_path.open("a") as f:
            for fact in facts:
                f.write(json.dumps(fact.model_dump(), default=str) + "\n")

    # ------------------------------------------------------------------ raw reads
    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def _events(self, run_id: str | None = None) -> list[dict[str, Any]]:
        rows = self._read(self.events_path)
        if run_id:
            rows = [r for r in rows if r.get("run_id") == run_id]
        return rows

    def _latest_facts(self, run_id: str) -> dict[str, Fact]:
        """Latest version per fact_id (mirrors argMax(..., ts) in the pipe)."""
        latest: dict[str, Fact] = {}
        for r in self._read(self.facts_path):
            if r.get("run_id", run_id) != run_id:
                continue
            f = Fact(**r)
            cur = latest.get(f.fact_id)
            if cur is None or f.ts >= cur.ts:
                latest[f.fact_id] = f
        return latest

    # ------------------------------------------------------------------ pipes
    async def current_ledger(self, run_id: str) -> Ledger | None:
        snaps = [r for r in self._events(run_id) if r["event_type"] == "ledger_snapshot"]
        if not snaps:
            return None
        payload = json.loads(snaps[-1]["payload"])
        return Ledger(**payload["ledger"])

    async def active_facts(self, run_id: str) -> list[Fact]:
        t = now()
        return [f for f in self._latest_facts(run_id).values() if f.status == "active" and not f.is_expired(t)]

    async def query(self, pipe: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        run_id = params.get("run_id", "")
        fn = getattr(self, f"_pipe_{pipe}", None)
        if fn is None:
            raise KeyError(f"unknown pipe {pipe}")
        return fn(run_id, params)

    # --- pipe: current_ledger
    def _pipe_current_ledger(self, run_id: str, p: dict) -> list[dict]:
        snaps = [r for r in self._events(run_id) if r["event_type"] == "ledger_snapshot"]
        return [{"ts": snaps[-1]["ts"], "payload": snaps[-1]["payload"]}] if snaps else []

    # --- pipe: active_facts
    def _pipe_active_facts(self, run_id: str, p: dict) -> list[dict]:
        t = now()
        return [f.model_dump() for f in self._latest_facts(run_id).values() if f.status == "active" and not f.is_expired(t)]

    # --- pipe: board (unit cards derived from the latest ledger + active facts)
    def _pipe_board(self, run_id: str, p: dict) -> list[dict]:
        snaps = self._pipe_current_ledger(run_id, p)
        if not snaps:
            return []
        ledger = json.loads(snaps[0]["payload"])["ledger"]
        return [
            {"unit_id": u["unit_id"], "label": u["label"], "status": u["status"], "our_price": u["our_price"],
             "current_reco": u["current_reco"], "confidence": u["confidence"], "comps_count": u["comps_count"],
             "last_checked": u["last_checked"], "strategy": u["strategy"], "reasoning": u.get("reasoning", "")}
            for u in ledger["units"]
        ]

    # --- pipe: token_metrics (input tokens per LLM call over time, per agent)
    def _pipe_token_metrics(self, run_id: str, p: dict) -> list[dict]:
        out = []
        counters: dict[str, int] = defaultdict(int)
        for r in self._read(self.events_path):
            if r["event_type"] != "llm_call":
                continue
            if run_id and r["run_id"] != run_id and r["agent"] != "naive":
                continue
            counters[r["agent"]] += 1
            out.append({"ts": r["ts"], "agent": r["agent"], "component": r["component"], "step": counters[r["agent"]],
                        "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                        "latency_ms": r["latency_ms"], "model": r["model"]})
        return out[-int(p.get("limit", 600)):]

    # --- pipe: fact_lifecycle (counts per minute per transition)
    def _pipe_fact_lifecycle(self, run_id: str, p: dict) -> list[dict]:
        buckets: dict[tuple[str, str], int] = defaultdict(int)
        kinds = {"fact_created", "fact_promoted", "fact_expired", "fact_evicted", "fact_contradicted", "fact_superseded"}
        for r in self._events(run_id):
            if r["event_type"] in kinds:
                minute = r["ts"][:16]
                buckets[(minute, r["event_type"])] += 1
        return [{"minute": m, "event_type": k, "n": n} for (m, k), n in sorted(buckets.items())]

    # --- pipe: memory_feed (live feed of promotions / evictions / contradictions)
    def _pipe_memory_feed(self, run_id: str, p: dict) -> list[dict]:
        kinds = {"fact_promoted", "fact_expired", "fact_evicted", "fact_contradicted", "fact_superseded"}
        rows = [r for r in self._events(run_id) if r["event_type"] in kinds]
        return [{"ts": r["ts"], "event_type": r["event_type"], "unit_id": r["unit_id"], "payload": r["payload"]}
                for r in rows[-int(p.get("limit", 40)):]][::-1]

    # --- pipe: lessons
    def _pipe_lessons(self, run_id: str, p: dict) -> list[dict]:
        rows = [r for r in self._events(run_id) if r["event_type"] in {"lesson", "strategy_changed", "self_correction"}]
        return [{"ts": r["ts"], "event_type": r["event_type"], "unit_id": r["unit_id"], "component": r["component"],
                 "payload": r["payload"]} for r in rows[-int(p.get("limit", 30)):]][::-1]

    # --- pipe: run_stats
    def _pipe_run_stats(self, run_id: str, p: dict) -> list[dict]:
        rows = self._events(run_id)
        if not rows:
            return []
        llm = [r for r in rows if r["event_type"] == "llm_call" and r["agent"] == "lethe"]
        pages = sum(json.loads(r["payload"]).get("pages_read", 0) for r in rows if r["event_type"] == "scout_done")
        cycles = sum(1 for r in rows if r["event_type"] == "cycle_done")
        steps = sum(1 for r in rows if r["event_type"] in {"chief_plan", "chief_update", "scout_done", "audit_done", "librarian_done", "gc_sweep"})
        in_tok = sum(r["input_tokens"] for r in llm)
        out_tok = sum(r["output_tokens"] for r in llm)
        # rough blended cost estimate (USD / 1M tokens) - documented in README
        cost = in_tok / 1e6 * 0.40 + out_tok / 1e6 * 1.60
        resumes = sum(1 for r in rows if r["event_type"] == "resumed")
        return [{
            "first_ts": rows[0]["ts"], "last_ts": rows[-1]["ts"], "cycles": cycles, "steps": steps,
            "llm_calls": len(llm), "pages_read": pages, "input_tokens": in_tok, "output_tokens": out_tok,
            "max_input_tokens": max((r["input_tokens"] for r in llm), default=0),
            "last_input_tokens": llm[-1]["input_tokens"] if llm else 0,
            "est_cost_usd": round(cost, 4), "resumes": resumes,
            "errors": sum(1 for r in rows if r["event_type"] == "error"),
        }]

    # --- pipe: recent_events
    def _pipe_recent_events(self, run_id: str, p: dict) -> list[dict]:
        rows = self._events(run_id)
        et = p.get("event_type")
        if et:
            rows = [r for r in rows if r["event_type"] == et]
        return [{k: r[k] for k in ("ts", "component", "event_type", "unit_id", "payload", "input_tokens", "output_tokens", "latency_ms", "model")}
                for r in rows[-int(p.get("limit", 50)):]][::-1]
