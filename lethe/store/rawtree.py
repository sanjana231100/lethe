"""RawTree store (the live state store + observability backend).

Writes: POST {api_url}/v1/tables/{table}   body = JSON array of objects, Bearer rt_... key.
        Tables auto-create on first insert (schema-free, dynamic columns).
Reads:  POST {api_url}/v1/query            body = {"sql": "...", "format": "JSON"}
        -> {"meta": [...], "data": [...], "rows": N, "statistics": {...}}

Buffered + batched with retry/backoff; an outage never crashes the loop (rule #7) - the buffer
is kept and retried on the next flush. All SQL lives in rawtree_sql.py.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..models import Event, Fact, Ledger, now, parse_ts
from . import rawtree_sql as sql
from .base import StateStore


def _ts_ms(ts: str) -> int:
    return int(parse_ts(ts).timestamp() * 1000)


class RawTreeStore(StateStore):
    name = "rawtree"

    def __init__(self, api_url: str, api_key: str, batch_size: int = 200):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self._events: list[dict[str, Any]] = []
        self._facts: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=30, headers={"Authorization": f"Bearer {self.api_key}"})

    # ------------------------------------------------------------------ writes
    async def append_events(self, events: list[Event]) -> None:
        for e in events:
            row = e.to_row()          # payload -> JSON string
            row["ts_ms"] = _ts_ms(e.ts)
            self._events.append(row)
        if len(self._events) >= self.batch_size:
            await self.flush()

    async def append_facts(self, facts: list[Fact]) -> None:
        for f in facts:
            row = f.model_dump()
            row["ts_ms"] = _ts_ms(f.ts)
            self._facts.append(row)
        if len(self._facts) >= self.batch_size:
            await self.flush()

    async def flush(self) -> None:
        async with self._lock:
            for table, buf in ((sql.EVENTS, self._events), (sql.FACTS, self._facts)):
                if not buf:
                    continue
                rows = list(buf)
                try:
                    await self._insert(table, rows)
                    del buf[: len(rows)]
                except Exception as exc:
                    print(f"[rawtree] insert of {len(rows)} rows into {table} failed: {exc!r} (kept in buffer)")

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=10),
           retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)), reraise=True)
    async def _insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        r = await self._client.post(f"{self.api_url}/v1/tables/{table}", content=json.dumps(rows, default=str).encode(),
                                    headers={"Content-Type": "application/json"})
        if r.status_code >= 500 or r.status_code == 429:
            r.raise_for_status()
        if r.status_code >= 400:
            raise RuntimeError(f"rawtree insert {r.status_code}: {r.text[:300]}")

    async def close(self) -> None:
        await self.flush()
        await self._client.aclose()

    # ------------------------------------------------------------------ reads
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6),
           retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)), reraise=True)
    async def sql(self, statement: str) -> list[dict[str, Any]]:
        r = await self._client.post(f"{self.api_url}/v1/query", json={"sql": statement, "format": "JSON"})
        if r.status_code >= 500 or r.status_code == 429:
            r.raise_for_status()
        low = r.text.lower()
        if r.status_code in (400, 404) and any(k in low for k in ("doesn't exist", "does not exist", "unknown table", "unknown_table", "not found", "no such table")):
            return []      # table not created yet (first boot)
        if r.status_code >= 400:
            raise RuntimeError(f"rawtree query {r.status_code}: {r.text[:300]}")
        return r.json().get("data", [])

    async def current_ledger(self, run_id: str) -> Ledger | None:
        rows = await self.sql(sql.current_ledger(run_id))
        if not rows:
            return None
        payload = rows[0]["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return Ledger(**payload["ledger"])

    async def active_facts(self, run_id: str) -> list[Fact]:
        rows = await self.sql(sql.latest_facts(run_id))
        t = now()
        out = []
        for r in rows:
            f = Fact(run_id=run_id, **{k: v for k, v in r.items() if k in Fact.model_fields and k != "run_id"})
            if not f.is_expired(t):
                out.append(f)
        return out

    async def query(self, pipe: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        p = params or {}
        run_id = p.get("run_id", "")
        limit = int(p.get("limit", 0) or 0)
        if pipe == "current_ledger":
            return await self.sql(sql.current_ledger(run_id))
        if pipe == "active_facts":
            return [f.model_dump() for f in await self.active_facts(run_id)]
        if pipe == "board":
            ledger = await self.current_ledger(run_id)
            if ledger is None:
                return []
            return [{"unit_id": u.unit_id, "label": u.label, "status": u.status, "our_price": u.our_price,
                     "current_reco": u.current_reco, "confidence": u.confidence, "comps_count": u.comps_count,
                     "last_checked": u.last_checked, "strategy": u.strategy, "reasoning": u.reasoning} for u in ledger.units]
        if pipe == "token_metrics":
            return await self.sql(sql.token_metrics(run_id, limit or 600))
        if pipe == "fact_lifecycle":
            return await self.sql(sql.fact_lifecycle(run_id))
        if pipe == "memory_feed":
            return await self.sql(sql.memory_feed(run_id, limit or 40))
        if pipe == "lessons":
            return await self.sql(sql.lessons(run_id, limit or 30))
        if pipe == "run_stats":
            rows = await self.sql(sql.run_stats(run_id))
            return [r for r in rows if r.get("first_ts")]
        if pipe == "recent_events":
            return await self.sql(sql.recent_events(run_id, limit or 50, p.get("event_type")))
        raise KeyError(f"unknown pipe {pipe}")
