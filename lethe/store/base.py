"""State store interface. Hard rule #2: state lives in RawTree, not in process memory.

Two implementations share one interface so the agent, the dashboard and the tests are
identical in MOCK=1 (local JSONL) and live (RawTree insert API + read-only SQL).
"""
from __future__ import annotations

import abc
from typing import Any

from ..config import Settings, settings as default_settings
from ..models import Event, Fact, Ledger


class StateStore(abc.ABC):
    name = "base"

    # ---- writes (append-only) ----
    @abc.abstractmethod
    async def append_events(self, events: list[Event]) -> None: ...

    @abc.abstractmethod
    async def append_facts(self, facts: list[Fact]) -> None: ...

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        await self.flush()

    # ---- reads (mirror the Tinybird pipes) ----
    @abc.abstractmethod
    async def current_ledger(self, run_id: str) -> Ledger | None: ...

    @abc.abstractmethod
    async def active_facts(self, run_id: str) -> list[Fact]: ...

    @abc.abstractmethod
    async def query(self, pipe: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


def make_store(cfg: Settings | None = None) -> StateStore:
    cfg = cfg or default_settings
    if cfg.mock or not cfg.rawtree_api_key:
        from .local import LocalStore
        return LocalStore(cfg.state_dir)
    from .rawtree import RawTreeStore
    return RawTreeStore(api_url=cfg.rawtree_api_url, api_key=cfg.rawtree_api_key)
