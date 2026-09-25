"""One Lethe cycle (see README §Loop):

  boot/resume -> chief.plan -> scouts (parallel) -> librarian -> auditor (+ one self-correcting
  retry) -> chief.update -> ledger_snapshot -> gc sweep -> sleep

Process memory holds nothing that matters: the ledger and the facts are re-read from the
store on every cycle, so `kill -9` at any point loses at most the in-flight cycle.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..agents.auditor import Auditor
from ..agents.chief import Chief
from ..agents.scout import Scout, run_scouts
from ..config import Settings, settings as default_settings
from ..llm.base import make_llm
from ..memory.librarian import Librarian
from ..models import AuditVerdict, Fact, FindingCard, Ledger, Lesson, iso
from ..store.base import make_store
from ..tools.nimble import MockNimble, make_nimble
from .events import EventLog
from .mission import load_mission, load_units, new_ledger, sync_units


class Runtime:
    def __init__(self, cfg: Settings | None = None, *, quiet: bool = False):
        self.cfg = cfg or default_settings
        self.mission = load_mission(self.cfg.mission_file)
        self.units = load_units(self.mission, self.cfg.units_limit)
        self.store = make_store(self.cfg)
        self.log = EventLog(self.store, self.cfg.run_id, quiet=quiet)
        self.llm = make_llm(self.cfg, purpose="chief")
        self.librarian_llm = make_llm(self.cfg, purpose="librarian")
        self.nimble = make_nimble(self.cfg)
        self.scout = Scout(self.nimble, make_llm(self.cfg, purpose="scout"), self.log)
        self.librarian = Librarian(self.librarian_llm, self.store, self.log, self.cfg.run_id, self.mission.get("refresh", {}),
                                   int(self.mission.get("comparability", {}).get("min_sources_for_canon", 2)))
        self.auditor = Auditor(make_llm(self.cfg, purpose="auditor"), self.log, self.mission)
        self.chief = Chief(self.llm, self.log, self.mission, self.units, self.cfg.token_budget)
        self.ledger: Ledger | None = None

    # ------------------------------------------------------------------ boot / resume
    async def boot(self) -> Ledger:
        t0 = time.perf_counter()
        ledger = await self.store.current_ledger(self.cfg.run_id)
        facts = await self.store.active_facts(self.cfg.run_id)
        if ledger is None:
            ledger = new_ledger(self.mission, self.units)
            await self.log.emit("system", "started", payload={"config": self.cfg.summary(), "units": list(self.units)})
        else:
            sync_units(ledger, self.units)
            await self.log.emit("system", "resumed", payload={"ms": int((time.perf_counter() - t0) * 1000), "cycles": ledger.kpis.cycles,
                                                              "facts_active": len(facts), "units": len(ledger.units), "config": self.cfg.summary(),
                                                              "note": "state rebuilt from store; no transcript replay"})
        self.ledger = ledger
        return ledger

    async def _snapshot(self, ledger: Ledger, facts: list[Fact]) -> None:
        ledger.kpis.facts_active = sum(1 for f in facts if f.status == "active")
        ledger.kpis.max_input_tokens = max(ledger.kpis.max_input_tokens, self.log.max_input_tokens)
        ledger.updated_at = iso()
        ledger.trim()
        await self.log.emit("system", "ledger_snapshot", payload={"ledger": ledger.model_dump()})
        await self.store.flush()

    # ------------------------------------------------------------------ one cycle
    async def cycle(self) -> Ledger:
        ledger = self.ledger or await self.boot()
        n = ledger.kpis.cycles + 1
        MockNimble.set_cycle(n - 1)
        await self.log.emit("system", "cycle_start", payload={"cycle": n})
        facts = await self.store.active_facts(self.cfg.run_id)

        # 1. plan
        jobs = await self.chief.plan(ledger, facts, max_units=self.cfg.units_limit)
        ledger.kpis.steps += 1

        # 2. scouts (parallel, disposable)
        cards = await run_scouts(self.scout, [(self.units[u], s) for u, s in jobs], self.cfg.max_concurrent_scouts)
        ledger.kpis.steps += len(cards)
        ledger.kpis.pages_read += sum(c.pages_read for c in cards)

        # 3. librarian, 4. auditor, with one self-correcting retry per failed card
        verdicts: list[AuditVerdict] = []
        final_cards: list[FindingCard] = []
        for card in cards:
            card, verdict = await self._ingest_and_audit(ledger, card)
            if not verdict.passed and verdict.retry_strategy and verdict.retry_strategy != card.strategy:
                unit = self.units[card.unit_id]
                await self.log.emit("chief", "strategy_changed", unit_id=card.unit_id,
                                    payload={"from": card.strategy, "to": verdict.retry_strategy, "because": verdict.lesson})
                ledger.unit(card.unit_id).strategy = verdict.retry_strategy
                retry = await self.scout.run(unit, verdict.retry_strategy)
                ledger.kpis.steps += 1
                ledger.kpis.pages_read += retry.pages_read
                retry, verdict2 = await self._ingest_and_audit(ledger, retry)
                if verdict2.passed:
                    await self.log.emit("auditor", "self_correction", unit_id=card.unit_id,
                                        payload={"lesson": verdict.lesson, "failed_strategy": card.strategy,
                                                 "new_strategy": verdict.retry_strategy, "comps": len(verdict2.accepted)})
                card, verdict = retry, verdict2
            verdicts.append(verdict)
            final_cards.append(card)

        # 5. chief updates recommendations + rewrites the ledger
        facts = await self.store.active_facts(self.cfg.run_id)
        await self.chief.update(ledger, facts, final_cards, verdicts)
        ledger.kpis.steps += 1
        ledger.kpis.cycles = n

        # 6. GC sweep + metrics
        stats = await self.librarian.gc_sweep(facts)
        ledger.kpis.facts_expired += stats["expired"]
        ledger.kpis.facts_evicted += stats["evicted"]
        ledger.kpis.steps += 1
        facts = await self.store.active_facts(self.cfg.run_id)
        await self._snapshot(ledger, facts)
        await self.log.emit("system", "cycle_done", payload={"cycle": n, "kpis": ledger.kpis.model_dump(),
                                                             "max_input_tokens_this_process": self.log.max_input_tokens})
        await self.store.flush()
        return ledger

    async def _ingest_and_audit(self, ledger: Ledger, card: FindingCard) -> tuple[FindingCard, AuditVerdict]:
        unit = self.units[card.unit_id]
        existing = await self.store.active_facts(self.cfg.run_id)
        try:
            await self.librarian.ingest(card, existing)
        except Exception as exc:
            await self.log.error("librarian", "ingest", exc, unit_id=card.unit_id)
        ledger.kpis.steps += 1
        verdict = await self.auditor.audit(unit, card, use_llm=True)
        ledger.kpis.steps += 1
        if verdict.lesson:
            ledger.lessons.append(Lesson(lesson=verdict.lesson, unit_id=card.unit_id, source="auditor"))
        try:
            facts = await self.store.active_facts(self.cfg.run_id)
            await self.librarian.apply_audit(verdict, facts)
        except Exception as exc:
            await self.log.error("librarian", "apply_audit", exc, unit_id=card.unit_id)
        return card, verdict

    # ------------------------------------------------------------------ forever
    async def run_forever(self) -> None:
        await self.boot()
        while True:
            t0 = time.perf_counter()
            try:
                await self.cycle()
            except Exception as exc:  # the loop must survive anything
                await self.log.error("system", "cycle", exc)
                await self.store.flush()
            elapsed = time.perf_counter() - t0
            sleep_for = max(5.0, self.cfg.cycle_interval - elapsed)
            await self.log.emit("system", "sleep", payload={"seconds": round(sleep_for), "cycle_seconds": round(elapsed, 1)})
            await self.store.flush()
            await asyncio.sleep(sleep_for)

    async def close(self) -> None:
        await self.store.close()
        await self.nimble.close()
