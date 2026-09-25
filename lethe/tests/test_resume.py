"""Resume-from-state: a new process rebuilds the same ledger from the store, no transcript replay."""
from __future__ import annotations

import asyncio

from lethe.config import Settings
from lethe.runtime.loop import Runtime


def test_kill_and_resume_same_ledger(tmp_path):
    cfg = Settings()
    cfg.mock = True
    cfg.run_id = "resume-test"
    cfg.state_dir = tmp_path / "state"
    cfg.units_limit = 2

    async def go():
        rt1 = Runtime(cfg, quiet=True)
        await rt1.boot()
        l1 = await rt1.cycle()
        await rt1.close()
        del rt1  # "kill -9": nothing survives in process memory

        rt2 = Runtime(cfg, quiet=True)
        l2 = await rt2.boot()
        assert l2.kpis.cycles == l1.kpis.cycles == 1
        assert [u.model_dump() for u in l2.units] == [u.model_dump() for u in l1.units]
        assert l2.decisions == l1.decisions and l2.lessons == l1.lessons
        resumed = [r for r in rt2.store._events(cfg.run_id) if r["event_type"] == "resumed"]
        assert len(resumed) == 1
        # and it keeps going from cycle 2, with the same facts
        l3 = await rt2.cycle()
        assert l3.kpis.cycles == 2
        await rt2.close()

    asyncio.run(go())


def test_finding_card_budget_and_token_budget(tmp_path):
    cfg = Settings()
    cfg.mock = True
    cfg.run_id = "budget-test"
    cfg.state_dir = tmp_path / "state"
    cfg.units_limit = 3

    async def go():
        rt = Runtime(cfg, quiet=True)
        await rt.boot()
        await rt.cycle()
        rows = rt.store._events(cfg.run_id)
        llm = [r for r in rows if r["event_type"] == "llm_call"]
        assert llm and max(r["input_tokens"] for r in llm) <= cfg.token_budget
        import json
        cards = [json.loads(r["payload"]) for r in rows if r["event_type"] == "scout_done"]
        assert cards and all(c.get("card_tokens", 0) <= 200 for c in cards)
        await rt.close()

    asyncio.run(go())
