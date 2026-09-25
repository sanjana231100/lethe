"""Librarian GC rules: supersede, contradict (+cascade), promote on 2nd source, evict on audit, expire on TTL."""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from lethe.llm.mock import MockLLM
from lethe.memory.librarian import Librarian
from lethe.models import AuditVerdict, Fact, FindingCard, Listing, iso, now
from lethe.runtime.events import EventLog
from lethe.store.local import LocalStore


def _rt(tmp_path):
    store = LocalStore(tmp_path / "state")
    log = EventLog(store, "t", quiet=True)
    lib = Librarian(MockLLM("mock", 6000), store, log, "t", {"price_ttl_seconds": 7200, "existence_ttl_seconds": 21600})
    return store, log, lib


def _card(price=109995.0, status="for_sale", url="https://www.rvtrader.com/listing/1", src="rvtrader.com", at=None):
    return FindingCard(unit_id="U01", strategy="broad", query="q", created_at=at or iso(),
                       comps=[Listing(listing_id="rvtrader-1", title="2022 Winnebago Solis 59P", year=2022, make="Winnebago", model="Solis",
                                      price=price, mileage=20000, location="Reno, NV", url=url, source=src, status=status)])


async def _events(store, kind):
    return [r for r in store._events("t") if r["event_type"] == kind]


def test_create_then_supersede(tmp_path):
    async def go():
        store, log, lib = _rt(tmp_path)
        await lib.ingest(_card(price=109995), [])
        facts = await store.active_facts("t")
        assert {f.predicate for f in facts} == {"asking_price", "specs", "status"}
        assert all(f.generation == "working" for f in facts)
        # price drop -> new version wins, old only in events
        await lib.ingest(_card(price=106995), facts)
        facts = await store.active_facts("t")
        price = [f for f in facts if f.predicate == "asking_price"]
        assert len(price) == 1 and price[0].value == "106995"
        assert len(await _events(store, "fact_superseded")) == 1
    asyncio.run(go())


def test_promote_on_independent_observation(tmp_path):
    async def go():
        store, log, lib = _rt(tmp_path)
        t0 = now()
        await lib.ingest(_card(at=iso(t0)), [])
        facts = await store.active_facts("t")
        # same URL, 1 minute later: NOT independent -> stays working
        await lib.ingest(_card(at=iso(t0 + timedelta(minutes=1))), facts)
        facts = await store.active_facts("t")
        assert all(f.generation == "working" for f in facts)
        # same value, 20 minutes later: independent -> canon
        await lib.ingest(_card(at=iso(t0 + timedelta(minutes=20))), facts)
        facts = await store.active_facts("t")
        assert all(f.generation == "canon" for f in facts)
        assert len(await _events(store, "fact_promoted")) == 3
    asyncio.run(go())


def test_contradiction_cascades_to_existing_price(tmp_path):
    async def go():
        store, log, lib = _rt(tmp_path)
        await lib.ingest(_card(price=109995), [])
        facts = await store.active_facts("t")
        await lib.ingest(_card(price=None, status="sold"), facts)
        facts = await store.active_facts("t")
        assert facts == [], "every fact about a sold listing must be contradicted, including the old price"
        kinds = [r["event_type"] for r in store._events("t")]
        assert kinds.count("fact_contradicted") >= 3
    asyncio.run(go())


def test_audit_evicts_and_promotes(tmp_path):
    async def go():
        store, log, lib = _rt(tmp_path)
        await lib.ingest(_card(), [])
        facts = await store.active_facts("t")
        v = AuditVerdict(card_id="c", unit_id="U01", passed=False, rejected={"rvtrader-1": "year outside ±1"})
        await lib.apply_audit(v, facts)
        assert await store.active_facts("t") == []
        # fresh run: accepted + second source -> canon
        store2, log2, lib2 = _rt(tmp_path / "b")
        await lib2.ingest(_card(), [])
        facts = await store2.active_facts("t")
        v = AuditVerdict(card_id="c", unit_id="U01", passed=True, accepted=["rvtrader-1"], second_source=True)
        await lib2.apply_audit(v, facts)
        facts = await store2.active_facts("t")
        assert facts and all(f.generation == "canon" for f in facts)
    asyncio.run(go())


def test_gc_sweep_expires_by_ttl(tmp_path):
    async def go():
        store, log, lib = _rt(tmp_path)
        await lib.ingest(_card(), [])
        facts = await store.active_facts("t")
        stats = await lib.gc_sweep(facts, at=now() + timedelta(hours=3))   # price ttl 2h, specs 6h
        assert stats["expired"] == 2 and stats["active"] == 1
        still = await store.active_facts("t")
        assert {f.predicate for f in still} == {"specs"} or all(f.is_expired(now() + timedelta(hours=3)) is False for f in still)
    asyncio.run(go())
