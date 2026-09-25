"""LIBRARIAN = memory garbage collector (Liquid LFM2.5 via Ollama in live mode).

Runs on every observation (Finding Card). Pipeline:

  1. deterministic candidate facts from the card's structured comps
  2. Liquid LLM assigns TTL + confidence per fact and flags contradictions
     (strict JSON, validated with Pydantic, retried on parse failure; falls back
     to mission defaults so a model hiccup never loses an observation)
  3. GC rules (pure functions, unit-tested):
       supersede    same subject+predicate, new value  -> new version wins, old kept in events only
       contradict   listing status sold/removed        -> every fact about that listing -> contradicted
       promote      >=2 independent observations       -> working -> canon
       evict        auditor rejected / confidence low  -> evicted
       expire       now > observed_at + ttl            -> expired   (gc_sweep)

Generations: scratch (inside scouts, never here) -> working -> canon.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..llm.base import LLMProvider
from ..models import AuditVerdict, Fact, FindingCard, Listing, iso, now, parse_ts
from ..runtime.events import EventLog
from ..store.base import StateStore

PROMOTION_MIN_GAP = timedelta(minutes=10)   # two observations of the same URL count as independent if this far apart
EVICT_BELOW_CONFIDENCE = 0.3


# ----------------------------------------------------------------------------- LLM schema
class FactProposal(BaseModel):
    subject: str
    predicate: str
    value: str
    confidence: float = Field(ge=0, le=1)
    ttl_seconds: int = Field(ge=60, le=7 * 86400)

    @field_validator("value", "subject", "predicate", mode="before")
    @classmethod
    def _str(cls, v):
        return "" if v is None else str(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("ttl_seconds", mode="before")
    @classmethod
    def _ttl(cls, v):
        try:
            return max(60, min(7 * 86400, int(float(v))))
        except (TypeError, ValueError):
            return 7200


class Contradiction(BaseModel):
    subject: str
    reason: str


class LibrarianOutput(BaseModel):
    facts: list[FactProposal]
    contradictions: list[Contradiction] = Field(default_factory=list)
    notes: str = ""


LIBRARIAN_SYSTEM = """ROLE: librarian
You are the memory garbage collector for a long-running pricing agent. You receive NEW OBSERVATIONS
(vehicle listings found on the web) and the EXISTING FACTS already in memory for the same unit.
Return ONLY a JSON object:
{"facts":[{"subject":"listing:<id>","predicate":"asking_price|specs|status","value":"...","confidence":0-1,"ttl_seconds":int}],
 "contradictions":[{"subject":"listing:<id>","reason":"..."}],
 "notes":"<=20 words"}
Rules:
- One fact per (subject,predicate). subject is "listing:<listing_id>" from the observation.
- asking_price: ttl 7200 (prices move fast). specs: ttl 21600. status: ttl 7200.
- confidence: 0.8 dealer page with price; 0.6 snippet-only; 0.4 if fields missing.
- If an observation says status sold/removed, add a contradiction for that subject.
- Never invent listings that are not in the observations."""


class Librarian:
    def __init__(self, llm: LLMProvider, store: StateStore, log: EventLog, run_id: str, refresh_cfg: dict[str, Any],
                 min_sources_for_canon: int = 2):
        self.llm = llm
        self.store = store
        self.log = log
        self.run_id = run_id
        self.cfg = refresh_cfg
        self.min_sources = min_sources_for_canon

    # ------------------------------------------------------------------ helpers
    def _default_ttl(self, predicate: str) -> int:
        if predicate == "asking_price":
            return int(self.cfg.get("price_ttl_seconds", 7200))
        if predicate == "status":
            return int(self.cfg.get("price_ttl_seconds", 7200))
        return int(self.cfg.get("existence_ttl_seconds", 21600))

    @staticmethod
    def candidate_facts(card: FindingCard, run_id: str) -> list[Fact]:
        """Deterministic extraction: every comp -> asking_price / specs / status facts."""
        out: list[Fact] = []
        for c in card.comps:
            subj = f"listing:{c.listing_id}"
            specs = f"{c.year or '?'}|{c.make} {c.model}|{c.mileage or 0}mi|{c.location}|{c.source}"
            rows = [("status", c.status), ("specs", specs)]
            if c.price:
                rows.append(("asking_price", f"{c.price:.0f}"))
            for pred, val in rows:
                out.append(Fact(fact_id=Fact.make_id(subj, pred), run_id=run_id, unit_id=card.unit_id, subject=subj,
                                predicate=pred, value=str(val), source_url=c.url, observed_at=card.created_at,
                                confidence=0.6, ttl_seconds=7200, generation="working", status="active"))
        return out

    # ------------------------------------------------------------------ ingest
    async def ingest(self, card: FindingCard, existing: list[Fact]) -> list[Fact]:
        """Turn one Finding Card into fact versions. Returns the new fact rows written."""
        candidates = self.candidate_facts(card, self.run_id)
        if not candidates:
            await self.log.emit("librarian", "librarian_done", unit_id=card.unit_id, payload={"card_id": card.card_id, "facts": 0})
            return []
        by_key = {(f.subject, f.predicate): f for f in existing if f.unit_id == card.unit_id}

        # --- 2. Liquid assigns TTL/confidence + contradictions (bounded prompt: <= 8 comps + existing facts for this unit)
        obs = [{"listing_id": c.listing_id, "price": c.price, "year": c.year, "mileage": c.mileage, "location": c.location,
                "source": c.source, "status": c.status, "snippet_only": not c.title} for c in card.comps[:8]]
        exist = [f.compact() for f in list(by_key.values())[:24]]
        user = f"NEW OBSERVATIONS (unit {card.unit_id}, strategy {card.strategy}):\n{obs}\n\nEXISTING FACTS:\n{exist}\n\nReturn the JSON."
        proposals: dict[tuple[str, str], FactProposal] = {}
        contradicted_subjects: dict[str, str] = {}
        try:
            out: LibrarianOutput = await self.log.llm_json(self.llm, "librarian", "librarian", LIBRARIAN_SYSTEM, user,
                                                          meta={"payload": {"observations": obs, "existing": exist}},
                                                          unit_id=card.unit_id, validator=LibrarianOutput, max_tokens=1200)
            proposals = {(p.subject, p.predicate): p for p in out.facts}
            # trust the model's contradictions only where the observation itself says sold/removed:
            # a 1B model must never be able to delete a live listing on its own
            gone = {f"listing:{c.listing_id}" for c in card.comps if c.status in ("sold", "removed")}
            contradicted_subjects = {c.subject: c.reason for c in out.contradictions if c.subject in gone}
        except Exception as exc:  # fallback: deterministic defaults (rule #7)
            await self.log.error("librarian", "llm_assign", exc, unit_id=card.unit_id)

        # rule: a sold/removed observation always contradicts, whatever the model said
        for c in card.comps:
            if c.status in ("sold", "removed"):
                contradicted_subjects.setdefault(f"listing:{c.listing_id}", f"listing {c.status}")

        # --- 3. GC rules
        new_rows: list[Fact] = []
        stats = {"created": 0, "superseded": 0, "promoted": 0, "contradicted": 0, "reobserved": 0}
        t_obs = parse_ts(card.created_at)
        for f in candidates:
            p = proposals.get((f.subject, f.predicate))
            if p:
                f.confidence = p.confidence
                f.ttl_seconds = p.ttl_seconds
            else:
                f.ttl_seconds = self._default_ttl(f.predicate)

            if f.subject in contradicted_subjects:
                f.status = "contradicted"
                f.generation = "working"
                new_rows.append(f)
                stats["contradicted"] += 1
                await self.log.emit("librarian", "fact_contradicted", unit_id=f.unit_id,
                                    payload={"fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                                             "reason": contradicted_subjects[f.subject], "source": f.source_url})
                continue

            prev = by_key.get((f.subject, f.predicate))
            if prev is None:
                new_rows.append(f); stats["created"] += 1
                await self.log.emit("librarian", "fact_created", unit_id=f.unit_id,
                                    payload={"fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate, "value": f.value,
                                             "ttl": f.ttl_seconds, "conf": f.confidence})
            elif prev.value != f.value:
                # supersede: new version wins; history stays in the events stream only
                f.generation = "working"
                new_rows.append(f); stats["superseded"] += 1
                await self.log.emit("librarian", "fact_superseded", unit_id=f.unit_id,
                                    payload={"fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                                             "old": prev.value, "new": f.value, "was": prev.generation})
            else:
                # re-observation of the same value: independent if different URL or far enough apart in time
                independent = prev.source_url != f.source_url or (t_obs - parse_ts(prev.observed_at)) >= PROMOTION_MIN_GAP
                if independent and prev.generation == "working":
                    f.generation = "canon"
                    f.confidence = max(prev.confidence, f.confidence, 0.85)
                    f.ttl_seconds = max(prev.ttl_seconds, f.ttl_seconds) * 2   # canon lives longer
                    new_rows.append(f); stats["promoted"] += 1
                    await self.log.emit("librarian", "fact_promoted", unit_id=f.unit_id,
                                        payload={"fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                                                 "value": f.value, "observations": 2, "sources": sorted({prev.source_url, f.source_url})})
                else:
                    # refresh observed_at (extends TTL), keep generation
                    f.generation = prev.generation
                    f.confidence = max(prev.confidence, f.confidence)
                    new_rows.append(f); stats["reobserved"] += 1

        # contradiction cascades: every *existing* active fact about a sold/removed listing dies too
        handled = {(f.subject, f.predicate) for f in new_rows}
        for (subj, pred), prev in by_key.items():
            if subj in contradicted_subjects and (subj, pred) not in handled and prev.status == "active":
                g = prev.model_copy(update={"status": "contradicted", "ts": iso()})
                new_rows.append(g); stats["contradicted"] += 1
                await self.log.emit("librarian", "fact_contradicted", unit_id=g.unit_id,
                                    payload={"fact_id": g.fact_id, "subject": g.subject, "predicate": g.predicate, "value": g.value,
                                             "reason": contradicted_subjects[subj], "was": prev.generation})

        if new_rows:
            await self.store.append_facts(new_rows)
        await self.log.emit("librarian", "librarian_done", unit_id=card.unit_id, payload={"card_id": card.card_id, **stats})
        return new_rows

    # ------------------------------------------------------------------ audit feedback
    async def apply_audit(self, verdict: AuditVerdict, facts: list[Fact]) -> list[Fact]:
        """Auditor verdict -> promotion (accepted + second source) or eviction (rejected)."""
        rows: list[Fact] = []
        latest = {f.fact_id: f for f in facts}
        for f in latest.values():
            if f.unit_id != verdict.unit_id or f.status != "active":
                continue
            lid = f.subject.split(":", 1)[1] if ":" in f.subject else ""
            if lid in verdict.rejected:
                g = f.model_copy(update={"status": "evicted", "ts": iso()})
                rows.append(g)
                await self.log.emit("librarian", "fact_evicted", unit_id=f.unit_id,
                                    payload={"fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                                             "reason": f"audit: {verdict.rejected[lid]}"})
            elif lid in verdict.accepted:
                conf = min(0.95, f.confidence + 0.15)
                update: dict[str, Any] = {"confidence": conf, "ts": iso()}
                if verdict.second_source and f.generation == "working" and conf >= 0.75:
                    update["generation"] = "canon"
                    await self.log.emit("librarian", "fact_promoted", unit_id=f.unit_id,
                                        payload={"fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                                                 "value": f.value, "reason": "audit passed with second source"})
                rows.append(f.model_copy(update=update))
        if rows:
            await self.store.append_facts(rows)
        return rows

    # ------------------------------------------------------------------ GC sweep
    async def gc_sweep(self, facts: list[Fact], at: datetime | None = None) -> dict[str, int]:
        """Expire TTL'd facts and evict low-confidence ones. Emits one event per transition."""
        at = at or now()
        rows: list[Fact] = []
        stats = {"expired": 0, "evicted": 0, "active": 0, "canon": 0, "working": 0}
        for f in facts:
            if f.status != "active":
                continue
            if f.is_expired(at):
                rows.append(f.model_copy(update={"status": "expired", "ts": iso(at)}))
                stats["expired"] += 1
                await self.log.emit("gc", "fact_expired", unit_id=f.unit_id,
                                    payload={"fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                                             "age_s": int((at - parse_ts(f.observed_at)).total_seconds()), "ttl": f.ttl_seconds})
            elif f.confidence < EVICT_BELOW_CONFIDENCE:
                rows.append(f.model_copy(update={"status": "evicted", "ts": iso(at)}))
                stats["evicted"] += 1
                await self.log.emit("gc", "fact_evicted", unit_id=f.unit_id,
                                    payload={"fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate, "reason": "low confidence"})
            else:
                stats["active"] += 1
                stats[f.generation] = stats.get(f.generation, 0) + 1
        if rows:
            await self.store.append_facts(rows)
        await self.log.emit("gc", "gc_sweep", payload=stats)
        return stats
