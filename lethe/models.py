"""Pydantic models. These are the *only* shapes that cross component boundaries.

Generational memory (see memory/librarian.py):
  scratch  -> raw pages inside a Scout; never persisted to a prompt
  working  -> facts extracted by the Librarian, each with a TTL
  canon    -> facts confirmed by >=2 independent sources, decisions, lessons
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or now()).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def parse_ts(s: str) -> datetime:
    s = s.replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.fromisoformat(s).astimezone(timezone.utc)


Generation = Literal["scratch", "working", "canon"]
FactStatus = Literal["active", "expired", "evicted", "contradicted"]
UnitStatus = Literal["scouting", "needs_verify", "verified", "action", "stale"]


# ----------------------------------------------------------------------------- inventory
class Unit(BaseModel):
    unit_id: str
    year: int
    make: str
    model: str
    trim: str = ""
    category: str = ""  # class_b | class_c | travel_trailer | truck ...
    mileage: int = 0
    our_price: float
    location: str = ""
    state: str = ""

    @property
    def label(self) -> str:
        return f"{self.year} {self.make} {self.model} {self.trim}".strip()


# ----------------------------------------------------------------------------- facts
class Fact(BaseModel):
    fact_id: str
    run_id: str = ""
    unit_id: str
    subject: str            # e.g. "listing:rvtrader:12345" or "unit:U01"
    predicate: str          # e.g. "asking_price", "exists", "mileage", "status"
    value: str
    source_url: str = ""
    observed_at: str = Field(default_factory=iso)
    confidence: float = 0.5
    ttl_seconds: int = 7200
    generation: Generation = "working"
    status: FactStatus = "active"
    ts: str = Field(default_factory=iso)

    @staticmethod
    def make_id(subject: str, predicate: str) -> str:
        return hashlib.sha1(f"{subject}|{predicate}".encode()).hexdigest()[:16]

    def expires_at(self) -> datetime:
        from datetime import timedelta
        return parse_ts(self.observed_at) + timedelta(seconds=self.ttl_seconds)

    def is_expired(self, at: datetime | None = None) -> bool:
        return (at or now()) >= self.expires_at()

    def compact(self) -> str:
        """One-line rendering used inside prompts (bounded)."""
        return f"[{self.generation[0]}|{self.confidence:.2f}] {self.subject} {self.predicate}={self.value}"


# ----------------------------------------------------------------------------- scout output
class Listing(BaseModel):
    listing_id: str
    title: str = ""
    year: int | None = None
    make: str = ""
    model: str = ""
    price: float | None = None
    mileage: int | None = None
    location: str = ""
    url: str = ""
    source: str = ""        # domain
    status: Literal["for_sale", "sold", "removed", "unknown"] = "for_sale"


class FindingCard(BaseModel):
    """<=200 tokens, structured. The only thing a Scout hands back."""
    card_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    unit_id: str
    strategy: str
    query: str
    sources: list[str] = Field(default_factory=list)
    comps: list[Listing] = Field(default_factory=list)
    pages_read: int = 0
    notes: str = ""
    error: str = ""
    created_at: str = Field(default_factory=iso)

    def compact(self) -> dict[str, Any]:
        """The <=200-token wire form: the only thing the Chief / Auditor ever see of a scout run."""
        d: dict[str, Any] = {
            "card": self.card_id[:8], "unit": self.unit_id, "strat": self.strategy, "n": len(self.comps),
            "comps": [
                {"id": c.listing_id, "y": c.year, "p": c.price, "mi": c.mileage, "loc": c.location, "st": c.status[:4], "src": c.source.split(".")[0]}
                for c in self.comps[:8]
            ],
        }
        if self.error:
            d["error"] = self.error[:100]
        if self.notes:
            d["notes"] = self.notes[:80]
        return d


# ----------------------------------------------------------------------------- audit
class AuditVerdict(BaseModel):
    card_id: str
    unit_id: str
    passed: bool
    accepted: list[str] = Field(default_factory=list)  # listing_ids
    rejected: dict[str, str] = Field(default_factory=dict)  # listing_id -> reason
    second_source: bool = False
    lesson: str = ""
    retry_strategy: str = ""
    reasoning: str = ""


# ----------------------------------------------------------------------------- ledger
class UnitState(BaseModel):
    unit_id: str
    label: str = ""
    our_price: float = 0
    status: UnitStatus = "scouting"
    strategy: str = "broad"
    current_reco: float = 0
    confidence: float = 0.0
    comps_count: int = 0
    reasoning: str = ""
    last_checked: str = ""
    next_check_after: str = ""
    failures: int = 0


class PlanStep(BaseModel):
    step: str
    status: Literal["todo", "doing", "done", "failed"] = "todo"


class Decision(BaseModel):
    what: str
    why: str
    at: str = Field(default_factory=iso)


class Lesson(BaseModel):
    lesson: str
    at: str = Field(default_factory=iso)
    unit_id: str = ""
    source: str = "auditor"


class KPIs(BaseModel):
    cycles: int = 0
    steps: int = 0
    pages_read: int = 0
    facts_active: int = 0
    facts_evicted: int = 0
    facts_expired: int = 0
    facts_contradicted: int = 0
    facts_promoted: int = 0
    max_input_tokens: int = 0


class Ledger(BaseModel):
    """Explicit mutable state. Rewritten (not appended) every cycle."""
    mission_id: str
    goal: str
    constraints: list[str] = Field(default_factory=list)
    units: list[UnitState] = Field(default_factory=list)
    plan: list[PlanStep] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    lessons: list[Lesson] = Field(default_factory=list)
    kpis: KPIs = Field(default_factory=KPIs)
    started_at: str = Field(default_factory=iso)
    updated_at: str = Field(default_factory=iso)

    MAX_DECISIONS: int = 12
    MAX_LESSONS: int = 15

    def unit(self, unit_id: str) -> UnitState:
        for u in self.units:
            if u.unit_id == unit_id:
                return u
        raise KeyError(unit_id)

    def trim(self) -> None:
        """Keep the ledger bounded; history lives in the events stream, not here."""
        self.decisions = self.decisions[-self.MAX_DECISIONS:]
        self.lessons = self.lessons[-self.MAX_LESSONS:]
        self.plan = [p for p in self.plan if p.status in {"todo", "doing"}] + [p for p in self.plan if p.status in {"done", "failed"}][-6:]

    def compact(self) -> dict[str, Any]:
        """Bounded snapshot for prompts (never the full history)."""
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "constraints": self.constraints,
            "units": [
                {"unit_id": u.unit_id, "label": u.label, "our_price": u.our_price, "status": u.status,
                 "strategy": u.strategy, "reco": u.current_reco, "conf": round(u.confidence, 2),
                 "comps": u.comps_count, "last_checked": u.last_checked[:16], "failures": u.failures}
                for u in self.units
            ],
            "plan": [p.model_dump() for p in self.plan[-8:]],
            "decisions": [d.model_dump() for d in self.decisions[-5:]],
            "kpis": self.kpis.model_dump(),
        }


# ----------------------------------------------------------------------------- events
class Event(BaseModel):
    ts: str = Field(default_factory=iso)
    run_id: str
    agent: Literal["lethe", "naive"] = "lethe"
    component: Literal["chief", "scout", "librarian", "auditor", "gc", "system", "naive"] = "system"
    event_type: str
    unit_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model: str = ""

    def to_row(self) -> dict[str, Any]:
        row = self.model_dump()
        row["payload"] = json.dumps(self.payload, default=str)[:60000]
        return row


# ----------------------------------------------------------------------------- llm
class LLMResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model: str
    provider: str

    def json(self) -> Any:  # type: ignore[override]
        return extract_json(self.text)


def extract_json(text: str) -> Any:
    """Tolerant JSON extraction: strips code fences and leading/trailing prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    opener = t[start]
    closer = "}" if opener == "{" else "]"
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unbalanced JSON")
