"""Budgeted context assembly (hard rule #1).

Every Chief/Auditor call gets a *ranked, budgeted* slice of state:

    1. mission ledger            (always; compact form)
    2. canon facts for the task  (verified by >=2 sources; highest trust)
    3. working facts, top-K by   relevance x recency x confidence
    4. last N lessons

Items are added in priority order until the token budget is hit; everything
else is dropped. What was included and what was dropped is returned so the
caller can log a `context_assembled` event - the judges can see the GC working.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

from ..llm.base import count_tokens
from ..models import Fact, Ledger, Lesson, now, parse_ts


@dataclass
class ContextReport:
    text: str
    tokens: int
    included: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    truncated_ledger: bool = False

    def summary(self) -> dict:
        return {"tokens": self.tokens, "included": self.included, "dropped": self.dropped,
                "truncated_ledger": self.truncated_ledger}


def score_fact(f: Fact, unit_ids: set[str], at: datetime) -> float:
    """relevance x recency x confidence. Relevance: facts about units in the current task
    rank first; unit-agnostic facts (e.g. market notes) rank in the middle."""
    relevance = 1.0 if f.unit_id in unit_ids else (0.6 if not f.unit_id else 0.15)
    age_s = max(0.0, (at - parse_ts(f.observed_at)).total_seconds())
    half_life = max(600.0, f.ttl_seconds / 2)
    recency = math.exp(-age_s / half_life)
    return relevance * (0.4 + 0.6 * recency) * f.confidence


def assemble_context(
    ledger: Ledger,
    facts: list[Fact],
    *,
    unit_ids: list[str] | None = None,
    budget_tokens: int,
    reserve_tokens: int = 1200,
    max_working: int = 40,
    max_lessons: int = 6,
    at: datetime | None = None,
) -> ContextReport:
    """Build the bounded state block for one LLM call.

    `budget_tokens` is the total input budget; `reserve_tokens` is left for the system prompt
    + the step's own evidence (finding cards etc.) that the caller appends afterwards.
    """
    at = at or now()
    units = set(unit_ids or [u.unit_id for u in ledger.units])
    avail = budget_tokens - reserve_tokens
    parts: list[str] = []
    included: dict[str, int] = {"ledger": 1, "canon": 0, "working": 0, "lessons": 0}
    dropped: dict[str, int] = {"canon": 0, "working": 0, "lessons": 0}

    # 1. ledger (always). If even the compact ledger is over budget, drop units outside the task.
    ledger_dict = ledger.compact()
    ledger_txt = "MISSION LEDGER:\n" + json.dumps(ledger_dict, separators=(",", ":"))
    truncated = False
    if count_tokens(ledger_txt) > avail * 0.6:
        ledger_dict["units"] = [u for u in ledger_dict["units"] if u["unit_id"] in units]
        ledger_dict["decisions"] = ledger_dict["decisions"][-2:]
        ledger_txt = "MISSION LEDGER (task slice):\n" + json.dumps(ledger_dict, separators=(",", ":"))
        truncated = True
    parts.append(ledger_txt)
    used = count_tokens(ledger_txt)
    HEADER = 14  # tokens reserved per section header + joiner

    # 2. canon facts relevant to the task
    canon = sorted((f for f in facts if f.generation == "canon" and f.status == "active" and (f.unit_id in units or not f.unit_id)),
                   key=lambda f: -score_fact(f, units, at))
    lines = []
    for f in canon:
        line = f.compact()
        t = count_tokens(line) + 1 + (HEADER if not lines else 0)
        if used + t > avail:
            dropped["canon"] += 1
            continue
        lines.append(line); used += t; included["canon"] += 1
    if lines:
        parts.append("CANON FACTS (verified by >=2 sources):\n" + "\n".join(lines))

    # 3. working facts: top-K by score
    working = sorted((f for f in facts if f.generation == "working" and f.status == "active" and not f.is_expired(at)),
                     key=lambda f: -score_fact(f, units, at))
    lines = []
    for i, f in enumerate(working):
        if i >= max_working:
            dropped["working"] += 1
            continue
        line = f.compact()
        t = count_tokens(line) + 1 + (HEADER if not lines else 0)
        if used + t > avail:
            dropped["working"] += 1
            continue
        lines.append(line); used += t; included["working"] += 1
    if lines:
        parts.append("WORKING FACTS (TTL-bound, unverified):\n" + "\n".join(lines))

    # 4. last N lessons
    lessons: list[Lesson] = ledger.lessons[-max_lessons:]
    dropped["lessons"] += max(0, len(ledger.lessons) - max_lessons)
    lines = []
    for l in lessons:
        line = f"- ({l.at[:16]}{' ' + l.unit_id if l.unit_id else ''}) {l.lesson}"
        t = count_tokens(line) + 1 + (HEADER if not lines else 0)
        if used + t > avail:
            dropped["lessons"] += 1
            continue
        lines.append(line); used += t; included["lessons"] += 1
    if lines:
        parts.append("RECENT LESSONS:\n" + "\n".join(lines))

    text = "\n\n".join(parts)
    # belt and braces: if the estimate was off, drop working-fact lines from the tail until it fits
    while count_tokens(text) > avail and included["working"] > 0:
        wi = next(i for i, p in enumerate(parts) if p.startswith("WORKING FACTS"))
        wlines = parts[wi].split("\n")
        if len(wlines) <= 2:
            parts.pop(wi)
        else:
            parts[wi] = "\n".join(wlines[:-1])
        included["working"] -= 1; dropped["working"] += 1
        text = "\n\n".join(parts)
    return ContextReport(text=text, tokens=count_tokens(text), included=included, dropped=dropped, truncated_ledger=truncated)
