"""AUDITOR: the critic. Checks Finding Cards against the mission's comparability rules and
for a second source; turns failures into LESSONS the Chief uses to change strategy.

Rules are code (tested); the LLM only phrases the lesson / reasoning in live mode."""
from __future__ import annotations

from typing import Any

from ..llm.base import LLMProvider
from ..models import AuditVerdict, FindingCard, Unit
from ..runtime.events import EventLog

AUDIT_SYSTEM = """ROLE: auditor
You are the critic for a vehicle-pricing agent. Given a rule-based audit of a Finding Card, write a short
reasoning and, if the card failed, a one-sentence LESSON that changes future strategy.
Return ONLY JSON: {"reasoning":"<=30 words","lesson":"<=25 words or empty","retry_strategy":"<one of the allowed strategies or empty>"}"""


class Auditor:
    def __init__(self, llm: LLMProvider, log: EventLog, mission: dict[str, Any]):
        self.llm = llm
        self.log = log
        self.cmp = mission.get("comparability", {})
        self.adj = mission.get("region_adjacency", {})
        self.strategies: list[str] = mission.get("strategies", ["broad"])

    # ------------------------------------------------------------------ rules
    def check_comp(self, unit: Unit, c) -> str | None:
        """Return a rejection reason or None if comparable."""
        tol = int(self.cmp.get("year_tolerance", 1))
        if c.year is None:
            return "year missing"
        if abs(c.year - unit.year) > tol:
            return f"year {c.year} outside ±{tol}"
        if c.model and unit.model.lower().split()[0] not in c.model.lower() and unit.model.lower() not in (c.title or "").lower():
            return f"model mismatch ({c.model})"
        if unit.trim and c.title:
            trim_key = unit.trim.split()[0].lower()
            if trim_key not in (c.title + " " + c.model).lower() and unit.category == "truck":
                return f"trim mismatch (need {trim_key})"
        if unit.mileage and c.mileage:
            pct = float(self.cmp.get("mileage_band_pct", 40)) / 100
            band = max(unit.mileage * pct, float(self.cmp.get("mileage_band_abs", 15000)))
            if abs(c.mileage - unit.mileage) > band:
                return f"mileage {c.mileage} outside band"
        if c.price is None:
            return "no price"
        if c.status != "for_sale":
            return f"listing {c.status}"
        st = _state(c.location)
        allowed = self.adj.get(unit.state)
        if st and allowed and st not in allowed and self.cmp.get("region_strict", False):
            return f"{st} outside {unit.state} radius"
        return None

    def is_local(self, unit: Unit, c) -> bool:
        st = _state(c.location)
        allowed = self.adj.get(unit.state)
        return bool(st and allowed and st in allowed)

    def rule_audit(self, unit: Unit, card: FindingCard) -> AuditVerdict:
        v = AuditVerdict(card_id=card.card_id, unit_id=unit.unit_id, passed=False)
        if card.error:
            v.reasoning = f"scout error: {card.error}"
            v.lesson = self._lesson_for_error(unit, card)
            v.retry_strategy = self._next_strategy(card.strategy, unit)
            return v
        for c in card.comps:
            why = self.check_comp(unit, c)
            if why:
                v.rejected[c.listing_id] = why
            else:
                v.accepted.append(c.listing_id)
        local = [c for c in card.comps if c.listing_id in v.accepted and self.is_local(unit, c)]
        sources = {c.source for c in card.comps if c.listing_id in v.accepted and c.source}
        v.second_source = len(sources) >= int(self.cmp.get("min_sources_for_canon", 2))
        need = int(self.cmp.get("min_comps_for_verified", 3))
        v.passed = len(v.accepted) >= min(need, 2)
        if not v.passed:
            v.reasoning = f"only {len(v.accepted)} comparable of {len(card.comps)}; rejected: {list(v.rejected.values())[:3]}"
            v.lesson = (f"Strategy '{card.strategy}' for {unit.label} yielded {len(v.accepted)} comparable comps "
                        f"({', '.join(sorted(set(v.rejected.values()))[:2]) or 'none found'}); try a different source")
            v.retry_strategy = self._next_strategy(card.strategy, unit)
        else:
            v.reasoning = f"{len(v.accepted)} comparable comps from {len(sources)} source(s), {len(local)} within region"
        return v

    def _lesson_for_error(self, unit: Unit, card: FindingCard) -> str:
        if "blocked" in card.error:
            return f"Source for strategy '{card.strategy}' blocks scraping (403); do not use it for {unit.category} units"
        if "no search results" in card.error:
            return f"Strategy '{card.strategy}' returned 0 results for '{card.query}'; use a site-specific source"
        return f"Strategy '{card.strategy}' failed ({card.error}); rotate strategy"

    def _next_strategy(self, current: str, unit: Unit) -> str:
        prefs = {"truck": ["autotrader", "cars_com", "broad"], "default": ["rvtrader", "rvusa", "broad"]}
        order = prefs["truck" if unit.category == "truck" else "default"]
        for s in order:
            if s != current and s in self.strategies:
                return s
        return "broad"

    # ------------------------------------------------------------------ entry
    async def audit(self, unit: Unit, card: FindingCard, use_llm: bool = True) -> AuditVerdict:
        v = self.rule_audit(unit, card)
        if use_llm:
            user = (f"UNIT: {unit.label}, {unit.mileage} mi, {unit.location} {unit.state}. RULES: {self.cmp}\n"
                    f"CARD: {card.compact()}\nRULE AUDIT: passed={v.passed} accepted={v.accepted} rejected={v.rejected} "
                    f"second_source={v.second_source} suggested_retry={v.retry_strategy}\nALLOWED STRATEGIES: {self.strategies}\nReturn the JSON.")
            try:
                out = await self.log.llm_json(self.llm, "auditor", "auditor", AUDIT_SYSTEM, user,
                                              meta={"payload": {"verdict": v.model_dump()}}, unit_id=unit.unit_id, max_tokens=300)
                if isinstance(out, dict):
                    v.reasoning = str(out.get("reasoning") or v.reasoning)[:200]
                    if not v.passed:
                        v.lesson = str(out.get("lesson") or v.lesson)[:200]
                        rs = str(out.get("retry_strategy") or "")
                        if rs in self.strategies and rs != card.strategy:
                            v.retry_strategy = rs
            except Exception as exc:
                await self.log.error("auditor", "llm", exc, unit_id=unit.unit_id)
        await self.log.emit("auditor", "audit_done", unit_id=unit.unit_id,
                            payload={"card_id": card.card_id, "passed": v.passed, "accepted": len(v.accepted), "rejected": v.rejected,
                                     "second_source": v.second_source, "reasoning": v.reasoning})
        if v.lesson:
            await self.log.emit("auditor", "lesson", unit_id=unit.unit_id,
                                payload={"lesson": v.lesson, "failed_strategy": card.strategy, "retry_strategy": v.retry_strategy})
        return v


def _state(location: str) -> str:
    parts = [p.strip() for p in location.split(",")]
    if len(parts) >= 2 and len(parts[-1]) == 2:
        return parts[-1].upper()
    return ""
