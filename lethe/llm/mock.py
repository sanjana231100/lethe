"""Mock LLM (MOCK=1): rule-based stand-ins for every role so the whole system runs offline.

Each agent passes its structured input via `meta["payload"]`; the mock reasons over that
the way the prompt asks the real model to. Token counts are still *real* estimates of the
prompt that would have been sent, so the token chart is meaningful in mock mode.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..models import LLMResult
from .base import LLMProvider, count_tokens


class MockLLM(LLMProvider):
    name = "mock"

    async def _complete(self, system: str, user: str, *, role: str, meta: dict[str, Any], max_tokens: int,
                        temperature: float, json_mode: bool) -> LLMResult:
        payload = meta.get("payload", {})
        handler = getattr(self, f"_{role}", None)
        data = handler(payload) if handler else {"ok": True, "echo": user[-200:]}
        text = json.dumps(data)
        return LLMResult(text=text, input_tokens=count_tokens(system) + count_tokens(user) + 8,
                         output_tokens=count_tokens(text), latency_ms=12, model="mock", provider="mock")

    # ------------------------------------------------------------------ scout
    def _scout_extract(self, p: dict[str, Any]) -> dict[str, Any]:
        listings = []
        for i, page in enumerate(p.get("pages", [])):
            t = page["text"]
            title = (re.search(r"^# (.+)$", t, re.M) or re.search(r"^(.+)$", t, re.M)).group(1).strip()
            status = "sold" if re.search(r"status:\s*sold|has been sold|no longer available", t, re.I) else "for_sale"
            price_m = re.search(r"Price:\s*\$?([\d,]{4,})", t)
            price = float(price_m.group(1).replace(",", "")) if price_m else None
            if price is None:
                sm = re.search(r"\$([\d,]{5,})", t)
                price = float(sm.group(1).replace(",", "")) if sm else None
            year_m = re.search(r"Year:\s*(\d{4})", t) or re.search(r"\b(20\d{2}|19\d{2})\b", t)
            make_m = re.search(r"Make:\s*([^\n]+?)\s{2,}Model:\s*([^\n]+?)\s{2,}Trim:\s*([^\n]*)", t)
            mil_m = re.search(r"Mileage:\s*([\d,]+)", t)
            loc_m = re.search(r"Location:\s*([^\n]+)", t)
            listings.append({"listing_id": f"{i}-1", "title": title[:80], "year": int(year_m.group(1)) if year_m else None,
                             "make": make_m.group(1).strip() if make_m else "", "model": make_m.group(2).strip() if make_m else "",
                             "price": price, "mileage": int(mil_m.group(1).replace(",", "")) if mil_m and mil_m.group(1) != "n/a" else None,
                             "location": loc_m.group(1).strip() if loc_m else "", "status": status})
        return {"listings": listings, "notes": f"{len(listings)} listings extracted"}

    # ------------------------------------------------------------------ chief
    def _chief_plan(self, p: dict[str, Any]) -> dict[str, Any]:
        actions = [{"tool": "request_scout", "args": {"unit_id": d["unit_id"], "strategy": d.get("suggested_strategy", "broad")}} for d in p.get("due", [])]
        actions.append({"tool": "update_plan", "args": {"steps": [
            {"step": f"scout {len(actions)} due units", "status": "doing"},
            {"step": "audit cards; promote 2-source facts to canon", "status": "todo"},
            {"step": "update recommendations; GC expired facts", "status": "todo"}]}})
        return {"thinking": f"{len(p.get('due', []))} units due; dispatching scouts with lesson-aware strategies.", "actions": actions}

    def _chief_update(self, p: dict[str, Any]) -> dict[str, Any]:
        actions = []
        for e in p.get("evidence", []):
            s = e["price_stats"]
            if e["audit"]["passed"]:
                why = (f"{s['n']} audited comps, median ${s['median']:,.0f} ({s['low']:,.0f}-{s['high']:,.0f}), "
                       f"{s.get('n_sources', 1)} source(s); target p45 {s['delta_pct']:+.1f}% vs ours")
                actions.append({"tool": "set_recommendation", "args": {"unit_id": e["unit_id"], "price": s["suggested"], "reasoning": why, "confidence": s["confidence"]}})
                if abs(s["suggested"] - (e.get("current_reco") or 0)) >= 250 and e.get("current_reco"):
                    actions.append({"tool": "record_decision", "args": {"what": f"{e['unit_id']}: move reco to ${s['suggested']:,.0f}", "why": f"market moved; now {s['delta_pct']:+.1f}% vs our price"}})
            else:
                actions.append({"tool": "mark_stale", "args": {"unit_id": e["unit_id"], "why": e["audit"]["reasoning"]}})
        actions.append({"tool": "update_plan", "args": {"steps": [
            {"step": "scout due units", "status": "done"}, {"step": "audit + promote", "status": "done"},
            {"step": "recommendations updated; next: refresh stale units", "status": "done"}]}})
        return {"thinking": "Applying audited price stats; stale units get re-scouted next cycle.", "actions": actions}

    # ------------------------------------------------------------------ auditor
    def _auditor(self, p: dict[str, Any]) -> dict[str, Any]:
        v = p.get("verdict", {})
        return {"reasoning": v.get("reasoning", ""), "lesson": v.get("lesson", ""), "retry_strategy": v.get("retry_strategy", "")}

    # ------------------------------------------------------------------ librarian
    def _librarian(self, p: dict[str, Any]) -> dict[str, Any]:
        facts, contradictions = [], []
        for o in p.get("observations", []):
            subj = f"listing:{o['listing_id']}"
            conf = 0.8 if (o.get("price") and o.get("mileage") is not None) else (0.6 if o.get("price") else 0.4)
            if o.get("status") in ("sold", "removed"):
                contradictions.append({"subject": subj, "reason": f"listing {o['status']}"})
            facts.append({"subject": subj, "predicate": "status", "value": o.get("status", "for_sale"), "confidence": conf, "ttl_seconds": 7200})
            facts.append({"subject": subj, "predicate": "specs", "value": "see observation", "confidence": conf, "ttl_seconds": 21600})
            if o.get("price"):
                facts.append({"subject": subj, "predicate": "asking_price", "value": f"{o['price']:.0f}", "confidence": conf, "ttl_seconds": 7200})
        return {"facts": facts, "contradictions": contradictions, "notes": f"{len(facts)} facts, {len(contradictions)} contradictions"}

    # ------------------------------------------------------------------ naive baseline
    def _naive(self, p: dict[str, Any]) -> dict[str, Any]:
        return {"thought": "Considering all previous observations...", "recommendation": p.get("suggested", 0), "next": p.get("next_unit", "")}
