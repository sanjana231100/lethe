"""CHIEF: orchestrator / planner. Never sees raw web data.

Sees only: Mission Ledger + canon facts + top working facts + recent lessons (budgeted by
memory/context.py) plus this step's evidence (compact Finding Cards + audit verdicts +
deterministic price stats). Edits its own context through explicit tools:

  update_plan, record_decision, add_lesson, request_scout, request_audit,
  set_recommendation, mark_stale
"""
from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider, count_tokens
from ..memory.context import assemble_context
from ..memory.pricing import price_stats
from ..models import AuditVerdict, Decision, Fact, FindingCard, Ledger, Lesson, PlanStep, Unit, UnitStatus, iso, now, parse_ts
from ..runtime.events import EventLog

TOOLS = ["update_plan", "record_decision", "add_lesson", "request_scout", "request_audit", "set_recommendation", "mark_stale"]


class Action(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ChiefOutput(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    thinking: str = ""


CHIEF_SYSTEM = """ROLE: chief
You are the Chief of a long-running pricing agent for a vehicle dealership. You never see raw web pages;
you see a bounded MISSION LEDGER, verified facts, and this step's evidence. You edit your own state ONLY
through tools. Return ONLY JSON: {"thinking":"<=40 words","actions":[{"tool":"...","args":{...}}]}
Tools:
- request_scout {unit_id, strategy}         dispatch a fresh scout (strategy from ALLOWED STRATEGIES; obey lessons)
- set_recommendation {unit_id, price, reasoning, confidence}   price is a number; reasoning <=30 words
- mark_stale {unit_id, why}
- update_plan {steps:[{step, status}]}       rewrite the short plan (<=6 steps)
- record_decision {what, why}
- add_lesson {lesson, unit_id}
Be decisive and terse."""


class Chief:
    def __init__(self, llm: LLMProvider, log: EventLog, mission: dict[str, Any], units: dict[str, Unit], token_budget: int):
        self.llm = llm
        self.log = log
        self.mission = mission
        self.units = units
        self.budget = token_budget
        self.strategies: list[str] = mission.get("strategies", ["broad"])
        self.refresh = mission.get("refresh", {})
        self.pricing = mission.get("pricing", {})
        self._tool_errors: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ due units (deterministic part of planning)
    def due_units(self, ledger: Ledger, max_units: int) -> list[dict[str, Any]]:
        t = now()
        stale_after = int(self.refresh.get("reco_stale_after_seconds", 1800))
        low = float(self.refresh.get("low_confidence_threshold", 0.55))
        due = []
        for u in ledger.units:
            why = None
            if not u.last_checked:
                why = "never checked"
            elif u.status == "stale":
                why = "stale"
            elif u.confidence < low:
                why = f"low confidence {u.confidence:.2f}"
            elif (t - parse_ts(u.last_checked)).total_seconds() > stale_after:
                why = "refresh interval elapsed"
            if why:
                due.append({"unit_id": u.unit_id, "why": why, "strategy": u.strategy, "failures": u.failures})
        due.sort(key=lambda d: (d["why"] != "never checked", d["why"] != "stale"))
        return due[:max_units]

    def _default_strategy(self, unit: Unit, ledger: Ledger, current: str) -> str:
        """Pick a strategy that lessons haven't ruled out for this unit."""
        bad = set()
        for l in ledger.lessons:
            if l.unit_id in ("", unit.unit_id):
                for s in self.strategies:
                    if f"'{s}'" in l.lesson and ("blocks" in l.lesson or "0 results" in l.lesson or "yielded" in l.lesson or "do not use" in l.lesson):
                        bad.add(s)
        order = ["autotrader", "cars_com", "broad", "rvtrader"] if unit.category == "truck" else ["rvtrader", "rvusa", "broad"]
        if current not in bad and current in self.strategies:
            return current
        for s in order:
            if s in self.strategies and s not in bad:
                return s
        return "broad"

    # ------------------------------------------------------------------ plan
    async def plan(self, ledger: Ledger, facts: list[Fact], max_units: int) -> list[tuple[str, str]]:
        due = self.due_units(ledger, max_units)
        for d in due:
            d["suggested_strategy"] = self._default_strategy(self.units[d["unit_id"]], ledger, d["strategy"])
        ctx = assemble_context(ledger, facts, unit_ids=[d["unit_id"] for d in due], budget_tokens=self.budget, reserve_tokens=900)
        await self.log.emit("chief", "context_assembled", payload={"phase": "plan", **ctx.summary()})
        user = (f"{ctx.text}\n\nUNITS DUE THIS CYCLE: {json.dumps(due)}\nALLOWED STRATEGIES: {self.strategies}\n"
                f"Decide which scouts to dispatch (request_scout, one per due unit, choose strategy) and update the plan. Return the JSON.")
        jobs: list[tuple[str, str]] = []
        try:
            out: ChiefOutput = await self.log.llm_json(self.llm, "chief", "chief_plan", CHIEF_SYSTEM, user,
                                                      meta={"payload": {"due": due, "ledger": ledger.compact()}}, validator=ChiefOutput, max_tokens=700)
            jobs = self.apply_actions(ledger, out.actions, phase="plan")
            await self.drain_tool_errors()
        except Exception as exc:
            await self.log.error("chief", "plan_llm", exc)
        if not jobs:  # deterministic fallback keeps the loop alive (rule #7)
            jobs = [(d["unit_id"], d["suggested_strategy"]) for d in due]
        for uid, strat in jobs:
            u = ledger.unit(uid)
            if u.strategy != strat:
                await self.log.emit("chief", "strategy_changed", unit_id=uid, payload={"from": u.strategy, "to": strat})
            u.strategy = strat
            u.status = "scouting"
        await self.log.emit("chief", "chief_plan", payload={"jobs": jobs, "due": [d["unit_id"] for d in due]})
        return jobs

    # ------------------------------------------------------------------ update
    async def update(self, ledger: Ledger, facts: list[Fact], cards: list[FindingCard], verdicts: list[AuditVerdict]) -> None:
        by_unit_cards = {c.unit_id: c for c in cards}
        by_unit_v = {v.unit_id: v for v in verdicts}
        evidence = []
        for uid, v in by_unit_v.items():
            unit = self.units[uid]
            comps = self._accepted_comps(uid, facts)
            stats = price_stats(unit, comps, target_percentile=float(self.pricing.get("target_percentile", 0.45)),
                                max_change_pct=float(self.pricing.get("max_change_pct", 12)))
            evidence.append({"unit_id": uid, "current_reco": ledger.unit(uid).current_reco, "audit": {"passed": v.passed, "accepted": len(v.accepted), "second_source": v.second_source,
                                                       "reasoning": v.reasoning, "lesson": v.lesson},
                             "card": by_unit_cards[uid].compact() if uid in by_unit_cards else None, "price_stats": stats})
        ctx = assemble_context(ledger, facts, unit_ids=list(by_unit_v), budget_tokens=self.budget, reserve_tokens=1600)
        await self.log.emit("chief", "context_assembled", payload={"phase": "update", **ctx.summary()})
        user = (f"{ctx.text}\n\nTHIS STEP'S EVIDENCE:\n{json.dumps(evidence, default=str)}\n\n"
                f"For each unit: set_recommendation if the audit passed (use price_stats.suggested unless you have a reason), "
                f"otherwise mark_stale. record_decision for notable changes. Return the JSON.")
        try:
            out: ChiefOutput = await self.log.llm_json(self.llm, "chief", "chief_update", CHIEF_SYSTEM, user,
                                                      meta={"payload": {"evidence": evidence, "ledger": ledger.compact()}}, validator=ChiefOutput, max_tokens=900)
            self.apply_actions(ledger, out.actions, phase="update")
            await self.drain_tool_errors()
            handled = {a.args.get("unit_id") for a in out.actions if a.tool in ("set_recommendation", "mark_stale")}
        except Exception as exc:
            await self.log.error("chief", "update_llm", exc)
            handled = set()
        # deterministic fallback for anything the model skipped
        for e in evidence:
            if e["unit_id"] in handled:
                continue
            if e["audit"]["passed"]:
                self._set_reco(ledger, e["unit_id"], e["price_stats"]["suggested"], "auto: median of audited comps", e["price_stats"]["confidence"])
            else:
                self._mark_stale(ledger, e["unit_id"], e["audit"]["reasoning"])
        for e in evidence:
            u = ledger.unit(e["unit_id"])
            u.last_checked = iso()
            u.comps_count = e["price_stats"]["n"]
            u.failures = 0 if e["audit"]["passed"] else u.failures + 1
        ledger.updated_at = iso()
        ledger.trim()
        await self.log.emit("chief", "chief_update", payload={"units": [{"unit_id": e["unit_id"], "reco": ledger.unit(e["unit_id"]).current_reco,
                                                                        "status": ledger.unit(e["unit_id"]).status} for e in evidence]})

    def _accepted_comps(self, unit_id: str, facts: list[Fact]) -> list[dict[str, Any]]:
        by_subject: dict[str, dict[str, Any]] = {}
        for f in facts:
            if f.unit_id != unit_id or f.status != "active":
                continue
            d = by_subject.setdefault(f.subject, {"listing_id": f.subject.split(":", 1)[-1], "confidence": f.confidence})
            if f.predicate == "asking_price":
                d["price"] = float(f.value)
            elif f.predicate == "specs":
                parts = f.value.split("|")
                d["year"] = parts[0]
                try:
                    d["mileage"] = int(parts[2].rstrip("mi") or 0) if len(parts) > 2 else 0
                except ValueError:
                    d["mileage"] = 0
                d["source"] = parts[4] if len(parts) > 4 else ""
                loc = parts[3] if len(parts) > 3 else ""
                st = loc.split(",")[-1].strip().upper()[:2] if "," in loc else ""
                d["local"] = st in self.mission.get("region_adjacency", {}).get(self.units[unit_id].state, [])
            elif f.predicate == "status":
                d["status"] = f.value
        return [d for d in by_subject.values() if d.get("price") and d.get("status", "for_sale") == "for_sale" and d["confidence"] >= 0.5]

    # ------------------------------------------------------------------ tools
    def apply_actions(self, ledger: Ledger, actions: list[Action], *, phase: str) -> list[tuple[str, str]]:
        jobs: list[tuple[str, str]] = []
        for a in actions:
            try:
                if a.tool == "request_scout":
                    uid = a.args["unit_id"]; ledger.unit(uid)
                    s = a.args.get("strategy", "broad")
                    if s not in self.strategies:
                        s = "broad"
                    jobs.append((uid, s))
                elif a.tool == "set_recommendation":
                    self._set_reco(ledger, a.args["unit_id"], float(a.args["price"]), str(a.args.get("reasoning", ""))[:200],
                                   float(a.args.get("confidence", 0.5)))
                elif a.tool == "mark_stale":
                    self._mark_stale(ledger, a.args["unit_id"], str(a.args.get("why", ""))[:200])
                elif a.tool == "update_plan":
                    norm = {"in_progress": "doing", "in-progress": "doing", "doing": "doing", "done": "done", "complete": "done",
                            "completed": "done", "failed": "failed", "todo": "todo", "pending": "todo"}
                    steps = []
                    for s in a.args.get("steps", [])[:6]:
                        if isinstance(s, str):
                            steps.append(PlanStep(step=s[:80]))
                        else:
                            steps.append(PlanStep(step=str(s.get("step", ""))[:80], status=norm.get(str(s.get("status", "todo")).lower(), "todo")))
                    ledger.plan = steps
                elif a.tool == "record_decision":
                    what = str(a.args.get("what", ""))[:160]
                    if what not in {d.what for d in ledger.decisions[-4:]}:   # dedupe repeats
                        ledger.decisions.append(Decision(what=what, why=str(a.args.get("why", ""))[:160]))
                elif a.tool == "add_lesson":
                    ledger.lessons.append(Lesson(lesson=str(a.args.get("lesson", ""))[:200], unit_id=str(a.args.get("unit_id", "")), source="chief"))
                elif a.tool == "request_audit":
                    pass  # audits are always run for every card in this runtime
                else:
                    raise KeyError(a.tool)
            except Exception as exc:
                # invalid tool call: recorded, never fatal (emitted by the caller via drain_tool_errors)
                self._tool_errors.append({"tool": a.tool, "args": a.args, "error": str(exc)[:120], "phase": phase})
        return jobs

    async def drain_tool_errors(self) -> None:
        errs, self._tool_errors = self._tool_errors, []
        for e in errs:
            await self.log.emit("chief", "tool_error", payload=e)

    def _set_reco(self, ledger: Ledger, uid: str, price: float, reasoning: str, confidence: float) -> None:
        u = ledger.unit(uid)
        unit = self.units[uid]
        cap = unit.our_price * float(self.pricing.get("max_change_pct", 12)) / 100
        price = max(unit.our_price - cap, min(unit.our_price + cap, price))   # guardrail regardless of the LLM
        changed = u.current_reco and abs(price - u.current_reco) >= 250
        u.current_reco = round(price)
        u.confidence = max(0.0, min(1.0, confidence))
        u.reasoning = reasoning
        u.status = "action" if abs(price - unit.our_price) / unit.our_price >= 0.03 else "verified"
        if changed:
            ledger.decisions.append(Decision(what=f"{uid} reco {u.current_reco:,.0f}", why=reasoning[:120]))

    def _mark_stale(self, ledger: Ledger, uid: str, why: str) -> None:
        u = ledger.unit(uid)
        u.status = "stale" if u.failures >= 1 else "needs_verify"
        u.reasoning = why
