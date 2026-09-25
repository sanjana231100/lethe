"""Context assembly stays under budget and drops by priority (working before canon, canon before ledger)."""
from __future__ import annotations

from lethe.llm.base import count_tokens
from lethe.memory.context import assemble_context
from lethe.models import Fact, Ledger, Lesson, UnitState


def _ledger(n_units=4):
    return Ledger(mission_id="m", goal="price things", constraints=["year_tolerance: 1"],
                  units=[UnitState(unit_id=f"U{i:02d}", label=f"unit {i}", our_price=1000.0 * i) for i in range(1, n_units + 1)],
                  lessons=[Lesson(lesson=f"lesson {i}", unit_id="U01") for i in range(10)])


def _facts(n, gen="working", unit="U01"):
    return [Fact(fact_id=f"f{i}", unit_id=unit, subject=f"listing:{i}", predicate="asking_price", value=str(1000 + i),
                 confidence=0.5 + (i % 5) / 10, generation=gen) for i in range(n)]


def test_under_budget_and_priority():
    ledger = _ledger()
    facts = _facts(300) + _facts(50, gen="canon")
    rep = assemble_context(ledger, facts, unit_ids=["U01"], budget_tokens=2000, reserve_tokens=500)
    assert rep.tokens <= 1500
    assert count_tokens(rep.text) == rep.tokens
    assert rep.included["ledger"] == 1
    assert rep.included["canon"] == 50, "canon facts have priority over working facts"
    assert rep.dropped["working"] > 0
    assert rep.included["lessons"] <= 6 and rep.dropped["lessons"] >= 4


def test_ranking_prefers_task_units_and_confidence():
    ledger = _ledger()
    facts = _facts(5, unit="U02") + _facts(5, unit="U01")
    rep = assemble_context(ledger, facts, unit_ids=["U01"], budget_tokens=6000, max_working=5)
    assert "listing:" in rep.text
    # only 5 working facts are allowed; they must all be U01's
    assert rep.included["working"] == 5 and rep.dropped["working"] == 5
    lines = [l for l in rep.text.splitlines() if l.startswith("[w|")]
    assert len(lines) == 5


def test_ledger_truncates_when_huge():
    ledger = _ledger(n_units=80)
    rep = assemble_context(ledger, [], unit_ids=["U01"], budget_tokens=1500, reserve_tokens=300)
    assert rep.truncated_ledger and rep.tokens <= 1200
    assert '"unit_id":"U01"' in rep.text and '"unit_id":"U50"' not in rep.text
