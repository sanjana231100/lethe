"""Mission loading: YAML spec + inventory CSV -> Units + a fresh Ledger."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from ..config import ROOT
from ..models import Ledger, PlanStep, Unit, UnitState


def load_mission(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_units(mission: dict[str, Any], limit: int | None = None) -> dict[str, Unit]:
    inv = ROOT / mission.get("inventory", "data/inventory.csv")
    units: dict[str, Unit] = {}
    with open(inv) as f:
        for row in csv.DictReader(f):
            u = Unit(unit_id=row["unit_id"], year=int(row["year"]), make=row["make"], model=row["model"], trim=row.get("trim", ""),
                     category=row.get("category", ""), mileage=int(row.get("mileage") or 0), our_price=float(row["our_price"]),
                     location=row.get("location", ""), state=row.get("state", ""))
            units[u.unit_id] = u
            if limit and len(units) >= limit:
                break
    return units


def new_ledger(mission: dict[str, Any], units: dict[str, Unit]) -> Ledger:
    cmp = mission.get("comparability", {})
    constraints = [f"{k}: {v}" for k, v in cmp.items()]
    return Ledger(
        mission_id=mission["mission_id"],
        goal=" ".join(mission["goal"].split()),
        constraints=constraints,
        units=[UnitState(unit_id=u.unit_id, label=u.label, our_price=u.our_price) for u in units.values()],
        plan=[PlanStep(step="scout every unit once"), PlanStep(step="audit comps, promote verified facts"),
              PlanStep(step="publish recommendations; refresh stale units")],
    )


def sync_units(ledger: Ledger, units: dict[str, Unit]) -> None:
    """When resuming with a changed inventory (scaling up), add missing units to the ledger."""
    have = {u.unit_id for u in ledger.units}
    for u in units.values():
        if u.unit_id not in have:
            ledger.units.append(UnitState(unit_id=u.unit_id, label=u.label, our_price=u.our_price))
