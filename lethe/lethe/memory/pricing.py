"""Deterministic pricing math the Chief uses as a hint (the LLM writes reasoning + confidence)."""
from __future__ import annotations

import statistics
from typing import Any

from ..models import Unit


def price_stats(unit: Unit, comps: list[dict[str, Any]], *, target_percentile: float = 0.45, max_change_pct: float = 12.0) -> dict[str, Any]:
    """comps: [{listing_id, price, mileage, year, source}] already accepted by the Auditor."""
    prices = sorted(c["price"] for c in comps if c.get("price"))
    if not prices:
        return {"n": 0, "suggested": unit.our_price, "median": None, "low": None, "high": None, "note": "no comps"}
    median = statistics.median(prices)
    idx = min(len(prices) - 1, max(0, int(round(target_percentile * (len(prices) - 1)))))
    target = prices[idx]
    # mileage adjustment: comps with more miles than ours pull the market down relative to our unit
    adj = 0.0
    if unit.mileage and unit.category != "travel_trailer":
        miles = [c.get("mileage") for c in comps if c.get("mileage")]
        if miles:
            avg = sum(miles) / len(miles)
            adj = (avg - unit.mileage) / 10000 * 0.015 * median  # ~1.5% per 10k mi
    suggested = target + adj
    cap = unit.our_price * max_change_pct / 100
    suggested = max(unit.our_price - cap, min(unit.our_price + cap, suggested))
    suggested = round(suggested / 100) * 100
    spread = (prices[-1] - prices[0]) / median if median else 1
    n_sources = len({c.get("source") for c in comps})
    n_local = sum(1 for c in comps if c.get("local"))
    confidence = min(0.95, 0.35 + 0.12 * min(len(prices), 5) + (0.1 if n_sources >= 2 else 0) - min(0.2, spread * 0.4)
                     - (0.1 if len(prices) and n_local == 0 else 0))
    return {"n": len(prices), "n_sources": n_sources, "median": median, "low": prices[0], "high": prices[-1],
            "target_percentile_price": target, "mileage_adjustment": round(adj), "suggested": suggested,
            "our_price": unit.our_price, "delta_pct": round((suggested - unit.our_price) / unit.our_price * 100, 1),
            "n_local": n_local, "confidence": round(max(0.2, confidence), 2)}
