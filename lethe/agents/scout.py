"""SCOUT: disposable worker (hard rule #3).

Fresh, empty context every time. Nimble search -> read top pages (scratch memory) ->
one bounded LLM extraction call -> Finding Card (<=200 tokens). Raw pages are discarded;
only a pointer (url + sha1 + size) is archived in the events stream.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider, count_tokens
from ..models import FindingCard, Listing, Unit
from ..runtime.events import EventLog
from ..tools.nimble import NimbleClient, PageContent

PAGE_TOKEN_CAP = 650      # per page, scratch -> prompt slice
MAX_PAGES = 3
CARD_TOKEN_CAP = 200

STRATEGY_DOMAINS: dict[str, list[str]] = {
    "broad": [],
    "rvtrader": ["rvtrader.com"],
    "rvusa": ["rvusa.com"],
    "autotrader": ["autotrader.com"],
    "cars_com": ["cars.com"],
    "marketplace": ["facebook.com", "craigslist.org"],
}


def build_query(unit: Unit, strategy: str) -> tuple[str, list[str]]:
    base = f"{unit.year} {unit.make} {unit.model} {unit.trim} for sale".replace("  ", " ").strip()
    if unit.category == "truck" and strategy == "broad":
        base = f"used {base}"
    domains = STRATEGY_DOMAINS.get(strategy, [])
    query = base if not domains else f"{base} site:{domains[0]}"
    return query, domains


from pydantic import field_validator


class ExtractedListing(BaseModel):
    """Tolerant: small models return null / strings where numbers are expected."""
    listing_id: str | None = ""
    title: str | None = ""
    year: int | None = None
    make: str | None = ""
    model: str | None = ""
    price: float | None = None
    mileage: int | None = None
    location: str | None = ""
    status: str | None = "for_sale"

    @field_validator("listing_id", "title", "make", "model", "location", "status", mode="before")
    @classmethod
    def _none_to_str(cls, v):
        return "" if v is None else str(v)

    @field_validator("price", "mileage", "year", mode="before")
    @classmethod
    def _num(cls, v):
        if v in (None, "", "null", "N/A", "n/a"):
            return None
        if isinstance(v, str):
            digits = re.sub(r"[^0-9.]", "", v)
            return float(digits) if digits else None
        return v


class ExtractionOutput(BaseModel):
    listings: list[ExtractedListing] = Field(default_factory=list)
    notes: str = ""


SCOUT_SYSTEM = """ROLE: scout_extract
You extract vehicle listings from web page text. For EACH page, return the listing(s) it describes.
Return ONLY JSON: {"listings":[{"listing_id":"<page index>-<n>","title":"","year":2022,"make":"","model":"","price":109995,"mileage":21500,"location":"City, ST","status":"for_sale|sold|removed"}],"notes":"<=15 words"}
Rules: price as a number (no $ or commas), null if absent. status=sold if the page says sold/no longer available.
At most 8 listings total; prefer the ones closest to the TARGET UNIT (same year +/-1, same model).
Ignore 'related listings' and navigation. Never invent values. Keep it compact."""


class Scout:
    def __init__(self, nimble: NimbleClient, llm: LLMProvider, log: EventLog):
        self.nimble = nimble
        self.llm = llm
        self.log = log

    async def run(self, unit: Unit, strategy: str) -> FindingCard:
        query, domains = build_query(unit, strategy)
        card = FindingCard(unit_id=unit.unit_id, strategy=strategy, query=query)
        await self.log.emit("scout", "scout_start", unit_id=unit.unit_id, payload={"strategy": strategy, "query": query})
        try:
            results = await self.nimble.search(query, count=8, include_domains=domains or None)
        except Exception as exc:
            await self.log.error("scout", "nimble.search", exc, unit_id=unit.unit_id)
            card.error = f"search failed: {type(exc).__name__}"
            await self.log.emit("scout", "scout_done", unit_id=unit.unit_id, payload={**card.compact(), "pages_read": 0})
            return card
        await self.log.emit("scout", "tool_call", unit_id=unit.unit_id,
                            payload={"tool": "nimble.search", "query": query, "n_results": len(results), "domains": domains})
        if not results:
            card.error = "no search results"
            card.notes = f"0 results for '{query}'"
            await self.log.emit("scout", "scout_done", unit_id=unit.unit_id, payload={**card.compact(), "pages_read": 0})
            return card

        # ---- read pages (scratch memory) ----
        pages: list[PageContent] = []
        blocked = 0
        for r in results[:MAX_PAGES]:
            if r.content:
                page = PageContent(url=r.url, text=(r.snippet + "\n" if r.snippet else "") + r.content)
            else:
                try:
                    page = await self.nimble.extract(r.url)
                except Exception as exc:
                    await self.log.error("scout", "nimble.extract", exc, unit_id=unit.unit_id)
                    continue
            await self.log.emit("scout", "tool_call", unit_id=unit.unit_id,
                                payload={"tool": "nimble.extract", "url": r.url, "status": page.status_code, "blocked": page.blocked,
                                         "bytes": len(page.text), "sha1": page.sha1})  # scratch pointer only
            if page.blocked:
                blocked += 1
                continue
            if page.status_code == 404 or not page.text.strip():
                pages.append(PageContent(url=r.url, text=f"{r.title}\n{r.snippet}\n(listing page unavailable: {page.status_code})", status_code=page.status_code))
                continue
            pages.append(page)
        card.pages_read = len(pages)
        card.sources = sorted({p.domain for p in pages})
        if blocked and not pages:
            card.error = f"blocked by source ({blocked} pages 403)"
            await self.log.emit("scout", "scout_done", unit_id=unit.unit_id, payload={**card.compact(), "pages_read": 0, "blocked": blocked})
            return card

        # ---- one fresh-context extraction call, bounded ----
        chunks = []
        for i, p in enumerate(pages):
            txt = clean_page(p.text, unit, PAGE_TOKEN_CAP)
            chunks.append(f"### PAGE {i} url={p.url}\n{txt}")
        user = (f"TARGET UNIT: {unit.label} ({unit.category}, {unit.mileage} mi, ours ${unit.our_price:,.0f}, {unit.location}, {unit.state})\n\n"
                + "\n\n".join(chunks) + "\n\nReturn the JSON.")
        try:
            out: ExtractionOutput = await self.log.llm_json(
                self.llm, "scout", "scout_extract", SCOUT_SYSTEM, user,
                meta={"payload": {"pages": [{"url": p.url, "text": p.text} for p in pages], "unit": unit.model_dump()}},
                unit_id=unit.unit_id, validator=ExtractionOutput, max_tokens=1600)
        except Exception as exc:
            await self.log.error("scout", "extract_llm", exc, unit_id=unit.unit_id)
            card.error = f"extraction failed: {type(exc).__name__}"
            out = ExtractionOutput()

        page_counts: dict[str, int] = {}
        for l in out.listings:
            pg = _page_for(l.listing_id or "", pages)
            if pg:
                page_counts[pg.url] = page_counts.get(pg.url, 0) + 1
        seen: set[str] = set()
        for i, l in enumerate(out.listings):
            src_page = _page_for(l.listing_id or "", pages)
            url = src_page.url if src_page else ""
            src = src_page.domain if src_page else ""
            lid = _stable_listing_id(url, l, i, shared_page=page_counts.get(url, 0) > 1)
            if lid in seen:
                continue
            seen.add(lid)
            st = (l.status or "").lower()
            status = "sold" if "sold" in st else "removed" if "remov" in st or "unavailable" in st else "for_sale" if st in ("", "for_sale", "available", "active") else "unknown"
            card.comps.append(Listing(listing_id=lid, title=(l.title or "")[:80], year=int(l.year) if l.year else None, make=l.make or "", model=l.model or "",
                                      price=l.price, mileage=int(l.mileage) if l.mileage else None, location=(l.location or "")[:40], url=url, source=src, status=status))  # type: ignore[arg-type]
        card.notes = out.notes[:120]
        card = _enforce_card_budget(card)
        if blocked:
            card.notes = (card.notes + f" | {blocked} page(s) blocked").strip(" |")
        await self.log.emit("scout", "scout_done", unit_id=unit.unit_id,
                            payload={**card.compact(), "pages_read": card.pages_read, "card_tokens": card_tokens(card), "blocked": blocked})
        return card


_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_PRICE = re.compile(r"\$\s?\d{1,3}(,\d{3})+|\$\s?\d{4,6}")
_NOISE = re.compile(r"(cookie|privacy|terms of use|sign in|log in|advertis|subscribe|newsletter|©|all rights|disclaimer|icc=|trader\.com/\?)", re.I)


def clean_page(text: str, unit: Unit, cap: int) -> str:
    """Scratch -> prompt slice. Index pages are 10k+ tokens of nav; keep the lines that carry
    listing signal (price / year / mileage / model name) so the extraction call stays small."""
    text = _LINK.sub(lambda m: m.group(1) or "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [l.strip(" -*#|>") for l in text.splitlines()]
    lines = [l for l in lines if len(l) > 3 and not _NOISE.search(l)]
    model_key = unit.model.lower().split()[0]
    year_re = re.compile(rf"\b({unit.year - 1}|{unit.year}|{unit.year + 1})\b")
    scored: list[tuple[int, int, str]] = []
    for i, l in enumerate(lines):
        sc = 0
        if _PRICE.search(l): sc += 3
        if year_re.search(l): sc += 2
        if model_key in l.lower(): sc += 2
        if re.search(r"\b(mi|miles|mileage)\b", l, re.I): sc += 1
        if re.search(r"\b(sold|no longer available)\b", l, re.I): sc += 2
        if re.search(r",\s?[A-Z]{2}\b", l): sc += 1
        if sc:
            scored.append((-sc, i, l))
    scored.sort()
    kept, used = [], 0
    for _, i, l in scored:
        l = l[:300]
        t = count_tokens(l) + 1
        if used + t > cap:
            continue
        kept.append((i, l)); used += t
    if not kept:  # nothing listing-like: fall back to the head of the page
        return _truncate_tokens("\n".join(lines), cap)
    return "\n".join(l for _, l in sorted(kept))   # original order, signal lines only


def _truncate_tokens(text: str, cap: int) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if count_tokens(text) <= cap:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # binary search on characters
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= cap:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + " ..."


def _page_for(listing_id: str, pages: list[PageContent]) -> PageContent | None:
    m = re.match(r"(\d+)", listing_id or "")
    if m and int(m.group(1)) < len(pages):
        return pages[int(m.group(1))]
    return pages[0] if len(pages) == 1 else None


def _stable_listing_id(url: str, l: ExtractedListing, i: int, shared_page: bool = False) -> str:
    """Short, stable id: <site>-<last path token>, e.g. rvtrader-5110231. Listings that come from the
    same index page get <site>-<hash of year|title|price|location> so they never collide."""
    if url and shared_page:
        host = url.split("//")[-1].split("/")[0].replace("www.", "").split(".")[0]
        key = f"{l.year}|{(l.title or '').lower()}|{l.price}|{(l.location or '').lower()}"
        return f"{host}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"
    if url:
        host = url.split("//")[-1].split("/")[0].replace("www.", "").split(".")[0]
        path = [t for t in url.split("//")[-1].split("/")[1:] if t]
        tail = re.sub(r"[^a-zA-Z0-9]+", "", path[-1])[-12:] if path else hashlib.sha1(url.encode()).hexdigest()[:8]
        return f"{host}-{tail or hashlib.sha1(url.encode()).hexdigest()[:8]}"
    return hashlib.sha1(f"{l.title}|{l.price}|{l.location}|{i}".encode()).hexdigest()[:10]


def card_tokens(card: FindingCard) -> int:
    import json
    return count_tokens(json.dumps(card.compact(), separators=(",", ":")))


def _enforce_card_budget(card: FindingCard) -> FindingCard:
    """Finding Cards are <=200 tokens on the wire: drop the least useful comps until it fits."""
    card.comps.sort(key=lambda c: (c.price is None, c.status != "for_sale"))
    card.notes = card.notes[:60]
    while card_tokens(card) > CARD_TOKEN_CAP and len(card.comps) > 1:
        card.comps.pop()
    return card


async def run_scouts(scout: Scout, jobs: list[tuple[Unit, str]], concurrency: int) -> list[FindingCard]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(unit: Unit, strategy: str) -> FindingCard:
        async with sem:
            try:
                return await scout.run(unit, strategy)
            except Exception as exc:  # a scout must never crash the loop
                await scout.log.error("scout", "run", exc, unit_id=unit.unit_id)
                return FindingCard(unit_id=unit.unit_id, strategy=strategy, query="", error=f"crash: {type(exc).__name__}")

    return list(await asyncio.gather(*(one(u, s) for u, s in jobs)))
