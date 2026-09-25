"""Nimble web data client (the Scouts' eyes) + recorded-fixture mock.

Interface:
  search(query, *, count)  -> list[SearchResult]   real-time SERP, parsed
  extract(url)             -> PageContent          real-time page fetch, text/markdown

Raw pages are *scratch* memory: they live only inside a Scout and are never persisted
to a prompt. Only a pointer (url + sha1) is archived in the events stream.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import ROOT, Settings, settings as default_settings


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    position: int = 0
    content: str = ""   # full page content when the search API returns it (saves an extract call)

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.replace("www.", "")


@dataclass
class PageContent:
    url: str
    text: str
    status_code: int = 200
    blocked: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def sha1(self) -> str:
        return hashlib.sha1(self.text.encode()).hexdigest()[:12]

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.replace("www.", "")


class NimbleClient:
    """Base interface. `NimbleAPI` is the live client, `MockNimble` replays fixtures."""
    name = "nimble"

    async def search(self, query: str, *, count: int = 10, include_domains: list[str] | None = None) -> list[SearchResult]:
        raise NotImplementedError

    async def extract(self, url: str) -> PageContent:
        raise NotImplementedError

    async def close(self) -> None:
        return None


# ============================================================================= live
class NimbleAPI(NimbleClient):
    """Nimble Web API. Endpoint shapes are in tools/nimble_api.py (kept separate so the
    exact request format can be verified against docs without touching the interface)."""
    name = "nimble-live"

    def __init__(self, api_key: str):
        from .nimble_api import build_extract_request, build_search_request, parse_extract_response, parse_search_response
        self._build_search = build_search_request
        self._build_extract = build_extract_request
        self._parse_search = parse_search_response
        self._parse_extract = parse_extract_response
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=60)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
           retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)), reraise=True)
    async def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        r = await self._client.post(url, json=body, headers=self._headers())
        if r.status_code >= 500 or r.status_code == 429:
            r.raise_for_status()
        if r.status_code >= 400:
            raise RuntimeError(f"nimble {r.status_code}: {r.text[:300]}")
        return r.json()

    async def search(self, query: str, *, count: int = 10, include_domains: list[str] | None = None) -> list[SearchResult]:
        url, body = self._build_search(query, count, include_domains)
        data = await self._post(url, body)
        return [SearchResult(**r) for r in self._parse_search(data)][:count]

    async def extract(self, url: str) -> PageContent:
        api_url, body = self._build_extract(url)
        try:
            data = await self._post(api_url, body)
        except RuntimeError as exc:
            return PageContent(url=url, text="", status_code=403 if "403" in str(exc) else 500, blocked=True, meta={"error": str(exc)[:200]})
        return PageContent(url=url, **self._parse_extract(data))

    async def close(self) -> None:
        await self._client.aclose()


# ============================================================================= mock
class MockNimble(NimbleClient):
    """Replays fixtures/nimble/listings.json.

    Listings evolve with `MockNimble.cycle` (set by the loop) so the demo shows real
    market dynamics offline: new comps appear, prices drop, listings get SOLD, and one
    source ("marketplace") is always blocked so the Auditor has to learn a lesson.
    """
    name = "nimble-mock"
    cycle: int = 0

    def __init__(self, fixture_path: Path | None = None):
        path = fixture_path or ROOT / "fixtures" / "nimble" / "listings.json"
        self.data = json.loads(path.read_text())
        self.listings: list[dict[str, Any]] = self.data["listings"]
        self.blocked_domains: set[str] = set(self.data.get("blocked_domains", []))

    @classmethod
    def set_cycle(cls, n: int) -> None:
        cls.cycle = n

    def _visible(self, l: dict[str, Any]) -> bool:
        return l.get("appears_cycle", 0) <= self.cycle

    def _price_at(self, l: dict[str, Any]) -> float:
        price = l["price"]
        for c, p in sorted(((int(k), v) for k, v in l.get("price_changes", {}).items())):
            if c <= self.cycle:
                price = p
        return price

    def _sold(self, l: dict[str, Any]) -> bool:
        sc = l.get("sold_cycle")
        return sc is not None and self.cycle >= sc

    @staticmethod
    def _site_filter(query: str) -> str | None:
        m = re.search(r"site:([\w.]+)", query)
        return m.group(1) if m else None

    async def search(self, query: str, *, count: int = 10, include_domains: list[str] | None = None) -> list[SearchResult]:
        q = query.lower()
        site = self._site_filter(q) or (include_domains[0] if include_domains else None)
        tokens = [t for t in re.sub(r"site:\S+", "", q).split() if t not in {"for", "sale", "used", "or", "listing"}]
        results = []
        for l in self.listings:
            if not self._visible(l):
                continue
            hay = f"{l['year']} {l['make']} {l['model']} {l.get('trim','')} {l['title']}".lower()
            if not all(t in hay for t in tokens if len(t) > 2 and not t.isdigit()):
                continue
            if any(t.isdigit() and len(t) == 4 and abs(int(t) - l["year"]) > 1 for t in tokens):
                continue
            if site and site not in l["source"]:
                continue
            if site is None and l.get("only_via_site"):
                continue
            price_txt = "SOLD" if self._sold(l) else f"${self._price_at(l):,.0f}"
            results.append(SearchResult(title=f"{l['title']} - {price_txt}", url=l["url"],
                                        snippet=f"{l['year']} {l['make']} {l['model']} {l.get('trim','')} | {l.get('mileage','')} mi | {l['location']}",
                                        position=len(results) + 1))
        return results[:count]

    async def extract(self, url: str) -> PageContent:
        domain = urlparse(url).netloc.replace("www.", "")
        if domain in self.blocked_domains:
            return PageContent(url=url, text="Access Denied - please verify you are human (403)", status_code=403, blocked=True)
        for l in self.listings:
            if l["url"] == url and self._visible(l):
                status = "SOLD" if self._sold(l) else "For Sale"
                lines = [
                    f"# {l['title']}",
                    f"Status: {status}",
                    f"Price: {'N/A - this listing has been sold' if self._sold(l) else '$' + format(self._price_at(l), ',.0f')}",
                    f"Year: {l['year']}  Make: {l['make']}  Model: {l['model']}  Trim: {l.get('trim','')}",
                    f"Mileage: {l.get('mileage', 'n/a')} miles",
                    f"Location: {l['location']}",
                    f"Seller: {l.get('seller','Private seller')}",
                    "",
                    l.get("description", "Well maintained, clean title, ready to go. Call for details."),
                    "",
                    "Related listings: " + ", ".join(o["title"] for o in self.listings[:3] if o is not l),
                    "Terms | Privacy | Contact | © listing site",
                ]
                return PageContent(url=url, text="\n".join(lines), status_code=200)
        return PageContent(url=url, text="404 Not Found - This listing is no longer available.", status_code=404)


def make_nimble(cfg: Settings | None = None) -> NimbleClient:
    cfg = cfg or default_settings
    if cfg.mock or not cfg.nimble_api_key:
        return MockNimble()
    return NimbleAPI(cfg.nimble_api_key)
