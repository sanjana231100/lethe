"""Exact Nimble v2 request/response shapes (verified against docs.nimbleway.com, Sept 2026).

Base URL : https://sdk.nimbleway.com/v2
Auth     : Authorization: Bearer <NIMBLE_API_KEY>
Search   : POST /search   {query, max_results, country, locale, search_depth, full_content, output_format, include_domains}
           -> {"request_id", "total_results", "results":[{"title","description","url","content","metadata":{"position"}}]}
Extract  : POST /extract  {url, formats:["markdown"], render:"auto", markdown_backend:"main_content", country}
           -> {"status","status_code","data":{"markdown","html",...}}
"""
from __future__ import annotations

from typing import Any

BASE = "https://sdk.nimbleway.com/v2"


def build_search_request(query: str, count: int, include_domains: list[str] | None = None,
                         full_content: bool = True) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "query": query,
        "max_results": max(1, min(count, 20)),
        "country": "US",
        "locale": "en",
        "search_depth": "standard",
        "full_content": full_content,
        "output_format": "plain_text",
    }
    if include_domains:
        body["include_domains"] = include_domains
    return f"{BASE}/search", body


def parse_search_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(data.get("results") or []):
        out.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": (r.get("description") or "")[:400],
            "position": int((r.get("metadata") or {}).get("position") or i + 1),
            "content": r.get("content") or "",
        })
    return [o for o in out if o["url"]]


def build_extract_request(url: str) -> tuple[str, dict[str, Any]]:
    body = {
        "url": url,
        "formats": ["markdown"],
        "render": "auto",
        "markdown_backend": "main_content",
        "country": "US",
        "locale": "en",
    }
    return f"{BASE}/extract", body


def parse_extract_response(data: dict[str, Any]) -> dict[str, Any]:
    d = data.get("data") or {}
    status_code = int(data.get("status_code") or 200)
    text = d.get("markdown") or ""
    if not text and d.get("html"):
        text = d["html"]
    return {"text": text, "status_code": status_code, "blocked": status_code in (401, 403, 429) or data.get("status") == "failed",
            "meta": {"task_id": data.get("task_id"), "warnings": data.get("warnings")}}
