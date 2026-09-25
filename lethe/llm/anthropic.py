"""Anthropic Messages API provider."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..models import LLMResult
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, token_budget: int, api_key: str):
        super().__init__(model, token_budget)
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
           retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)), reraise=True)
    async def _complete(self, system: str, user: str, *, role: str, meta: dict[str, Any], max_tokens: int,
                        temperature: float, json_mode: bool) -> LLMResult:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system + ("\nRespond with a single JSON object and nothing else." if json_mode else ""),
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
            if r.status_code >= 500 or r.status_code == 429:
                r.raise_for_status()
            if r.status_code >= 400:
                raise RuntimeError(f"anthropic {r.status_code}: {r.text[:300]}")
            data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return LLMResult(text=text, input_tokens=int(usage.get("input_tokens") or 0),
                         output_tokens=int(usage.get("output_tokens") or 0), latency_ms=0,
                         model=data.get("model", self.model), provider="anthropic")
