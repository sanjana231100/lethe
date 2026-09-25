"""OpenAI-compatible chat completions (OpenAI, OpenRouter, Liquid hosted, etc.)."""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..models import LLMResult
from .base import LLMProvider


class _RateGate:
    """Tracks x-ratelimit-remaining-tokens / reset per (provider, model) and sleeps before a call
    that would not fit. Groq's free tier is 8k tokens/min per model, so the runtime spreads roles
    across models and this gate keeps each bucket from tripping 429s."""
    _state: dict[str, dict[str, float]] = {}
    _locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def lock(cls, key: str) -> asyncio.Lock:
        return cls._locks.setdefault(key, asyncio.Lock())

    @classmethod
    async def wait_for(cls, key: str, need: int) -> None:
        st = cls._state.get(key)
        if not st:
            return
        remaining = st["remaining"] - (time.time() - st["at"]) / max(st["reset_s"], 0.001) * 0 if False else st["remaining"]
        if time.time() >= st["at"] + st["reset_s"]:
            return  # bucket refilled
        if remaining < need + 200:
            delay = max(0.0, st["at"] + st["reset_s"] - time.time())
            print(f"[rate] {key}: {int(remaining)} tokens left, need {need}; waiting {delay:.1f}s")
            await asyncio.sleep(min(delay, 90))

    @classmethod
    def update(cls, key: str, headers: httpx.Headers, used: int) -> None:
        rem = headers.get("x-ratelimit-remaining-tokens")
        reset = headers.get("x-ratelimit-reset-tokens")
        if rem is None:
            return
        cls._state[key] = {"remaining": float(rem), "reset_s": _parse_duration(reset or "60s"), "at": time.time()}


def _parse_duration(s: str) -> float:
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|s|m|h)", s):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total or 60.0


class OpenAICompatProvider(LLMProvider):
    def __init__(self, model: str, token_budget: int, base_url: str, api_key: str, name: str = "openai"):
        super().__init__(model, token_budget)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = name
        self._gate_key = f"{name}:{model}"

    async def _complete(self, system: str, user: str, *, role: str, meta: dict[str, Any], max_tokens: int,
                        temperature: float, json_mode: bool) -> LLMResult:
        from .base import count_tokens
        need = count_tokens(system) + count_tokens(user) + max_tokens
        async with _RateGate.lock(self._gate_key):          # one in-flight call per model bucket
            await _RateGate.wait_for(self._gate_key, need)
            return await self._post(system, user, role=role, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode, need=need)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30),
           retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)), reraise=True)
    async def _post(self, system: str, user: str, *, role: str, max_tokens: int, temperature: float, json_mode: bool, need: int) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/lethe-agent"
            headers["X-Title"] = "Lethe"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            _RateGate.update(self._gate_key, r.headers, need)
            if r.status_code == 429:
                ra = r.headers.get("retry-after")
                await asyncio.sleep(min(float(ra) if ra and ra.replace(".", "").isdigit() else 10.0, 60))
                r.raise_for_status()
            if r.status_code >= 500:
                r.raise_for_status()
            if r.status_code == 400 and json_mode and "json" in r.text.lower():
                # provider-side JSON validation failed (e.g. output truncated): retry once in free-form mode;
                # our tolerant extract_json() + Pydantic validation still guard the result
                body.pop("response_format", None)
                body["max_tokens"] = int(max_tokens * 1.5)
                r = await client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                _RateGate.update(self._gate_key, r.headers, need)
            if r.status_code >= 400:
                raise RuntimeError(f"{self.name} {r.status_code}: {r.text[:300]}")
            data = r.json()
        usage = data.get("usage") or {}
        return LLMResult(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=0,
            model=data.get("model", self.model),
            provider=self.name,
        )
