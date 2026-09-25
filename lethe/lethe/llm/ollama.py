"""Ollama provider: runs Liquid AI LFM2.5 locally (LIQUID_MODEL=hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF).

Uses POST /api/chat with format="json" for strict JSON output. Token counts come from
Ollama's prompt_eval_count / eval_count.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..models import LLMResult
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "liquid-ollama"

    def __init__(self, model: str, token_budget: int, base_url: str = "http://localhost:11434"):
        super().__init__(model, token_budget)
        self.base_url = base_url.rstrip("/")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6),
           retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)), reraise=True)
    async def _complete(self, system: str, user: str, *, role: str, meta: dict[str, Any], max_tokens: int,
                        temperature: float, json_mode: bool) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            # LFM2.5 model-card sampling: temp 0.1, top_k 50, repeat_penalty 1.05
            "options": {"temperature": min(temperature, 0.1), "top_k": 50, "repeat_penalty": 1.05,
                        "num_predict": max_tokens, "num_ctx": 8192},
        }
        if json_mode:
            body["format"] = "json"
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
        return LLMResult(
            text=(data.get("message") or {}).get("content", ""),
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            latency_ms=int((data.get("total_duration") or 0) / 1_000_000),
            model=data.get("model", self.model),
            provider="liquid-ollama",
        )
