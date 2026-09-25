"""LLM provider interface.

Hard rule #1: the prompt is not the memory. Every call is built fresh from
system + bounded state + step evidence, and `complete()` refuses to send more
than `token_budget` input tokens. The actual token count of every call is
returned in LLMResult and logged by the caller as an `llm_call` event.
"""
from __future__ import annotations

import abc
import time
from typing import Any

from ..config import Settings, settings as default_settings
from ..models import LLMResult

try:  # tiktoken is used for a provider-neutral pre-flight count
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback when tiktoken is unavailable
    _ENC = None


def count_tokens(text: str) -> int:
    if _ENC is None:
        return max(1, len(text) // 4)
    return len(_ENC.encode(text, disallowed_special=()))


class TokenBudgetExceeded(RuntimeError):
    pass


class LLMProvider(abc.ABC):
    name: str = "base"

    def __init__(self, model: str, token_budget: int):
        self.model = model
        self.token_budget = token_budget

    async def complete(self, system: str, user: str, *, role: str, meta: dict[str, Any] | None = None,
                       max_tokens: int = 900, temperature: float = 0.2, json_mode: bool = True) -> LLMResult:
        """Budget-enforced entry point. Providers implement `_complete`."""
        est = count_tokens(system) + count_tokens(user) + 8
        if est > self.token_budget:
            raise TokenBudgetExceeded(f"{role}: {est} estimated input tokens > budget {self.token_budget}")
        t0 = time.perf_counter()
        result = await self._complete(system, user, role=role, meta=meta or {}, max_tokens=max_tokens,
                                      temperature=temperature, json_mode=json_mode)
        result.latency_ms = result.latency_ms or int((time.perf_counter() - t0) * 1000)
        if not result.input_tokens:  # provider didn't report usage -> use our estimate
            result.input_tokens = est
        if not result.output_tokens:
            result.output_tokens = count_tokens(result.text)
        return result

    @abc.abstractmethod
    async def _complete(self, system: str, user: str, *, role: str, meta: dict[str, Any], max_tokens: int,
                        temperature: float, json_mode: bool) -> LLMResult: ...


GROQ_DEFAULTS = {"chief": "openai/gpt-oss-120b", "scout": "qwen/qwen3.8-27b", "auditor": "openai/gpt-oss-20b"}


def make_llm(cfg: Settings | None = None, *, purpose: str = "chief") -> LLMProvider:
    """Factory. purpose: 'chief' | 'scout' | 'auditor' (pluggable provider) or 'librarian' (Liquid).
    Roles can run on different models so each has its own rate-limit bucket."""
    cfg = cfg or default_settings
    if cfg.mock:
        from .mock import MockLLM
        return MockLLM(model="mock", token_budget=cfg.token_budget)

    if purpose == "librarian":
        from .ollama import OllamaProvider
        from .openai_compat import OpenAICompatProvider
        if cfg.liquid_base_url.rstrip("/").endswith("11434") or "localhost" in cfg.liquid_base_url:
            return OllamaProvider(model=cfg.liquid_model, token_budget=cfg.token_budget, base_url=cfg.liquid_base_url)
        return OpenAICompatProvider(model=cfg.liquid_model, token_budget=cfg.token_budget,
                                    base_url=cfg.liquid_base_url, api_key=cfg.liquid_api_key, name="liquid")

    p = cfg.llm_provider.lower()
    if p == "openai":
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(model=cfg.llm_model or "gpt-4.1-mini", token_budget=cfg.token_budget,
                                    base_url="https://api.openai.com/v1", api_key=cfg.openai_api_key, name="openai")
    if p == "openrouter":
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(model=cfg.llm_model or "openai/gpt-4.1-mini", token_budget=cfg.token_budget,
                                    base_url="https://openrouter.ai/api/v1", api_key=cfg.openrouter_api_key, name="openrouter")
    if p == "groq":
        from .openai_compat import OpenAICompatProvider
        override = {"scout": cfg.llm_scout_model, "auditor": cfg.llm_auditor_model}.get(purpose, "") or cfg.llm_model
        return OpenAICompatProvider(model=override or GROQ_DEFAULTS.get(purpose, GROQ_DEFAULTS["chief"]), token_budget=cfg.token_budget,
                                    base_url="https://api.groq.com/openai/v1", api_key=cfg.groq_api_key, name="groq")
    if p == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(model=cfg.llm_model or "claude-sonnet-5", token_budget=cfg.token_budget,
                                 api_key=cfg.anthropic_api_key)
    if p == "liquid":  # run the Chief on Liquid too (stretch)
        return make_llm(cfg, purpose="librarian")
    if p == "mock":
        from .mock import MockLLM
        return MockLLM(model="mock", token_budget=cfg.token_budget)
    raise ValueError(f"unknown LLM_PROVIDER={cfg.llm_provider}")
