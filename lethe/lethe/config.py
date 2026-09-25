"""Configuration. Secrets come only from the environment / .env (never hardcoded)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    if " #" in v or v.startswith("#"):   # tolerate inline comments in .env
        v = v.split("#", 1)[0]
    return v.strip()


def _flag(name: str, default: str = "0") -> bool:
    return _env(name, default).lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    mock: bool = field(default_factory=lambda: _flag("MOCK", "1"))
    run_id: str = field(default_factory=lambda: _env("RUN_ID", "dev"))
    cycle_interval: int = field(default_factory=lambda: int(_env("CYCLE_INTERVAL_SECONDS", "120")))
    token_budget: int = field(default_factory=lambda: int(_env("TOKEN_BUDGET", "6000")))
    max_concurrent_scouts: int = field(default_factory=lambda: int(_env("MAX_CONCURRENT_SCOUTS", "3")))
    units_limit: int = field(default_factory=lambda: int(_env("UNITS_LIMIT", "4")))
    mission_file: Path = field(default_factory=lambda: ROOT / _env("MISSION_FILE", "missions/rv_pricing.yaml"))
    inventory_file: Path = field(default_factory=lambda: ROOT / _env("INVENTORY_FILE", "data/inventory.csv"))
    state_dir: Path = field(default_factory=lambda: ROOT / _env("STATE_DIR", ".lethe_state"))

    # LLM (chief / auditor / scout extraction)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_scout_model: str = field(default_factory=lambda: _env("LLM_SCOUT_MODEL", ""))     # separate rate buckets
    llm_auditor_model: str = field(default_factory=lambda: _env("LLM_AUDITOR_MODEL", ""))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))

    # Liquid (librarian)
    liquid_base_url: str = field(default_factory=lambda: _env("LIQUID_BASE_URL", "http://localhost:11434"))
    liquid_model: str = field(default_factory=lambda: _env("LIQUID_MODEL", "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF"))
    liquid_api_key: str = field(default_factory=lambda: _env("LIQUID_API_KEY"))

    # Nimble
    nimble_api_key: str = field(default_factory=lambda: _env("NIMBLE_API_KEY"))

    # RawTree (state store + observability)
    rawtree_api_key: str = field(default_factory=lambda: _env("RAWTREE_API_KEY") or _env("TINYBIRD_TOKEN"))
    rawtree_api_url: str = field(default_factory=lambda: _env("RAWTREE_API_URL", "https://api.rawtree.com"))

    # BFL
    bfl_api_key: str = field(default_factory=lambda: _env("BFL_API_KEY"))

    @property
    def effective_llm_provider(self) -> str:
        return "mock" if self.mock else self.llm_provider

    def summary(self) -> dict:
        return {
            "mock": self.mock,
            "run_id": self.run_id,
            "llm_provider": self.effective_llm_provider,
            "llm_model": self.llm_model or "(provider default)",
            "librarian": "mock" if self.mock else f"{self.liquid_model} @ {self.liquid_base_url}",
            "store": "local-jsonl" if (self.mock or not self.rawtree_api_key) else self.rawtree_api_url,
            "nimble": "mock" if (self.mock or not self.nimble_api_key) else "live",
            "token_budget": self.token_budget,
            "cycle_interval": self.cycle_interval,
            "units_limit": self.units_limit,
        }


settings = Settings()
