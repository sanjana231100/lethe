# Lethe — agents that forget on purpose

> Every agent today dies from remembering too much. **Lethe** is an agent runtime with
> *garbage-collected memory*: it keeps what matters, forgets what doesn't, and has been
> running unattended since noon.

Built for the Long Horizon Agents Hackathon ("build agents that preserve what matters").
The flagship mission keeps a dealership's used RV / truck inventory priced against the
**live web market**, continuously, for hours, with a context window that never grows.

## Why

Long-horizon agents drown in their own history: observations, actions and stale context
pile up until every call is slower, costlier and less reliable. Lethe replaces the
ever-growing transcript with **explicit mutable state** (a Mission Ledger that is rewritten,
not appended), **generational memory with TTLs and eviction**, and a **hard token budget per
LLM call**. State lives in RawTree, not in the process: `kill -9` it and it resumes in
seconds with no transcript replay.

## Architecture

```
                 ┌──────────────────────────────────────────┐
                 │  CHIEF (orchestrator, planner)           │
                 │  sees: Mission Ledger + canon facts +    │
                 │        top working facts + lessons       │
                 │  never sees raw web data                 │
                 └───────┬───────────────────────▲──────────┘
             assigns tasks│                       │finding cards / verdicts
            ┌─────────────▼─────────┐   ┌─────────┴───────────┐
            │ SCOUTS (Nimble)       │   │ AUDITOR (critic)    │
            │ fresh context each,   │   │ comparability rules │
            │ parallel, disposable  │   │ + 2nd source; writes│
            │ → Finding Card ≤200t  │   │ LESSONS on failure  │
            └─────────────┬─────────┘   └─────────▲───────────┘
                raw observations                  │
            ┌─────────────▼───────────────────────┴───────────┐
            │ LIBRARIAN (Liquid LFM2.5) = memory garbage      │
            │ collector: facts → TTL/confidence → promote /    │
            │ expire / evict / contradict                      │
            └─────────────┬────────────────────────────────────┘
                          │ events + facts (append-only)
            ┌─────────────▼────────────────────────────────────┐
            │ RAWTREE: lethe_events, lethe_facts → SQL queries:  │
            │ current_ledger, active_facts, token_metrics …     │
            └─────────────┬────────────────────────────────────┘
                          │ read-only SQL
            ┌─────────────▼────────────────────────────────────┐
            │ DASHBOARD: board, token chart (Lethe vs naive),   │
            │ fact lifecycle, GC feed, lessons, uptime          │
            └──────────────────────────────────────────────────┘
```

### Generational memory (the core idea)

Modeled on JVM generational GC (`lethe/memory/librarian.py`):

| Generation | What | Lifetime |
|---|---|---|
| **scratch** | raw pages inside a Scout | dies with the Scout; only `url + sha1` is archived in events |
| **working** | facts the Librarian extracted (`asking_price`, `specs`, `status` per listing) | TTL (price 2h, specs 6h) |
| **canon** | facts confirmed by **≥2 independent observations** (different URL, or the same URL ≥10 min apart, or an audit with a second source), plus decisions, mission spec, lessons | until contradicted |

Transitions (every one is an event): `fact_created`, `fact_superseded` (new price replaces old;
history stays in events only), `fact_promoted`, `fact_contradicted` (a listing marked *sold*
kills every fact about it, cascading to the old price), `fact_evicted` (audit rejection or low
confidence), `fact_expired` (TTL, by the GC sweep).

### Context assembly (`lethe/memory/context.py`)

Each Chief call gets a **ranked, budgeted slice**: ledger (always) → canon facts for the task →
top-K working facts by `relevance × recency × confidence` → last N lessons. Anything over
budget is dropped by priority, and a `context_assembled` event logs what was included and
dropped. The provider layer refuses any call over `TOKEN_BUDGET` (default 6,000 input tokens)
and logs the real token count of every call.

### The Mission Ledger (`lethe/models.py: Ledger`)

One structured object, **rewritten each cycle** and stored as the latest `ledger_snapshot`
event. The Chief edits it only through tools: `update_plan`, `record_decision`, `add_lesson`,
`request_scout(unit_id, strategy)`, `set_recommendation(unit_id, price, reasoning, confidence)`,
`mark_stale(unit_id)`. Guardrails in code cap any recommendation at ±12% per cycle.

### One cycle (`lethe/runtime/loop.py`)

1. **Boot/resume** — latest ledger + active facts from RawTree SQL queries (`resumed` event with ms).
2. **Chief plans** — units due (never checked, stale, low confidence, refresh interval); picks a
   strategy per unit that lessons haven't ruled out.
3. **Scouts** run in parallel (asyncio semaphore): Nimble search → read top pages → one
   fresh-context extraction call → Finding Card (≤200 tokens on the wire).
4. **Librarian** turns every card into fact versions, with TTL/confidence from Liquid LFM2.5.
5. **Auditor** checks comparability (year ±1, model/trim, mileage band, region radius, status) and
   a second source. Fail → lesson + **an immediate retry with a different strategy**; success
   after a retry is logged as `self_correction`.
6. **Chief updates** recommendations from audited comps (deterministic price stats + LLM
   reasoning), rewrites the ledger, snapshot.
7. **GC sweep** — expire TTLs, evict low-confidence facts, emit metrics.
8. Sleep `CYCLE_INTERVAL_SECONDS` (default 120). Repeat, unattended, for hours.

## Sponsor tools

| Sponsor | Component | Role |
|---|---|---|
| **Nimble** | Scouts | real-time web search (`POST sdk.nimbleway.com/v2/search`, `full_content`) + page extraction (`/v2/extract`, markdown) of live vehicle listings — `lethe/tools/nimble.py`, `nimble_api.py` |
| **Liquid AI** | Librarian | LFM2.5-1.2B-Instruct running locally via Ollama (`hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF`, JSON mode, model-card sampling) assigns TTL + confidence per fact and flags contradictions on **every** observation — `lethe/llm/ollama.py`, `lethe/memory/librarian.py` |
| **RawTree** | State + observability | schema-free tables `lethe_events` + `lethe_facts`, batched inserts with retry (`POST /v1/tables/{table}`), 8 read-only SQL queries (`lethe/store/rawtree_sql.py`) for boot/resume and every dashboard panel — `lethe/store/rawtree.py`, `rawtree/README.md` |
| **Black Forest Labs** (stretch) | Visual worker | FLUX.2 [pro] hero image for repriced units — `lethe/tools/flux.py` |

Chief / Auditor / Scout-extraction LLM is pluggable: `LLM_PROVIDER=openai|anthropic|openrouter|liquid|mock`.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # MOCK=1 by default: fully offline
.venv/bin/python -m lethe.runtime.main once       # one cycle, prints the ledger
.venv/bin/python -m lethe.runtime.main run        # unattended loop (Ctrl-C / kill -9 any time)
.venv/bin/python -m lethe.runtime.main resume     # same as run: boots from the store
.venv/bin/python -m lethe.runtime.main status
.venv/bin/python -m baseline.naive_agent --steps 24   # the append-everything baseline
.venv/bin/uvicorn dashboard.app:app --port 8000       # http://localhost:8000
.venv/bin/python -m pytest -q                         # GC rules, context budget, resume
```

### Going live

Groq's free tier allows 8k tokens/min per model, so roles run on separate models (chief `openai/gpt-oss-120b`,
scouts `qwen/qwen3.8-27b`, auditor `openai/gpt-oss-20b`; override with `LLM_MODEL`, `LLM_SCOUT_MODEL`,
`LLM_AUDITOR_MODEL`) and a per-model rate gate waits instead of tripping 429s.


1. **RawTree**: put your cluster key in `.env` as `RAWTREE_API_KEY` (tables auto-create on first
   insert; see `rawtree/README.md`).
2. **Nimble**: `NIMBLE_API_KEY` from the Nimble dashboard.
3. **Liquid**: `brew install ollama && ollama pull hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF` (keep
   `LIQUID_BASE_URL=http://localhost:11434`).
4. **Chief LLM**: `LLM_PROVIDER=groq` + `GROQ_API_KEY` (default model `openai/gpt-oss-120b`), or openai / anthropic / openrouter.
5. `MOCK=0 RUN_ID=demo .venv/bin/python -m lethe.runtime.main run`

Any integration without a key silently falls back to its mock, so partial setups still run.

### The kill -9 demo

```bash
RUN_ID=demo .venv/bin/python -m lethe.runtime.main run     # in one terminal
kill -9 $(pgrep -f "lethe.runtime.main run")                 # in another
RUN_ID=demo .venv/bin/python -m lethe.runtime.main resume   # back in ~1s; look for the `resumed` event
```

## Metrics

Live cycle (Nimble + Groq + Liquid, 3 units): 17 LLM calls, max input 2,876 tokens, 0 errors, one
self-correction (U03: `broad` → 0 priced comps → lesson → `rvtrader` → 3 comps → recommendation).

Mock run, 4 units:

| | Lethe | Naive baseline |
|---|---|---|
| input tokens, call 1 | ~0.9k | ~0.5k |
| input tokens, call 24 | ~3.4k | ~10.2k and climbing linearly |
| max context over 4 cycles (71 calls) | 3.5k (budget 6k) | unbounded |
| facts: created / promoted / superseded / contradicted / evicted | live on the dashboard | n/a |

Cost estimate on the dashboard uses a blended $0.40/M input, $1.60/M output.

## Repo layout

```
lethe/
  config.py, models.py           settings; Fact / FindingCard / Ledger / Event / AuditVerdict
  llm/                           provider interface + openai-compatible / anthropic / ollama (Liquid) / mock
  tools/nimble.py, nimble_api.py Nimble v2 client + recorded-fixture mock;  tools/flux.py (stretch)
  store/rawtree.py, rawtree_sql.py, local.py   RawTree insert + SQL reader; JSONL mirror with the same queries
  memory/librarian.py            generational GC rules;  memory/context.py budgeted assembly;  pricing.py
  agents/chief.py, scout.py, auditor.py
  runtime/loop.py, main.py, events.py, mission.py
baseline/naive_agent.py          append-everything baseline (agent="naive")
rawtree/                         table layout + handy SQL
dashboard/                       FastAPI + single-page UI (6 panels, 5s refresh)
missions/rv_pricing.yaml         the mission is config; data/inventory.csv seed units
fixtures/nimble/listings.json    recorded market that evolves per cycle (price drops, SOLD, new comps, a blocked source)
tests/                           GC rules, context budget, resume-from-state, card/token budgets
```

## 3-minute demo

1. *"Agents die from remembering too much."*
2. Dashboard hero: running since HH:MM, N cycles, N pages, max context X tokens.
3. Token chart: naive climbing, Lethe flat. Two sentences on scratch → working → canon.
4. `kill -9` live → `resume` → back in a second. *"The memory was never in the prompt."*
5. Lessons feed: `broad` → 0 results → lesson → `rvtrader` → 3 comps (self-correction). GC feed:
   listing SOLD → facts contradicted → recommendation moved.
6. Nimble = eyes, Liquid = memory GC, RawTree = state + observability. *Lethe: agents that forget on purpose.*
