"""SQL behind every Lethe "pipe" on RawTree (ClickHouse dialect, read-only via POST /v1/query).

Tables are created automatically on first insert (schema-free). Two tables:
  lethe_events  every action / tool call / fact transition / llm_call / ledger_snapshot
  lethe_facts   fact *versions*; the latest row per fact_id wins

Every row carries `ts` (string, 'YYYY-MM-DD HH:MM:SS.mmm' UTC) and `ts_ms` (epoch millis) so
ordering never depends on how the dynamic column typed the string. `payload` is a JSON string
(queried with JSONExtract*), not a nested object, so the ledger snapshot doesn't fan out into
hundreds of dynamic columns.
"""
from __future__ import annotations

EVENTS = "lethe_events"
FACTS = "lethe_facts"

FACT_KINDS = "('fact_created','fact_promoted','fact_expired','fact_evicted','fact_contradicted','fact_superseded')"
GC_KINDS = "('fact_promoted','fact_expired','fact_evicted','fact_contradicted','fact_superseded')"
LESSON_KINDS = "('lesson','strategy_changed','self_correction')"
STEP_KINDS = "('chief_plan','chief_update','scout_done','audit_done','librarian_done','gc_sweep')"


def q(s: str) -> str:
    """Quote a string literal for ClickHouse SQL."""
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def current_ledger(run_id: str) -> str:
    return f"""SELECT toString(ts) AS ts, toString(payload) AS payload
FROM {EVENTS}
WHERE run_id = {q(run_id)} AND event_type = 'ledger_snapshot'
ORDER BY ts_ms DESC LIMIT 1"""


def latest_facts(run_id: str) -> str:
    """Latest version per fact_id (TTL expiry is applied in Python by the store)."""
    return f"""SELECT fact_id,
  argMax(unit_id, ts_ms) AS unit_id, argMax(subject, ts_ms) AS subject, argMax(predicate, ts_ms) AS predicate,
  argMax(toString(value), ts_ms) AS value, argMax(source_url, ts_ms) AS source_url,
  argMax(toString(observed_at), ts_ms) AS observed_at, argMax(confidence, ts_ms) AS confidence,
  argMax(ttl_seconds, ts_ms) AS ttl_seconds, argMax(generation, ts_ms) AS generation, argMax(status, ts_ms) AS status,
  argMax(toString(ts), ts_ms) AS ts
FROM {FACTS}
WHERE run_id = {q(run_id)}
GROUP BY fact_id
HAVING status = 'active'
LIMIT 10000"""


def token_metrics(run_id: str, limit: int = 600) -> str:
    return f"""SELECT toString(ts) AS ts, agent, component, input_tokens, output_tokens, latency_ms, model,
  row_number() OVER (PARTITION BY agent ORDER BY ts_ms) AS step
FROM {EVENTS}
WHERE event_type = 'llm_call' AND (run_id = {q(run_id)} OR run_id = {q('naive-' + run_id)})
ORDER BY ts_ms
LIMIT {int(limit)}"""


def fact_lifecycle(run_id: str) -> str:
    return f"""SELECT substring(toString(ts), 1, 16) AS minute, event_type, count() AS n
FROM {EVENTS}
WHERE run_id = {q(run_id)} AND event_type IN {FACT_KINDS}
GROUP BY minute, event_type
ORDER BY minute, event_type
LIMIT 5000"""


def memory_feed(run_id: str, limit: int = 40) -> str:
    return f"""SELECT toString(ts) AS ts, event_type, unit_id, toString(payload) AS payload
FROM {EVENTS}
WHERE run_id = {q(run_id)} AND event_type IN {GC_KINDS}
ORDER BY ts_ms DESC LIMIT {int(limit)}"""


def lessons(run_id: str, limit: int = 30) -> str:
    return f"""SELECT toString(ts) AS ts, event_type, unit_id, component, toString(payload) AS payload
FROM {EVENTS}
WHERE run_id = {q(run_id)} AND event_type IN {LESSON_KINDS}
ORDER BY ts_ms DESC LIMIT {int(limit)}"""


def run_stats(run_id: str) -> str:
    return f"""SELECT
  toString(min(ts)) AS first_ts, toString(max(ts)) AS last_ts,
  countIf(event_type = 'cycle_done') AS cycles,
  countIf(event_type IN {STEP_KINDS}) AS steps,
  countIf(event_type = 'llm_call' AND agent = 'lethe') AS llm_calls,
  sumIf(JSONExtractInt(toString(payload), 'pages_read'), event_type = 'scout_done') AS pages_read,
  sumIf(input_tokens, event_type = 'llm_call' AND agent = 'lethe') AS input_tokens,
  sumIf(output_tokens, event_type = 'llm_call' AND agent = 'lethe') AS output_tokens,
  maxIf(input_tokens, event_type = 'llm_call' AND agent = 'lethe') AS max_input_tokens,
  argMaxIf(input_tokens, ts_ms, event_type = 'llm_call' AND agent = 'lethe') AS last_input_tokens,
  round(input_tokens / 1e6 * 0.40 + output_tokens / 1e6 * 1.60, 4) AS est_cost_usd,
  countIf(event_type = 'resumed') AS resumes,
  countIf(event_type = 'error') AS errors
FROM {EVENTS}
WHERE run_id = {q(run_id)}"""


def recent_events(run_id: str, limit: int = 50, event_type: str | None = None) -> str:
    et = f" AND event_type = {q(event_type)}" if event_type else ""
    return f"""SELECT toString(ts) AS ts, component, event_type, unit_id, toString(payload) AS payload,
  input_tokens, output_tokens, latency_ms, model
FROM {EVENTS}
WHERE run_id = {q(run_id)}{et}
ORDER BY ts_ms DESC LIMIT {int(limit)}"""
