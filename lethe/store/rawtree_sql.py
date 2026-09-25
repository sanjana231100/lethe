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
    # RawTree columns are Dynamic; aggregates need explicit casts.
    return f"""SELECT toString(fact_id) AS fid,
  argMax(toString(unit_id), toInt64(ts_ms)) AS unit_id, argMax(toString(subject), toInt64(ts_ms)) AS subject,
  argMax(toString(predicate), toInt64(ts_ms)) AS predicate, argMax(toString(value), toInt64(ts_ms)) AS value,
  argMax(toString(source_url), toInt64(ts_ms)) AS source_url, argMax(toString(observed_at), toInt64(ts_ms)) AS observed_at,
  argMax(toFloat64(confidence), toInt64(ts_ms)) AS confidence, argMax(toUInt32(ttl_seconds), toInt64(ts_ms)) AS ttl_seconds,
  argMax(toString(generation), toInt64(ts_ms)) AS generation, argMax(toString(status), toInt64(ts_ms)) AS status,
  argMax(toString(ts), toInt64(ts_ms)) AS ts
FROM {FACTS}
WHERE run_id = {q(run_id)}
GROUP BY fid
HAVING status = 'active'
LIMIT 10000"""


def token_metrics(run_id: str, limit: int = 600) -> str:
    return f"""SELECT toString(ts) AS ts, agent, component, input_tokens, output_tokens, latency_ms, model,
  row_number() OVER (PARTITION BY toString(agent) ORDER BY toInt64(ts_ms)) AS step
FROM {EVENTS}
WHERE event_type = 'llm_call' AND (run_id = {q(run_id)} OR run_id = {q('naive-' + run_id)})
ORDER BY ts_ms
LIMIT {int(limit)}"""


def fact_lifecycle(run_id: str) -> str:
    return f"""SELECT substring(toString(ts), 1, 16) AS minute, toString(event_type) AS et, count() AS n
FROM {EVENTS}
WHERE run_id = {q(run_id)} AND toString(event_type) IN {FACT_KINDS}
GROUP BY minute, et
ORDER BY minute, et
LIMIT 5000"""


def memory_feed(run_id: str, limit: int = 40) -> str:
    return f"""SELECT toString(ts) AS ts, event_type, unit_id, toString(payload) AS payload
FROM {EVENTS}
WHERE run_id = {q(run_id)} AND toString(event_type) IN {GC_KINDS}
ORDER BY ts_ms DESC LIMIT {int(limit)}"""


def lessons(run_id: str, limit: int = 30) -> str:
    return f"""SELECT toString(ts) AS ts, event_type, unit_id, component, toString(payload) AS payload
FROM {EVENTS}
WHERE run_id = {q(run_id)} AND toString(event_type) IN {LESSON_KINDS}
ORDER BY ts_ms DESC LIMIT {int(limit)}"""


def run_stats(run_id: str) -> str:
    return f"""SELECT
  min(toString(ts)) AS first_ts, max(toString(ts)) AS last_ts,
  countIf(event_type = 'cycle_done') AS cycles,
  countIf(toString(event_type) IN {STEP_KINDS}) AS steps,
  countIf(event_type = 'llm_call' AND agent = 'lethe') AS llm_calls,
  sumIf(JSONExtractInt(toString(payload), 'pages_read'), event_type = 'scout_done') AS pages_read,
  sumIf(toUInt64(input_tokens), event_type = 'llm_call' AND agent = 'lethe') AS in_tok,
  sumIf(toUInt64(output_tokens), event_type = 'llm_call' AND agent = 'lethe') AS out_tok,
  maxIf(toUInt64(input_tokens), event_type = 'llm_call' AND agent = 'lethe') AS max_input_tokens,
  argMaxIf(toUInt64(input_tokens), toInt64(ts_ms), event_type = 'llm_call' AND agent = 'lethe') AS last_input_tokens,
  round(in_tok / 1e6 * 0.40 + out_tok / 1e6 * 1.60, 4) AS est_cost_usd,
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
