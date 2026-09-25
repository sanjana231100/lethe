# RawTree for Lethe

RawTree (https://rawtree.com) is the state store and observability backend. It is schema-free:
tables are created on first insert, so there is nothing to deploy.

```
RAWTREE_API_KEY=rt_...              # cluster API key (read_write)
RAWTREE_API_URL=https://api.rawtree.com
```

| Table | Rows |
|---|---|
| `lethe_events` | every action, tool call, fact transition, LLM call (token counts, latency, model), lesson, error, and the `ledger_snapshot` the agent boots from |
| `lethe_facts` | fact *versions* (latest per `fact_id` wins): subject, predicate, value, source_url, observed_at, confidence, ttl_seconds, generation, status |

Every row has `ts` (UTC string) and `ts_ms` (epoch millis) for ordering; `payload` is a JSON string.

Writes: `POST /v1/tables/{table}` with a JSON array. Reads: `POST /v1/query {"sql": ..., "format": "JSON"}`.
The SQL behind each dashboard panel and the resume path is in `lethe/store/rawtree_sql.py`:
`current_ledger`, `latest_facts`, `token_metrics`, `fact_lifecycle`, `memory_feed`, `lessons`, `run_stats`, `recent_events`.

Handy ad-hoc queries (ClickHouse dialect, read-only):

```sql
-- context never grows: max input tokens per cycle
SELECT substring(toString(ts),1,16) AS minute, max(input_tokens) FROM lethe_events
WHERE run_id='demo' AND event_type='llm_call' GROUP BY minute ORDER BY minute;

-- what got forgotten, and why
SELECT toString(ts), event_type, unit_id, JSONExtractString(toString(payload),'subject') AS subject,
       JSONExtractString(toString(payload),'reason') AS reason
FROM lethe_events WHERE run_id='demo' AND event_type IN ('fact_evicted','fact_contradicted','fact_expired')
ORDER BY ts_ms DESC LIMIT 20;
```
