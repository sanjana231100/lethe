"""Lethe dashboard: FastAPI serving a static page + thin proxies to the store's pipes.

In live mode every /api/<pipe> call hits the published RawTree endpoint; in MOCK=1 it hits
the local JSONL mirror. The page polls every 5s.

    uvicorn dashboard.app:app --port 8000 --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lethe.config import settings
from lethe.store.base import make_store

app = FastAPI(title="Lethe dashboard")
STATIC = Path(__file__).parent / "static"
store = make_store(settings)

PIPES = {"run_stats", "board", "token_metrics", "fact_lifecycle", "memory_feed", "lessons", "recent_events", "current_ledger", "active_facts"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
async def config() -> dict:
    return {**settings.summary(), "naive_run_id": f"naive-{settings.run_id}"}


@app.get("/api/{pipe}")
async def pipe(pipe: str, run_id: str | None = Query(default=None), limit: int | None = None, event_type: str | None = None) -> JSONResponse:
    if pipe not in PIPES:
        raise HTTPException(404, f"unknown pipe {pipe}")
    params: dict = {"run_id": run_id or settings.run_id}
    if limit:
        params["limit"] = limit
    if event_type:
        params["event_type"] = event_type
    try:
        rows = await store.query(pipe, params)
    except Exception as exc:  # the dashboard must keep rendering through a store hiccup
        return JSONResponse({"data": [], "error": str(exc)[:300]}, status_code=200)
    return JSONResponse({"data": rows})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
