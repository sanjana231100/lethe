"""CLI:  python -m lethe.runtime.main once|run|resume|status|reset"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import signal
import sys

from rich.console import Console
from rich.panel import Panel

from ..config import settings
from .loop import Runtime

console = Console()


async def _once(rt: Runtime) -> None:
    await rt.boot()
    ledger = await rt.cycle()
    console.print(Panel(json.dumps(ledger.compact(), indent=1), title=f"ledger after cycle {ledger.kpis.cycles}", expand=False))
    console.print(f"max input tokens this process: {rt.log.max_input_tokens} (budget {rt.cfg.token_budget}); llm calls: {rt.log.n_llm_calls}")


async def _status(rt: Runtime) -> None:
    ledger = await rt.store.current_ledger(rt.cfg.run_id)
    stats = await rt.store.query("run_stats", {"run_id": rt.cfg.run_id})
    console.print(Panel(json.dumps(rt.cfg.summary(), indent=1), title="config", expand=False))
    console.print(Panel(json.dumps(stats[0] if stats else {}, indent=1), title="run_stats", expand=False))
    console.print(Panel(json.dumps(ledger.compact(), indent=1) if ledger else "no ledger yet", title="ledger", expand=False))


async def _run(rt: Runtime) -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    task = asyncio.create_task(rt.run_forever())
    await stop.wait()
    task.cancel()
    await rt.store.flush()
    console.print("[yellow]stopped; state is in the store — `resume` picks up from the last ledger_snapshot[/yellow]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lethe", description="Lethe: agents that forget on purpose")
    ap.add_argument("command", choices=["once", "run", "resume", "status", "reset"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.command == "reset":
        if not (settings.mock or not settings.rawtree_token):
            console.print("[red]reset only clears the local JSONL store; RawTree is append-only (use a new RUN_ID)[/red]")
            return 1
        shutil.rmtree(settings.state_dir, ignore_errors=True)
        console.print(f"cleared {settings.state_dir}")
        return 0

    rt = Runtime(settings, quiet=args.quiet)
    console.print(Panel(json.dumps(rt.cfg.summary(), indent=1), title="lethe", expand=False))
    try:
        if args.command == "once":
            asyncio.run(_once(rt))
        elif args.command in ("run", "resume"):
            asyncio.run(_run(rt))
        elif args.command == "status":
            asyncio.run(_status(rt))
    finally:
        asyncio.run(rt.close())
    return 0


if __name__ == "__main__":
    sys.exit(main())
