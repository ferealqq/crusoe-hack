"""CLI for research runs.

uv run python -m app.orchestrator --dataset churn --setup shared_brain
uv run python -m app.orchestrator --report
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from app.config import ResearchConfig, get_settings
from app.data import load_dataset
from app.lab import SETUPS, Setup
from app.loop import research_run
from app.services import inference
from app.store import DATA_DIR, read_jsonl


def _m(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 100_000 else f"{n / 1e3:.0f}k"


def results_table(root: Path = DATA_DIR) -> str:
    rows: list[dict[str, Any]] = read_jsonl(root / "research_runs.jsonl")
    if not rows:
        return "No finished research runs yet."
    header = (
        f"{'research run':<40} {'setup':<13} {'fitness':>8} {'val gain':>9}  "
        f"{'best val loss by generation':<44} {'memory':>6}  tokens (prompt/cached/output)"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        gens = read_jsonl(root / "research_runs" / r["run_id"] / "generations.jsonl")
        trail = " ".join(f"{g['end_best']:.4f}" for g in gens)
        fitness = f"{r['fitness']:+.2%}" if r.get("fitness") is not None else "n/a"
        u = r["usage"]
        lines.append(
            f"{r['run_id']:<40} {r['setup']:<13} {fitness:>8} {r['val_gain']:>+9.2%}  "
            f"{trail:<44} {r['memory_tokens']:>6}  "
            f"{_m(u['prompt'])}/{_m(u['cached'])}/{_m(u['output'])}"
        )
    return "\n".join(lines)


async def check_models(setup: Setup) -> None:
    s = get_settings()
    wanted = [s.researcher_model] + ([s.curator_model] if setup == "shared_brain" else [])
    available = set(await inference.list_models())
    missing = [m for m in wanted if m not in available]
    if missing:
        sys.exit(f"model(s) not on Crusoe: {', '.join(missing)}. Check GET /playground/models.")


async def main(args: argparse.Namespace) -> None:
    overrides = {
        k: v
        for k, v in {
            "generations": args.generations,
            "researchers": args.researchers,
            "researcher_runs": args.runs,
            "researcher_turns": args.turns,
        }.items()
        if v is not None
    }
    config = ResearchConfig(**overrides)
    await check_models(args.setup)
    dataset = load_dataset(args.dataset)
    await research_run(dataset, args.setup, config, commit=not args.no_commit)
    print()
    print(results_table())


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run Shared Brain research runs.")
    parser.add_argument("--dataset", default="churn")
    parser.add_argument("--setup", choices=SETUPS, default="shared_brain")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--researchers", type=int)
    parser.add_argument("--runs", type=int, help="runs per No memory researcher (default 7)")
    parser.add_argument("--turns", type=int, help="max turns per researcher (default 30)")
    parser.add_argument("--no-commit", action="store_true", help="don't commit outputs to git")
    parser.add_argument("--report", action="store_true", help="print the results table and exit")
    args = parser.parse_args()
    if args.report:
        print(results_table())
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "openai", "hpack", "grpc", "modal"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
