"""CLI that runs research runs. A plain Python script, not an agent.

    uv run python -m app.orchestrator --dataset churn --setup shared_brain
    uv run python -m app.orchestrator --dataset churn --setup all

With --setup all, the setups run in parallel and share one baseline measurement (same L0
and noise threshold). Results go to data/runs/<timestamp>-<dataset>/, then report.html.
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from app import report
from app.config import SETUPS, ResearchConfig, Setup, get_settings
from app.lab import Dataset
from app.loop import SetupRun, measure_baseline
from app.services import inference
from app.store import Store

log = logging.getLogger("orchestrator")


async def check_models() -> dict[str, dict[str, str]]:
    """Stop if a model ID is missing on Crusoe. Returns pricing per model ($ per 1M tokens)."""
    s = get_settings()
    models = (await inference.get_client().models.list()).data
    ids = {m.id for m in models}
    missing = [m for m in (s.researcher_model, s.curator_model) if m not in ids]
    if missing:
        sys.exit(f"models not on Crusoe: {missing}. Available: {sorted(ids)}")
    return {
        m.id: (m.model_dump().get("pricing") or {})
        for m in models
        if m.id in (s.researcher_model, s.curator_model)
    }


async def main_async(args: argparse.Namespace) -> Path:
    s = get_settings()
    overrides = {
        k: v
        for k, v in {
            "generations": args.generations,
            "researchers": args.researchers,
            "max_turns": args.max_turns,
        }.items()
        if v is not None
    }
    cfg = ResearchConfig(**overrides)
    setups: list[Setup] = list(SETUPS) if args.setup == "all" else [args.setup]
    dataset = Dataset(Path(s.data_dir) / args.dataset)
    pricing = await check_models()

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    root = Path(s.data_dir) / "runs" / f"{stamp}-{args.dataset}"
    store = Store(root)
    logging.getLogger().addHandler(logging.FileHandler(root / "run.log"))
    store.write_json(
        "config",
        {
            "dataset": args.dataset,
            "setups": setups,
            "research": cfg.model_dump(),
            "researcher_model": s.researcher_model,
            "curator_model": s.curator_model,
            "pricing": pricing,
            "web_search": bool(s.ydc_api_key),
            "sandbox": {
                "timeout_s": s.sandbox_timeout_seconds,
                "cpu": s.sandbox_cpu,
                "memory_mb": s.sandbox_memory_mb,
            },
            "started": stamp,
        },
    )
    log.info("results in %s", root)

    code = (Path("baselines") / args.dataset / "train.py").read_text()
    baseline = await measure_baseline(dataset, code, cfg, store)
    store.write_json("baseline", baseline.__dict__)
    log.info(
        "baseline: L0 %.5f, noise threshold %.5f, test %s",
        baseline.l0,
        baseline.threshold,
        baseline.test_losses,
    )

    runs = [SetupRun(setup, dataset, baseline, cfg, root) for setup in setups]
    results = await asyncio.gather(*(r.run() for r in runs), return_exceptions=True)
    summary = []
    for setup, res in zip(setups, results, strict=True):
        if isinstance(res, BaseException):
            log.error("[%s] failed: %r", setup, res, exc_info=res)
            summary.append({"setup": setup, "error": repr(res)})
        else:
            summary.append({k: v for k, v in res.items() if k != "best_code"})
    store.write_json("summary", summary)
    store.write_json("rate_limited", inference.RATE_LIMITED)
    if inference.RATE_LIMITED:
        log.warning("Crusoe rate limits (429) hit during the run: %s", inference.RATE_LIMITED)

    path = report.write_report(root)
    print("\nsetup         fitness   val gain  best  runs")
    for row in summary:
        if "error" in row:
            print(f"{row['setup']:<13} error: {row['error'][:60]}")
            continue
        fit = f"{row['fitness']:+.2%}" if row["fitness"] is not None else "n/a"
        print(
            f"{row['setup']:<13} {fit:>8} {row['val_gain']:>+9.2%}  v{row['best_version']:<3} "
            f"{row['runs']:>4}"
        )
    print(f"\nreport: {path}")
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="churn")
    ap.add_argument("--setup", choices=[*SETUPS, "all"], default="shared_brain")
    ap.add_argument("--generations", type=int)
    ap.add_argument("--researchers", type=int)
    ap.add_argument("--max-turns", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "modal", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
