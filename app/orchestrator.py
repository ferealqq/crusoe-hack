"""CLI that runs research runs. A plain Python script, not an agent.

    uv run python -m app.orchestrator --dataset churn --setup shared_brain
    uv run python -m app.orchestrator --dataset churn --setup all

With --setup all, the setups run in parallel and share one baseline measurement (same L0
and noise threshold). Results go to data/runs/<timestamp>-<dataset>/, then report.html.
"""

import argparse
import asyncio
import logging
import signal
from datetime import UTC, datetime
from pathlib import Path

from app import events, report
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
        raise RuntimeError(f"models not on Crusoe: {missing}. Available: {sorted(ids)}")
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
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    root = Path(s.data_dir) / "runs" / f"{stamp}-{args.dataset}"
    store = Store(root)
    event_log = events.EventLog(root)
    handler = logging.FileHandler(root / "run.log")
    logging.getLogger().addHandler(handler)
    with events.bind(event_log):
        events.emit(
            "run.started",
            dataset=args.dataset,
            setups=setups,
            research=cfg.model_dump(),
            researcher_model=s.researcher_model,
            curator_model=s.curator_model,
        )
        heartbeat = asyncio.create_task(_heartbeat())
        terminal, terminal_data = "run.failed", {}
        try:
            success = await _execute(args, cfg, setups, root, store, stamp)
            terminal = "run.completed" if success else "run.failed"
        except asyncio.CancelledError:
            terminal = "run.cancelled"
            raise
        except Exception as e:
            terminal_data = {"error_type": type(e).__name__}
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            events.emit(terminal, **terminal_data)
            logging.getLogger().removeHandler(handler)
            handler.close()
    return root


async def _heartbeat() -> None:
    while True:
        await asyncio.sleep(5)
        events.emit("run.heartbeat")


async def _execute(
    args: argparse.Namespace,
    cfg: ResearchConfig,
    setups: list[Setup],
    root: Path,
    store: Store,
    stamp: str,
) -> bool:
    s = get_settings()
    dataset = Dataset(Path(s.data_dir) / args.dataset)
    pricing = await check_models()
    store.write_json(
        "config",
        {
            "dataset": args.dataset,
            "loss": dataset.loss,
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
    with events.scope(phase="baseline", role="orchestrator"):
        events.emit("baseline.started", seeds=cfg.baseline_seeds)
        baseline = await measure_baseline(dataset, code, cfg, store)
        events.emit(
            "baseline.completed",
            l0=baseline.l0,
            threshold=baseline.threshold,
            val_losses=baseline.val_losses,
            test_losses=baseline.test_losses,
        )
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
    return all("error" not in row for row in summary)


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
    try:
        asyncio.run(_cli(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("research run cancelled")
        raise SystemExit(130) from None


async def _cli(args: argparse.Namespace) -> None:
    # SIGINT is handled by asyncio.run; handle SIGTERM as cancellation as well so
    # sandbox finally blocks finish before the terminal event is recorded.
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await main_async(args)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


if __name__ == "__main__":
    main()
