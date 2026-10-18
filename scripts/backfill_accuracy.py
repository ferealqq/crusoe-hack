"""Re-score finished research runs on the test split to record test accuracy.

Runs from before accuracy was tracked only stored test log loss. This reruns their starting
and best train.py in Modal sandboxes (test seeds, same harness as `loop.final_test`) and
appends the test runs, with accuracy, to each run's experiments.jsonl. `app/replay.py` reads
them from there. Runs that already have accuracy are skipped.

    uv run python scripts/backfill_accuracy.py               # every finished run
    uv run python scripts/backfill_accuracy.py <run_id> ...  # just these
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import scoring  # noqa: E402
from app.config import ResearchConfig  # noqa: E402
from app.data import load_dataset  # noqa: E402
from app.schemas import Experiment  # noqa: E402
from app.services import sandbox  # noqa: E402
from app.store import DATA_DIR, append_jsonl, read_jsonl  # noqa: E402


def needs_backfill(row: dict, exps: list[dict]) -> bool:
    if row.get("fitness") is None or row.get("test_best_accuracy"):
        return False
    return not any(e["split"] == "test" and e.get("accuracy") is not None for e in exps)


async def backfill(row: dict) -> None:
    run_dir = DATA_DIR / "research_runs" / row["run_id"]
    exps = read_jsonl(run_dir / "experiments.jsonl")
    dataset = load_dataset(row["dataset"])
    cfg = ResearchConfig(**row["config"])
    start = next(e["code"] for e in exps if e["generation"] == 0)
    versions = {0: start}
    if row["best_version"]:
        versions[row["best_version"]] = (run_dir / "best_train.py").read_text()

    async def score(version: int, code: str, seed: int) -> Experiment:
        res = await sandbox.run_train(code, dataset.test_files, dataset.target, seed)
        ok = res.status == "ok" and res.preds is not None
        preds = np.asarray(res.preds) if ok else None
        return Experiment(
            run_id=f"test-v{version}-backfill-{seed}",
            generation=cfg.generations + 1,
            agent="orchestrator",
            seed=seed,
            split="test",
            status=res.status,
            loss=scoring.log_loss(dataset.test_y, preds) if ok else None,
            accuracy=scoring.accuracy(dataset.test_y, preds) if ok else None,
            runtime_s=res.runtime_s,
            note=f"test: train.py v{version} (accuracy backfill)",
            output_tail=res.output_tail,
            code=code,
        )

    done = await asyncio.gather(
        *(score(v, code, s) for v, code in versions.items() for s in cfg.test_seeds)
    )
    for exp in done:
        append_jsonl(run_dir / "experiments.jsonl", exp.model_dump())
        acc = f"{exp.accuracy:.4f}" if exp.accuracy is not None else exp.status
        print(f"{row['run_id']} {exp.run_id}: accuracy {acc}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_ids", nargs="*")
    args = parser.parse_args()
    rows = [
        r
        for r in read_jsonl(DATA_DIR / "research_runs.jsonl")
        if (not args.run_ids or r["run_id"] in args.run_ids)
        and needs_backfill(
            r, read_jsonl(DATA_DIR / "research_runs" / r["run_id"] / "experiments.jsonl")
        )
    ]
    print(f"{len(rows)} runs to re-score: {', '.join(r['run_id'] for r in rows) or 'none'}")
    await asyncio.gather(*(backfill(r) for r in rows))


if __name__ == "__main__":
    asyncio.run(main())
