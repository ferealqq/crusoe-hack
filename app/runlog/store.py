"""Where run event logs live.

- `app/runlog/samples/*.jsonl`: recorded runs committed to the repo (always available for demos)
- `var/runs/*.jsonl`: runs written locally by RunLogger (live while being written)
- `brain-runs/*.jsonl` in the Neon bucket: finished runs uploaded by RunLogger
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.services import storage

log = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent / "samples"
LOCAL_DIR = Path("var/runs")
BUCKET_PREFIX = "brain-runs/"


@dataclass
class RunSummary:
    run_id: str
    source: str  # sample | local | bucket
    setup: str | None
    status: str  # live | completed | failed
    best_val_loss: float | None
    baseline_val_loss: float | None
    fitness: float | None
    generations: int | None


def summarize(run_id: str, source: str, lines: list[str]) -> RunSummary:
    s = RunSummary(run_id, source, None, "live", None, None, None, None)
    for line in lines:
        if not line.strip():
            continue
        e = json.loads(line)
        match e["t"]:
            case "run.started":
                s.setup = e["setup"]
                s.baseline_val_loss = e["baseline_val_loss"]
                s.generations = e["config"]["generations"]
            case "gen.finished":
                s.best_val_loss = e["best_val_loss"]
            case "eval.finished":
                s.fitness = e["fitness"]
            case "run.finished":
                s.status = e["status"]
    return s


def local_path(run_id: str) -> Path:
    return LOCAL_DIR / f"{run_id}.jsonl"


def _local_files() -> dict[str, tuple[str, Path]]:
    files = {p.stem: ("sample", p) for p in SAMPLES_DIR.glob("*.jsonl")}
    if LOCAL_DIR.exists():
        files |= {p.stem: ("local", p) for p in LOCAL_DIR.glob("*.jsonl")}
    return files


async def list_runs() -> list[RunSummary]:
    runs = {
        run_id: summarize(run_id, source, path.read_text().splitlines())
        for run_id, (source, path) in _local_files().items()
    }
    try:
        for key in await storage.list_keys(BUCKET_PREFIX):
            run_id = Path(key).stem
            if run_id not in runs:
                text = (await storage.get_bytes(key)).decode()
                runs[run_id] = summarize(run_id, "bucket", text.splitlines())
    except Exception as e:  # bucket is optional for the demo (e.g. no .env yet)
        log.info("skipping bucket runs: %s", e)
    return sorted(runs.values(), key=lambda r: r.run_id)


async def read_lines(run_id: str) -> list[str] | None:
    if found := _local_files().get(run_id):
        return found[1].read_text().splitlines()
    try:
        return (await storage.get_bytes(f"{BUCKET_PREFIX}{run_id}.jsonl")).decode().splitlines()
    except Exception:
        return None
