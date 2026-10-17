"""Where run event logs live.

- `app/runlog/samples/*.jsonl`: recorded runs committed to the repo (always available for demos)
- `var/runs/*.jsonl`: runs written locally by RunLogger (live while being written)
"""

import json
from dataclasses import dataclass
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent / "samples"
LOCAL_DIR = Path("var/runs")


@dataclass
class RunSummary:
    run_id: str
    source: str  # sample | local
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
    runs = [
        summarize(run_id, source, path.read_text().splitlines())
        for run_id, (source, path) in _local_files().items()
    ]
    return sorted(runs, key=lambda r: r.run_id)


async def read_lines(run_id: str) -> list[str] | None:
    if found := _local_files().get(run_id):
        return found[1].read_text().splitlines()
    return None
