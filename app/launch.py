"""Start a research run from the UI: one orchestrator process per setup, all with the same
timestamp, so app/replay.py groups them into one run (`<dataset>-<stamp>`).

The processes are the same CLI as `uv run python -m app.orchestrator`: they write data/ and
commit their own files. Their output goes to var/logs/ (not committed).
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.store import DATA_DIR

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "var" / "logs"
SETUPS = ("shared_brain", "no_memory")
START_TIMEOUT_S = 90.0  # imports, the Crusoe model check, then config.json appears


class LaunchError(RuntimeError):
    pass


@dataclass
class Launch:
    run_id: str
    procs: list[subprocess.Popen]
    logs: list[Path]

    def running(self) -> bool:
        return any(p.poll() is None for p in self.procs)


_launches: list[Launch] = []


def active() -> Launch | None:
    """A run started here whose processes are still going."""
    return next((x for x in _launches if x.running()), None)


def _tail(path: Path, chars: int = 1500) -> str:
    try:
        return path.read_text(errors="replace")[-chars:]
    except OSError:
        return ""


def start(prompt: str, generations: int, dataset: str = "churn") -> Launch:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    for setup in SETUPS:
        log = LOG_DIR / f"{dataset}-{setup}-{stamp}.log"
        cmd = [sys.executable, "-m", "app.orchestrator", "--dataset", dataset, "--setup", setup,
               "--generations", str(generations), "--stamp", stamp, "--goal", prompt]  # fmt: skip
        with log.open("w") as out:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
            )
        procs.append(proc)
        logs.append(log)
    launch = Launch(f"{dataset}-{stamp}", procs, logs)
    _launches.append(launch)
    return launch


def wait_started(launch: Launch, dataset: str = "churn", root: Path | None = None) -> None:
    """Block until every setup has written its config.json, so the run is visible to the API.

    Raises LaunchError with the log tail if a process exits first (bad key, missing model).
    """
    root = root or ROOT / DATA_DIR
    stamp = launch.run_id.removeprefix(f"{dataset}-")
    configs = [root / "research_runs" / f"{dataset}-{s}-{stamp}" / "config.json" for s in SETUPS]
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        for proc, log in zip(launch.procs, launch.logs, strict=True):
            if proc.poll() is not None and proc.returncode != 0:
                for p in launch.procs:
                    p.terminate()
                raise LaunchError(f"orchestrator exited with {proc.returncode}:\n{_tail(log)}")
        if all(c.exists() for c in configs):
            return
        time.sleep(0.5)
    raise LaunchError(f"run did not start within {START_TIMEOUT_S:.0f} s; see {launch.logs[0]}")
