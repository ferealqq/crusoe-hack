"""Re-emit a recorded run through RunLogger in (sped-up) real time, as a stand-in for the
orchestrator. Use it to rehearse the "Watch live" demo or to test the SSE path.

    uv run python scripts/simulate_live_run.py [--sample sample-shared-brain] [--speed 20]
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.runlog import store  # noqa: E402
from app.runlog.events import RunStarted, event_adapter  # noqa: E402
from app.runlog.logger import RunLogger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="sample-shared-brain")
    parser.add_argument("--speed", type=float, default=20.0, help="run-time seconds per second")
    parser.add_argument("--max-gap", type=float, default=3.0, help="longest wait, in seconds")
    args = parser.parse_args()

    lines = (store.SAMPLES_DIR / f"{args.sample}.jsonl").read_text().splitlines()
    events = [event_adapter.validate_json(line) for line in lines]
    start = events[0]
    assert isinstance(start, RunStarted)
    run_id = f"live-{datetime.now():%H%M%S}"
    print(
        f"streaming {len(events)} events as {run_id}; watch at http://localhost:5173/run/{run_id}?live=1"
    )

    with RunLogger.start(
        run_id, setup=start.setup, config=start.config, baseline_val_loss=start.baseline_val_loss
    ) as runlog:
        prev = 0.0
        for e in events[1:]:
            time.sleep(min(args.max_gap, max(0.0, e.ts - prev) / args.speed))
            prev = e.ts
            if e.t != "run.finished":
                runlog.emit(e)


if __name__ == "__main__":
    main()
