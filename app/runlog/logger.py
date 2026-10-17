"""RunLogger: how the orchestrator emits events.

    with RunLogger.start("sb-2026-10-17-a", setup="shared_brain", config=cfg,
                         baseline_val_loss=2.31) as runlog:
        runlog.emit(GenStarted(gen=1, start_best_val_loss=2.31))
        ...

Events are validated, numbered and timestamped, then appended to var/runs/<run_id>.jsonl,
which the frontend can stream live.
"""

import time
from typing import Self

from app.runlog import store
from app.runlog.events import Event, RunConfig, RunFinished, RunStarted, Setup, event_adapter


class RunLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.path = store.local_path(run_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._t0 = time.monotonic()
        self._finished = False

    @classmethod
    def start(
        cls, run_id: str, *, setup: Setup, config: RunConfig, baseline_val_loss: float
    ) -> Self:
        logger = cls(run_id)
        logger.path.write_text("")
        logger.emit(
            RunStarted(
                run_id=run_id, setup=setup, config=config, baseline_val_loss=baseline_val_loss
            )
        )
        return logger

    def emit(self, event: Event) -> None:
        event.seq, event.ts = self._seq, round(time.monotonic() - self._t0, 3)
        self._seq += 1
        line = event_adapter.dump_json(event, exclude_none=True).decode()
        with self.path.open("a") as f:
            f.write(line + "\n")
        if isinstance(event, RunFinished):
            self._finished = True

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        if not self._finished:
            self.emit(RunFinished(status=status, error=error))  # type: ignore[arg-type]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish("failed" if exc else "completed", str(exc) if exc else None)
