"""State for one research run: dataset, budget, run log, best train.py and memory."""

import asyncio
import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app import gitlog, scoring
from app.config import ResearchConfig
from app.data import Dataset
from app.memory import Memory
from app.schemas import Experiment
from app.services import sandbox
from app.store import DATA_DIR, append_jsonl

Setup = Literal["shared_brain", "no_memory", "one_shot"]
SETUPS: tuple[Setup, ...] = ("shared_brain", "no_memory", "one_shot")

log = logging.getLogger(__name__)


def researcher_runs(config: ResearchConfig, setup: Setup) -> int:
    """Shared Brain researchers give up runs for the curator, so it and No memory use the same
    total. One-shot is the floor: a single edit."""
    if setup == "one_shot":
        return 1
    if setup == "no_memory":
        return config.researcher_runs
    return config.researcher_runs - math.ceil(config.curator_runs / config.researchers)


def runs_per_generation(config: ResearchConfig, setup: Setup) -> int:
    curator = config.curator_runs if setup == "shared_brain" else 0
    return config.researchers * researcher_runs(config, setup) + curator


@dataclass(frozen=True)
class Best:
    version: int
    code: str
    val_loss: float  # verified score (baseline: mean of seeds 0-4; curator: mean of seeds 1-3)
    run_id: str
    note: str
    seed_loss: float  # its score on the research seed, which researchers' runs use


class Lab:
    def __init__(
        self,
        dataset: Dataset,
        setup: Setup,
        config: ResearchConfig,
        *,
        root: Path = DATA_DIR,
        commit: bool = False,
        run_id: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.setup = setup
        self.config = config
        self.root = root
        self.run_id = run_id or f"{dataset.name}-{setup}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
        self.dir = root / "research_runs" / self.run_id
        self.commit_enabled = commit
        self.memory = Memory(self.dir / "memory.jsonl", config.memory_cap_tokens)
        self.experiments: list[Experiment] = []
        self.baseline_code = ""
        self.l0 = 0.0
        self.threshold = 0.0
        self._best: Best | None = None
        self._counters: Counter[str] = Counter()

    @property
    def best(self) -> Best:
        if self._best is None:
            raise RuntimeError("run the baseline first")
        return self._best

    @best.setter
    def best(self, best: Best) -> None:
        self._best = best
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "best_train.py").write_text(best.code)

    @property
    def loss(self) -> str:
        return self.dataset.meta.get("loss", "log_loss")

    @property
    def researcher_runs(self) -> int:
        return researcher_runs(self.config, self.setup)

    def new_run_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}-{self._counters[prefix]}"

    async def run_experiment(
        self,
        code: str,
        *,
        prefix: str,
        agent: str,
        generation: int,
        seed: int,
        note: str = "",
        split: Literal["val", "test"] = "val",
    ) -> Experiment:
        """Run train.py in a sandbox and score it here, outside the sandbox."""
        run_id = self.new_run_id(prefix)
        files = self.dataset.research_files if split == "val" else self.dataset.test_files
        result = await sandbox.run_train(code, files, self.dataset.target, seed, self.loss)
        loss = None
        if result.status == "ok" and result.preds is not None:
            labels = self.dataset.val_y if split == "val" else self.dataset.test_y
            loss = scoring.score(self.loss, labels, np.asarray(result.preds))
        exp = Experiment(
            run_id=run_id,
            generation=generation,
            agent=agent,
            seed=seed,
            split=split,
            status=result.status,
            loss=loss,
            runtime_s=result.runtime_s,
            note=note,
            output_tail=result.output_tail,
            code=code,
        )
        self.experiments.append(exp)
        append_jsonl(self.dir / "experiments.jsonl", exp.model_dump())
        score = f" loss {loss:.5f}" if loss is not None else ""
        log.info("%s %s seed %d: %s%s  %s", run_id, split, seed, result.status, score, note[:80])
        return exp

    def find_run(self, run_id: str) -> Experiment | None:
        """A research (val) run from this research run. Never returns a test run."""
        return next((e for e in self.experiments if e.run_id == run_id and e.split == "val"), None)

    def log_agent(self, row: dict[str, Any]) -> None:
        append_jsonl(self.dir / "agents.jsonl", row)

    def log_generation(self, row: dict[str, Any]) -> None:
        append_jsonl(self.dir / "generations.jsonl", row)

    async def checkpoint(self, message: str) -> None:
        """Commit this run's files to git (no-op unless commits are on)."""
        if not self.commit_enabled:
            return
        paths = [self.dir, self.root / "research_runs.jsonl"]
        full = f"chore(brain): {message} [{self.run_id}]"
        sha = await asyncio.to_thread(gitlog.commit, paths, full)
        if sha:
            log.info("committed %s: %s", sha, message)
