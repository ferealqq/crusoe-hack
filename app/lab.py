"""Runs train.py in the sandbox and scores it. Labels stay here, outside the sandbox."""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app import scoring
from app.services import sandbox
from app.store import Store

log = logging.getLogger(__name__)

OUTPUT_TAIL_CHARS = 2000


class Dataset:
    """A prepared dataset (scripts/prepare_data.py). Shared by every setup in a run."""

    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir
        self.meta: dict[str, Any] = json.loads((data_dir / "meta.json").read_text())
        target = self.meta["target"]
        self.loss: str = self.meta["loss"]
        self.val_y = pd.read_parquet(data_dir / "val_y.parquet")[target].to_numpy(np.float64)
        self.test_y = pd.read_parquet(data_dir / "test_y.parquet")[target].to_numpy(np.float64)
        self._test_cache: dict[str, asyncio.Task[list[float | None]]] = {}

    def card(self) -> str:
        """The dataset card agents read. No val or test labels or scores."""
        return json.dumps(self.meta, indent=2)

    async def test_losses(self, code: str, seeds: list[int]) -> list[float | None]:
        """Test loss per seed (None for a failed seed). Cached by code, across setups."""
        key = hashlib.sha256(code.encode()).hexdigest() + str(seeds)
        if key not in self._test_cache:
            self._test_cache[key] = asyncio.create_task(self._score_test(code, seeds))
        return await self._test_cache[key]

    async def _score_test(self, code: str, seeds: list[int]) -> list[float | None]:
        results = await asyncio.gather(
            *(sandbox.run_train(code, self.dir, seed, mode="test") for seed in seeds)
        )
        losses: list[float | None] = []
        for seed, r in zip(seeds, results, strict=True):
            try:
                if r.status != "ok" or r.preds is None:
                    raise ValueError(r.status + ": " + r.output[-500:])
                losses.append(scoring.score(self.loss, self.test_y, r.preds))
            except ValueError as e:
                log.warning("test scoring failed for seed %s: %s", seed, e)
                losses.append(None)
        return losses


class Experiment(BaseModel):
    id: str
    gen: int
    agent: str
    seed: int
    status: Literal["ok", "error", "timeout"]
    val_loss: float | None
    runtime_s: float
    note: str
    output_tail: str
    code: str
    started_from: int  # best version the agent started from


class Lab:
    """One research run's experiments: run, score on val, record."""

    def __init__(self, dataset: Dataset, store: Store) -> None:
        self.dataset = dataset
        self.store = store
        self.experiments: dict[str, Experiment] = {}
        self._count = 0

    async def run(
        self, code: str, seed: int, agent: str, note: str, gen: int, started_from: int
    ) -> Experiment:
        self._count += 1
        exp_id = f"e{self._count}"
        result = await sandbox.run_train(code, self.dataset.dir, seed, mode="research")
        status, val_loss, output = result.status, None, result.output
        if status == "ok" and result.preds is not None:
            try:
                val_loss = scoring.score(self.dataset.loss, self.dataset.val_y, result.preds)
            except ValueError as e:
                status, output = "error", output + f"\n[bad predictions: {e}]"
        exp = Experiment(
            id=exp_id,
            gen=gen,
            agent=agent,
            seed=seed,
            status=status,
            val_loss=val_loss,
            runtime_s=round(result.runtime_s, 1),
            note=note[:300],
            output_tail=output[-OUTPUT_TAIL_CHARS:],
            code=code,
            started_from=started_from,
        )
        self.experiments[exp_id] = exp
        self.store.append("experiments", exp.model_dump())
        return exp
