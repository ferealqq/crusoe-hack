import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import ResearchConfig
from app.data import Dataset, load_dataset
from app.lab import Lab


def write_dataset(root: Path, name: str = "toy", n: int = 200) -> None:
    """A tiny prepared dataset: 20% positives, two numeric features, target `y`."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = (np.arange(n) % 5 == 0).astype(int)
    folder = root / name
    folder.mkdir(parents=True)
    train, val, test = df.iloc[:140], df.iloc[140:170], df.iloc[170:]
    train.to_parquet(folder / "train.parquet", index=False)
    for split, part in (("val", val), ("test", test)):
        part.drop(columns=["y"]).to_parquet(folder / f"{split}_X.parquet", index=False)
        part[["y"]].to_parquet(folder / f"{split}_y.parquet", index=False)
    meta = {"name": name, "target": "y", "loss": "log_loss", "rows": {"train": 140, "val": 30}}
    (folder / "meta.json").write_text(json.dumps(meta))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"  # async tests run with the anyio plugin (installed with httpx)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    write_dataset(tmp_path)
    return load_dataset("toy", root=tmp_path)


@pytest.fixture
def lab(dataset: Dataset, tmp_path: Path) -> Lab:
    return Lab(dataset, "shared_brain", ResearchConfig(), root=tmp_path, run_id="test-run")
