"""Load a prepared dataset (scripts/prepare_data.py) and build the files each sandbox gets."""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.store import DATA_DIR


@dataclass(frozen=True)
class Dataset:
    name: str
    meta: dict[str, Any]
    target: str
    val_y: np.ndarray
    test_y: np.ndarray
    # What the sandbox gets. Research: train + val inputs. Test: train + val (with labels) +
    # test inputs. Labels for X.parquet never go in.
    research_files: dict[str, bytes]
    test_files: dict[str, bytes]


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def load_dataset(name: str, root: Path = DATA_DIR) -> Dataset:
    folder = root / name
    if not (folder / "meta.json").exists():
        raise FileNotFoundError(
            f"{folder} not found: run `uv run python scripts/prepare_data.py {name}`"
        )
    meta = json.loads((folder / "meta.json").read_text())
    target = meta["target"]
    train = pd.read_parquet(folder / "train.parquet")
    val_X = pd.read_parquet(folder / "val_X.parquet")
    val_y = pd.read_parquet(folder / "val_y.parquet")[target].to_numpy()
    test_X = pd.read_parquet(folder / "test_X.parquet")
    test_y = pd.read_parquet(folder / "test_y.parquet")[target].to_numpy()
    train_val = pd.concat([train, val_X.assign(**{target: val_y})], ignore_index=True)
    return Dataset(
        name=name,
        meta=meta,
        target=target,
        val_y=val_y,
        test_y=test_y,
        research_files={"train.parquet": _parquet(train), "X.parquet": _parquet(val_X)},
        test_files={"train.parquet": _parquet(train_val), "X.parquet": _parquet(test_X)},
    )
