"""Download a dataset, split it 70/15/15 and save it to data/<name>/.

uv run python scripts/prepare_data.py churn
"""

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.store import DATA_DIR  # noqa: E402

DATASETS = {
    "churn": {
        "url": "https://data.openml.org/datasets/0004/46371/dataset_46371.pq",
        "source": "OpenML 46371 (Customer_Churn_Classification_Nominal)",
        "target": "exited",
        "positive": "churned",
        "task": "Predict the chance (0 to 1) that a bank customer leaves.",
        "loss": "log_loss",
    },
    "superconductivity": {
        "url": "https://data.openml.org/datasets/0004/44148/dataset_44148.pq",
        "source": "OpenML 44148 (superconduct)",
        "target": "criticaltemp",
        "positive": None,  # regression
        "task": "Predict a superconductor's critical temperature (kelvin) from 79 material "
        "features.",
        "loss": "rmse",
    },
}
SEED = 42


def random_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 split of row indices."""
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(n * 0.70), int(n * 0.85)
    return idx[:a], idx[a:b], idx[b:]


def stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 split of row indices, stratified by label."""
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[], [], []]
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        a, b = int(len(idx) * 0.70), int(len(idx) * 0.85)
        for part, chunk in zip(parts, (idx[:a], idx[a:b], idx[b:]), strict=True):
            part.append(chunk)
    train, val, test = (rng.permutation(np.concatenate(p)) for p in parts)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(DATASETS))
    name = parser.parse_args().name
    spec = DATASETS[name]

    with urllib.request.urlopen(spec["url"]) as resp:
        df = pd.read_parquet(io.BytesIO(resp.read()))
    target = spec["target"]
    classification = spec["positive"] is not None
    if classification:
        df[target] = (df[target].astype(str) == spec["positive"]).astype(int)
    df = df.reset_index(drop=True)

    if classification:
        train_idx, val_idx, test_idx = stratified_split(df[target].to_numpy(), SEED)
    else:
        train_idx, val_idx, test_idx = random_split(len(df), SEED)
    out = DATA_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    df.iloc[train_idx].to_parquet(out / "train.parquet", index=False)
    for split, idx in (("val", val_idx), ("test", test_idx)):
        part = df.iloc[idx]
        part.drop(columns=[target]).to_parquet(out / f"{split}_X.parquet", index=False)
        part[[target]].to_parquet(out / f"{split}_y.parquet", index=False)

    meta = {
        "name": name,
        "source": spec["source"],
        "task": spec["task"],
        "target": target,
        "loss": spec["loss"],
        "split": "random 70/15/15, "
        + ("stratified by target, " if classification else "")
        + f"seed {SEED}",
        "rows": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "columns": {c: str(t) for c, t in df.drop(columns=[target]).dtypes.items()},
    }
    train_y = df[target].iloc[train_idx]
    if classification:
        meta["positive_label"] = f"{spec['positive']} (encoded as 1)"
        meta["positive_rate_train"] = round(float(train_y.mean()), 4)
    else:
        meta["target_train"] = {
            "mean": round(float(train_y.mean()), 3),
            "std": round(float(train_y.std()), 3),
            "min": round(float(train_y.min()), 3),
            "max": round(float(train_y.max()), 3),
        }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {name} to {out}: {meta['rows']}")


if __name__ == "__main__":
    main()
