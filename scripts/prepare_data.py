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
    }
}
SEED = 42


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
    df[target] = (df[target].astype(str) == spec["positive"]).astype(int)
    df = df.reset_index(drop=True)

    train_idx, val_idx, test_idx = stratified_split(df[target].to_numpy(), SEED)
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
        "positive_label": f"{spec['positive']} (encoded as 1)",
        "loss": spec["loss"],
        "split": "random 70/15/15, stratified by target, seed 42",
        "rows": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "positive_rate_train": round(float(df[target].iloc[train_idx].mean()), 4),
        "columns": {c: str(t) for c, t in df.drop(columns=[target]).dtypes.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {name} to {out}: {meta['rows']}, positive rate {meta['positive_rate_train']}")


if __name__ == "__main__":
    main()
