"""Download a dataset, split it and save it under data/<name>/.

    uv run python scripts/prepare_data.py churn

Writes train.parquet (with the target), val_X/val_y, test_X/test_y, trainval.parquet
(train + val with labels, used by final scoring and model export) and meta.json (the dataset card).
"""

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

SEED = 42


def load_churn() -> tuple[pd.DataFrame, dict]:
    url = "https://data.openml.org/datasets/0004/46371/dataset_46371.pq"
    with urllib.request.urlopen(url, timeout=120) as r:
        df = pd.read_parquet(io.BytesIO(r.read()))
    df["exited"] = (df["exited"].astype(str) == "churned").astype("int8")
    meta = {
        "name": "churn",
        "source": "OpenML 46371 (Customer Churn, CC BY 4.0)",
        "target": "exited",
        "loss": "log_loss",
        "predict": "probability that the customer leaves (exited = 1), between 0 and 1",
        "split": "random 70/15/15, stratified on the target, seed 42",
    }
    return df, meta


def load_superconductivity() -> tuple[pd.DataFrame, dict]:
    url = "https://data.openml.org/datasets/0004/44148/dataset_44148.pq"
    with urllib.request.urlopen(url, timeout=120) as r:
        df = pd.read_parquet(io.BytesIO(r.read()))
    meta = {
        "name": "superconductivity",
        "source": "OpenML 44148 (Superconductivity, CC BY 4.0)",
        "target": "criticaltemp",
        "loss": "rmse",
        "predict": "critical temperature in kelvin at which the material superconducts",
        "split": "random 70/15/15, seed 42",
    }
    return df, meta


DATASETS = {"churn": load_churn, "superconductivity": load_superconductivity}


def split(df: pd.DataFrame, target: str, stratify: bool) -> tuple[pd.DataFrame, ...]:
    strat = df[target] if stratify else None
    train, rest = train_test_split(df, test_size=0.30, random_state=SEED, stratify=strat)
    strat = rest[target] if stratify else None
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=strat)
    return tuple(x.reset_index(drop=True) for x in (train, val, test))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(DATASETS))
    args = ap.parse_args()

    df, meta = DATASETS[args.name]()
    target = meta["target"]
    train, val, test = split(df, target, stratify=meta["loss"] == "log_loss")

    out = Path(get_settings().data_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet")
    val.drop(columns=[target]).to_parquet(out / "val_X.parquet")
    val[[target]].to_parquet(out / "val_y.parquet")
    test.drop(columns=[target]).to_parquet(out / "test_X.parquet")
    test[[target]].to_parquet(out / "test_y.parquet")
    pd.concat([train, val], ignore_index=True).to_parquet(out / "trainval.parquet")

    meta |= {
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "train_target_mean": round(float(train[target].mean()), 4),
        "columns": {c: str(t) for c, t in train.dtypes.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: train {len(train)}, val {len(val)}, test {len(test)}")


if __name__ == "__main__":
    main()
