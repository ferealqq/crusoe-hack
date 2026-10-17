"""Locked research harness. Runs inside the sandbox; agents can't edit it.

    python /harness/harness.py --seed 0

Fits /work/train.py on /data/train.parquet, predicts /data/val_X.parquet and saves the
predictions to /out/preds.npy. The sandbox never has the val labels: the orchestrator
scores the predictions outside it.
"""

import argparse
import importlib.util
import json
import os
import random
import time

import numpy as np
import pandas as pd

# Tests point this at a temp dir; in the sandbox everything lives under /.
ROOT = os.environ.get("HARNESS_ROOT", "/")


def load_user_module(path: str):
    spec = importlib.util.spec_from_file_location("train", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(ROOT, "data/meta.json")))
    train = pd.read_parquet(os.path.join(ROOT, "data/train.parquet"))
    X = pd.read_parquet(os.path.join(ROOT, "data/val_X.parquet"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    user = load_user_module(os.path.join(ROOT, "work/train.py"))
    started = time.perf_counter()
    model = user.fit(train, meta["target"], args.seed)
    fitted = time.perf_counter()
    preds = np.asarray(user.predict(model, X), dtype=np.float64).reshape(-1)
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    np.save(os.path.join(ROOT, "out/preds.npy"), preds)
    print(f"[harness] fit {fitted - started:.1f}s, predict {time.perf_counter() - fitted:.1f}s")


if __name__ == "__main__":
    main()
