"""Locked final scorer. Only the orchestrator runs it; agents never see it.

    python /harness/test.py --seed 1

Fits /work/train.py on train + val (with labels), predicts /data/test_X.parquet and saves
the predictions to /out/preds.npy. The test labels never enter the sandbox.
"""

import argparse
import importlib.util
import json
import os
import random

import numpy as np
import pandas as pd

# Tests point this at a temp dir; in the sandbox everything lives under /.
ROOT = os.environ.get("HARNESS_ROOT", "/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(ROOT, "data/meta.json")))
    train = pd.read_parquet(os.path.join(ROOT, "data/trainval.parquet"))
    X = pd.read_parquet(os.path.join(ROOT, "data/test_X.parquet"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    spec = importlib.util.spec_from_file_location("train", os.path.join(ROOT, "work/train.py"))
    assert spec and spec.loader
    user = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user)
    model = user.fit(train, meta["target"], args.seed)
    preds = np.asarray(user.predict(model, X), dtype=np.float64).reshape(-1)
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    np.save(os.path.join(ROOT, "out/preds.npy"), preds)


if __name__ == "__main__":
    main()
