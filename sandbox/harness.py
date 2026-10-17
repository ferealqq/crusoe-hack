"""Locked. Runs inside the sandbox: fit train.py on train.parquet, predict X.parquet.

    python harness.py --target exited --seed 0

Writes predictions to preds.json. Never sees labels for X.parquet: the orchestrator scores
the predictions outside the sandbox.
"""

import argparse
import json
import math

import pandas as pd
import train  # the agent's train.py, next to this file

parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

train_df = pd.read_parquet("train.parquet")
X = pd.read_parquet("X.parquet")

model = train.fit(train_df, args.target, args.seed)
preds = [float(p) for p in train.predict(model, X)]

if len(preds) != len(X):
    raise SystemExit(f"predict returned {len(preds)} values for {len(X)} rows")
if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in preds):
    raise SystemExit("predict must return probabilities between 0 and 1")

with open("preds.json", "w") as f:
    json.dump(preds, f)
