"""Prediction helper included in the model bundle: python predict.py inputs.parquet output.csv."""

import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent


class Predictor:
    def __init__(self, bundle: str | Path = BUNDLE_DIR) -> None:
        bundle = Path(bundle)
        self.runtime = json.loads((bundle / "runtime.json").read_text())
        with (bundle / "model.pkl").open("rb") as f:
            self._state = cloudpickle.load(f)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        columns = [f["name"] for f in self.runtime["features"]]
        missing = set(columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        values = self._state["predict"](self._state["model"], X.loc[:, columns].copy())
        preds = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(preds) != len(X) or not np.isfinite(preds).all():
            raise ValueError("Model must return one finite prediction per input row")
        return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--bundle", type=Path, default=BUNDLE_DIR)
    args = ap.parse_args()
    X = pd.read_csv(args.input) if args.input.suffix == ".csv" else pd.read_parquet(args.input)
    preds = Predictor(args.bundle).predict(X)
    pd.DataFrame({"prediction": preds}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
