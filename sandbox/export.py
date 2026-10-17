"""Fit the accepted train.py on train + val and serialize it inside the sandbox."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("HARNESS_ROOT", "/"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    meta = json.loads((ROOT / "data/meta.json").read_text())
    train = pd.read_parquet(ROOT / "data/trainval.parquet")
    target = meta["target"]
    features = train.drop(columns=[target])
    sample = features.head(5).copy()
    schema = [{"name": c, "dtype": str(t)} for c, t in features.dtypes.items()]
    rows = len(train)

    random.seed(args.seed)
    np.random.seed(args.seed)
    spec = importlib.util.spec_from_file_location("train", ROOT / "work/train.py")
    assert spec and spec.loader
    user = importlib.util.module_from_spec(spec)
    sys.modules["train"] = user
    spec.loader.exec_module(user)
    model = user.fit(train, target, args.seed)
    # Retain custom classes, closures, preprocessing and predict() globals as well as weights.
    cloudpickle.register_pickle_by_value(user)
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    with (out / "model.pkl").open("wb") as f:
        cloudpickle.dump({"model": model, "predict": user.predict}, f)

    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"].lower() not in {"pip", "setuptools", "wheel"}
    }
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "seed": args.seed,
        "training_split": "train+val",
        "training_rows": rows,
        "target": target,
        "features": schema,
    }
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    requirements = "--extra-index-url https://download.pytorch.org/whl/cpu\n"
    requirements += "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))
    (out / "requirements.txt").write_text(requirements)

    # Verify the downloadable loader in a fresh process rather than relying on the
    # already-imported training module. Only training inputs are used.
    expected = np.asarray(user.predict(model, sample.copy()), dtype=np.float64).reshape(-1)
    sample.to_parquet(out / "sample.parquet")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("predict.py")),
            str(out / "sample.parquet"),
            str(out / "sample.csv"),
            "--bundle",
            str(out),
        ],
        check=True,
        cwd=out,
    )
    actual = pd.read_csv(out / "sample.csv")["prediction"].to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
    (out / "sample.parquet").unlink()
    (out / "sample.csv").unlink()


if __name__ == "__main__":
    main()
