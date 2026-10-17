"""Starting point: ridge regression on all columns, standardized."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    model.fit(X, y)
    return model


def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)
