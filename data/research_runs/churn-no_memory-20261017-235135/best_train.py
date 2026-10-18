import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        random_seed=seed,
        verbose=False,
        thread_count=2
    )
    model.fit(X, y)
    return model

def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]