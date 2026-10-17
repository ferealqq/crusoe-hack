import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=6,
        loss_function='Logloss',
        random_seed=seed,
        verbose=False,
        l2_leaf_reg=3,
    )
    model.fit(X, y)
    return model

def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]