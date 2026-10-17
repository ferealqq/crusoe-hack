import numpy as np
import pandas as pd
import lightgbm as lgb

def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]