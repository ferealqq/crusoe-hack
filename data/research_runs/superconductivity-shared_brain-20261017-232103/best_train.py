import numpy as np
import pandas as pd
import xgboost as xgb


def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)