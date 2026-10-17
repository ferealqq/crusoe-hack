import numpy as np
import pandas as pd
import lightgbm as lgb

def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)