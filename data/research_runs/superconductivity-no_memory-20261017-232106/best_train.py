import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model = CatBoostRegressor(
        iterations=20000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        border_count=64,
        loss_function='RMSE',
        random_seed=seed,
        verbose=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=200,
        verbose=False,
        plot=False
    )
    # Refit on full data with best iteration
    best_model = CatBoostRegressor(
        iterations=model.best_iteration_,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        border_count=64,
        loss_function='RMSE',
        random_seed=seed,
        verbose=False,
    )
    best_model.fit(X, y, verbose=False)
    return best_model

def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)