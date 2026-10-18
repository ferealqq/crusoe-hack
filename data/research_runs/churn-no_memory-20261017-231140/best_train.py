import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    # Split for early stopping
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=5,
        loss_function='Logloss',
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
    return model

def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]