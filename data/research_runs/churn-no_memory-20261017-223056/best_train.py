import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split


def fit(train: pd.DataFrame, target: str, seed: int) -> object:
    X = train.drop(columns=[target])
    y = train[target]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        border_count=128,
        bagging_temperature=1,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50, verbose=False)
    return model


def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
