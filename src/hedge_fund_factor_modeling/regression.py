from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


def time_series_split(
    df: pd.DataFrame,
    factor_cols: list[str],
    target_col: str,
    train_end: str,
    test_start: str,
) -> SplitData:
    """Split data chronologically to avoid lookahead bias."""
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    train = df.loc[:train_end_ts]
    test = df.loc[test_start_ts:]

    if train.empty or test.empty:
        raise ValueError(
            f"Empty train or test split. train={len(train)}, test={len(test)}. "
            "Check train_end and test_start against your data range."
        )
    if train.index.max() >= test.index.min():
        raise ValueError("Train period overlaps test period — possible lookahead bias.")

    X_train = train[factor_cols]
    y_train = train[target_col]
    X_test = test[factor_cols]
    y_test = test[target_col]

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_index=train.index,
        test_index=test.index,
    )


def fit_ols(X_train: pd.DataFrame, y_train: pd.Series):
    X = sm.add_constant(X_train)
    return sm.OLS(y_train, X).fit()


def predict_ols(model, X: pd.DataFrame) -> np.ndarray:
    X_const = sm.add_constant(X)
    return model.predict(X_const)


def fit_ridge(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def fit_lasso(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 0.1) -> Lasso:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def fit_elastic_net(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> ElasticNet:
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def fit_ols_sklearn(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def grid_search_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> GridSearchCV:
    param_grid = {"alpha": np.logspace(-4, 2, 20)}
    cv = TimeSeriesSplit(n_splits=n_splits)
    grid = GridSearchCV(
        Ridge(),
        param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
    )
    grid.fit(X_train, y_train)
    return grid


def grid_search_elastic_net(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> GridSearchCV:
    param_grid = {
        "alpha": np.logspace(-4, 2, 10),
        "l1_ratio": np.linspace(0, 1, 10),
    }
    cv = TimeSeriesSplit(n_splits=n_splits)
    grid = GridSearchCV(
        ElasticNet(max_iter=10000, random_state=42),
        param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
    )
    grid.fit(X_train, y_train)
    return grid


def get_factor_exposures(model, factor_cols: list[str]) -> pd.Series:
    """Extract coefficients from sklearn or statsmodels model."""
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    if hasattr(model, "coef_"):
        return pd.Series(model.coef_, index=factor_cols, name="exposure")
    if hasattr(model, "params"):
        params = model.params.drop("const", errors="ignore")
        return params.reindex(factor_cols)
    raise TypeError(f"Unsupported model type: {type(model)}")


MODEL_FITTERS = {
    "ols": lambda X, y, cfg: fit_ols_sklearn(X, y),
    "ridge": lambda X, y, cfg: fit_ridge(X, y, alpha=cfg["regression"]["ridge_alpha"]),
    "lasso": lambda X, y, cfg: fit_lasso(X, y, alpha=cfg["regression"]["lasso_alpha"]),
    "elastic_net": lambda X, y, cfg: fit_elastic_net(
        X,
        y,
        alpha=cfg["regression"]["elastic_net_alpha"],
        l1_ratio=cfg["regression"]["elastic_net_l1_ratio"],
    ),
}


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict") and type(model).__module__.startswith("statsmodels"):
        return predict_ols(model, X)
    return model.predict(X)


def train_model(
    model_name: str,
    split: SplitData,
    config: dict[str, Any],
):
    if model_name not in MODEL_FITTERS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_FITTERS)}")
    return MODEL_FITTERS[model_name](split.X_train, split.y_train, config)
