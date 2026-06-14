from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    ic = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    if ic is None or np.isnan(ic):
        ic = 0.0

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "residual_volatility": float(np.std(residuals, ddof=1)),
        "information_coefficient": float(ic),
    }


def evaluate_split(
    model,
    split,
    model_name: str,
    predict_fn,
) -> dict[str, Any]:
    y_train_pred = predict_fn(model, split.X_train)
    y_test_pred = predict_fn(model, split.X_test)

    train_metrics = compute_metrics(split.y_train, y_train_pred)
    test_metrics = compute_metrics(split.y_test, y_test_pred)

    return {
        "model": model_name,
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "residual_volatility": test_metrics["residual_volatility"],
        "information_coefficient": test_metrics["information_coefficient"],
    }


def evaluate_predictions(
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    model_name: str,
    notes: str = "",
) -> dict[str, Any]:
    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    return {
        "model": model_name,
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "residual_volatility": test_metrics["residual_volatility"],
        "information_coefficient": test_metrics["information_coefficient"],
        "notes": notes,
    }


NOTES = {
    "ols": "Baseline factor model",
    "ridge": "Regularized factor model",
    "lasso": "Sparse factor selection",
    "elastic_net": "Mixed L1/L2 regularization",
    "rolling_ridge": "Time-varying factor exposure",
    "expanding_ridge": "Expanding-window factor exposure",
    "ridge_cv": "Ridge with time-series cross-validation",
}
