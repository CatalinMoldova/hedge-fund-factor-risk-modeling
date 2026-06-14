from __future__ import annotations

import pandas as pd

from hedge_fund_factor_modeling.regression import SplitData, fit_ridge


def rolling_factor_betas(
    df: pd.DataFrame,
    factor_cols: list[str],
    target_col: str,
    window: int = 36,
    min_periods: int = 24,
    alpha: float = 1.0,
    mode: str = "rolling",
) -> pd.DataFrame:
    """
    Estimate time-varying factor exposures using rolling or expanding Ridge regression.

    Returns a DataFrame indexed by date with one column per factor plus intercept.
    """
    records: list[dict] = []
    dates: list[pd.Timestamp] = []

    for end_idx in range(min_periods, len(df) + 1):
        if mode == "expanding":
            window_df = df.iloc[:end_idx]
        else:
            start_idx = max(0, end_idx - window)
            window_df = df.iloc[start_idx:end_idx]

        if len(window_df) < min_periods:
            continue

        X = window_df[factor_cols]
        y = window_df[target_col]
        model = fit_ridge(X, y, alpha=alpha)

        row = {col: coef for col, coef in zip(factor_cols, model.coef_)}
        row["intercept"] = model.intercept_
        records.append(row)
        dates.append(window_df.index[-1])

    return pd.DataFrame(records, index=pd.DatetimeIndex(dates))


def expanding_factor_betas(
    df: pd.DataFrame,
    factor_cols: list[str],
    target_col: str,
    min_periods: int = 24,
    alpha: float = 1.0,
) -> pd.DataFrame:
    return rolling_factor_betas(
        df,
        factor_cols,
        target_col,
        window=len(df),
        min_periods=min_periods,
        alpha=alpha,
        mode="expanding",
    )


def walk_forward_predict(
    df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    factor_cols: list[str],
    target_col: str,
    window: int = 36,
    min_periods: int = 24,
    alpha: float = 1.0,
    mode: str = "rolling",
) -> tuple[pd.Series, pd.Series]:
    """
    Predict returns at each date using betas estimated only on prior observations.
    Skips dates without sufficient prior history.
    """
    actuals = []
    predictions = []
    valid_dates = []

    for date in dates:
        history = df.loc[:date]
        if len(history) < min_periods + 1:
            continue

        train_hist = history.iloc[:-1]
        features = history.iloc[[-1]][factor_cols]

        if mode == "expanding":
            window_df = train_hist
        else:
            window_df = train_hist.iloc[-window:]

        if len(window_df) < min_periods:
            continue

        model = fit_ridge(window_df[factor_cols], window_df[target_col], alpha=alpha)
        valid_dates.append(date)
        actuals.append(float(history.iloc[-1][target_col]))
        predictions.append(float(model.predict(features)[0]))

    if not valid_dates:
        raise ValueError("No dates had sufficient history for walk-forward prediction.")

    index = pd.DatetimeIndex(valid_dates)
    return pd.Series(actuals, index=index), pd.Series(predictions, index=index)


def evaluate_walk_forward_ridge(
    df: pd.DataFrame,
    split: SplitData,
    factor_cols: list[str],
    target_col: str,
    window: int = 36,
    min_periods: int = 24,
    alpha: float = 1.0,
    mode: str = "rolling",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Walk-forward actuals and predictions on train and test periods."""
    y_train, y_train_pred = walk_forward_predict(
        df,
        split.train_index,
        factor_cols,
        target_col,
        window=window,
        min_periods=min_periods,
        alpha=alpha,
        mode=mode,
    )
    y_test, y_test_pred = walk_forward_predict(
        df,
        split.test_index,
        factor_cols,
        target_col,
        window=window,
        min_periods=min_periods,
        alpha=alpha,
        mode=mode,
    )
    return y_train, y_train_pred, y_test, y_test_pred
