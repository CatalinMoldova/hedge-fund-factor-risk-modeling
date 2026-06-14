from __future__ import annotations

import pandas as pd


def split_fama_monthly_yearly(fama: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw Fama-French CSV into monthly (YYYYMM) and yearly (YYYY) rows."""
    date_col = fama.columns[0]
    dates = fama[date_col].astype(str).str.strip()
    fama_monthly = fama[dates.str.match(r"^\d{6}$")].copy()
    fama_early = fama[dates.str.match(r"^\d{4}$")].copy()
    return fama_monthly, fama_early


def parse_monthly_index(df: pd.DataFrame, date_col: str = "Unnamed: 0") -> pd.DataFrame:
    """Convert YYYYMM date column to DatetimeIndex."""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = out[date_col].astype(str).str.strip()
        out = out[out[date_col].str.match(r"^\d{6}$")]
        out[date_col] = pd.to_datetime(out[date_col], format="%Y%m")
        out = out.set_index(date_col)
    return out


def merge_factors_and_fund(
    fama_monthly: pd.DataFrame,
    fund_returns: pd.DataFrame,
    factor_cols: list[str] | None = None,
    target_col: str = "Fund",
) -> pd.DataFrame:
    """Align and merge factor returns with fund returns on monthly dates."""
    fama = fama_monthly.copy()
    funds = fund_returns.copy()

    if not isinstance(fama.index, pd.DatetimeIndex):
        fama = parse_monthly_index(fama.reset_index() if fama.index.name else fama)
    if not isinstance(funds.index, pd.DatetimeIndex):
        funds = parse_monthly_index(funds.reset_index() if funds.index.name else funds)

    combined = pd.concat([fama, funds], axis=1, join="inner")
    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.dropna()
    combined = combined.sort_index()

    if factor_cols is None:
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    required = factor_cols + [target_col]
    missing = [c for c in required if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required columns after merge: {missing}")

    return combined


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for all numeric columns."""
    return pd.DataFrame(
        {
            "Mean": df.mean(),
            "Std Dev": df.std(),
            "Skewness": df.skew(),
            "Kurtosis": df.kurtosis(),
            "Max": df.max(),
            "Min": df.min(),
        }
    )
