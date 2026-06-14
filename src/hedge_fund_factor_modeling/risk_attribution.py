from __future__ import annotations

import pandas as pd


def attribute_returns(
    df: pd.DataFrame,
    betas: pd.DataFrame,
    factor_cols: list[str],
    target_col: str = "Fund",
) -> pd.DataFrame:
    """
    Decompose fund returns into factor-driven return and residual alpha.

    Uses the most recent available beta row at or before each date.
    """
    aligned_betas = betas.reindex(df.index, method="ffill")
    factor_contrib = (df[factor_cols] * aligned_betas[factor_cols]).sum(axis=1)
    intercept = aligned_betas["intercept"].fillna(0)
    predicted = intercept + factor_contrib
    residual = df[target_col] - predicted

    return pd.DataFrame(
        {
            "actual": df[target_col],
            "predicted": predicted,
            "factor_return": factor_contrib,
            "intercept": intercept,
            "residual_alpha": residual,
        },
        index=df.index,
    )
