import pandas as pd


def test_monthly_alignment(merged):
    assert isinstance(merged.index, pd.DatetimeIndex)
    assert merged.index.is_monotonic_increasing
    assert len(merged) > 24
    required = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Fund"]
    assert all(col in merged.columns for col in required)
