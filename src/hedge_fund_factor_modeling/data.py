from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_fama_french(path: str | Path, skiprows: int = 0) -> pd.DataFrame:
    """Load Fama-French factor data from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fama-French file not found: {path}")

    # Kenneth French raw files have a 3-line header; sample files use a normal header.
    if "sample" in str(path):
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    df = pd.read_csv(path, skiprows=skiprows or 3)
    return df


def load_fund_returns(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Load hedge fund returns from CSV or Excel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fund returns file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if date_col in df.columns:
        if df[date_col].dtype == object and df[date_col].astype(str).str.len().eq(6).all():
            df[date_col] = pd.to_datetime(df[date_col], format="%Y%m")
        else:
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    elif "Unnamed: 0" in df.columns:
        col = df["Unnamed: 0"]
        if col.astype(str).str.len().eq(6).all():
            df["Unnamed: 0"] = pd.to_datetime(col, format="%Y%m")
        else:
            df["Unnamed: 0"] = pd.to_datetime(col)
        df = df.set_index("Unnamed: 0")

    return df
