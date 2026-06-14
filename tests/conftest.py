import pandas as pd
import pytest

from hedge_fund_factor_modeling.data import load_config
from hedge_fund_factor_modeling.factors import merge_factors_and_fund


@pytest.fixture
def config():
    return load_config("configs/default.yaml")


@pytest.fixture
def merged(config):
    fama = pd.read_csv(config["data"]["fama_path"], parse_dates=["date"]).set_index("date")
    funds = pd.read_csv(config["data"]["fund_returns_path"], parse_dates=["date"]).set_index("date")
    return merge_factors_and_fund(
        fama,
        funds,
        factor_cols=config["factors"]["factor_cols"],
        target_col=config["factors"]["target_col"],
    )
