
def test_train_dates_precede_test_dates(merged, config):
    from hedge_fund_factor_modeling.regression import time_series_split

    split = time_series_split(
        merged,
        factor_cols=config["factors"]["factor_cols"],
        target_col=config["factors"]["target_col"],
        train_end=config["split"]["train_end"],
        test_start=config["split"]["test_start"],
    )
    assert split.train_index.max() < split.test_index.min()


def test_no_lookahead_in_features(merged, config):
    from hedge_fund_factor_modeling.regression import time_series_split

    split = time_series_split(
        merged,
        factor_cols=config["factors"]["factor_cols"],
        target_col=config["factors"]["target_col"],
        train_end=config["split"]["train_end"],
        test_start=config["split"]["test_start"],
    )
    assert split.X_train.index.max() < split.X_test.index.min()
    assert split.y_train.index.max() < split.y_test.index.min()
