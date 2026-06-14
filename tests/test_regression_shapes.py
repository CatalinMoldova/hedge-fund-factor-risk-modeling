
def test_regression_shapes(merged, config):
    from hedge_fund_factor_modeling.regression import predict_model, time_series_split, train_model

    split = time_series_split(
        merged,
        factor_cols=config["factors"]["factor_cols"],
        target_col=config["factors"]["target_col"],
        train_end=config["split"]["train_end"],
        test_start=config["split"]["test_start"],
    )
    model = train_model("ridge", split, config)
    preds = predict_model(model, split.X_test)
    assert len(preds) == len(split.y_test)
    assert split.X_train.shape[1] == len(config["factors"]["factor_cols"])
