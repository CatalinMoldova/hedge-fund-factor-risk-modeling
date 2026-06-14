
def test_factor_merge_columns(merged, config):
    for col in config["factors"]["factor_cols"]:
        assert col in merged.columns
    assert config["factors"]["target_col"] in merged.columns
    assert merged[config["factors"]["factor_cols"]].notna().all().all()
