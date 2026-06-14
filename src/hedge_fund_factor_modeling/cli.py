from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hedge_fund_factor_modeling.data import load_config, load_fama_french, load_fund_returns
from hedge_fund_factor_modeling.evaluation import NOTES, evaluate_predictions, evaluate_split
from hedge_fund_factor_modeling.factors import (
    compute_summary_stats,
    merge_factors_and_fund,
    split_fama_monthly_yearly,
)
from hedge_fund_factor_modeling.plots import generate_all_plots
from hedge_fund_factor_modeling.regression import (
    MODEL_FITTERS,
    get_factor_exposures,
    grid_search_ridge,
    predict_model,
    time_series_split,
    train_model,
)
from hedge_fund_factor_modeling.risk_attribution import attribute_returns
from hedge_fund_factor_modeling.rolling_exposures import (
    evaluate_walk_forward_ridge,
    expanding_factor_betas,
    rolling_factor_betas,
)


def prepare_data(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    fama = load_fama_french(config["data"]["fama_path"])
    funds = load_fund_returns(
        config["data"]["fund_returns_path"],
        date_col=config["data"].get("date_col", "date"),
    )

    if "sample" not in config["data"]["fama_path"]:
        fama_monthly, _ = split_fama_monthly_yearly(fama)
        merged = merge_factors_and_fund(
            fama_monthly,
            funds,
            factor_cols=config["factors"]["factor_cols"],
            target_col=config["factors"]["target_col"],
        )
    else:
        merged = merge_factors_and_fund(
            fama,
            funds,
            factor_cols=config["factors"]["factor_cols"],
            target_col=config["factors"]["target_col"],
        )

    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(results_dir / "merged_data.csv")
    compute_summary_stats(merged).to_csv(results_dir / "summary_statistics.csv")

    return merged


def run_training(config_path: str, model_name: str = "ridge") -> None:
    config = load_config(config_path)
    merged = prepare_data(config_path)

    split = time_series_split(
        merged,
        factor_cols=config["factors"]["factor_cols"],
        target_col=config["factors"]["target_col"],
        train_end=config["split"]["train_end"],
        test_start=config["split"]["test_start"],
    )

    results_dir = Path(config["output"]["results_dir"])
    performance_rows = []

    models_to_run = list(MODEL_FITTERS.keys()) if model_name == "all" else [model_name]
    exposures = {}

    for name in models_to_run:
        model = train_model(name, split, config)
        row = evaluate_split(model, split, name, predict_model)
        row["notes"] = NOTES.get(name, "")
        performance_rows.append(row)
        exposures[name] = get_factor_exposures(model, config["factors"]["factor_cols"])

    if model_name == "all":
        ridge_cv = grid_search_ridge(
            split.X_train,
            split.y_train,
            n_splits=config["regression"].get("cv_splits", 5),
        )
        row = evaluate_split(ridge_cv, split, "ridge_cv", predict_model)
        row["notes"] = NOTES["ridge_cv"]
        performance_rows.append(row)
        exposures["ridge_cv"] = get_factor_exposures(ridge_cv, config["factors"]["factor_cols"])

        for wf_name, wf_mode in [("rolling_ridge", "rolling"), ("expanding_ridge", "expanding")]:
            y_train, y_train_pred, y_test, y_test_pred = evaluate_walk_forward_ridge(
                merged,
                split,
                factor_cols=config["factors"]["factor_cols"],
                target_col=config["factors"]["target_col"],
                window=config["rolling"]["window_months"],
                min_periods=config["rolling"]["min_periods"],
                alpha=config["regression"]["ridge_alpha"],
                mode=wf_mode,
            )
            row = evaluate_predictions(
                y_train,
                y_train_pred,
                y_test,
                y_test_pred,
                wf_name,
                notes=NOTES[wf_name],
            )
            performance_rows.append(row)

    rolling = rolling_factor_betas(
        merged,
        factor_cols=config["factors"]["factor_cols"],
        target_col=config["factors"]["target_col"],
        window=config["rolling"]["window_months"],
        min_periods=config["rolling"]["min_periods"],
        alpha=config["regression"]["ridge_alpha"],
        mode="rolling",
    )
    expanding = expanding_factor_betas(
        merged,
        factor_cols=config["factors"]["factor_cols"],
        target_col=config["factors"]["target_col"],
        min_periods=config["rolling"]["min_periods"],
        alpha=config["regression"]["ridge_alpha"],
    )

    primary_model = train_model(model_name if model_name != "all" else "ridge", split, config)
    attribution = attribute_returns(
        merged.loc[split.test_index],
        rolling,
        factor_cols=config["factors"]["factor_cols"],
        target_col=config["factors"]["target_col"],
    )

    pd.DataFrame(performance_rows).to_csv(results_dir / "model_performance.csv", index=False)
    pd.DataFrame(exposures).to_csv(results_dir / "factor_exposures.csv")
    rolling.to_csv(results_dir / "rolling_betas.csv")
    expanding.to_csv(results_dir / "expanding_betas.csv")
    attribution.to_csv(results_dir / "residual_diagnostics.csv")

    factor_exposure_df = pd.DataFrame(exposures)
    generate_all_plots(
        merged,
        split,
        primary_model,
        rolling,
        factor_exposure_df,
        Path(config["output"]["assets_dir"]),
    )


def run_evaluate(results_path: str) -> None:
    df = pd.read_csv(results_path)
    print(df.to_string(index=False))


def run_plot(config_path: str, output_dir: str) -> None:
    run_training(config_path, model_name="ridge")
    print(f"Plots saved to {output_dir}/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hedge fund factor risk modeling CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-data", help="Load and merge factor and fund data")
    prep.add_argument("--config", default="configs/default.yaml")

    train = sub.add_parser("train", help="Train regression models")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--model", default="all", choices=["all", "ols", "ridge", "lasso", "elastic_net"])

    evaluate = sub.add_parser("evaluate", help="Print model performance table")
    evaluate.add_argument("--results", default="results/model_performance.csv")

    plot = sub.add_parser("plot", help="Generate diagnostic plots")
    plot.add_argument("--config", default="configs/default.yaml")
    plot.add_argument("--output", default="assets")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-data":
        merged = prepare_data(args.config)
        print(f"Prepared {len(merged)} monthly observations.")
    elif args.command == "train":
        run_training(args.config, model_name=args.model)
        print("Training complete. Results saved to results/")
    elif args.command == "evaluate":
        run_evaluate(args.results)
    elif args.command == "plot":
        run_plot(args.config, args.output)


if __name__ == "__main__":
    main()
