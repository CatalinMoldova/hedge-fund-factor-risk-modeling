from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge

from hedge_fund_factor_modeling.regression import predict_model


def plot_factor_exposure_heatmap(
    exposures: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 4))
    sns.heatmap(exposures.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Factor Exposures by Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rolling_betas(rolling_betas: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for col in rolling_betas.columns:
        if col != "intercept":
            plt.plot(rolling_betas.index, rolling_betas[col], label=col)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title("Rolling Factor Exposures (Ridge)")
    plt.xlabel("Date")
    plt.ylabel("Beta")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Predicted vs Actual Returns",
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_regularization_path(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: Path,
) -> None:
    alphas = np.logspace(-2, 3, 30)
    coefs = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        coefs.append(model.coef_)
    coefs = np.array(coefs)

    plt.figure(figsize=(8, 5))
    for i, col in enumerate(X_train.columns):
        plt.plot(alphas, coefs[:, i], label=col)
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Coefficient")
    plt.title("Ridge Regularization Path")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(residuals.index, residuals.values)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_title("Residuals Over Time")
    axes[0].set_xlabel("Date")

    axes[1].hist(residuals, bins=15, edgecolor="black", alpha=0.7)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_plots(
    merged: pd.DataFrame,
    split,
    model,
    rolling_betas: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    assets_dir: Path,
) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    y_pred = predict_model(model, split.X_test)

    plot_factor_exposure_heatmap(factor_exposures, assets_dir / "factor_exposure_heatmap.png")
    plot_rolling_betas(rolling_betas, assets_dir / "rolling_beta_chart.png")
    plot_predicted_vs_actual(split.y_test, y_pred, assets_dir / "predicted_vs_actual_returns.png")
    plot_regularization_path(split.X_train, split.y_train, assets_dir / "regularization_path.png")
    plot_residuals(split.y_test, y_pred, assets_dir / "residuals_plot.png")
