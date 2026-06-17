# Hedge Fund Factor Risk Modeling

> A quantitative finance project that models a **simulated** monthly fund-return series against the **real** Fama-French five factors, using regularized regression, rolling factor exposures, and out-of-sample validation.

## Overview

This project demonstrates the research workflow behind factor risk modeling: estimating a fund's exposure to systematic risk factors (Mkt-RF, SMB, HML, RMW, CMA), comparing regularized regression models, and evaluating out-of-sample stability with time-based validation.

**About the data:** the fund return series used here (`FundReturns.xlsx`) is a **simulated** monthly series spanning the Fama-French history; it is **not** a real fund. The five factors are real (Kenneth French Data Library). The purpose is to demonstrate the modeling and validation *process* on a shareable dataset, not to analyze any real manager's performance. This is **not** a trading strategy.

## Research Question

Given a fund return series and the Fama-French five factors, how much of the return variation is explained by systematic factor exposure, and does regularization (Ridge / Lasso / Elastic Net) improve out-of-sample stability relative to OLS?

## Why This Matters

Fund returns often reflect exposure to common risk factors rather than pure manager skill. Estimating those exposures, validating them out of sample, and inspecting the residuals is the standard first step in separating factor-driven returns from any residual ("alpha") component. Here that workflow is applied to a simulated series so the full pipeline can be run and shared publicly.

## Quick Start

### Option A: One-command environment setup (recommended)

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh          # uses conda if available, otherwise venv
# or explicitly:
./scripts/setup_env.sh conda    # creates .conda-env/
./scripts/setup_env.sh venv     # creates .venv/
```

Activate the environment:

```bash
# conda
conda activate .conda-env

# venv (macOS/Linux)
source .venv/bin/activate
```

The setup script installs all dependencies and the package in editable mode. Environments live in `.conda-env/` or `.venv/` (gitignored, local to your machine).

### Option B: Manual install

```bash
pip install -r requirements.txt
pip install -e .
```

### Run the pipeline

```bash
# Demo (bundled synthetic sample data)
python -m hedge_fund_factor_modeling.cli train --model all --config configs/default.yaml

# Analysis on the simulated fund series (real Fama-French factors + FundReturns.xlsx)
./scripts/download_fama_french.sh
python -m hedge_fund_factor_modeling.cli train --model all --config configs/real_fund.yaml
python -m hedge_fund_factor_modeling.cli evaluate --results results/model_performance.csv
python -m hedge_fund_factor_modeling.cli plot --output assets/
pytest
```

<!-- TODO: confirm the config filename. The data/README previously referred to configs/real_fama.yaml while this file says configs/real_fund.yaml. Make both READMEs match the file that actually exists in configs/. -->

## Methods

- OLS, Ridge, Lasso, Elastic Net, Ridge CV (TimeSeriesSplit), rolling Ridge, expanding Ridge
- Time-based train/test split (no random shuffling)
- Rolling-window and expanding-window Ridge betas
- Out-of-sample evaluation with residual diagnostics
- Factor exposure heatmaps and regularization path analysis

## Results

**Data:** 744 monthly observations (Jul 1963 – Jun 2025): real Kenneth French 5-factor returns + the simulated `FundReturns.xlsx` series. Train: 1963–2018 · Test: 2019–2025.

<!-- TODO: confirm every number below was generated from the FundReturns.xlsx run (configs/real_fund.yaml), NOT from the synthetic CI demo (configs/default.yaml). -->

| Model           | Train R² | Test R² | RMSE | MAE  | Residual Volatility | Notes                                   |
| --------------- | -------- | ------- | ---- | ---- | ------------------- | --------------------------------------- |
| OLS             | 0.76     | 0.82    | 0.19 | 0.16 | 0.18                | Baseline factor model                   |
| Ridge           | 0.76     | 0.82    | 0.19 | 0.16 | 0.18                | Regularized factor model                |
| Lasso           | 0.73     | 0.72    | 0.24 | 0.20 | 0.24                | Sparse factor selection                 |
| Elastic Net     | 0.62     | 0.57    | 0.29 | 0.24 | 0.29                | Mixed L1/L2 regularization              |
| Ridge CV        | 0.76     | 0.82    | 0.19 | 0.16 | 0.19                | Ridge with time-series cross-validation |
| Rolling Ridge   | 0.71     | 0.83    | 0.19 | 0.15 | 0.19                | Time-varying factor exposure            |
| Expanding Ridge | 0.75     | 0.83    | 0.18 | 0.16 | 0.18                | Expanding-window factor exposure        |

**Key takeaway:** the denser models (OLS, Ridge, and the rolling/expanding Ridge variants) hold their out-of-sample fit, while the sparser Lasso and Elastic Net degrade — evidence that the smaller factor loadings carry signal rather than noise worth shrinking away. Expanding Ridge achieves the best out-of-sample fit (Test R² = 0.83).

**Factor exposures (OLS):** `[ regenerate and paste the coefficients from the FundReturns.xlsx OLS run here ]`

<!-- TODO: the previously published exposures (Mkt-RF 0.07, SMB/HML/RMW/CMA 0.01) cannot coexist with a Test R² of ~0.82 given the series' volatility — coefficients that small would explain almost no variance. Re-pull the coefficients and R² from the SAME regression and make sure they reconcile before publishing. -->

## Factor Exposure Analysis

![Factor exposure heatmap](assets/factor_exposure_heatmap.png)

Estimated betas show how each model loads on the Fama-French five factors. Ridge and OLS produce similar exposures on this sample; Lasso can zero out weaker factors.

## Rolling Betas

![Rolling factor exposures](assets/rolling_beta_chart.png)

Rolling Ridge regression shows how factor exposures evolve over time — a key step before attributing performance or building signals.

## Model Fit and Diagnostics

| Chart | Description |
| --- | --- |
| ![Predicted vs actual](assets/predicted_vs_actual_returns.png) | Out-of-sample predicted vs actual fund returns |
| ![Regularization path](assets/regularization_path.png) | How Ridge coefficients shrink as λ increases |
| ![Residuals](assets/residuals_plot.png) | Unexplained returns after factor attribution |

## Relevance to Quantitative Research

This project is not a trading strategy. It focuses on the research habits needed before turning any signal into a tradable idea:

- No-lookahead checks (time-based splits)
- Rolling-window validation
- Out-of-sample evaluation
- Factor exposure stability
- Residual diagnostics

## Project Structure

```
hedge-fund-factor-risk-modeling/
├── src/hedge_fund_factor_modeling/   # Core package
├── notebooks/                         # Theory + demo notebook
├── configs/default.yaml               # Experiment configuration
├── data/sample/                       # Synthetic data for end-to-end runs
├── results/                           # Model outputs (CSV)
├── assets/                            # Diagnostic plots
└── tests/                             # pytest suite + GitHub Actions CI
```

## Engineering Practices

- Modular Python package with config-driven experiments
- CLI for prepare → train → evaluate → plot
- pytest suite with no-lookahead and alignment tests
- GitHub Actions CI on every push

## Data

See [data/README.md](data/README.md). The repo ships synthetic sample data so the full pipeline runs without any external files. The `FundReturns.xlsx` series used for the main results is a simulated course dataset and is not redistributed here. Download the real Fama-French factors from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## Limitations

Single-fund series, and a **simulated** one — results illustrate the methodology rather than describe a real fund. Factor exposures may shift across market regimes. This is a factor-risk research project, not a trading strategy.

## Future Work

- Add real, multi-fund panels
- Regime detection and nonlinear models
- Bayesian shrinkage for exposure estimation
- Transaction-cost-aware strategy extension (separate from this research repo)

## License

MIT — see [LICENSE](LICENSE).
