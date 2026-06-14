#!/usr/bin/env bash
# Create a local project environment (conda or venv) and install dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV_DIR="$ROOT/.conda-env"
VENV_DIR="$ROOT/.venv"

usage() {
  cat <<EOF
Usage: ./scripts/setup_env.sh [conda|venv]

Creates a project-local environment and installs this package in editable mode.

  conda   Create .conda-env/ using environment.yml (default if conda is available)
  venv    Create .venv/ using python -m venv

After setup, activate with:

  conda:  conda activate "$CONDA_ENV_DIR"
  venv:   source .venv/bin/activate   (macOS/Linux)
          .venv\\Scripts\\activate     (Windows)

Then run:

  python -m hedge_fund_factor_modeling.cli train --model all
  pytest
EOF
}

install_editable() {
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install -e .
  echo ""
  echo "Environment ready."
  python -m hedge_fund_factor_modeling.cli prepare-data --config configs/default.yaml
}

setup_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda/Anaconda or run: ./scripts/setup_env.sh venv"
    exit 1
  fi

  if [[ -d "$CONDA_ENV_DIR" ]]; then
    echo "Conda env already exists at .conda-env/"
  else
    echo "Creating conda environment in .conda-env/ ..."
    conda env create -f environment.yml -p "$CONDA_ENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_DIR"
  install_editable
  echo ""
  echo "Activate later with: conda activate $CONDA_ENV_DIR"
}

setup_venv() {
  if [[ -d "$VENV_DIR" ]]; then
    echo "venv already exists at .venv/"
  else
    echo "Creating venv in .venv/ ..."
    python3 -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  install_editable
  echo ""
  echo "Activate later with: source .venv/bin/activate"
}

MODE="${1:-auto}"

case "$MODE" in
  conda)
    setup_conda
    ;;
  venv)
    setup_venv
    ;;
  auto)
    if command -v conda >/dev/null 2>&1; then
      setup_conda
    else
      setup_venv
    fi
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown option: $MODE"
    usage
    exit 1
    ;;
esac
