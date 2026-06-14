#!/usr/bin/env bash
# Download Kenneth French Fama-French 5-factor monthly data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw"
ZIP_URL="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
ZIP_PATH="$RAW_DIR/fama_french_5f.zip"
CSV_PATH="$RAW_DIR/F-F_Research_Data_5_Factors_2x3.csv"

mkdir -p "$RAW_DIR"

if [[ -f "$CSV_PATH" ]]; then
  echo "Fama-French file already exists: $CSV_PATH"
  exit 0
fi

echo "Downloading Fama-French 5-factor data..."
curl -fsSL "$ZIP_URL" -o "$ZIP_PATH"
unzip -o -j "$ZIP_PATH" "F-F_Research_Data_5_Factors_2x3.csv" -d "$RAW_DIR"
rm -f "$ZIP_PATH"
echo "Saved to $CSV_PATH"
