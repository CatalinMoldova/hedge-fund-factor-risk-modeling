# Data

## Included

- `sample/sample_fama_factors.csv` — synthetic monthly Fama-French-style factor returns for running the pipeline end-to-end.
- `sample/sample_fund_returns.csv` — synthetic monthly hedge fund returns aligned to the sample factor dates.

## Not included

- Original `FundReturns (1).xlsx` (course/private data)
- Full proprietary hedge fund dataset

## External data

Download Fama-French 5-factor monthly data automatically:

```bash
chmod +x scripts/download_fama_french.sh
./scripts/download_fama_french.sh
```

Or manually from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

Place files in `data/raw/` (gitignored):

- `F-F_Research_Data_5_Factors_2x3.csv` (via script above)
- `fund_returns.csv` or `FundReturns.xlsx` (your own data)

Then use `configs/real_fama.yaml` or update `configs/default.yaml` paths.
