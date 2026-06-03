# mlops-deploy

## Run Wikipedia SARIMAX on Cloud Run MLflow

This repository includes a script that applies the preprocessing described in the article and runs SARIMAX tracked in MLflow (default target: English).

Implemented preprocessing:
- Drop rows with all daily values null.
- Drop rows with fewer than 300 observed days.
- Fill remaining nulls with 0.
- Parse Page metadata into article title, language, access type, and access origin.
- Mark commons.wikimedia and mediawiki rows as invalid language and exclude them.
- Aggregate to daily mean pageviews per language.

### 1) Create and activate a Python environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r examples/requirements-forecast.txt
```

### 3) Set tracking URI to your Cloud Run service URL

Replace the value with your own service URL.

```powershell
$env:MLFLOW_TRACKING_URI = "https://mlflow-xxxxx-uc.a.run.app"
```

### 4) Run a quick local dry-run (no MLflow tracking)

```powershell
python examples/train_sarimax_mlflow.py --dry-run --max-rows 4000 --target-language zh --steps 30
```

### 5) Run a single English-only SARIMAX run with MLflow tracking

```powershell
python examples/train_sarimax_mlflow.py `
	--tracking-uri $env:MLFLOW_TRACKING_URI `
	--experiment-name "wikipedia-language-forecasting" `
	--train-path "dataset/train_1.csv" `
	--exog-path "dataset/Exog_Campaign_eng" `
	--target-language en `
	--min-observed-days 300 `
	--steps 60
```

### 6) Optional: subset test

```powershell
python examples/train_sarimax_mlflow.py `
	--dry-run `
	--max-rows 4000 `
	--target-language zh `
	--steps 30
```

### Outputs

- One run for the selected language (default: en).
- Input dataset tracked with `mlflow.log_input(...)`.
- Artifacts: ts_dataset_<language>.csv, forecast_vs_actual_<language>.csv, preprocessing_summary.json, feature_enriched_pages_sample.csv.
- Metrics: RMSE, MAPE, wMAPE.

### Notes

- The script uses mlflow.statsmodels.log_model for SARIMAX.
- The intended production run is English-only (`--target-language en`) using Exog_Campaign_eng.
- If Cloud Run is public, no auth token is needed.
- If you later restrict Cloud Run IAM, you will need authenticated requests from your client.