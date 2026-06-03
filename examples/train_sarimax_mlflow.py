import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass

import mlflow
import mlflow.statsmodels
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class PreprocessStats:
    rows_total: int = 0
    rows_all_null_dropped: int = 0
    rows_low_observed_dropped: int = 0
    rows_invalid_language_dropped: int = 0
    rows_kept: int = 0


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = float(np.abs(y_true).sum())
    if denom == 0:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / denom * 100)


def parse_order(value: str, expected_len: int, arg_name: str) -> tuple[int, ...]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"{arg_name} must contain {expected_len} integers separated by commas")
    return tuple(int(x) for x in parts)


def parse_page_metadata(page_series: pd.Series) -> pd.DataFrame:
    # Split from right so underscores in article titles are preserved.
    parts = page_series.astype(str).str.rsplit("_", n=3, expand=True)
    while parts.shape[1] < 4:
        parts[parts.shape[1]] = ""
    parts.columns = ["article_title", "domain", "access_type", "access_origin"]

    domain = parts["domain"].str.lower()
    language = domain.str.split(".").str[0]

    invalid_domain = domain.str.contains("commons.wikimedia.org|mediawiki", regex=True, na=False)
    language = language.mask(invalid_domain, np.nan)

    parts["language"] = language
    parts["invalid_domain"] = invalid_domain
    return parts


def build_feature_enriched_dataframe(
    train_path: str,
    min_observed_days: int,
    chunksize: int,
    max_rows: int | None,
) -> pd.DataFrame:
    header = pd.read_csv(train_path, nrows=0)
    date_cols = [c for c in header.columns if c != "Page"]
    if not date_cols:
        raise ValueError("No date columns found in train dataset")

    enriched_chunks: list[pd.DataFrame] = []
    rows_processed = 0

    for chunk in pd.read_csv(train_path, chunksize=chunksize, low_memory=False):
        if max_rows is not None and rows_processed >= max_rows:
            break
        if max_rows is not None and rows_processed + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - rows_processed].copy()

        rows_processed += len(chunk)
        metadata = parse_page_metadata(chunk["Page"])
        values = chunk[date_cols].apply(pd.to_numeric, errors="coerce")

        observed_days = values.notna().sum(axis=1)
        all_null = values.isna().all(axis=1)

        enriched = pd.concat(
            [
                chunk[["Page"]].reset_index(drop=True),
                metadata[["article_title", "language", "access_type", "access_origin"]].reset_index(drop=True),
                pd.DataFrame(
                    {
                        "observed_days": observed_days.values,
                        "all_null": all_null.values,
                        "passes_min_observed": (observed_days >= min_observed_days).values,
                    }
                ),
            ],
            axis=1,
        )
        enriched_chunks.append(enriched)

    if not enriched_chunks:
        return pd.DataFrame(
            columns=[
                "Page",
                "article_title",
                "language",
                "access_type",
                "access_origin",
                "observed_days",
                "all_null",
                "passes_min_observed",
            ]
        )

    return pd.concat(enriched_chunks, ignore_index=True)


def read_exog_series(exog_path: str, expected_len: int) -> pd.Series:
    exog_df = pd.read_csv(exog_path)
    if "Exog" not in exog_df.columns:
        raise ValueError("Exogenous file must contain a column named 'Exog'")
    exog = pd.to_numeric(exog_df["Exog"], errors="coerce").fillna(0.0)
    if len(exog) != expected_len:
        raise ValueError(f"Exogenous length mismatch: expected {expected_len}, got {len(exog)}")
    return exog


def aggregate_language_daily_means(
    train_path: str,
    min_observed_days: int,
    chunksize: int,
    max_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessStats]:
    header = pd.read_csv(train_path, nrows=0)
    if "Page" not in header.columns:
        raise ValueError("train_1.csv must contain a 'Page' column")

    date_cols = [c for c in header.columns if c != "Page"]
    if len(date_cols) != 550:
        raise ValueError(f"Expected 550 daily columns, found {len(date_cols)}")

    dates = pd.to_datetime(date_cols, format="%Y-%m-%d", errors="raise")

    stats = PreprocessStats()
    lang_sum: dict[str, np.ndarray] = {}
    lang_count: dict[str, int] = {}
    rows_processed = 0

    for chunk in pd.read_csv(train_path, chunksize=chunksize, low_memory=False):
        if max_rows is not None and rows_processed >= max_rows:
            break
        if max_rows is not None and rows_processed + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - rows_processed].copy()

        rows_processed += len(chunk)
        stats.rows_total += len(chunk)

        metadata = parse_page_metadata(chunk["Page"])
        values = chunk[date_cols].apply(pd.to_numeric, errors="coerce")

        all_null_mask = values.isna().all(axis=1)
        stats.rows_all_null_dropped += int(all_null_mask.sum())

        observed_days = values.notna().sum(axis=1)
        low_observed_mask = observed_days < min_observed_days
        stats.rows_low_observed_dropped += int((~all_null_mask & low_observed_mask).sum())

        invalid_language_mask = metadata["language"].isna()
        stats.rows_invalid_language_dropped += int((~all_null_mask & ~low_observed_mask & invalid_language_mask).sum())

        keep_mask = ~all_null_mask & ~low_observed_mask & ~invalid_language_mask
        if keep_mask.sum() == 0:
            continue

        clean_values = values.loc[keep_mask].fillna(0.0)
        clean_languages = metadata.loc[keep_mask, "language"]

        grouped = clean_values.groupby(clean_languages).sum()
        counts = clean_languages.value_counts()

        for language, row in grouped.iterrows():
            row_values = row.to_numpy(dtype=float)
            if language in lang_sum:
                lang_sum[language] += row_values
                lang_count[language] += int(counts[language])
            else:
                lang_sum[language] = row_values
                lang_count[language] = int(counts[language])

        stats.rows_kept += int(keep_mask.sum())

    if not lang_sum:
        raise ValueError("No data left after preprocessing. Adjust filters or verify source data.")

    languages = sorted(lang_sum.keys())
    daily_by_language = pd.DataFrame(
        {language: lang_sum[language] / max(lang_count[language], 1) for language in languages},
        index=dates,
    )
    daily_by_language.index.name = "date"

    lang_meta = pd.DataFrame(
        {
            "language": languages,
            "pages_kept": [lang_count[language] for language in languages],
            "avg_daily_views": [float(daily_by_language[language].mean()) for language in languages],
        }
    ).sort_values("avg_daily_views", ascending=False)

    return daily_by_language, lang_meta, stats


def build_english_dataset(
    train_path: str,
    exog_path: str,
    target_language: str,
    min_observed_days: int,
    chunksize: int,
    max_rows: int | None,
) -> tuple[pd.DataFrame, PreprocessStats]:
    daily_by_language, _, stats = aggregate_language_daily_means(
        train_path=train_path,
        min_observed_days=min_observed_days,
        chunksize=chunksize,
        max_rows=max_rows,
    )

    if target_language not in daily_by_language.columns:
        available = sorted(daily_by_language.columns.tolist())
        preview = ", ".join(available[:10])
        if max_rows is not None:
            raise ValueError(
                f"Language '{target_language}' not found after preprocessing for current subset. "
                f"Increase --max-rows or run full dataset. Available sample languages: {preview}"
            )
        raise ValueError(f"Language '{target_language}' not found after preprocessing")

    exog = read_exog_series(exog_path, expected_len=len(daily_by_language))
    exog.index = daily_by_language.index

    english_ts = pd.DataFrame(
        {
            "date": daily_by_language.index,
            "y": daily_by_language[target_language].values,
            "exog": exog.values,
        }
    )
    return english_ts, stats


def train_eval_sarimax(
    series: pd.Series,
    exog_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    steps: int,
) -> tuple[pd.DataFrame, dict[str, float], object]:
    if len(series) <= steps + 20:
        raise ValueError("Series too short for selected horizon")

    train_y = series.iloc[:-steps]
    test_y = series.iloc[-steps:]

    train_exog = exog_series.iloc[:-steps].to_frame("exog")
    test_exog = exog_series.iloc[-steps:].to_frame("exog")

    model = SARIMAX(
        train_y,
        order=order,
        seasonal_order=seasonal_order,
        exog=train_exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    forecast = fitted.get_forecast(steps=len(test_y), exog=test_exog).predicted_mean

    pred_df = pd.DataFrame(
        {
            "date": test_y.index,
            "actual": test_y.values,
            "forecast": forecast.values,
        }
    )
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(test_y, forecast))),
        "mape": mape(test_y, forecast),
        "wmape": wmape(test_y, forecast),
    }
    return pred_df, metrics, fitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia language-level SARIMAX benchmark with MLflow")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"), help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="wikipedia-language-forecasting")
    parser.add_argument("--train-path", default="dataset/train_1.csv")
    parser.add_argument("--exog-path", default="dataset/Exog_Campaign_eng")
    parser.add_argument("--min-observed-days", type=int, default=300)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for fast local tests")
    parser.add_argument("--target-language", default="en", help="Language code to model (default: en)")
    parser.add_argument(
        "--feature-analysis-rows",
        type=int,
        default=10000,
        help="Rows used to build the feature-enriched analysis dataset artifact",
    )
    parser.add_argument("--order", default="4,1,3", help="SARIMAX p,d,q")
    parser.add_argument("--seasonal-order", default="3,0,2,7", help="SARIMAX P,D,Q,s")
    parser.add_argument("--dry-run", action="store_true", help="Skip MLflow tracking and run locally")
    args = parser.parse_args()

    if not args.dry_run and not args.tracking_uri:
        raise ValueError("Set --tracking-uri or MLFLOW_TRACKING_URI, or use --dry-run")

    order = parse_order(args.order, 3, "--order")
    seasonal_order = parse_order(args.seasonal_order, 4, "--seasonal-order")

    english_ts, stats = build_english_dataset(
        train_path=args.train_path,
        exog_path=args.exog_path,
        target_language=args.target_language,
        min_observed_days=args.min_observed_days,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )

    if not args.dry_run:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        feature_enriched_df = build_feature_enriched_dataframe(
            train_path=args.train_path,
            min_observed_days=args.min_observed_days,
            chunksize=args.chunksize,
            max_rows=args.feature_analysis_rows,
        )

        preprocess_summary = asdict(stats)
        preprocess_summary.update(
            {
                "language_selected": args.target_language,
                "dataset_rows": int(len(english_ts)),
                "steps": args.steps,
                "min_observed_days": args.min_observed_days,
                "feature_analysis_rows": args.feature_analysis_rows,
            }
        )

        preprocess_path = os.path.join(tmpdir, "preprocessing_summary.json")
        with open(preprocess_path, "w", encoding="utf-8") as f:
            json.dump(preprocess_summary, f, indent=2)

        english_dataset_path = os.path.join(tmpdir, f"ts_dataset_{args.target_language}.csv")
        english_ts.to_csv(english_dataset_path, index=False)

        feature_dataset_path = os.path.join(tmpdir, "feature_enriched_pages_sample.csv")
        feature_enriched_df.to_csv(feature_dataset_path, index=False)

        y_series = pd.Series(english_ts["y"].values, index=pd.to_datetime(english_ts["date"]), name="y")
        exog_series = pd.Series(english_ts["exog"].values, index=pd.to_datetime(english_ts["date"]), name="exog")

        run = mlflow.start_run(run_name=f"SARIMAX_forecast_{args.target_language}") if not args.dry_run else None
        try:
            pred_df, metrics, fitted = train_eval_sarimax(
                series=y_series,
                exog_series=exog_series,
                order=order,
                seasonal_order=seasonal_order,
                steps=args.steps,
            )
            pred_path = os.path.join(tmpdir, f"forecast_vs_actual_{args.target_language}.csv")
            pred_df.to_csv(pred_path, index=False)

            print(
                f"[{args.target_language}] RMSE={metrics['rmse']:.4f} "
                f"MAPE={metrics['mape']:.2f}% "
                f"wMAPE={metrics['wmape']:.2f}%"
            )

            if not args.dry_run:
                dataset_for_tracking = mlflow.data.from_pandas(english_ts, source=args.train_path)
                mlflow.log_input(dataset_for_tracking, context="timeseries")

                mlflow.log_params(
                    {
                        "language": args.target_language,
                        "order": str(order),
                        "seasonal_order": str(seasonal_order),
                        "min_observed_days": args.min_observed_days,
                        "steps": args.steps,
                        "dataset_rows": len(english_ts),
                        "sample_mode": "full_language_only" if args.max_rows is None else "language_subset",
                    }
                )
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(preprocess_path)
                mlflow.log_artifact(english_dataset_path)
                mlflow.log_artifact(feature_dataset_path)
                mlflow.log_artifact(pred_path)
                mlflow.statsmodels.log_model(
                    statsmodels_model=fitted,
                    name=f"sarimax_model_{args.target_language}",
                )
        finally:
            if run is not None:
                mlflow.end_run()


if __name__ == "__main__":
    main()
