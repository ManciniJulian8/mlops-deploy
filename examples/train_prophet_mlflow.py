import argparse
import os
import tempfile

import mlflow
import mlflow.prophet
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error

from train_sarimax_mlflow import build_english_dataset, mape, wmape


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_plot(fig, path: str) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.clf()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prophet experiments with MLflow tracking")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"), help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="wikipedia-language-forecasting-prophet")
    parser.add_argument("--run-name", default="Prophet_Forecast")
    parser.add_argument("--train-path", default="dataset/train_1.csv")
    parser.add_argument("--exog-path", default="dataset/Exog_Campaign_eng")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--min-observed-days", type=int, default=300)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional subset for quick tests")
    parser.add_argument("--seasonality-prior-scale", type=float, default=50.0)
    parser.add_argument("--n-changepoints", type=int, default=30)
    parser.add_argument("--changepoint-prior-scale", type=float, default=0.1)
    parser.add_argument("--holidays-prior-scale", type=float, default=50.0)
    parser.add_argument("--dry-run", action="store_true", help="Run without MLflow tracking")
    args = parser.parse_args()

    if not args.dry_run and not args.tracking_uri:
        raise ValueError("Set --tracking-uri or MLFLOW_TRACKING_URI, or use --dry-run")

    english_ts, _stats = build_english_dataset(
        train_path=args.train_path,
        exog_path=args.exog_path,
        target_language=args.target_language,
        min_observed_days=args.min_observed_days,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )

    if len(english_ts) <= args.steps + 20:
        raise ValueError("Not enough points after preprocessing for selected horizon")

    df_prophet = english_ts.rename(columns={"date": "ds"}).copy()
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    df_prophet_ex = df_prophet[["ds", "y", "exog"]].rename(columns={"exog": "Exog"})

    train_default = df_prophet.iloc[:-args.steps][["ds", "y"]].copy()
    train_exog = df_prophet_ex.iloc[:-args.steps][["ds", "y", "Exog"]].copy()

    y_true = df_prophet["y"].iloc[-args.steps:].reset_index(drop=True)

    if not args.dry_run:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        prophet_dataset_path = os.path.join(tmp_dir, f"prophet_ts_dataset_{args.target_language}.csv")
        prophet_dataset_exog_path = os.path.join(tmp_dir, f"prophet_ts_dataset_exog_{args.target_language}.csv")
        df_prophet.to_csv(prophet_dataset_path, index=False)
        df_prophet_ex.to_csv(prophet_dataset_exog_path, index=False)

        run = mlflow.start_run(run_name=args.run_name) if not args.dry_run else None
        try:
            # Prophet default (without exogenous regressor)
            model_default = Prophet(
                seasonality_prior_scale=args.seasonality_prior_scale,
                weekly_seasonality=True,
            )
            fit_default = model_default.fit(train_default)
            future_default = model_default.make_future_dataframe(periods=args.steps, freq="D")
            forecast_default = fit_default.predict(future_default)

            yhat_default = forecast_default["yhat"].iloc[-args.steps:].reset_index(drop=True)
            default_rmse = rmse(y_true, yhat_default)
            default_mape = mape(y_true, yhat_default)
            default_wmape = wmape(y_true, yhat_default)

            default_forecast_path = os.path.join(tmp_dir, f"prophet_default_forecast_{args.target_language}.csv")
            forecast_default.to_csv(default_forecast_path, index=False)
            default_plot_path = os.path.join(tmp_dir, f"prophet_default_plot_{args.target_language}.png")
            save_plot(fit_default.plot(forecast_default), default_plot_path)

            # Prophet tuned with holiday/campaign effect via external regressor
            model_tuned = Prophet(
                n_changepoints=args.n_changepoints,
                changepoint_prior_scale=args.changepoint_prior_scale,
                seasonality_prior_scale=args.seasonality_prior_scale,
                holidays_prior_scale=args.holidays_prior_scale,
                weekly_seasonality=True,
            )
            model_tuned.add_regressor("Exog")
            fit_tuned = model_tuned.fit(train_exog)
            forecast_tuned = fit_tuned.predict(df_prophet_ex[["ds", "Exog"]])

            yhat_tuned = forecast_tuned["yhat"].iloc[-args.steps:].reset_index(drop=True)
            tuned_rmse = rmse(y_true, yhat_tuned)
            tuned_mape = mape(y_true, yhat_tuned)
            tuned_wmape = wmape(y_true, yhat_tuned)

            tuned_forecast_path = os.path.join(tmp_dir, f"prophet_tuned_forecast_{args.target_language}.csv")
            forecast_tuned.to_csv(tuned_forecast_path, index=False)
            tuned_plot_path = os.path.join(tmp_dir, f"prophet_tuned_plot_{args.target_language}.png")
            save_plot(fit_tuned.plot(forecast_tuned), tuned_plot_path)

            print(
                f"[{args.target_language}] Prophet default RMSE={default_rmse:.4f} MAPE={default_mape:.2f}% "
                f"wMAPE={default_wmape:.2f}%"
            )
            print(
                f"[{args.target_language}] Prophet tuned+Exog RMSE={tuned_rmse:.4f} MAPE={tuned_mape:.2f}% "
                f"wMAPE={tuned_wmape:.2f}%"
            )

            if not args.dry_run:
                ts_dataset = mlflow.data.from_pandas(df_prophet, source=args.train_path, targets="y")
                ts_dataset_exog = mlflow.data.from_pandas(df_prophet_ex, source=args.exog_path, targets="y")
                mlflow.log_input(ts_dataset, context="prophet_timeseries")
                mlflow.log_input(ts_dataset_exog, context="prophet_timeseries_with_exog")

                mlflow.log_params(
                    {
                        "language": args.target_language,
                        "steps": args.steps,
                        "min_observed_days": args.min_observed_days,
                        "sample_mode": "full_language_only" if args.max_rows is None else "language_subset",
                        "default_seasonality_prior_scale": args.seasonality_prior_scale,
                        "tuned_n_changepoints": args.n_changepoints,
                        "tuned_changepoint_prior_scale": args.changepoint_prior_scale,
                        "tuned_seasonality_prior_scale": args.seasonality_prior_scale,
                        "tuned_holidays_prior_scale": args.holidays_prior_scale,
                    }
                )
                mlflow.log_metrics(
                    {
                        "default_rmse": default_rmse,
                        "default_mape": default_mape,
                        "default_wmape": default_wmape,
                        "tuned_exog_rmse": tuned_rmse,
                        "tuned_exog_mape": tuned_mape,
                        "tuned_exog_wmape": tuned_wmape,
                    }
                )

                mlflow.log_artifact(prophet_dataset_path)
                mlflow.log_artifact(prophet_dataset_exog_path)
                mlflow.log_artifact(default_forecast_path)
                mlflow.log_artifact(default_plot_path)
                mlflow.log_artifact(tuned_forecast_path)
                mlflow.log_artifact(tuned_plot_path)

                mlflow.prophet.log_model(
                    pr_model=fit_default,
                    name=f"prophet_default_model_{args.target_language}",
                    input_example=train_default.head(),
                )
                mlflow.prophet.log_model(
                    pr_model=fit_tuned,
                    name=f"prophet_tuned_exog_model_{args.target_language}",
                    input_example=train_exog.head(),
                )
        finally:
            if run is not None:
                mlflow.end_run()


if __name__ == "__main__":
    main()
