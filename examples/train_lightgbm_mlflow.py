import argparse
import os
import tempfile

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from train_sarimax_mlflow import build_english_dataset, mape, wmape


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_lags(value: str) -> list[int]:
    lags = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not lags:
        raise ValueError("--lags must include at least one lag")
    if min(lags) <= 0:
        raise ValueError("All lags must be positive integers")
    return lags


def build_supervised_train_frame(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    feat = df.copy()
    feat["ds"] = pd.to_datetime(feat["date"])
    feat["dow"] = feat["ds"].dt.dayofweek
    feat["dom"] = feat["ds"].dt.day
    feat["month"] = feat["ds"].dt.month

    for lag in lags:
        feat[f"y_lag_{lag}"] = feat["y"].shift(lag)

    feat = feat.dropna().reset_index(drop=True)
    return feat


def forecast_recursive(
    model: lgb.LGBMRegressor,
    history: list[float],
    future_dates: pd.Series,
    future_exog: pd.Series,
    lags: list[int],
    feature_cols: list[str],
) -> np.ndarray:
    preds: list[float] = []
    max_lag = max(lags)
    if len(history) < max_lag:
        raise ValueError(f"Not enough history for max lag {max_lag}")

    for ds, exog_val in zip(future_dates, future_exog):
        row = {
            "exog": float(exog_val),
            "dow": int(ds.dayofweek),
            "dom": int(ds.day),
            "month": int(ds.month),
        }
        for lag in lags:
            row[f"y_lag_{lag}"] = float(history[-lag])

        x_row = pd.DataFrame([row], columns=feature_cols)
        pred = float(model.predict(x_row)[0])
        preds.append(pred)
        history.append(pred)

    return np.array(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightGBM experiment with MLflow tracking")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"), help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="wikipedia-language-forecasting-lightgbm")
    parser.add_argument("--run-name", default="LightGBM_Forecast")
    parser.add_argument("--train-path", default="dataset/train_1.csv")
    parser.add_argument("--exog-path", default="dataset/Exog_Campaign_eng")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--min-observed-days", type=int, default=300)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional subset for quick tests")
    parser.add_argument("--lags", default="1,2,3,7,14,21,28")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Run without MLflow tracking")
    args = parser.parse_args()

    if not args.dry_run and not args.tracking_uri:
        raise ValueError("Set --tracking-uri or MLFLOW_TRACKING_URI, or use --dry-run")

    lags = parse_lags(args.lags)

    english_ts, _stats = build_english_dataset(
        train_path=args.train_path,
        exog_path=args.exog_path,
        target_language=args.target_language,
        min_observed_days=args.min_observed_days,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )

    if len(english_ts) <= args.steps + max(lags):
        raise ValueError("Not enough points after preprocessing for selected horizon and lags")

    df = english_ts.copy().sort_values("date").reset_index(drop=True)
    train_df = df.iloc[:-args.steps].copy()
    test_df = df.iloc[-args.steps:].copy()

    supervised = build_supervised_train_frame(train_df, lags)
    feature_cols = ["exog", "dow", "dom", "month"] + [f"y_lag_{lag}" for lag in lags]
    X_train = supervised[feature_cols]
    y_train = supervised["y"]

    model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    preds = forecast_recursive(
        model=model,
        history=train_df["y"].astype(float).tolist(),
        future_dates=pd.to_datetime(test_df["date"]),
        future_exog=test_df["exog"].astype(float),
        lags=lags,
        feature_cols=feature_cols,
    )

    y_true = test_df["y"].reset_index(drop=True)
    y_pred = pd.Series(preds)

    metric_rmse = rmse(y_true, y_pred)
    metric_mape = mape(y_true, y_pred)
    metric_wmape = wmape(y_true, y_pred)

    print(
        f"[{args.target_language}] LightGBM RMSE={metric_rmse:.4f} "
        f"MAPE={metric_mape:.2f}% wMAPE={metric_wmape:.2f}%"
    )

    if args.dry_run:
        return

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, f"lightgbm_ts_dataset_{args.target_language}.csv")
        pred_path = os.path.join(tmp_dir, f"lightgbm_forecast_{args.target_language}.csv")
        fi_path = os.path.join(tmp_dir, f"lightgbm_feature_importance_{args.target_language}.csv")

        df.to_csv(dataset_path, index=False)

        pred_df = pd.DataFrame(
            {
                "date": test_df["date"].values,
                "actual": y_true.values,
                "forecast": y_pred.values,
            }
        )
        pred_df.to_csv(pred_path, index=False)

        fi_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                "importance_split": model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
        fi_df.to_csv(fi_path, index=False)

        with mlflow.start_run(run_name=args.run_name):
            ts_dataset = mlflow.data.from_pandas(df, source=args.train_path, targets="y")
            mlflow.log_input(ts_dataset, context="lightgbm_timeseries")

            mlflow.log_params(
                {
                    "language": args.target_language,
                    "steps": args.steps,
                    "min_observed_days": args.min_observed_days,
                    "lags": args.lags,
                    "n_estimators": args.n_estimators,
                    "learning_rate": args.learning_rate,
                    "num_leaves": args.num_leaves,
                    "max_depth": args.max_depth,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "sample_mode": "full_language_only" if args.max_rows is None else "language_subset",
                }
            )
            mlflow.log_metrics(
                {
                    "rmse": metric_rmse,
                    "mape": metric_mape,
                    "wmape": metric_wmape,
                }
            )

            mlflow.log_artifact(dataset_path)
            mlflow.log_artifact(pred_path)
            mlflow.log_artifact(fi_path)

            mlflow.lightgbm.log_model(
                lgb_model=model,
                name=f"lightgbm_model_{args.target_language}",
                input_example=X_train.head(),
            )


if __name__ == "__main__":
    main()