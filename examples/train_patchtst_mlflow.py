import argparse
import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from train_sarimax_mlflow import build_english_dataset, mape, wmape


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PatchTST experiment with MLflow tracking")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"), help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="wikipedia-language-forecasting-patchtst")
    parser.add_argument("--run-name", default="PatchTST_Forecast")
    parser.add_argument("--train-path", default="dataset/train_1.csv")
    parser.add_argument("--exog-path", default="dataset/Exog_Campaign_eng")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--min-observed-days", type=int, default=300)
    parser.add_argument("--steps", type=int, default=20, help="Forecast horizon")
    parser.add_argument("--history", type=int, default=30, help="Window length")
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional subset for fast tests")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-chunks", type=int, default=2)
    parser.add_argument("--show-graph", action="store_true", help="Enable tsai ShowGraph callback")
    parser.add_argument("--dry-run", action="store_true", help="Run without MLflow tracking")
    args = parser.parse_args()

    if not args.dry_run and not args.tracking_uri:
        raise ValueError("Set --tracking-uri or MLFLOW_TRACKING_URI, or use --dry-run")

    # Import tsai lazily so the script fails with a clear message if dependency is missing.
    try:
        from tsai.all import PatchTST, ShowGraph, SlidingWindow, TSForecaster, mae
    except Exception as exc:
        raise RuntimeError(
            "tsai/torch dependencies are not installed. Run: pip install -r examples/requirements-patchtst.txt"
        ) from exc

    english_ts, _stats = build_english_dataset(
        train_path=args.train_path,
        exog_path=args.exog_path,
        target_language=args.target_language,
        min_observed_days=args.min_observed_days,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )

    df_ex = english_ts.copy()
    df_ex["date"] = pd.to_datetime(df_ex["date"])
    df_ex = df_ex.set_index("date")

    # Input matrix for deep learning model.
    # We use both target and exogenous signal as features to predict future y.
    x_cols = ["y", "exog"]
    y_col = "y"

    X, y = SlidingWindow(
        window_len=args.history,
        horizon=args.steps,
        stride=1,
        get_x=x_cols,
        get_y=x_cols,
    )(df_ex)

    n_samples = len(X)
    if n_samples < 2:
        raise ValueError("Not enough samples after sliding-window transform")

    # Target behavior from the reference notebook:
    # keep the last window as validation/test block.
    cut = min(500, n_samples - 1)
    train_idx = np.arange(0, cut)
    valid_idx = np.arange(cut, n_samples)
    splits = (train_idx, valid_idx)

    if not args.dry_run:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        splits_path = os.path.join(tmp_dir, "ts_splits.npz")
        np.savez(splits_path, train_idx=train_idx, valid_idx=valid_idx)

        dataset_path = os.path.join(tmp_dir, f"tsai_dataset_{args.target_language}.csv")
        df_ex.reset_index().to_csv(dataset_path, index=False)

        callbacks = [ShowGraph()] if args.show_graph else []
        fcst = TSForecaster(
            X,
            y,
            splits=splits,
            path=os.path.join(tmp_dir, "models"),
            batch_size=args.batch_size,
            arch=PatchTST,
            arch_config={"n_layers": args.n_chunks},
            metrics=mae,
            cbs=callbacks,
        )

        run = mlflow.start_run(run_name=args.run_name) if not args.dry_run else None
        try:
            preds, *_ = (None,)
            fcst.fit_one_cycle(args.epochs, args.learning_rate)
            preds, targets, *_ = fcst.get_X_preds(X[valid_idx], y[valid_idx])

            # Use the last predicted horizon block as final forecast series.
            preds_arr = np.array(preds)
            targets_arr = np.array(targets)

            if preds_arr.ndim == 3:
                final_pred = preds_arr[-1, 0, :].reshape(-1)
            else:
                final_pred = preds_arr[-1].reshape(-1)

            if targets_arr.ndim == 3:
                y_true = targets_arr[-1, 0, :].reshape(-1)
            else:
                y_true = targets_arr[-1].reshape(-1)

            metric_rmse = rmse(y_true, final_pred)
            metric_mape = mape(pd.Series(y_true), pd.Series(final_pred))
            metric_wmape = wmape(pd.Series(y_true), pd.Series(final_pred))

            pred_df = pd.DataFrame(
                {
                    "step": np.arange(1, len(final_pred) + 1),
                    "actual": y_true,
                    "forecast": final_pred,
                }
            )
            pred_path = os.path.join(tmp_dir, f"patchtst_forecast_{args.target_language}.csv")
            pred_df.to_csv(pred_path, index=False)

            model_export_path = os.path.join(tmp_dir, f"patchtst_model_{args.target_language}.pkl")
            fcst.export(model_export_path)

            print(
                f"[{args.target_language}] PatchTST RMSE={metric_rmse:.4f} "
                f"MAPE={metric_mape:.2f}% wMAPE={metric_wmape:.2f}%"
            )

            if not args.dry_run:
                tsai_ex_dataset = mlflow.data.from_pandas(df_ex.reset_index(), source=args.train_path, targets="y")
                mlflow.log_input(tsai_ex_dataset, context="numpy_timeseries")

                mlflow.log_params(
                    {
                        "language": args.target_language,
                        "arch": "PatchTST",
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                        "history": args.history,
                        "horizon": args.steps,
                        "test_size": args.steps,
                        "split_cutoff": int(cut),
                        "n_samples": int(n_samples),
                        "features": ",".join(x_cols),
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
                mlflow.log_artifact(splits_path)
                mlflow.log_artifact(pred_path)
                mlflow.log_artifact(model_export_path)
        finally:
            if run is not None:
                mlflow.end_run()


if __name__ == "__main__":
    main()
