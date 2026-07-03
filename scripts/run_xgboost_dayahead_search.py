"""
run_xgboost_dayahead_search.py — XGBoost day-ahead random search.

Usage:
    conda run -n epf-2 python scripts/run_xgboost_dayahead_search.py
"""

import logging, os, sys, json, time, random, yaml
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.data_loader import load_data
from src.common.metrics import compute_all_metrics
from src.common.feature_builder_dayahead import build_features_dayahead
from src.common.output_schema import make_long_table

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

_OUTPUT = os.path.join(_PROJECT_DIR, "outputs", "dayahead_xgboost_search_30d")
WINDOWS = {"75d": 75, "90d": 90, "120d": 120, "150d": 150, "all": 99999}
SEARCH_DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-02-01", "2026-02-20")]
CONFIRM_DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-02-21", "2026-03-02")]
FULL_DAYS = SEARCH_DAYS + CONFIRM_DAYS

FEATURE_COLS = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
    "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
    "morning_trend", "is_info_fresh",
    "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
    "same_hour_mean_7d", "same_hour_mean_14d", "same_hour_std_7d",
    "same_hour_max_7d", "same_hour_min_7d",
    "price_momentum_24_168", "net_load_rank_30d", "bidding_space_rank_30d",
    "is_spring_festival_window", "days_to_spring_festival",
    "days_after_spring_festival", "is_month_start", "is_month_end",
]


def sample_params(random_state: int) -> dict:
    """Random sample XGBoost params."""
    rng = random.Random(random_state)
    objectives = ["reg:absoluteerror", "reg:squarederror"]
    if hasattr(xgb, "__version__"):
        v = xgb.__version__.split(".")
        if int(v[0]) >= 2:
            objectives.append("reg:pseudohubererror")
    
    return {
        "objective": rng.choice(objectives),
        "max_depth": rng.choice([4, 6, 8, 10]),
        "learning_rate": rng.choice([0.015, 0.03, 0.05]),
        "subsample": rng.choice([0.75, 0.85, 0.95]),
        "colsample_bytree": rng.choice([0.75, 0.85, 0.95]),
        "lambda": rng.choice([1, 2, 5, 10]),
        "alpha": rng.choice([0, 0.1, 0.5, 1.0]),
        "min_child_weight": rng.choice([1, 5, 10, 20]),
        "n_estimators": rng.choice([1000, 2000]),
        "tree_method": "hist", "device": "cuda",
        "early_stopping_rounds": 20,
        "verbosity": 0,
    }


def _prepare_X(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from dataframe."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    for col in ["hour", "month", "day_of_week", "is_weekend"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.values


def _train_eval(dtrain: xgb.DMatrix, deval: xgb.DMatrix, params: dict) -> dict:
    """Train XGBoost and return best iteration + model."""
    n_rounds = params.pop("n_estimators", 1000)
    es = params.pop("early_stopping_rounds", 20)
    
    model = xgb.train(
        params, dtrain, num_boost_round=n_rounds,
        evals=[(deval, "eval")],
        early_stopping_rounds=es, verbose_eval=False,
    )
    best_iter = model.best_iteration if model.best_iteration else n_rounds
    return {"model": model, "best_iter": best_iter, "params": params}


def eval_config(params: dict, window_label: str, window_days: int,
                train_dates: list[str], eval_dates: list[str],
                df_full: pd.DataFrame) -> dict | None:
    """Evaluate one config on a window: train on train_dates, eval on eval_dates."""
    all_preds = []
    for target_date in eval_dates:
        target_dt = pd.Timestamp(target_date)
        train_start = target_dt - timedelta(days=window_days)
        train_end = target_dt - timedelta(hours=1)
        val_start = target_dt - timedelta(days=7)

        train_mask = (df_full["ds"] >= train_start) & (df_full["ds"] < train_end)
        val_mask = (df_full["ds"] >= val_start) & (df_full["ds"] < train_end)

        train_df = df_full[train_mask]
        val_df = df_full[val_mask]
        if len(train_df) < 100 or len(val_df) < 24:
            continue

        try:
            X_tr, y_tr = _prepare_X(train_df), train_df["y"].values
            X_va, y_va = _prepare_X(val_df), val_df["y"].values
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            deval = xgb.DMatrix(X_va, label=y_va)

            result = _train_eval(dtrain, deval, params.copy())
            model = result["model"]

            # Predict target day
            start_ds = target_dt + timedelta(hours=1)
            end_ds = target_dt + timedelta(days=1)
            mask = (df_full["ds"] >= start_ds) & (df_full["ds"] < end_ds)
            day_df = df_full[mask]
            if len(day_df) == 0:
                continue

            X_test = _prepare_X(day_df)
            dtest = xgb.DMatrix(X_test)
            y_pred = model.predict(dtest)

            day_df = day_df.copy()
            day_df["y_pred"] = y_pred
            day_df["y_true"] = day_df["y"].values
            day_df["hour_business"] = day_df["hour"].values  # hour 1-24 maps directly
            day_df["target_day"] = target_date
            all_preds.append(day_df[["ds", "y_true", "y_pred", "hour_business", "target_day"]])
        except Exception as e:
            logger.warning(f"      {target_date} failed: {e}")
            continue

    if len(all_preds) < 5:
        return None

    merged = pd.concat(all_preds, ignore_index=True)
    valid = ~np.isnan(merged["y_pred"].values) & ~np.isnan(merged["y_true"].values)
    if valid.sum() < 24:
        return None

    metrics = compute_all_metrics(merged["y_true"].values[valid], merged["y_pred"].values[valid])
    return {
        "sMAPE": metrics["sMAPE_floor50"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "n": int(valid.sum()),
        "params": params,
    }


def main():
    if not _HAS_XGB:
        logger.error("xgboost not installed")
        return

    os.makedirs(os.path.join(_OUTPUT, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT, "reports"), exist_ok=True)

    # Load data
    with open(os.path.join(_PROJECT_DIR, "configs", "paths.yaml"), encoding="utf-8") as f:
        data_path = yaml.safe_load(f)["default_data"]
    df = load_data(data_path, target="dayahead")
    df = build_features_dayahead(df)
    df["ds"] = pd.to_datetime(df["ds"])
    logger.info(f"Data: {len(df)} rows, {df['ds'].min()} ~ {df['ds'].max()}")

    all_results = []

    for window_label, window_days in WINDOWS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Window: {window_label} ({window_days}d)")
        logger.info(f"{'='*50}")

        for trial in range(30):
            params = sample_params(trial + hash(window_label) % 10000)
            logger.info(f"  Trial {trial+1}/30: objective={params['objective']}, "
                         f"md={params['max_depth']}, lr={params['learning_rate']}")

            # Search window
            sr = eval_config(params, window_label, window_days,
                              SEARCH_DAYS[:15], SEARCH_DAYS[15:], df)
            if sr is None:
                logger.info(f"    Search: failed")
                continue

            # Confirm window
            cr = eval_config(params, window_label, window_days,
                              SEARCH_DAYS, CONFIRM_DAYS, df)
            if cr is None:
                logger.info(f"    Confirm: failed (sr={sr['sMAPE']:.2f}%)")
                continue

            # Full 30d
            fr = eval_config(params, window_label, window_days,
                              FULL_DAYS[:20], FULL_DAYS[20:], df)

            result = {
                "window": window_label,
                "trial": trial + 1,
                "search_sMAPE": sr["sMAPE"],
                "confirm_sMAPE": cr["sMAPE"],
                "full_sMAPE": fr["sMAPE"] if fr else cr["sMAPE"],
                "search_n": sr["n"],
                "confirm_n": cr["n"],
            }
            # Add params
            for k, v in params.items():
                if k in ("tree_method", "gpu_id", "verbosity", "early_stopping_rounds"):
                    continue
                result[f"p_{k}"] = v
            
            all_results.append(result)
            logger.info(f"    Search={sr['sMAPE']:.2f}% Confirm={cr['sMAPE']:.2f}% "
                         f"Full={result['full_sMAPE']:.2f}%")

    # Save config search results
    if not all_results:
        logger.error("No results. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(_OUTPUT, "metrics", "config_search_results.csv"),
                       index=False, encoding="utf-8-sig")
    logger.info(f"\nConfig search saved: {len(results_df)} trials")

    # Find best
    best_row = results_df.loc[results_df["full_sMAPE"].idxmin()]
    logger.info(f"\nBest (full): window={best_row['window']}, "
                 f"sMAPE={best_row['full_sMAPE']:.4f}%, "
                 f"search={best_row['search_sMAPE']:.2f}%, "
                 f"confirm={best_row['confirm_sMAPE']:.2f}%")

    # Save summary
    summary = results_df.groupby("window").agg(
        best_smape=("full_sMAPE", "min"),
        mean_smape=("full_sMAPE", "mean"),
        n_trials=("full_sMAPE", "count"),
    ).reset_index().sort_values("best_smape")
    summary.to_csv(os.path.join(_OUTPUT, "metrics", "summary.csv"),
                    index=False, encoding="utf-8-sig")

    # Generate report
    lines = [
        "# XGBoost Day-Ahead Search Report",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Trials: {len(results_df)}",
        "",
        "## 1. Best Config",
        "",
    ]
    best = results_df.loc[results_df["full_sMAPE"].idxmin()]
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Best full sMAPE | {best['full_sMAPE']:.4f}% |")
    lines.append(f"| Window | {best['window']} |")
    lines.append(f"| Search sMAPE | {best['search_sMAPE']:.2f}% |")
    lines.append(f"| Confirm sMAPE | {best['confirm_sMAPE']:.2f}% |")
    for k in ["objective", "max_depth", "learning_rate", "subsample",
               "colsample_bytree", "lambda", "alpha", "min_child_weight", "n_estimators"]:
        pk = f"p_{k}"
        if pk in best:
            lines.append(f"| {k} | {best[pk]} |")

    lines += [
        "",
        "## 2. Window Summary",
        "",
        "| Window | Best sMAPE | Mean sMAPE | Trials |",
        "|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r['window']} | {r['best_smape']:.4f}% | {r['mean_smape']:.4f}% | {int(r['n_trials'])} |")

    lines += [
        "",
        "## 3. 结论",
        "",
        f"| 问题 | 回答 |",
        f"|---|---|",
        f"| XGBoost 最佳 sMAPE | {best['full_sMAPE']:.2f}% |",
        f"| 最优窗口 | {best['window']} |",
        f"| 最优 objective | {best.get('p_objective', 'N/A')} |",
        f"| 优于 12.47% | {'✅' if best['full_sMAPE'] < 12.47 else '❌'} |",
        f"| 优于 LightGBM 单模型 11.97% | {'✅' if best['full_sMAPE'] < 11.97 else '❌'} |",
        f"| 优于 11.73% fusion | {'✅' if best['full_sMAPE'] < 11.73 else '❌'} |",
        f"| 低于 11.5% | {'✅' if best['full_sMAPE'] < 11.5 else '❌'} |",
        f"| 低于 11% | {'✅' if best['full_sMAPE'] < 11 else '❌'} |",
        f"| 建议进入模型池 | {'✅' if best['full_sMAPE'] < 12.5 else '❌'} |",
    ]

    report = "\n".join(lines)
    report_path = os.path.join(_OUTPUT, "reports", "xgboost_dayahead_search_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"\n{report}")
    logger.info(f"\nDone. Report: {report_path}")


if __name__ == "__main__":
    main()
