"""
run_xgboost_sentinel.py — XGBoost sentinel experiment for day-ahead.

Small-scale search: 8 trials, 3 windows (90/120/150d).
Stop if full_30d sMAPE > 11.8% and no hour advantage.

Usage:
    python scripts/run_xgboost_sentinel.py
    python scripts/run_xgboost_sentinel.py --n-trials 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import smape_floor50, compute_all_metrics
from src.common.data_loader import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="XGBoost sentinel experiment")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_xgboost_sentinel_30d")
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


EVAL_START = "2026-02-01"
EVAL_END = "2026-03-02"
SEARCH_END = "2026-02-20"
CONFIRM_START = "2026-02-21"
WINDOWS = [90, 120, 150]


def _fast_rank_rolling(series: pd.Series, window: int = 720) -> pd.Series:
    """Fast approximate rank using rolling percentile."""
    return series.rolling(window, min_periods=max(10, window // 4)).apply(
        lambda x: (x < x[-1]).sum() / len(x) if len(x) >= 10 else 0.5,
        raw=True,
    ).fillna(0.5)


def build_v3_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Reuse v3 feature builder (optimized, no O(n^2) ranks)."""
    from src.common.feature_builder import build_features as build_base
    from src.common.feature_builder_dayahead import (
        _add_lag_features, _add_same_hour_stats, _add_price_momentum,
        _add_calendar_features,
    )
    from src.common.feature_builder_dayahead_v3 import (
        _add_volatility, _add_change_features,
        _add_exact_spring_festival, _add_interaction_features,
    )

    df = build_base(raw)
    adjusted = df["ds"] - pd.Timedelta(seconds=1)
    df["hour_business"] = adjusted.dt.hour + 1
    df["period"] = np.select(
        [df["hour_business"].between(1, 8), df["hour_business"].between(9, 16)],
        ["1_8", "9_16"], default="17_24",
    )
    df = _add_lag_features(df)
    df = _add_same_hour_stats(df)
    df = _add_price_momentum(df)
    df = _add_calendar_features(df)

    df = df.sort_values("ds").reset_index(drop=True)
    df["net_load_rank_30d"] = _fast_rank_rolling(df["net_load"], 720)
    df["bidding_space_rank_30d"] = _fast_rank_rolling(df["bidding_space"], 720)

    df = _add_volatility(df)
    df = _add_change_features(df)
    df = _add_exact_spring_festival(df)
    df = _add_interaction_features(df)
    df["target_day"] = df["ds"].dt.date.astype(str)
    df["business_day"] = df["target_day"]
    df = df.ffill().fillna(0).reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"ds", "y", "target_day", "business_day", "hour_business", "period",
               "date_only", "y_pred", "y_true", "model_name", "task"}
    numeric = df.select_dtypes(include=[np.float64, np.int64, np.float32, np.int32, np.int8, bool]).columns
    return [c for c in numeric if c not in exclude and c != "y"]


def evaluate_xgb_config(params: dict, window, df: pd.DataFrame, feat_cols: list[str],
                        all_days: list[str], search_days: list[str], confirm_days: list[str],
                        config_name: str) -> dict | None:
    """Train rolling XGBoost and evaluate."""
    import xgboost as xgb

    preds = []
    for day in all_days:
        target_dt = pd.Timestamp(day)
        train_all = df[df["target_day"] < day]
        if len(train_all) < 200:
            continue

        if window == "all":
            train_df = train_all.copy()
        else:
            train_start = target_dt - timedelta(days=window)
            train_df = train_all[train_all["ds"] >= train_start].copy()

        if len(train_df) < 100:
            continue

        X_tr = train_df[feat_cols].values.astype(float)
        y_tr = train_df["y"].values.astype(float)

        try:
            xgb_params = {
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "lambda": params["lambda"],
                "alpha": params["alpha"],
                "min_child_weight": params["min_child_weight"],
                "objective": params["objective"],
                "tree_method": "hist",
                "verbosity": 0,
            }

            n_rounds = params.get("n_estimators", 1000)

            # Use last 30 days as validation for early stopping
            val_start = target_dt - timedelta(days=30)
            val_data = train_all[train_all["ds"].between(val_start, target_dt - timedelta(hours=1))]

            if len(val_data) >= 50:
                X_val = val_data[feat_cols].values.astype(float)
                y_val = val_data["y"].values.astype(float)
                dtrain = xgb.DMatrix(X_tr, y_tr)
                dval = xgb.DMatrix(X_val, y_val)
                model = xgb.train(
                    xgb_params, dtrain,
                    num_boost_round=n_rounds,
                    evals=[(dval, "eval")],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
            else:
                dtrain = xgb.DMatrix(X_tr, y_tr)
                model = xgb.train(xgb_params, dtrain, num_boost_round=n_rounds, verbose_eval=False)

            day_df = df[df["target_day"] == day].copy()
            X_pred = day_df[feat_cols].values.astype(float)
            dtest = xgb.DMatrix(X_pred)
            day_df = day_df.copy()
            day_df["y_pred"] = model.predict(dtest)
            day_df["y_true"] = day_df["y"].values
            day_df["model_name"] = config_name
            preds.append(day_df)
        except Exception as e:
            logger.debug(f"    Day {day}: {e}")
            continue

    if not preds:
        return None

    full = pd.concat(preds, ignore_index=True)
    search_mask = full["target_day"].isin(search_days)
    confirm_mask = full["target_day"].isin(confirm_days)

    s_smape = smape_floor50(full.loc[search_mask, "y_true"].values,
                            full.loc[search_mask, "y_pred"].values) if search_mask.sum() >= 10 else None
    c_smape = smape_floor50(full.loc[confirm_mask, "y_true"].values,
                            full.loc[confirm_mask, "y_pred"].values) if confirm_mask.sum() >= 10 else None
    f_smape = smape_floor50(full["y_true"].values, full["y_pred"].values)

    return {
        "config_name": config_name, "window": window, "params": params,
        "predictions": full,
        "search_smape": s_smape, "confirm_smape": c_smape, "full_smape": f_smape,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.output_root)
    for sub in ["predictions", "metrics", "reports", "debug"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    data_path = args.data_path
    if data_path is None:
        from src.common.repo_paths import get_data_path
        data_path = str(get_data_path())

    logger.info("Loading data and building v3 features...")
    raw = load_data(data_path, target="dayahead")
    df = build_v3_features(raw)
    df = df[df["ds"] >= "2025-08-01"].reset_index(drop=True)
    logger.info(f"Feature DF: {len(df)} rows")

    feat_cols = get_feature_cols(df)
    logger.info(f"Features: {len(feat_cols)}")

    all_days = sorted(df[(df["ds"] >= EVAL_START) & (df["ds"] <= f"{EVAL_END} 23:00:00")]["target_day"].unique())
    search_days = [d for d in all_days if d <= SEARCH_END]
    confirm_days = [d for d in all_days if d >= CONFIRM_START]
    logger.info(f"Eval: {len(all_days)} days, Search: {len(search_days)}, Confirm: {len(confirm_days)}")

    all_results = {}
    config_log = []

    def store_result(result):
        if result is None:
            return
        name = result["config_name"]
        all_results[name] = result
        entry = {
            "config": name, "window": result["window"],
            "objective": result["params"].get("objective", "reg:squarederror"),
            "max_depth": result["params"]["max_depth"],
            "learning_rate": result["params"]["learning_rate"],
            "subsample": result["params"]["subsample"],
            "colsample_bytree": result["params"]["colsample_bytree"],
            "lambda": result["params"]["lambda"],
            "alpha": result["params"]["alpha"],
            "min_child_weight": result["params"]["min_child_weight"],
            "n_estimators": result["params"].get("n_estimators", 1000),
            "search_smape": round(result["search_smape"], 4) if result["search_smape"] else None,
            "confirm_smape": round(result["confirm_smape"], 4) if result["confirm_smape"] else None,
            "full_smape": round(result["full_smape"], 4),
        }
        config_log.append(entry)
        s_str = f"{result['search_smape']:.4f}%" if result["search_smape"] else "N/A"
        c_str = f"{result['confirm_smape']:.4f}%" if result["confirm_smape"] else "N/A"
        logger.info(f"  {name}: full={result['full_smape']:.4f}% search={s_str} confirm={c_str}")

    # ── XGBoost search space ──
    OBJECTIVES = ["reg:absoluteerror", "reg:squarederror"]

    def objective(trial):
        params = {
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.015, 0.03, 0.05]),
            "subsample": trial.suggest_categorical("subsample", [0.75, 0.85, 0.95]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.75, 0.85, 0.95]),
            "lambda": trial.suggest_categorical("lambda", [1, 2, 5, 10]),
            "alpha": trial.suggest_categorical("alpha", [0, 0.1, 0.5, 1.0]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 5, 10]),
            "n_estimators": trial.suggest_categorical("n_estimators", [1000, 2000]),
            "objective": trial.suggest_categorical("objective", OBJECTIVES),
        }
        window = trial.suggest_categorical("window", WINDOWS)

        name = f"xgb_sentinel_{trial.number+1:02d}"
        result = evaluate_xgb_config(params, window, df, feat_cols, all_days, search_days, confirm_days, name)
        store_result(result)

        if result is None:
            return 99.0
        score = result["full_smape"]
        if result["search_smape"] and result["confirm_smape"]:
            gap = result["confirm_smape"] - result["search_smape"]
            if gap > 3.0:
                score += 1.0
        return score

    logger.info(f"\nXGBoost sentinel: {args.n_trials} trials")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # ── Save ──
    for name, result in all_results.items():
        path = output_root / "predictions" / f"{name}_dayahead.csv"
        result["predictions"].to_csv(str(path), index=False, encoding="utf-8-sig")

    config_df = pd.DataFrame(config_log).sort_values("full_smape")
    config_df.to_csv(str(output_root / "metrics" / "config_search_results.csv"),
                     index=False, encoding="utf-8-sig")

    summary_rows = []
    for name, result in all_results.items():
        full = result["predictions"]
        m = compute_all_metrics(full["y_true"].values, full["y_pred"].values)
        m["model_name"] = name
        m["search_smape"] = result["search_smape"]
        m["confirm_smape"] = result["confirm_smape"]
        m["window"] = result["window"]
        summary_rows.append(m)
    summary_df = pd.DataFrame(summary_rows).sort_values("sMAPE_floor50")
    summary_df.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    # Hour/period metrics for best
    if len(summary_df) > 0:
        best_name = summary_df.iloc[0]["model_name"]
        best_full = all_results[best_name]["predictions"]
        hour_rows = []
        for h, grp in best_full.groupby("hour_business"):
            m = compute_all_metrics(grp["y_true"].values, grp["y_pred"].values)
            m["hour_business"] = h
            hour_rows.append(m)
        pd.DataFrame(hour_rows).to_csv(str(output_root / "metrics" / "hour_metrics.csv"),
                                        index=False, encoding="utf-8-sig")
        period_rows = []
        for p, grp in best_full.groupby("period"):
            m = compute_all_metrics(grp["y_true"].values, grp["y_pred"].values)
            m["period"] = p
            period_rows.append(m)
        pd.DataFrame(period_rows).to_csv(str(output_root / "metrics" / "period_metrics.csv"),
                                          index=False, encoding="utf-8-sig")

    # ── Report ──
    lines = []
    def w(s=""):
        lines.append(s)

    best_s = config_df.iloc[0]["full_smape"] if len(config_df) > 0 else 99.0

    w("# XGBoost Sentinel Day-Ahead Report")
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Trials**: {args.n_trials}")
    w(f"**Windows**: {WINDOWS}")
    w()
    w("## Results")
    w("| Config | Full sMAPE | Search | Confirm | Window | Obj | Depth | LR |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in config_df.iterrows():
        s_str = f"{r['search_smape']:.2f}%" if r['search_smape'] else "N/A"
        c_str = f"{r['confirm_smape']:.2f}%" if r['confirm_smape'] else "N/A"
        w(f"| {r['config']} | {r['full_smape']:.2f}% | {s_str} | {c_str} | {r['window']} | {r['objective']} | {r['max_depth']} | {r['learning_rate']} |")
    w()
    w("## Continue/Stop Decision")
    w(f"- Best XGBoost: {best_s:.4f}%")
    w(f"- Continue threshold: full_30d <= 11.5% or confirm advantage over LightGBM")
    w(f"- Stop threshold: full_30d > 11.8% and no hour advantage")
    if best_s <= 11.5:
        w("- **DECISION: CONTINUE** - XGBoost below 11.5%")
    elif best_s > 11.8:
        w("- **DECISION: STOP** - XGBoost above 11.8%, no advantage")
    else:
        w("- **DECISION: MARGINAL** - Between 11.5% and 11.8%, check hour metrics")
    w()

    report = "\n".join(lines)
    report_path = output_root / "reports" / "dayahead_xgboost_sentinel_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    docs_report = Path(_PROJECT_DIR) / "docs" / "reports" / "dayahead_xgboost_sentinel_report.md"
    with open(str(docs_report), "w", encoding="utf-8") as f:
        f.write(report)

    manifest = {
        "n_trials": args.n_trials,
        "best_full_smape": config_df.iloc[0]["full_smape"] if len(config_df) > 0 else None,
        "decision": "continue" if best_s <= 11.5 else ("stop" if best_s > 11.8 else "marginal"),
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nXGBoost Sentinel: best={best_s:.4f}%, decision={manifest['decision']}")


if __name__ == "__main__":
    main()
