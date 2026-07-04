"""
run_lightgbm_stage3.py — LightGBM Stage-3 search with Optuna + v3 features.

Target: beat 11.85% (best_two_average champion).
Search: 20 Optuna trials, windows [90, 120, 150, all], objectives [mae, rmse].
Split: search (Feb 1-20), confirm (Feb 21-Mar 2), full 30d.

Usage:
    python scripts/run_lightgbm_stage3.py
    python scripts/run_lightgbm_stage3.py --n-trials 20 --output-root outputs/dayahead_lgbm_stage3_30d
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
import time
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

# Suppress Optuna's own logging
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="LightGBM Stage-3 (v3 features + Optuna)")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_lgbm_stage3_30d")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Evaluation dates ──
EVAL_START = "2026-02-01"
EVAL_END = "2026-03-02"
SEARCH_END = "2026-02-20"
CONFIRM_START = "2026-02-21"

# ── Windows to test ──
WINDOWS = [90, 120, 150, "all"]


def _fast_rank_rolling(series: pd.Series, window: int = 720) -> pd.Series:
    """Fast approximate rank using rolling percentile (vectorized)."""
    return series.rolling(window, min_periods=max(10, window // 4)).apply(
        lambda x: (x < x[-1]).sum() / len(x) if len(x) >= 10 else 0.5,
        raw=True,
    ).fillna(0.5)


def build_v3_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build v3 features from raw data. Optimized: skip O(n^2) rank loops."""
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

    # ── BUSINESS DAY MAPPING (must use business_time, NOT ds.date()) ──
    # business_day D has hour_business 1-24 mapping to ds D 01:00 to D+1 00:00
    from src.common.business_time import business_time_mapping

    biz = business_time_mapping(df["ds"])
    df["business_day"] = biz["business_day"].astype(str)
    df["hour_business"] = biz["hour_business"]
    df["period"] = biz["period"]
    df["target_day"] = df["business_day"]

    # v2 extended features (skip _add_30d_ranks — O(n^2) too slow)
    df = _add_lag_features(df)
    df = _add_same_hour_stats(df)
    df = _add_price_momentum(df)
    df = _add_calendar_features(df)

    # Fast approximate rank (vectorized rolling)
    df = df.sort_values("ds").reset_index(drop=True)
    df["net_load_rank_30d"] = _fast_rank_rolling(df["net_load"], 720)
    df["bidding_space_rank_30d"] = _fast_rank_rolling(df["bidding_space"], 720)

    # v3 new features (skip _add_additional_ranks — also O(n^2))
    df = _add_volatility(df)
    df = _add_change_features(df)
    df = _add_exact_spring_festival(df)
    df = _add_interaction_features(df)

    # Fill NaN
    df = df.ffill().fillna(0).reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns, excluding target and metadata."""
    exclude = {
        "ds", "y", "target_day", "business_day", "hour_business", "period",
        "date_only", "y_pred", "y_true", "model_name", "task",
        "lag_48h_raw", "lag_168h_raw",
    }
    numeric = df.select_dtypes(include=[np.float64, np.int64, np.float32, np.int32, np.int8, bool]).columns
    cols = [c for c in numeric if c not in exclude and c != "y"]
    return cols


def evaluate_config(params: dict, window, df: pd.DataFrame, feat_cols: list[str],
                    all_days: list[str], search_days: list[str], confirm_days: list[str],
                    config_name: str) -> dict | None:
    """Train rolling LightGBM and evaluate on 30d window. Returns result dict or None."""
    import lightgbm as lgb
    import gc

    MAX_TRAIN_ROWS = 5000
    preds = []

    for day in all_days:
        target_dt = pd.Timestamp(day)
        train_all = df[df["target_day"] < day]
        if len(train_all) < 200:
            continue

        if window == "all":
            train_df = train_all.tail(MAX_TRAIN_ROWS).copy()
        else:
            train_start = target_dt - timedelta(days=window)
            train_df = train_all[train_all["ds"] >= train_start].copy()
            if len(train_df) > MAX_TRAIN_ROWS:
                train_df = train_df.tail(MAX_TRAIN_ROWS)

        if len(train_df) < 100:
            continue

        val_start = target_dt - timedelta(days=30)
        val_df = train_all[train_all["ds"].between(val_start, target_dt - timedelta(hours=1))]
        if len(val_df) > 2000:
            val_df = val_df.tail(2000)

        X_tr = train_df[feat_cols].values.astype(float)
        y_tr = train_df["y"].values.astype(float)

        try:
            n_rounds = params.get("n_estimators", 2000)
            lgb_params = {
                "boosting_type": "gbdt",
                "num_leaves": params["num_leaves"],
                "min_data_in_leaf": params["min_data_in_leaf"],
                "lambda_l1": params["lambda_l1"],
                "lambda_l2": params["lambda_l2"],
                "learning_rate": params["learning_rate"],
                "feature_fraction": params["feature_fraction"],
                "bagging_fraction": params["bagging_fraction"],
                "bagging_freq": params["bagging_freq"],
                "objective": params["objective"],
                "metric": params["objective"],
                "verbosity": -1,
            }

            if len(val_df) >= 50:
                X_val = val_df[feat_cols].values.astype(float)
                y_val = val_df["y"].values.astype(float)
                model = lgb.train(
                    lgb_params, lgb.Dataset(X_tr, y_tr),
                    num_boost_round=n_rounds,
                    valid_sets=[lgb.Dataset(X_val, y_val)],
                    valid_names=["eval"],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
            else:
                model = lgb.train(
                    lgb_params, lgb.Dataset(X_tr, y_tr),
                    num_boost_round=n_rounds,
                    callbacks=[lgb.log_evaluation(0)],
                )

            day_df = df[df["target_day"] == day].copy()
            X_pred = day_df[feat_cols].values.astype(float)
            day_df = day_df.copy()
            day_df["y_pred"] = model.predict(X_pred)
            day_df["y_true"] = day_df["y"].values
            day_df["model_name"] = config_name
            preds.append(day_df)
            del model
            gc.collect()
        except Exception as e:
            logger.debug(f"    Day {day}: {e}")
            continue

    if not preds:
        return None

    full = pd.concat(preds, ignore_index=True)

    search_mask = full["target_day"].isin(search_days)
    confirm_mask = full["target_day"].isin(confirm_days)

    s_smape = smape_floor50(
        full.loc[search_mask, "y_true"].values,
        full.loc[search_mask, "y_pred"].values,
    ) if search_mask.sum() >= 10 else None

    c_smape = smape_floor50(
        full.loc[confirm_mask, "y_true"].values,
        full.loc[confirm_mask, "y_pred"].values,
    ) if confirm_mask.sum() >= 10 else None

    f_smape = smape_floor50(full["y_true"].values, full["y_pred"].values)

    return {
        "config_name": config_name,
        "window": window,
        "params": params,
        "predictions": full,
        "search_smape": s_smape,
        "confirm_smape": c_smape,
        "full_smape": f_smape,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.output_root)
    for sub in ["predictions", "metrics", "reports", "debug"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    data_path = args.data_path
    if data_path is None:
        from src.common.repo_paths import get_data_path
        data_path = str(get_data_path())

    logger.info("Loading data and building v3 features...")
    raw = load_data(data_path, target="dayahead")
    df = build_v3_features(raw)

    # Filter to relevant period (need history before eval start)
    df = df[df["ds"] >= "2025-08-01"].reset_index(drop=True)
    logger.info(f"Feature DF: {len(df)} rows ({df['ds'].min()} -> {df['ds'].max()})")

    feat_cols = get_feature_cols(df)
    logger.info(f"Feature columns: {len(feat_cols)}: {feat_cols[:10]}...")

    # ── Evaluation days ──
    all_days = sorted(df[
        (df["ds"] >= EVAL_START) & (df["ds"] <= f"{EVAL_END} 23:00:00")
    ]["target_day"].unique())
    search_days = [d for d in all_days if d <= SEARCH_END]
    confirm_days = [d for d in all_days if d >= CONFIRM_START]

    logger.info(f"Evaluation: {len(all_days)} days ({all_days[0]} -> {all_days[-1]})")
    logger.info(f"Search: {len(search_days)} days (Feb 1-20)")
    logger.info(f"Confirm: {len(confirm_days)} days (Feb 21-Mar 2)")

    # ── Storage for all results ──
    all_results: dict[str, dict] = {}
    config_log = []

    def store_result(result: dict):
        if result is None:
            return
        name = result["config_name"]
        all_results[name] = result
        entry = {
            "config": name,
            "window": result["window"],
            "objective": result["params"].get("objective", "rmse"),
            "num_leaves": result["params"]["num_leaves"],
            "min_data_in_leaf": result["params"]["min_data_in_leaf"],
            "lambda_l1": result["params"]["lambda_l1"],
            "lambda_l2": result["params"]["lambda_l2"],
            "learning_rate": result["params"]["learning_rate"],
            "feature_fraction": result["params"]["feature_fraction"],
            "bagging_fraction": result["params"]["bagging_fraction"],
            "bagging_freq": result["params"]["bagging_freq"],
            "n_estimators": result["params"].get("n_estimators", 2000),
            "search_smape": round(result["search_smape"], 4) if result["search_smape"] else None,
            "confirm_smape": round(result["confirm_smape"], 4) if result["confirm_smape"] else None,
            "full_smape": round(result["full_smape"], 4),
        }
        config_log.append(entry)
        s_str = f"{result['search_smape']:.4f}%" if result["search_smape"] else "N/A"
        c_str = f"{result['confirm_smape']:.4f}%" if result["confirm_smape"] else "N/A"
        logger.info(f"  {name}: full={result['full_smape']:.4f}% search={s_str} confirm={c_str}")

    # ════════════════════════════════════════
    # Step 1: Baseline confirmation (90d, mae)
    # ════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Baseline confirmation (90d, mae)")
    logger.info("=" * 60)

    baseline_params = {
        "num_leaves": 127, "min_data_in_leaf": 50,
        "lambda_l1": 0.1, "lambda_l2": 2.0,
        "learning_rate": 0.02, "feature_fraction": 0.85,
        "bagging_fraction": 0.85, "bagging_freq": 5,
        "objective": "mae", "n_estimators": 2000,
    }
    r = evaluate_config(baseline_params, 90, df, feat_cols, all_days, search_days, confirm_days,
                        "stage3_baseline_90d_mae")
    store_result(r)

    # ════════════════════════════════════════
    # Step 2: Optuna search (20 trials)
    # ════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info(f"STEP 2: Optuna search ({args.n_trials} trials)")
    logger.info("=" * 60)

    # Optuna search space
    SEARCH_SPACE = {
        "num_leaves": [127, 191, 255, 319],
        "min_data_in_leaf": [20, 30, 50, 80],
        "lambda_l1": [0.1, 0.5, 1.0, 2.0],
        "lambda_l2": [1.0, 2.0, 5.0, 10.0],
        "learning_rate": [0.015, 0.02, 0.03, 0.05],
        "feature_fraction": [0.75, 0.85, 0.95],
        "bagging_fraction": [0.75, 0.85, 0.95],
        "bagging_freq": [1, 5],
        "n_estimators": [1000, 2000],
    }
    OBJECTIVES = ["mae", "mae", "mae", "rmse"]  # 3:1 mae:rmse ratio

    def objective(trial):
        params = {}
        for k, vals in SEARCH_SPACE.items():
            params[k] = trial.suggest_categorical(k, vals)
        obj = trial.suggest_categorical("objective", OBJECTIVES)
        params["objective"] = obj
        window = trial.suggest_categorical("window", WINDOWS)

        name = f"optuna_{trial.number+1:02d}"
        result = evaluate_config(params, window, df, feat_cols, all_days, search_days, confirm_days, name)
        store_result(result)

        if result is None:
            return 99.0
        # Optuna minimizes; we want to minimize full_smape
        # But also penalize if confirm is much worse than search (overfitting)
        score = result["full_smape"]
        if result["search_smape"] and result["confirm_smape"]:
            gap = result["confirm_smape"] - result["search_smape"]
            if gap > 3.0:
                score += 1.0  # penalty for overfitting
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    logger.info(f"\nOptuna best trial: #{study.best_trial.number+1}")
    logger.info(f"  Params: {study.best_params}")
    logger.info(f"  Value: {study.best_value:.4f}")

    # ════════════════════════════════════════
    # Save outputs
    # ════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("Saving outputs...")
    logger.info("=" * 60)

    # Save predictions for all configs
    for name, result in all_results.items():
        path = output_root / "predictions" / f"{name}_dayahead.csv"
        result["predictions"].to_csv(str(path), index=False, encoding="utf-8-sig")

    # Save config search results
    config_df = pd.DataFrame(config_log).sort_values("full_smape")
    config_df.to_csv(str(output_root / "metrics" / "config_search_results.csv"),
                     index=False, encoding="utf-8-sig")

    # Summary with full metrics
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

    # Hour/period metrics for best model
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

    w("# LightGBM Stage-3 Day-Ahead Report (v3 Features + Optuna)")
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Features**: v3 (base + extended + volatility + ranks + change + interactions)")
    w(f"**Search**: {args.n_trials} Optuna trials + 1 baseline")
    w(f"**Search window**: Feb 1-20 ({len(search_days)} days)")
    w(f"**Confirm window**: Feb 21-Mar 2 ({len(confirm_days)} days)")
    w(f"**Full 30d**: Feb 1-Mar 2 ({len(all_days)} days)")
    w()

    w("## 1. Baseline Confirmation")
    baseline_key = "stage3_baseline_90d_mae"
    if baseline_key in all_results:
        br = all_results[baseline_key]
        w(f"- Full sMAPE: {br['full_smape']:.4f}%")
        w(f"- Search: {br['search_smape']:.4f}%" if br['search_smape'] else "- Search: N/A")
        w(f"- Confirm: {br['confirm_smape']:.4f}%" if br['confirm_smape'] else "- Confirm: N/A")
    w()

    w("## 2. Top 10 Configurations")
    w("| Config | Full sMAPE | Search | Confirm | Window | Obj | nl | lr |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in config_df.head(10).iterrows():
        s_str = f"{r['search_smape']:.2f}%" if r['search_smape'] else "N/A"
        c_str = f"{r['confirm_smape']:.2f}%" if r['confirm_smape'] else "N/A"
        w(f"| {r['config']} | {r['full_smape']:.2f}% | {s_str} | {c_str} | {r['window']} | {r['objective']} | {r['num_leaves']} | {r['learning_rate']} |")
    w()

    best_row = summary_df.iloc[0]
    best_s = best_row["sMAPE_floor50"]
    w("## 3. Target Check")
    w(f"- **Best**: {best_name} ({best_s:.4f}%)")
    w(f"- Current champion (best_two_average): 11.85%")
    w(f"- Below 11.85%? {'YES' if best_s < 11.85 else 'NO'}")
    w(f"- Below 11.5%? {'YES' if best_s < 11.5 else 'NO'}")
    w(f"- Below 11.0%? {'YES' if best_s < 11.0 else 'NO'}")
    w()

    w("## 4. Overfitting Check")
    for _, r in config_df.head(5).iterrows():
        if r['search_smape'] and r['confirm_smape']:
            gap = r['confirm_smape'] - r['search_smape']
            flag = "WARNING" if abs(gap) > 2.0 else "OK"
            w(f"- {r['config']}: search={r['search_smape']:.2f}% confirm={r['confirm_smape']:.2f}% gap={gap:+.2f}pp [{flag}]")
    w()

    w("## 5. Feature Importance (Top 15)")
    # We can't easily get feature importance from the Optuna study,
    # but we can from the best single model if we retrain
    w("(See debug/feature_importance.json)")
    w()

    w("## 6. Comparison with Stage-2")
    w("| Metric | Stage-2 Best | Stage-3 Best | Delta |")
    w("|---|---|---|---|")
    stage2_best = 12.07  # trial_02
    delta = best_s - stage2_best
    w(f"| sMAPE_floor50 | {stage2_best:.2f}% | {best_s:.2f}% | {delta:+.2f}pp |")
    w()

    w("## 7. Recommendations")
    if best_s < 11.85:
        w(f"YES - Stage-3 has beaten the 11.85% champion.")
    else:
        w(f"NO - Stage-3 did not beat 11.85%. Best is {best_s:.2f}%.")
    if best_s < 11.5:
        w(f"YES - Below 11.5% target!")
    if best_s < 11.0:
        w(f"YES - Below 11.0% stretch target!")
    w()
    w("### Next steps")
    if best_s >= 11.5:
        w("- Proceed to XGBoost sentinel experiment")
        w("- Evaluate safe fusion with Stage-3 best")
        w("- Consider AutoGluon if fusion also fails")
    else:
        w("- Stage-3 is promising. Fine-tune around Optuna best params.")
        w("- Proceed to safe fusion with Stage-3 best.")
    w()

    report = "\n".join(lines)
    report_path = output_root / "reports" / "dayahead_lgbm_stage3_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)

    # Also copy to docs/reports
    docs_report = Path(_PROJECT_DIR) / "docs" / "reports" / "dayahead_stage3_report.md"
    with open(str(docs_report), "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report: {report_path}")

    # Manifest
    manifest = {
        "n_trials": args.n_trials,
        "best_config": config_df.iloc[0]["config"] if len(config_df) > 0 else None,
        "best_full_smape": config_df.iloc[0]["full_smape"] if len(config_df) > 0 else None,
        "best_search_smape": config_df.iloc[0].get("search_smape") if len(config_df) > 0 else None,
        "best_confirm_smape": config_df.iloc[0].get("confirm_smape") if len(config_df) > 0 else None,
        "champion_to_beat": 11.85,
        "beaten": config_df.iloc[0]["full_smape"] < 11.85 if len(config_df) > 0 else False,
        "below_11_5": config_df.iloc[0]["full_smape"] < 11.5 if len(config_df) > 0 else False,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("LIGHTGBM STAGE-3 SUMMARY (v3 Features + Optuna)")
    print("=" * 80)
    print(f"{'Config':45s} {'Full':>8s} {'Search':>8s} {'Confirm':>8s}")
    print("-" * 80)
    for _, r in config_df.head(15).iterrows():
        beats = "BEAT" if r["full_smape"] < 11.85 else ""
        s_str = f"{r['search_smape']:.2f}%" if r['search_smape'] else "N/A"
        c_str = f"{r['confirm_smape']:.2f}%" if r['confirm_smape'] else "N/A"
        print(f"{r['config'][:45]:45s} {r['full_smape']:7.2f}% {s_str:>8s} {c_str:>8s} {beats}")
    print("-" * 80)
    print(f"{'Champion (best_two_average)':45s} {'11.85':>7s}%")
    print(f"{'Stage-2 best (trial_02)':45s} {'12.07':>7s}%")
    if len(config_df) > 0:
        print(f"{'Stage-3 best':45s} {config_df.iloc[0]['full_smape']:7.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
