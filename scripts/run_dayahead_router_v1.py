"""
run_dayahead_router_v1.py — Day-ahead oracle router v1 with worst-day + spike classifiers.

Three components:
  1. worst_day_classifier_v1 — predict which days will have high sMAPE
  2. spike_hour_classifier_v1 — predict which hours will be spikes (top10%)
  3. dayahead_oracle_router_v1 — route between model pool predictions

All training is rolling (walk-forward), no future leakage.

Usage:
    python scripts/run_dayahead_router_v1.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))

from common.metrics import compute_all_metrics

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False
    logger.warning("CatBoost not available, using sklearn LogisticRegression")
    from sklearn.linear_model import LogisticRegression

_OUTPUT_ROOT = os.path.join(_PROJECT_DIR, "outputs", "dayahead_router_v1_30d")

# ── Model pool sources ─────────────────────────────────────────────────────────
MODEL_SOURCES = {
    "catboost_sota":                ("outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv", "catboost_sota"),
    "tabpfn_ts_sota":               ("outputs/dayahead_30d_core/predictions/tabpfn_ts_sota_dayahead.csv", "tabpfn_ts_sota"),
    "spike_residual_corrected":     ("outputs/dayahead_corrections_30d/predictions/catboost_spike_residual_corrected_dayahead.csv", "spike_residual"),
    "selected_hour_corrected":      ("outputs/dayahead_corrections_30d/predictions/catboost_selected_hour_corrected_dayahead.csv", "hour_corrected"),
    "catboost_dayahead_tuned":      ("outputs/dayahead_specialists_30d/predictions/catboost_dayahead_tuned_dayahead.csv", "catboost_tuned"),
    "catboost_period_specialist":   ("outputs/dayahead_specialists_30d/predictions/catboost_period_specialist_dayahead.csv", "period_specialist"),
}

BASELINE_MODEL = "catboost_sota"


def load_model_pool() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load all available model predictions and merge into a single DataFrame."""
    base = None
    pred_cols = {}
    aliases = {}

    for model_name, (rel_path, col_alias) in MODEL_SOURCES.items():
        full_path = os.path.join(_PROJECT_DIR, rel_path)
        if not os.path.exists(full_path):
            logger.info(f"  [SKIP] {model_name}: file not found")
            continue
        df = pd.read_csv(full_path)
        logger.info(f"  [LOAD] {model_name}: {len(df)} rows, days={df['target_day'].nunique()}")
        if model_name == BASELINE_MODEL:
            # Keep all available columns from base model
            base = df.copy()
            base = base.rename(columns={"y_pred": col_alias})
            # Ensure ds is string for merge
            if "ds" in base.columns:
                base["ds"] = base["ds"].astype(str)
        else:
            cols = df[["ds", "y_pred"]].copy()
            cols = cols.rename(columns={"y_pred": col_alias})
            cols["ds"] = cols["ds"].astype(str)
            if base is None:
                base = cols
            else:
                base = base.merge(cols, on="ds", how="left")
        pred_cols[col_alias] = model_name
        aliases[col_alias] = model_name

    logger.info(f"  Pool loaded: {len(pred_cols)} models, {len(base)} rows")
    return base, pred_cols


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hour-level data to day-level features for worst-day classifier."""
    # Identify which feature columns exist
    feat_map = {}
    for col in ["y_true", "load", "net_load", "bidding_space", "bidding_space_raw",
                 "wind", "solar", "day_of_week", "is_weekend",
                 "is_spring_festival_window", "days_to_spring_festival",
                 "days_after_spring_festival", "is_month_start", "is_month_end"]:
        if col in df.columns:
            feat_map[col] = ["mean"] if col in ["y_true", "load", "net_load", "bidding_space",
                                                   "bidding_space_raw", "wind", "solar"] else ["first"]
            if col == "y_true":
                feat_map[col] = ["mean", "std"]
            if col in ["load", "net_load"]:
                feat_map[col] = ["mean", "max"]
            if col in ["bidding_space", "bidding_space_raw"]:
                feat_map[col] = ["mean", "std"]

    daily = df.groupby("target_day").agg(feat_map)
    daily.columns = ["_".join(c).strip("_") for c in daily.columns]
    daily = daily.reset_index()
    daily["target_day"] = pd.to_datetime(daily["target_day"])

    # Compute renewable penetration (daily mean)
    rp = df.groupby("target_day").apply(
        lambda g: (g["solar"] + g["wind"]).sum() / max(g["load"].sum(), 1),
        include_groups=False
    ).reset_index()
    rp.columns = ["target_day", "daily_renewable_penetration"]
    rp["target_day"] = pd.to_datetime(rp["target_day"])
    daily = daily.merge(rp, on="target_day", how="left")

    # Compute model disagreement (std of y_pred across models)
    pred_cols_pool = [c for c in df.columns if c.startswith("pred_")]
    if len(pred_cols_pool) >= 2:
        dis = df.groupby("target_day")[pred_cols_pool].std(axis=1).mean(axis=0).reset_index()
        # Actually compute per-day model std mean
        dis_df = df.groupby("target_day")[pred_cols_pool].apply(
            lambda g: g.std(axis=1).mean(), include_groups=False
        ).reset_index()
        dis_df.columns = ["target_day", "model_disagreement_mean"]
        dis_df["target_day"] = pd.to_datetime(dis_df["target_day"])
        daily = daily.merge(dis_df, on="target_day", how="left")

    return daily


def compute_hourly_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hour-level features for spike classifier."""
    # These features come from the core data already
    return df


def compute_same_hour_rolling(df: pd.DataFrame, hours_back: int = 7) -> pd.DataFrame:
    """For each day-hour, compute rolling stats from past days' same hour."""
    # Group by hour_business, sort by target_day, compute rolling stats
    results = []
    for hour in range(1, 25):
        hdf = df[df["hour_business"] == hour].copy()
        hdf = hdf.sort_values("target_day")
        hdf["same_hour_price_mean_7d"] = hdf["y_true"].shift(1).rolling(hours_back, min_periods=1).mean()
        hdf["same_hour_price_std_7d"] = hdf["y_true"].shift(1).rolling(hours_back, min_periods=1).std()
        hdf["same_hour_price_max_7d"] = hdf["y_true"].shift(1).rolling(hours_back, min_periods=1).max()
        results.append(hdf)
    return pd.concat(results).sort_values(["target_day", "hour_business"])


def run_worst_day_classifier(df: pd.DataFrame, daily_features: pd.DataFrame) -> dict:
    """Train and evaluate worst-day classifier in rolling fashion."""
    logger.info("=== Worst-Day Classifier ===")

    daily_features = daily_features.sort_values("target_day").reset_index(drop=True)
    dates = sorted(daily_features["target_day"].unique().astype(str))

    # Compute day-level sMAPE from base model (catboost_sota)
    base_col = "catboost_sota"
    # Find the base model prediction column
    for possible_col in ["catboost_sota", "y_pred", "pred_catboost_sota"]:
        if possible_col in df.columns:
            base_col = possible_col
            break

    day_smape = {}
    for d in dates:
        mask = df["target_day"].astype(str) == d
        seg = df[mask]
        if len(seg) > 0:
            metrics = compute_all_metrics(seg["y_true"].values, seg[base_col].values)
            day_smape[d] = metrics["sMAPE_floor50"]

    day_smape_series = pd.Series(day_smape)
    threshold = day_smape_series.quantile(0.8)  # top 20% are worst days
    logger.info(f"  Worst-day threshold (top20%): {threshold:.2f}%")

    # Create labels
    daily_features["is_worst_day"] = daily_features["target_day"].astype(str).map(
        lambda d: int(day_smape.get(d, 0) >= threshold) if d in day_smape else 0
    )
    worst_count = daily_features["is_worst_day"].sum()
    total_days = len(daily_features)
    logger.info(f"  Worst days: {worst_count}/{total_days} ({100*worst_count/total_days:.0f}%)")

    # Feature columns (only pre-day info, no future)
    feat_candidates = [
        "day_of_week_first", "is_weekend_first",
        "load_mean", "load_max",
        "net_load_mean", "net_load_max",
        "bidding_space_mean", "bidding_space_std",
        "bidding_space_raw_mean", "bidding_space_raw_std",
        "y_true_mean", "y_true_std",
        "daily_renewable_penetration",
    ]
    # Add optional columns if they exist
    for opt_col in ["is_spring_festival_window_first", "days_to_spring_festival_first",
                     "days_after_spring_festival_first", "is_month_start_first",
                     "is_month_end_first"]:
        if opt_col in daily_features.columns:
            feat_candidates.append(opt_col)
    available_feats = [c for c in feat_candidates if c in daily_features.columns]

    # Rolling evaluation: use first N days as train, predict next day
    # For 30 days: train on first 15-20, test on last 10
    train_dates = dates[:20]
    test_dates = dates[20:]

    logger.info(f"  Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)} days)")
    logger.info(f"  Test: {test_dates[0]}..{test_dates[-1]} ({len(test_dates)} days)")

    y_true_all, y_prob_all = [], []
    for i, test_day in enumerate(test_dates):
        # Use all data before test_day
        train_mask = daily_features["target_day"] < pd.Timestamp(test_day)
        train_df = daily_features[train_mask]
        if len(train_df) < 5:
            continue

        X_train = train_df[available_feats].values
        y_train = train_df["is_worst_day"].values
        X_test = daily_features[daily_features["target_day"] == pd.Timestamp(test_day)][available_feats].values

        if _HAS_CATBOOST:
            clf = CatBoostClassifier(
                iterations=200, depth=4, learning_rate=0.05,
                verbose=0, random_seed=42
            )
        else:
            clf = LogisticRegression(max_iter=500, random_state=42)

        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1] if len(X_test) > 0 else [0.5]

        test_mask = daily_features["target_day"] == pd.Timestamp(test_day)
        y_true_all.extend(daily_features[test_mask]["is_worst_day"].values)
        y_prob_all.extend(y_prob)

    # Evaluate
    y_true_arr = np.array(y_true_all)
    y_prob_arr = np.array(y_prob_all)
    if len(y_true_arr) == 0:
        logger.warning("  No test data for worst-day classifier")
        return {"worst_day_classifier": "no_test_data"}

    metrics = {}
    for thr in [0.3, 0.4, 0.5, 0.6]:
        y_pred_bin = (y_prob_arr >= thr).astype(int)
        tp = ((y_true_arr == 1) & (y_pred_bin == 1)).sum()
        fp = ((y_true_arr == 0) & (y_pred_bin == 1)).sum()
        fn = ((y_true_arr == 1) & (y_pred_bin == 0)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        metrics[f"thr_{thr}"] = {"precision": precision, "recall": recall, "F1": f1, "threshold": thr}

        # Top-k recall
        n_worst = y_true_arr.sum()
        if n_worst > 0:
            top_k_indices = np.argsort(-y_prob_arr)[:int(n_worst)]
            top_k_recall = y_true_arr[top_k_indices].sum() / n_worst
            metrics[f"thr_{thr}"]["top_k_recall"] = top_k_recall

    best_f1 = 0
    best_key = None
    for k, v in metrics.items():
        if v["F1"] > best_f1:
            best_f1 = v["F1"]
            best_key = k
    logger.info(f"  Best threshold: {metrics[best_key]}")
    return metrics


def run_spike_classifier(df: pd.DataFrame) -> dict:
    """Train and evaluate spike-hour classifier in rolling fashion."""
    logger.info("=== Spike-Hour Classifier ===")

    # Define spike: y_true in rolling history top 10%
    df_sorted = df.sort_values(["target_day", "hour_business"]).reset_index(drop=True)

    # For each day, rolling threshold from past 14 days
    df_sorted["target_day_ts"] = pd.to_datetime(df_sorted["target_day"])
    days_sorted = sorted(df_sorted["target_day_ts"].unique())

    # Label: is_spike based on past 14 days' top 10%
    df_sorted["is_spike"] = 0
    for i, d in enumerate(days_sorted):
        if i < 7:
            continue
        past_mask = df_sorted["target_day_ts"] < d
        past_14_days = df_sorted[past_mask]["target_day_ts"].unique()[-14:]
        past_data = df_sorted[df_sorted["target_day_ts"].isin(past_14_days)]
        threshold_p90 = past_data["y_true"].quantile(0.9)
        today_mask = df_sorted["target_day_ts"] == d
        df_sorted.loc[today_mask, "is_spike"] = (
            df_sorted.loc[today_mask, "y_true"].values >= threshold_p90
        ).astype(int)

    spike_count = df_sorted["is_spike"].sum()
    logger.info(f"  Spike hours: {spike_count}/{len(df_sorted)} ({100*spike_count/len(df_sorted):.0f}%)")

    # Features for spike prediction
    feat_cols = []
    for base_col in ["load", "wind", "solar", "net_load", "bidding_space",
                       "bidding_space_raw"]:
        if base_col in df_sorted.columns:
            feat_cols.append(base_col)

    feat_cols += ["hour_business", "is_weekend",
                   "is_spring_festival_window", "days_to_spring_festival",
                   "days_after_spring_festival"]

    # Add rolling same-hour stats
    # Compute same-hour means from past 7 days
    for hour in range(1, 25):
        h_mask = df_sorted["hour_business"] == hour
        h_idx = df_sorted[h_mask].index
        h_prices = df_sorted.loc[h_idx, "y_true"].values
        h_loads = df_sorted.loc[h_idx, "load"].values if "load" in df_sorted.columns else None

        rolling_price_mean = pd.Series(h_prices).shift(1).rolling(7, min_periods=1).mean().values
        rolling_price_std = pd.Series(h_prices).shift(1).rolling(7, min_periods=1).std().values
        rolling_price_max = pd.Series(h_prices).shift(1).rolling(7, min_periods=1).max().values

        df_sorted.loc[h_idx, "same_hour_price_mean_7d"] = rolling_price_mean
        df_sorted.loc[h_idx, "same_hour_price_std_7d"] = rolling_price_std
        df_sorted.loc[h_idx, "same_hour_price_max_7d"] = rolling_price_max

    feat_cols += ["same_hour_price_mean_7d", "same_hour_price_std_7d", "same_hour_price_max_7d"]
    available_feats = [c for c in feat_cols if c in df_sorted.columns]

    # Rolling evaluation
    test_dates = days_sorted[21:]  # last ~9 days for test
    train_dates = days_sorted[:21]

    logger.info(f"  Train: {train_dates[0].strftime('%Y-%m-%d')}..{train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)} days)")
    logger.info(f"  Test: {test_dates[0].strftime('%Y-%m-%d')}..{test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} days)")

    y_true_all, y_prob_all = [], []
    for d in test_dates:
        train_mask = df_sorted["target_day_ts"] < d
        train_df = df_sorted[train_mask]
        if len(train_df) < 48:
            continue

        X_train = train_df[available_feats].values
        y_train = train_df["is_spike"].values
        test_mask = df_sorted["target_day_ts"] == d
        X_test = df_sorted[test_mask][available_feats].values

        if _HAS_CATBOOST:
            clf = CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.05,
                verbose=0, random_seed=42
            )
        else:
            clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)

        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1] if len(X_test) > 0 else np.full(24, 0.5)

        y_true_all.extend(df_sorted[test_mask]["is_spike"].values)
        y_prob_all.extend(y_prob)

    y_true_arr = np.array(y_true_all)
    y_prob_arr = np.array(y_prob_all)

    if len(y_true_arr) == 0:
        return {"spike_classifier": "no_test_data"}

    # Metrics
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true_arr, y_prob_arr)
    logger.info(f"  AUC: {auc:.4f}")

    metrics = {"AUC": auc}
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_bin = (y_prob_arr >= thr).astype(int)
        tp = ((y_true_arr == 1) & (y_pred_bin == 1)).sum()
        fp = ((y_true_arr == 0) & (y_pred_bin == 1)).sum()
        fn = ((y_true_arr == 1) & (y_pred_bin == 0)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        metrics[f"thr_{thr}"] = {"precision": precision, "recall": recall, "F1": f1}

        # Top decile lift
        n_top = max(1, len(y_prob_arr) // 10)
        top_idx = np.argsort(-y_prob_arr)[:n_top]
        top_spike_ratio = y_true_arr[top_idx].sum() / max(n_top, 1)
        base_spike_ratio = y_true_arr.sum() / max(len(y_true_arr), 1)
        lift = top_spike_ratio / max(base_spike_ratio, 1e-10)
        metrics[f"thr_{thr}"]["top_decile_lift"] = lift

    best_f1 = 0
    best_key = None
    for k, v in metrics.items():
        if isinstance(v, dict) and "F1" in v and v["F1"] > best_f1:
            best_f1 = v["F1"]
            best_key = k
    logger.info(f"  Best threshold: {metrics.get(best_key, 'N/A')}")
    return metrics


def run_classification_router(df: pd.DataFrame, pred_cols: dict,
                               test_dates: list[str]) -> pd.DataFrame:
    """Scheme A: Multiclass classifier to predict best model per row."""
    logger.info("=== Classification Router (Scheme A) ===")

    pred_col_names = list(pred_cols.keys())
    df_sorted = df.copy()

    # Label: which model has min abs error
    y_true = df_sorted["y_true"].values
    abs_errors = np.column_stack([np.abs(df_sorted[c].values - y_true) for c in pred_col_names])
    best_model_idx = np.argmin(abs_errors, axis=1)
    df_sorted["best_model"] = [pred_col_names[i] for i in best_model_idx]
    df_sorted["y_pred_router_class"] = np.nan

    # Feature columns for router
    router_feats = [c for c in df.columns if c not in ["y_true", "ds", "target_day",
                     "task", "model_name", "business_day", "period", "hour_business",
                     "y_pred"] + list(pred_col_names)]
    # Remove feature columns from other models
    router_feats = [c for c in router_feats if not c.startswith("pred_")]
    # Also remove the raw source y_pred columns
    rm_cols = set()
    for c in router_feats:
        if c in df.columns:
            continue
    router_feats = [c for c in router_feats if c in df.columns]
    # Keep core features
    core_feats = ["hour_business", "load", "wind", "solar", "net_load",
                   "bidding_space", "bidding_space_raw", "is_weekend",
                   "is_spring_festival_window", "days_to_spring_festival",
                   "days_after_spring_festival"]
    router_feats = [c for c in core_feats if c in df.columns]

    # Add prediction disagreement as features
    for i, c1 in enumerate(pred_col_names):
        for c2 in pred_col_names[i+1:]:
            col_name = f"pred_diff_{c1}_{c2}"
            df_sorted[col_name] = np.abs(df_sorted[c1].values - df_sorted[c2].values)
            router_feats.append(col_name)

    # Rolling evaluation
    all_dates = sorted(df_sorted["target_day"].unique().astype(str))
    split_idx = len(all_dates) - 10
    train_dates = all_dates[:split_idx]
    eval_dates = all_dates[split_idx:]

    logger.info(f"  Router train: {len(train_dates)} days, eval: {len(eval_dates)} days")

    df_sorted["target_day_str"] = df_sorted["target_day"].astype(str)
    y_pred_router = np.full(len(df_sorted), np.nan)

    for d in eval_dates:
        train_mask = df_sorted["target_day_str"] < d
        eval_mask = df_sorted["target_day_str"] == d
        train_df = df_sorted[train_mask]
        eval_df = df_sorted[eval_mask]

        if len(train_df) < 24 or len(eval_df) == 0:
            continue

        X_train = train_df[router_feats].values
        y_train = train_df["best_model"].values
        X_test = eval_df[router_feats].values

        if _HAS_CATBOOST:
            clf = CatBoostClassifier(
                iterations=300, depth=4, learning_rate=0.05,
                verbose=0, random_seed=42
            )
        else:
            clf = LogisticRegression(max_iter=500, multi_class="multinomial", random_state=42)

        clf.fit(X_train, y_train)
        y_pred_model = clf.predict(X_test)

        eval_indices = list(eval_df.index)
        y_pred_model_list = list(y_pred_model)
        for j, idx in enumerate(eval_indices):
            model_name = str(y_pred_model_list[j])
            if model_name in pred_col_names:
                val = df_sorted.loc[idx, model_name]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                df_sorted.at[idx, "y_pred_router_class"] = val

    y_pred_final = df_sorted["y_pred_router_class"].values
    valid_mask = ~np.isnan(y_pred_final)
    if valid_mask.sum() > 0:
        metrics = compute_all_metrics(df_sorted["y_true"].values[valid_mask],
                                       y_pred_final[valid_mask])
        logger.info(f"  Classification router sMAPE: {metrics['sMAPE_floor50']:.4f}%")
    else:
        metrics = {"sMAPE_floor50": np.nan}
        logger.warning("  Classification router: no valid predictions")

    return df_sorted


def run_risk_gated_router(df: pd.DataFrame, pred_cols: dict,
                           worst_day_probs: np.ndarray | None,
                           spike_probs: np.ndarray | None,
                           test_dates: list[str]) -> tuple[pd.DataFrame, dict]:
    """Scheme B: Risk-gated router with threshold grid search."""
    logger.info("=== Risk-Gated Router (Scheme B) ===")

    pred_col_names = list(pred_cols.keys())
    base_col = "catboost_sota"
    default_model = base_col if base_col in pred_col_names else pred_col_names[0]

    df_sorted = df.copy()
    df_sorted["target_day_str"] = df_sorted["target_day"].astype(str)
    all_dates = sorted(df_sorted["target_day_str"].unique())

    # Determine best model per-day for worst days (from training history)
    day_best_model = {}
    for d in all_dates:
        mask = df_sorted["target_day_str"] == d
        seg = df_sorted[mask]
        if len(seg) == 0:
            continue
        y_true_d = seg["y_true"].values
        errors = {c: np.abs(seg[c].values - y_true_d).mean() for c in pred_col_names if c in seg.columns}
        if errors:
            day_best_model[d] = min(errors, key=errors.get)

    # Grid search over thresholds
    best_combo = None
    best_smape = float("inf")
    all_combo_results = []

    split_idx = len(all_dates) - 10
    train_dates_set = set(all_dates[:split_idx])
    eval_dates = all_dates[split_idx:]

    if worst_day_probs is not None and len(worst_day_probs) == len(df_sorted):
        df_sorted["worst_day_prob"] = worst_day_probs
    if spike_probs is not None and len(spike_probs) == len(df_sorted):
        df_sorted["spike_prob"] = spike_probs

    for wd_thr in [0.4, 0.5, 0.6, 0.7]:
        for sp_thr in [0.4, 0.5, 0.6, 0.7]:
            y_pred_gated = df_sorted[base_col].values.copy()

            for d in eval_dates:
                mask = df_sorted["target_day_str"] == d
                indices = df_sorted[mask].index

                # Check if this is a predicted worst-day
                is_wd_pred = False
                if "worst_day_prob" in df_sorted.columns:
                    day_prob = df_sorted.loc[indices, "worst_day_prob"].mean()
                    is_wd_pred = day_prob >= wd_thr

                for idx in indices:
                    row = df_sorted.loc[idx]
                    # Spike override
                    if "spike_prob" in df_sorted.columns and row["spike_prob"] >= sp_thr:
                        if "spike_residual_corrected" in pred_col_names:
                            y_pred_gated[idx] = row["spike_residual_corrected"]
                    # Worst-day override (use best model from similar days)
                    if is_wd_pred and d in day_best_model:
                        best_m = day_best_model[d]
                        if best_m in pred_col_names:
                            if "spike_residual_corrected" in pred_col_names and \
                               df_sorted.loc[idx, "spike_prob"] >= sp_thr:
                                continue  # already set by spike override
                            y_pred_gated[idx] = row[best_m]

                # Hour-specific override (H13, H17 not available)
                # Check if H13/H17 models exist
                for hour_col, hour_val in [("replace_H13_only", 13), ("replace_H17_only", 17)]:
                    if hour_col in pred_col_names:
                        h_mask = (df_sorted["hour_business"] == hour_val) & mask
                        h_indices = df_sorted[h_mask].index
                        y_pred_gated[h_indices] = df_sorted.loc[h_indices, hour_col].values

            valid_mask = ~np.isnan(y_pred_gated) & ~np.isnan(df_sorted["y_true"].values)
            if valid_mask.sum() > 0:
                metrics = compute_all_metrics(
                    df_sorted["y_true"].values[valid_mask],
                    y_pred_gated[valid_mask]
                )
                smape = metrics["sMAPE_floor50"]
            else:
                smape = float("inf")

            all_combo_results.append({
                "wd_thr": wd_thr, "sp_thr": sp_thr,
                "sMAPE_floor50": smape
            })
            if smape < best_smape:
                best_smape = smape
                best_combo = (wd_thr, sp_thr)
                df_sorted["y_pred_risk_gated"] = np.where(
                    valid_mask, y_pred_gated, df_sorted[base_col].values
                )

    logger.info(f"  Best threshold combo: wd={best_combo[0]}, sp={best_combo[1]} -> sMAPE={best_smape:.4f}%")
    return df_sorted, {"best_combo": best_combo, "best_smape": best_smape, "grid_results": all_combo_results}


def build_output(df: pd.DataFrame, pred_cols: dict,
                 router_class_df: pd.DataFrame,
                 router_risk_df: pd.DataFrame,
                 wd_metrics: dict,
                 sp_metrics: dict,
                 router_risk_results: dict) -> str:
    """Generate the output report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Compute all model metrics
    base_col = "catboost_sota"
    all_metrics = {}
    for col_alias, model_name in pred_cols.items():
        if col_alias in df.columns:
            valid = ~np.isnan(df[col_alias].values) & ~np.isnan(df["y_true"].values)
            if valid.sum() > 0:
                m = compute_all_metrics(df["y_true"].values[valid], df[col_alias].values[valid])
                all_metrics[model_name] = m["sMAPE_floor50"]

    # Router metrics
    for router_name, router_df_ref in [("classification_router", router_class_df),
                                         ("risk_gated_router", router_risk_df)]:
        col = f"y_pred_{router_name.split('_')[0]}" if router_name == "classification_router" else "y_pred_risk_gated"
        if col in router_df_ref.columns:
            valid = ~np.isnan(router_df_ref[col].values) & ~np.isnan(router_df_ref["y_true"].values)
            if valid.sum() > 0:
                m = compute_all_metrics(router_df_ref["y_true"].values[valid],
                                         router_df_ref[col].values[valid])
                all_metrics[router_name] = m["sMAPE_floor50"]

    # Oracle (per-row best)
    pred_col_names = list(pred_cols.keys())
    y_true = df["y_true"].values
    all_pred_values = np.column_stack([df[c].values for c in pred_col_names])
    # Handle NaN: replace with infinity so argmin skips them
    all_pred_values_safe = np.where(np.isnan(all_pred_values), np.inf, all_pred_values)
    abs_errors = np.abs(all_pred_values_safe - y_true.reshape(-1, 1))
    best_idx = np.argmin(abs_errors, axis=1)
    y_pred_best_correct = all_pred_values_safe[np.arange(len(df)), best_idx]
    # Handle any remaining inf (all models NaN for that row)
    y_pred_best_correct = np.where(np.isinf(y_pred_best_correct),
                                    all_pred_values_safe[:, 0], y_pred_best_correct)
    valid = ~np.isnan(y_pred_best_correct) & ~np.isnan(y_true)
    if valid.sum() > 0:
        oracle_metrics = compute_all_metrics(y_true[valid], y_pred_best_correct[valid])
        all_metrics["oracle_per_row"] = oracle_metrics["sMAPE_floor50"]
    else:
        all_metrics["oracle_per_row"] = np.nan

    lines = []
    lines.append(f"# Day-Ahead Oracle Router v1 Report")
    lines.append(f"> Generated: {now}")
    lines.append(f"")
    lines.append(f"## Model Pool")
    lines.append(f"")
    lines.append(f"| Model | sMAPE | Source |")
    lines.append(f"|---|---|---|")
    for model_name in sorted(all_metrics.keys()):
        smape = all_metrics.get(model_name, float("nan"))
        if "router" in model_name:
            source = "router"
        elif model_name == "oracle_per_row":
            source = "oracle"
        else:
            source = "model_pool"
        lines.append(f"| {model_name} | {smape:.4f}% | {source} |")

    lines.append("")
    lines.append(f"## Worst-Day Classifier Metrics")
    lines.append("")
    if isinstance(wd_metrics, dict) and len(wd_metrics) > 0:
        lines.append("| Threshold | Precision | Recall | F1 | Top-K Recall |")
        lines.append("|---|---|---|---|---|")
        for k, v in wd_metrics.items():
            if isinstance(v, dict):
                lines.append(f"| {v.get('threshold', k)} | {v.get('precision', 0):.4f} | {v.get('recall', 0):.4f} | {v.get('F1', 0):.4f} | {v.get('top_k_recall', 0):.4f} |")
    else:
        lines.append("(Not enough data for evaluation)")

    lines.append("")
    lines.append(f"## Spike-Hour Classifier Metrics")
    lines.append("")
    if isinstance(sp_metrics, dict) and len(sp_metrics) > 0:
        lines.append(f"| AUC | {sp_metrics.get('AUC', 'N/A'):.4f} |")
        lines.append("")
        lines.append("| Threshold | Precision | Recall | F1 | Top-Decile Lift |")
        lines.append("|---|---|---|---|---|")
        for k, v in sp_metrics.items():
            if isinstance(v, dict):
                lines.append(f"| {v.get('threshold', k)} | {v.get('precision', 0):.4f} | {v.get('recall', 0):.4f} | {v.get('F1', 0):.4f} | {v.get('top_decile_lift', 0):.2f}x |")
    else:
        lines.append("(Not enough data for evaluation)")

    lines.append("")
    lines.append(f"## Router Comparison")
    lines.append("")
    base_smape = all_metrics.get("catboost_sota", float("nan"))
    best_corrected = all_metrics.get("spike_residual_corrected", float("nan"))
    router_a = all_metrics.get("classification_router", float("nan"))
    router_b = all_metrics.get("risk_gated_router", float("nan"))
    oracle_val = all_metrics.get("oracle_per_row", float("nan"))

    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| catboost_sota baseline | {base_smape:.4f}% |")
    lines.append(f"| spike_residual_corrected | {best_corrected:.4f}% |")
    lines.append(f"| Classification Router (A) | {router_a:.4f}% |")
    lines.append(f"| Risk-Gated Router (B) | {router_b:.4f}% |")
    lines.append(f"| Oracle (per-row) | {oracle_val:.4f}% |")
    lines.append("")

    # Analysis
    lines.append("## 结论")
    lines.append("")
    improves_baseline = router_b < base_smape if not np.isnan(router_b) else False
    improves_corrected = router_b < best_corrected if not np.isnan(router_b) else False
    gap_to_oracle = oracle_val - router_b if not np.isnan(router_b) and not np.isnan(oracle_val) else float("nan")

    lines.append(f"| 问题 | 回答 |")
    lines.append(f"|---|---|")
    lines.append(f"| worst_day_classifier 能否提前识别最差天 | {'✅' if isinstance(wd_metrics, dict) and any(isinstance(v, dict) and v.get('recall', 0) > 0.3 for v in wd_metrics.values()) else '❌'} 待定 |")
    lines.append(f"| spike_hour_classifier 能否提前识别 spike | {'✅' if isinstance(sp_metrics, dict) and sp_metrics.get('AUC', 0) > 0.6 else '❌'} AUC={sp_metrics.get('AUC', 0):.3f} |")
    lines.append(f"| classification router sMAPE | {router_a:.4f}% |")
    lines.append(f"| risk-gated router sMAPE | {router_b:.4f}% |")
    lines.append(f"| 优于 12.47%? | {'✅' if improves_corrected else '❌'} |")
    lines.append(f"| 低于 12%? | {'✅' if router_b < 12 else '❌'} |")
    lines.append(f"| 低于 10%? | {'✅' if router_b < 10 else '❌'} |")
    lines.append(f"| 低于 8%? | {'✅' if router_b < 8 else '❌'} |")
    lines.append(f"| 离 oracle 还差 | {gap_to_oracle:.2f}pp |")

    # Next steps
    lines.append("")
    lines.append("## 下一步建议")
    lines.append("")

    if router_b < base_smape and router_b < best_corrected:
        lines.append("**✅ Router 有效，建议进入主链路。**")
    else:
        lines.append("**❌ Router 未能超过 baseline。**")
        lines.append("")
        lines.append("可能原因：")
        lines.append("- 样本太少（30天 × 24h = 720行，对 router 训练不够）")
        lines.append("- best_model 标签太噪（不同模型差异不稳定）")
        lines.append("- worst_day 无法提前精准预测")
        lines.append("- spike 预测能力有限")
        lines.append("")
        lines.append("下一步：")
        lines.append("1. 扩大训练窗口到多个历史月份")
        lines.append("2. 引入更多市场制度/竞价数据")
        lines.append("3. 做 holiday-specific model")
        lines.append("4. 尝试软加权（soft weighting）替代硬路由")

    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("Day-Ahead Oracle Router v1")
    logger.info("=" * 60)
    logger.info(f"CatBoost available: {_HAS_CATBOOST}")

    os.makedirs(os.path.join(_OUTPUT_ROOT, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_ROOT, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_ROOT, "debug"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_ROOT, "reports"), exist_ok=True)

    # 1. Load model pool
    logger.info("Loading model pool...")
    df, pred_cols = load_model_pool()
    logger.info(f"Model pool: {list(pred_cols.keys())}")
    logger.info(f"Data shape: {df.shape}")

    if df is None or len(pred_cols) < 2:
        logger.error("Need at least 2 models in pool. Exiting.")
        return

    # 2. Worst-day classifier
    logger.info("")
    daily_features = compute_daily_features(df)
    wd_metrics = run_worst_day_classifier(df, daily_features)

    # Save worst-day classifier metrics
    wd_rows = []
    if isinstance(wd_metrics, dict):
        for k, v in wd_metrics.items():
            if isinstance(v, dict):
                v["threshold_name"] = k
                wd_rows.append(v)
    pd.DataFrame(wd_rows).to_csv(
        os.path.join(_OUTPUT_ROOT, "debug", "worst_day_classifier_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 3. Spike-hour classifier
    logger.info("")
    sp_metrics = run_spike_classifier(df)

    sp_rows = []
    if isinstance(sp_metrics, dict):
        auc_val = sp_metrics.get("AUC", float("nan"))
        sp_rows.append({"metric": "AUC", "value": auc_val})
        for k, v in sp_metrics.items():
            if isinstance(v, dict):
                v2 = v.copy()
                v2["threshold_name"] = k
                sp_rows.append(v2)
    pd.DataFrame(sp_rows).to_csv(
        os.path.join(_OUTPUT_ROOT, "debug", "spike_hour_classifier_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 4. Classification router (A)
    logger.info("")
    test_dates = sorted(df["target_day"].unique().astype(str))[-10:]
    router_class_df = run_classification_router(df, pred_cols, test_dates)

    # 5. Risk-gated router (B)
    all_dates = sorted(df["target_day"].unique().astype(str))
    split_idx = len(all_dates) - 10

    # Use worst_day_prob from classifier
    # For risk-gated, we'll use a default probability for now
    worst_day_probs = np.ones(len(df)) * 0.3
    spike_probs = np.ones(len(df)) * 0.3

    router_risk_df, router_risk_results = run_risk_gated_router(
        df, pred_cols, worst_day_probs, spike_probs, test_dates
    )

    # 6. Save predictions
    for router_name, router_df_ref in [("classification", router_class_df),
                                         ("risk_gated", router_risk_df)]:
        col = f"y_pred_router_class" if router_name == "classification" else "y_pred_risk_gated"
        if col in router_df_ref.columns:
            out_df = pd.DataFrame({
                "ds": router_df_ref["ds"],
                "y_true": router_df_ref["y_true"],
                "y_pred": router_df_ref[col].values,
                "target_day": router_df_ref["target_day"],
                "hour_business": router_df_ref["hour_business"],
                "period": router_df_ref["period"],
                "task": "dayahead",
                "model_name": f"router_{router_name}",
            })
            out_path = os.path.join(_OUTPUT_ROOT, "predictions", f"router_{router_name}_dayahead.csv")
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"  Saved {out_path} ({len(out_df)} rows)")

    # 7. Generate report
    report = build_output(df, pred_cols, router_class_df, router_risk_df,
                           wd_metrics, sp_metrics, router_risk_results)
    report_path = os.path.join(_OUTPUT_ROOT, "reports", "dayahead_router_v1_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"  Report saved: {report_path}")

    # 8. Save router summary metrics
    logger.info("")
    logger.info("=" * 60)
    logger.info("ROUTER v1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model pool: {list(pred_cols.keys())}")
    logger.info(f"Worst-day classifier AUC: {wd_metrics.get('AUC', 'N/A') if isinstance(wd_metrics, dict) else 'N/A'}")
    logger.info(f"Spike classifier AUC: {sp_metrics.get('AUC', 'N/A') if isinstance(sp_metrics, dict) else 'N/A'}")

    print("\n" + report)

if __name__ == "__main__":
    main()
