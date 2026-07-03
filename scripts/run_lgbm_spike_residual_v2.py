"""
run_lgbm_spike_residual_v2.py — LightGBM spike residual v2 with asymmetric/hour-aware/guardrail correction.
"""

import logging, os, sys, json, time, yaml
from datetime import datetime, timedelta
from collections import defaultdict

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

from catboost import CatBoostRegressor

_OUTPUT = os.path.join(_PROJECT_DIR, "outputs", "dayahead_lgbm_corrections_v2_30d")
os.makedirs(os.path.join(_OUTPUT, "predictions"), exist_ok=True)
os.makedirs(os.path.join(_OUTPUT, "metrics"), exist_ok=True)
os.makedirs(os.path.join(_OUTPUT, "reports"), exist_ok=True)


def load_data_with_features():
    """Load original data, build features, and align with LGBM predictions."""
    # Load original data
    with open(os.path.join(_PROJECT_DIR, "configs", "paths.yaml"), encoding="utf-8") as f:
        data_path = yaml.safe_load(f)["default_data"]
    df = load_data(data_path, target="dayahead")
    df = build_features_dayahead(df)
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Load LGBM predictions
    lgbm_path = os.path.join(_PROJECT_DIR, "outputs", "dayahead_lgbm_90d",
                               "predictions", "lightgbm_90d_high_leaf_dayahead.csv")
    lgbm = pd.read_csv(lgbm_path)
    lgbm["ds"] = pd.to_datetime(lgbm["ds"])
    
    # Merge LGBM predictions onto feature data
    feat_cols = [c for c in df.columns if c not in ("ds",)]
    merged = df[["ds"] + feat_cols].merge(
        lgbm[["ds", "y_pred"]], on="ds", how="inner"
    )
    merged["y_true"] = merged["y"].values
    merged["hour_business"] = merged["hour"]
    merged["period"] = merged["hour_business"].apply(lambda h: "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24")
    merged["target_day"] = merged["ds"].dt.date.astype(str)
    
    # Fix: target_day should be the business day, not the exact timestamp
    # For hour 1-23: target_day = ds.date
    # For hour 24 (00:00 D+1): target_day = ds.date - 1 day
    merged["target_day"] = merged.apply(
        lambda r: (r["ds"] - timedelta(days=1)).strftime("%Y-%m-%d") if r["hour_business"] == 24 else r["ds"].strftime("%Y-%m-%d"),
        axis=1
    )
    
    return merged


CORRECTION_FEATURES = [
    "hour", "load", "net_load", "bidding_space", "bidding_space_raw",
    "wind", "solar", "renew_penetration", "ramp_load",
    "lag_price_target", "lag_price_week", "is_weekend",
    "same_hour_mean_7d", "same_hour_std_7d", "same_hour_max_7d",
    "price_momentum_24_168", "net_load_rank_30d", "bidding_space_rank_30d",
    "is_spring_festival_window",
]


def train_residual_corrector(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Train CatBoost residual corrector in rolling fashion, returning corrected predictions."""
    dates = sorted(df["target_day"].unique())
    n = len(df)
    y_true = df["y_true"].values
    y_base = df["y_pred"].values  # LightGBM base prediction
    residual = y_true - y_base
    
    # Base correction (CatBoost rolling)
    y_corrected = y_base.copy()
    valid_days = 0
    
    avail_feats = [c for c in CORRECTION_FEATURES if c in df.columns]
    logger.info(f"  Correction features ({len(avail_feats)}): {avail_feats}")
    
    for i, d in enumerate(dates):
        if i < 2:
            continue  # need at least 2 days of history
        past = df[df["target_day"] < d]
        today = df[df["target_day"] == d]
        if len(past) < 48 or len(today) == 0:
            continue
        
        X_train = past[avail_feats].fillna(0).values
        y_train = residual[past.index]
        X_test = today[avail_feats].fillna(0).values
        
        cbr = CatBoostRegressor(iterations=300, depth=5, learning_rate=0.03,
                                 verbose=0, random_seed=42)
        cbr.fit(X_train, y_train)
        corr = cbr.predict(X_test)
        y_corrected[today.index] = y_base[today.index] + corr
        valid_days += 1
    
    base_metrics = compute_all_metrics(y_true, y_base)
    v1_metrics = compute_all_metrics(y_true, y_corrected)
    logger.info(f"  Base: {base_metrics['sMAPE_floor50']:.4f}% -> V1: {v1_metrics['sMAPE_floor50']:.4f}%")
    
    return y_corrected, {"base": base_metrics, "v1": v1_metrics, "valid_days": valid_days}


def apply_asymmetric(y_pred: np.ndarray, residual: np.ndarray, 
                     alpha_up: float, alpha_down: float) -> np.ndarray:
    """Apply asymmetric correction: boost positive residuals, dampen negative."""
    result = y_pred.copy()
    pos_mask = residual > 0
    neg_mask = residual < 0
    result[pos_mask] = y_pred[pos_mask] + residual[pos_mask] * (alpha_up - 1)
    result[neg_mask] = y_pred[neg_mask] + residual[neg_mask] * (alpha_down - 1)
    return result


def apply_hour_aware(y_pred: np.ndarray, residual: np.ndarray,
                     df: pd.DataFrame, alphas: dict) -> np.ndarray:
    """Apply hour-aware correction strengths."""
    result = y_pred.copy()
    for hour, alpha in alphas.items():
        hour_mask = df["hour_business"] == hour
        if hour_mask.sum() > 0:
            result[hour_mask] = y_pred[hour_mask] + residual[hour_mask] * alpha
    return result


def apply_guardrail(y_pred: np.ndarray, base_pred: np.ndarray,
                    df: pd.DataFrame, max_delta_high: float,
                    max_delta_low: float) -> np.ndarray:
    """Limit correction magnitude based on predicted price quantile."""
    result = y_pred.copy()
    delta = result - base_pred
    threshold = np.percentile(base_pred, 90)
    high_mask = base_pred >= threshold
    low_mask = ~high_mask
    
    over_high = np.abs(delta) > max_delta_high
    over_low = np.abs(delta) > max_delta_low
    
    result[high_mask & over_high] = base_pred[high_mask & over_high] + np.sign(delta[high_mask & over_high]) * max_delta_high
    result[low_mask & over_low] = base_pred[low_mask & over_low] + np.sign(delta[low_mask & over_low]) * max_delta_low
    return result


def apply_no_worsen(y_pred: np.ndarray, base_pred: np.ndarray,
                    df: pd.DataFrame, past_hour_metrics: dict) -> np.ndarray:
    """Revert to base prediction for hours that got worse."""
    result = y_pred.copy()
    for hour, smape_diff in past_hour_metrics.items():
        if smape_diff > 0:  # correction made it worse
            hour_mask = df["hour_business"] == hour
            result[hour_mask] = base_pred[hour_mask]
    return result


def rolling_v2_search(df: pd.DataFrame) -> np.ndarray:
    """Apply v2 strategies in rolling fashion with parameter search."""
    dates = sorted(df["target_day"].unique())
    y_true = df["y_true"].values
    y_base = df["y_pred"].values
    n = len(df)
    y_v2 = y_base.copy()
    
    avail_feats = [c for c in CORRECTION_FEATURES if c in df.columns]
    
    # Parameter grid
    alpha_ups = [0.5, 0.75, 1.0, 1.25]
    alpha_downs = [0.1, 0.25, 0.5]
    hour_alphas = {h: [0.25, 0.5, 0.75, 1.0] for h in [10, 12, 13, 18, 1]}
    max_deltas_high = [100, 150, 200]
    max_deltas_low = [30, 50, 80]
    
    # Track hour performance for no-worsen filter
    hour_performance = {}  # hour -> list of sMAPE improvements
    
    for i, d in enumerate(dates):
        if i < 5:
            continue  # need enough history
        past = df[df["target_day"] < d]
        today = df[df["target_day"] == d]
        if len(past) < 72 or len(today) == 0:
            continue
        
        # Train CatBoost on past residuals
        X_train = past[avail_feats].fillna(0).values
        y_train = (y_true[past.index] - y_base[past.index])
        X_test = today[avail_feats].fillna(0).values
        
        cbr = CatBoostRegressor(iterations=300, depth=5, learning_rate=0.03,
                                 verbose=0, random_seed=42)
        cbr.fit(X_train, y_train)
        raw_corr = cbr.predict(X_test)
        raw_pred = y_base[today.index] + raw_corr
        
        # Evaluate on PAST to find best params (no future leakage)
        past_pred = y_base[past.index] + cbr.predict(X_train)
        past_true = y_true[past.index]
        
        # Asymmetric alpha search (on past data)
        best_au, best_ad = 1.0, 0.5
        best_smape = float("inf")
        for au in alpha_ups:
            for ad in alpha_downs:
                test_pred = apply_asymmetric(past_pred, past_true - past_pred, au, ad)
                m = compute_all_metrics(past_true, test_pred)
                if m["sMAPE_floor50"] < best_smape:
                    best_smape = m["sMAPE_floor50"]
                    best_au, best_ad = au, ad
        
        # Apply asymmetric to current day
        corrected_pred = apply_asymmetric(raw_pred, raw_corr, best_au, best_ad)
        
        # Hour-aware search
        best_hour_alphas = {}
        for h in [10, 12, 13, 18, 1]:
            ha_best, ha_smape = 1.0, float("inf")
            for ha in hour_alphas[h]:
                h_mask_past = past["hour_business"] == h
                if h_mask_past.sum() < 5:
                    continue
                hp = past_pred[h_mask_past.values]
                ht = past_true[past.index][h_mask_past.values]
                hc = hp + (ht - hp) * ha
                m = compute_all_metrics(ht, hc)
                if m["sMAPE_floor50"] < ha_smape:
                    ha_smape = m["sMAPE_floor50"]
                    ha_best = ha
            best_hour_alphas[h] = ha_best
        
        corrected_pred = apply_hour_aware(corrected_pred, raw_corr, today, best_hour_alphas)
        
        # Guardrail search
        best_mh, best_ml = 200, 80
        best_gs = float("inf")
        for mh in max_deltas_high:
            for ml in max_deltas_low:
                gp = apply_guardrail(past_pred, y_base[past.index], past, mh, ml)
                m = compute_all_metrics(past_true, gp)
                if m["sMAPE_floor50"] < best_gs:
                    best_gs = m["sMAPE_floor50"]
                    best_mh, best_ml = mh, ml
        
        corrected_pred = apply_guardrail(corrected_pred, y_base[today.index], today, best_mh, best_ml)
        
        # No-worsen hour filter
        hour_smape_base = {}
        hour_smape_corrected = {}
        for h in range(1, 25):
            h_mask = past["hour_business"] == h
            if h_mask.sum() < 3:
                continue
            hm_base = compute_all_metrics(y_true[past.index][h_mask.values], y_base[past.index][h_mask.values])
            hm_corr = compute_all_metrics(y_true[past.index][h_mask.values], past_pred[h_mask.values])
            if hm_base["sMAPE_floor50"] < hm_corr["sMAPE_floor50"]:
                # Correction made it worse, revert this hour
                h_mask_today = today["hour_business"] == h
                corrected_pred[h_mask_today.values] = y_base[today.index][h_mask_today.values]
        
        y_v2[today.index] = corrected_pred
        if (i + 1) % 5 == 0:
            seg_m = compute_all_metrics(y_true[df["target_day"] <= d], y_v2[df["target_day"] <= d])
            logger.info(f"  [{i+1}/{len(dates)}] {d}: sMAPE={seg_m['sMAPE_floor50']:.2f}%")
    
    return y_v2


def main():
    logger.info("=" * 60)
    logger.info("LightGBM Spike Residual v2")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading data and features...")
    df = load_data_with_features()
    logger.info(f"  Data: {len(df)} rows, {df['target_day'].nunique()} days")
    logger.info(f"  Available: hour={24 in df['hour'].values}")
    
    y_true = df["y_true"].values
    y_base = df["y_pred"].values
    
    # Compute base metrics
    base_m = compute_all_metrics(y_true, y_base)
    logger.info(f"  LightGBM base sMAPE: {base_m['sMAPE_floor50']:.4f}%")
    
    # V1 rolling correction (CatBoost)
    y_v1, v1_info = train_residual_corrector(df)
    v1_m = v1_info["v1"]
    
    # V2 with all strategies
    logger.info("Running V2 correction (asymmetric + hour-aware + guardrail + no-worsen)...")
    y_v2 = rolling_v2_search(df)
    v2_m = compute_all_metrics(y_true, y_v2)
    logger.info(f"  V2 sMAPE: {v2_m['sMAPE_floor50']:.4f}%")
    
    # Save predictions
    out_df = pd.DataFrame({
        "ds": df["ds"], "y_true": y_true, "y_pred": y_v2,
        "hour_business": df["hour_business"], "period": df["period"],
        "target_day": df["target_day"], "task": "dayahead",
        "model_name": "lgbm_spike_residual_v2",
    })
    out_path = os.path.join(_OUTPUT, "predictions", "lgbm_spike_residual_v2_dayahead.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"  Saved: {out_path} ({len(out_df)} rows)")
    
    # Hour and period metrics
    hour_rows = []
    for h in range(1, 25):
        h_mask = df["hour_business"] == h
        if h_mask.sum() > 0:
            mb = compute_all_metrics(y_true[h_mask], y_base[h_mask])
            mv = compute_all_metrics(y_true[h_mask], y_v2[h_mask])
            hour_rows.append({
                "hour": h, "base_smape": mb["sMAPE_floor50"],
                "v2_smape": mv["sMAPE_floor50"],
                "improvement": mb["sMAPE_floor50"] - mv["sMAPE_floor50"],
                "n": int(h_mask.sum()),
            })
    hour_df = pd.DataFrame(hour_rows)
    hour_df.to_csv(os.path.join(_OUTPUT, "metrics", "hour_metrics.csv"), index=False, encoding="utf-8-sig")
    
    period_rows = []
    for p in ["1_8", "9_16", "17_24"]:
        p_mask = df["period"] == p
        if p_mask.sum() > 0:
            mb = compute_all_metrics(y_true[p_mask], y_base[p_mask])
            mv = compute_all_metrics(y_true[p_mask], y_v2[p_mask])
            period_rows.append({
                "period": p, "base_smape": mb["sMAPE_floor50"],
                "v2_smape": mv["sMAPE_floor50"],
                "improvement": mb["sMAPE_floor50"] - mv["sMAPE_floor50"],
                "n": int(p_mask.sum()),
            })
    period_df = pd.DataFrame(period_rows)
    period_df.to_csv(os.path.join(_OUTPUT, "metrics", "period_metrics.csv"), index=False, encoding="utf-8-sig")
    
    # Summary
    summary = pd.DataFrame([
        {"model": "lightgbm_base", "sMAPE_floor50": base_m["sMAPE_floor50"], "MAE": base_m["MAE"], "RMSE": base_m["RMSE"]},
        {"model": "lgbm_spike_residual_v1", "sMAPE_floor50": v1_m["sMAPE_floor50"], "MAE": 0, "RMSE": 0},
        {"model": "lgbm_spike_residual_v2", "sMAPE_floor50": v2_m["sMAPE_floor50"], "MAE": v2_m["MAE"], "RMSE": v2_m["RMSE"]},
    ])
    summary.to_csv(os.path.join(_OUTPUT, "metrics", "summary.csv"), index=False, encoding="utf-8-sig")
    
    # Report
    beats_v1 = v2_m["sMAPE_floor50"] < v1_m["sMAPE_floor50"]
    below_11 = v2_m["sMAPE_floor50"] < 11
    below_105 = v2_m["sMAPE_floor50"] < 10.5
    
    improved_hours = hour_df[hour_df["improvement"] > 0].sort_values("improvement", ascending=False)
    worsened_hours = hour_df[hour_df["improvement"] < 0].sort_values("improvement")
    
    report = f"""# LightGBM Spike Residual v2 Report
> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. Summary

| Model | sMAPE |
|---|---|
| LightGBM base | {base_m['sMAPE_floor50']:.4f}% |
| lgbm_spike_residual_v1 | {v1_m['sMAPE_floor50']:.4f}% |
| lgbm_spike_residual_v2 | {v2_m['sMAPE_floor50']:.4f}% |

## 2. 结论

| 问题 | 回答 |
|---|---|
| V2 sMAPE | {v2_m['sMAPE_floor50']:.2f}% |
| 优于 11.27% | {'✅' if beats_v1 else '❌'} |
| 低于 11% | {'✅' if below_11 else '❌'} |
| 低于 10.5% | {'✅' if below_105 else '❌'} |
| 建议替换 v1 | {'✅' if beats_v1 else '❌'} |

## 3. Hour 改善 Top 5
{improved_hours.head(5).to_string(index=False) if len(improved_hours) > 0 else 'N/A'}

## 4. Hour 变差 Top 5
{worsened_hours.head(5).to_string(index=False) if len(worsened_hours) > 0 else 'N/A'}
"""
    
    report_path = os.path.join(_OUTPUT, "reports", "lgbm_spike_residual_v2_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved: {report_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
