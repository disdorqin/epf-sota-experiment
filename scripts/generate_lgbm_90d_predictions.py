"""
generate_lgbm_90d_predictions.py — Generate LightGBM 90d high_leaf predictions and all downstream analysis.

Usage:
    D:/computer_download/environment/conda/epf-2/python.exe scripts/generate_lgbm_90d_predictions.py
"""

import logging, os, sys, json, time
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
from src.models.lightgbm_dayahead_adapter import LightGBMDayaheadAdapter

OUTPUT = os.path.join(_PROJECT_DIR, "outputs")

WINDOW_DAYS = 90
CONFIG = "high_leaf_regularized"
N_ROUNDS = 2000

# ── 1. Generate LightGBM predictions ──────────────────────────────────────
def gen_lgbm_predictions():
    """Generate LightGBM 90d high_leaf predictions."""
    import yaml
    yaml_path = os.path.join(_PROJECT_DIR, "configs", "paths.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_path = cfg["default_data"]

    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path, target="dayahead")
    logger.info(f"  Raw: {len(df)} rows")

    logger.info("Building enhanced features...")
    df = build_features_dayahead(df)
    logger.info(f"  Features: {len(df)} rows, {len(df.columns)} cols")

    df["ds"] = pd.to_datetime(df["ds"])

    dates = pd.date_range("2026-02-01", "2026-03-02", freq="D")
    dates_str = [d.strftime("%Y-%m-%d") for d in dates]

    all_days = []
    for i, target_date in enumerate(dates_str):
        t0 = time.time()
        target_dt = pd.Timestamp(target_date)
        train_start = target_dt - timedelta(days=WINDOW_DAYS)
        train_end = target_dt - timedelta(hours=1)

        train_mask = (df["ds"] >= train_start) & (df["ds"] < train_end)
        train_df = df[train_mask].copy()

        # Validation = last 7 days
        val_start = target_dt - timedelta(days=7)
        val_mask = (df["ds"] >= val_start) & (df["ds"] < train_end)
        val_df = df[val_mask].copy()

        if len(train_df) < 100:
            logger.warning(f"  {target_date}: only {len(train_df)} train rows, skipping")
            continue

        adapter = LightGBMDayaheadAdapter(config_name=CONFIG)
        params = adapter.params.copy()
        params["num_boost_round"] = N_ROUNDS
        params["early_stopping_rounds"] = 50
        # Remove GPU for full accuracy
        for k in ["device", "gpu_platform_id", "gpu_device_id"]:
            params.pop(k, None)
        adapter.params = params

        adapter.train(train_df, val_df)

        # Predict target day
        start_ds = target_dt + timedelta(hours=1)
        end_ds = target_dt + timedelta(days=1)
        mask = (df["ds"] >= start_ds) & (df["ds"] < end_ds)
        day_df = df[mask].copy()

        if len(day_df) == 0:
            logger.warning(f"  {target_date}: no prediction data")
            continue

        y_pred = adapter.predict(day_df)
        day_df["y_pred"] = y_pred
        day_df["y_true"] = day_df.get("y", np.nan)
        day_df["task"] = "dayahead"
        day_df["model_name"] = "lightgbm_90d_high_leaf"
        day_df["hour_business"] = ((day_df["ds"].dt.hour + 23) % 24 + 1)
        day_df["period"] = day_df["hour_business"].apply(lambda h: "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24")
        day_df["target_day"] = target_date
        day_df["business_day"] = target_date

        out = day_df[["ds", "y_true", "y_pred", "hour_business", "period",
                       "business_day", "target_day", "task", "model_name"]].copy()
        all_days.append(out)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            seg_metrics = compute_all_metrics(out["y_true"].values, out["y_pred"].values)
            logger.info(f"  [{i+1}/{len(dates_str)}] {target_date}: {seg_metrics['sMAPE_floor50']:.2f}% ({elapsed:.1f}s)")

    if all_days:
        merged = pd.concat(all_days, ignore_index=True).sort_values(["target_day", "hour_business"]).reset_index(drop=True)
        out_dir = os.path.join(OUTPUT, "dayahead_lgbm_90d", "predictions")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "lightgbm_90d_high_leaf_dayahead.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"  Saved {out_path} ({len(merged)} rows)")

        # Compute overall metrics
        metrics = compute_all_metrics(merged["y_true"].values, merged["y_pred"].values)
        logger.info(f"  Overall sMAPE: {metrics['sMAPE_floor50']:.4f}%")
        return merged
    return pd.DataFrame()

# ── 2. Run model pool + oracle ─────────────────────────────────────────────
def run_model_pool(df_lgbm):
    """Load all model predictions, compute unified ranking and oracle."""
    logger.info("\n=== Model Pool v2 ===")

    sources = {
        "catboost_sota": "outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv",
        "tabpfn_ts_sota": "outputs/dayahead_30d_core/predictions/tabpfn_ts_sota_dayahead.csv",
        "spike_residual": "outputs/dayahead_corrections_30d/predictions/catboost_spike_residual_corrected_dayahead.csv",
        "hour_corrected": "outputs/dayahead_corrections_30d/predictions/catboost_selected_hour_corrected_dayahead.csv",
        "catboost_tuned": "outputs/dayahead_specialists_30d/predictions/catboost_dayahead_tuned_dayahead.csv",
        "period_specialist": "outputs/dayahead_specialists_30d/predictions/catboost_period_specialist_dayahead.csv",
    }

    # Build merged table
    merged = df_lgbm[["ds", "y_true", "hour_business", "period", "target_day"]].copy()
    merged["ds"] = merged["ds"].astype(str)
    model_names = {"lightgbm_90d_high_leaf": df_lgbm["y_pred"].values}
    model_aliases = {"lightgbm_90d_high_leaf": "LightGBM 90d high_leaf"}

    for name, path in sources.items():
        full = os.path.join(_PROJECT_DIR, path)
        if not os.path.exists(full):
            logger.info(f"  [SKIP] {name}: not found")
            continue
        mdf = pd.read_csv(full)
        m = mdf[["ds", "y_pred"]].copy()
        m["ds"] = m["ds"].astype(str)
        m.columns = ["ds", name]
        merged = merged.merge(m, on="ds", how="left")
        model_names[name] = merged[name].values
        model_aliases[name] = name

    n_models = len(model_names)
    logger.info(f"  Models loaded: {n_models}")

    # Compute metrics per model
    y_true = merged["y_true"].values
    pool_rows = []
    for name in model_names:
        yp = merged[name].values
        n_valid = (~np.isnan(yp) & ~np.isnan(y_true)).sum()
        if n_valid > 0:
            m = compute_all_metrics(y_true[~np.isnan(yp) & ~np.isnan(y_true)],
                                     yp[~np.isnan(yp) & ~np.isnan(y_true)])
            pool_rows.append({"model_name": name, "alias": model_aliases.get(name, name),
                               "sMAPE_floor50": m["sMAPE_floor50"], "MAE": m["MAE"],
                               "RMSE": m["RMSE"], "n": n_valid})

    pool_df = pd.DataFrame(pool_rows).sort_values("sMAPE_floor50").reset_index(drop=True)

    # Oracle per-row
    all_preds = np.column_stack([merged[c].values for c in model_names])
    all_preds_safe = np.where(np.isnan(all_preds), np.inf, all_preds)
    abs_errors = np.abs(all_preds_safe - y_true.reshape(-1, 1))
    best_idx = np.argmin(abs_errors, axis=1)
    y_best = all_preds_safe[np.arange(len(merged)), best_idx]
    valid = ~np.isinf(y_best) & ~np.isnan(y_true)
    oracle_m = compute_all_metrics(y_true[valid], y_best[valid])
    oracle_smape = oracle_m["sMAPE_floor50"]

    # Oracle per-hour
    hour_best = np.zeros(len(merged))
    for h in range(1, 25):
        h_mask = merged["hour_business"] == h
        if h_mask.sum() == 0:
            continue
        h_preds = all_preds_safe[h_mask.values]
        h_errors = np.abs(h_preds - y_true[h_mask.values].reshape(-1, 1))
        h_bi = np.argmin(h_errors, axis=1)
        hour_best[h_mask.values] = h_preds[np.arange(h_mask.sum()), h_bi]
    hour_valid = ~np.isnan(hour_best) & ~np.isnan(y_true)
    hour_oracle = compute_all_metrics(y_true[hour_valid], hour_best[hour_valid])

    # Save pool summary
    out_dir = os.path.join(OUTPUT, "dayahead_model_pool_v2_30d")
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)

    pool_df.to_csv(os.path.join(out_dir, "metrics", "model_pool_summary.csv"),
                    index=False, encoding="utf-8-sig")

    oracle_df = pd.DataFrame([
        {"oracle_type": "per_row", "sMAPE_floor50": oracle_smape},
        {"oracle_type": "per_hour", "sMAPE_floor50": hour_oracle["sMAPE_floor50"]},
    ])
    oracle_df.to_csv(os.path.join(out_dir, "metrics", "model_pool_oracle.csv"),
                      index=False, encoding="utf-8-sig")

    logger.info(f"\n  Model Pool Ranking:")
    for _, r in pool_df.iterrows():
        marker = " 🏆" if r["sMAPE_floor50"] == pool_df["sMAPE_floor50"].min() else ""
        logger.info(f"    {r['sMAPE_floor50']:.4f}%  {r['model_name']}{marker}")
    logger.info(f"  Oracle per-row: {oracle_smape:.4f}%")
    logger.info(f"  Oracle per-hour: {hour_oracle['sMAPE_floor50']:.4f}%")

    return pool_df, oracle_smape, merged, model_names


# ── 3. Run fusion ──────────────────────────────────────────────────────────
def run_fusion(merged, model_names):
    """Run all fusion methods."""
    logger.info("\n=== Fusion v2 ===")

    y_true = merged["y_true"].values
    pred_cols = list(model_names.keys())
    n = len(merged)

    base_a = "lightgbm_90d_high_leaf"
    base_b = "spike_residual"

    results = []

    # 1. simple_average: lightgbm + spike_residual
    ca, cb = merged[base_a].values, merged[base_b].values
    fused_simple = 0.5 * ca + 0.5 * cb
    m = compute_all_metrics(y_true, fused_simple)
    results.append(("best_two_average", m["sMAPE_floor50"]))

    # 2. inverse_smape_period: past 7 days weighting per period
    for method_name, group_col in [("inv_smape_period", "period"),
                                     ("inv_smape_hour", "hour_business")]:
        y_pred = np.full(n, np.nan)
        dates = sorted(merged["target_day"].unique())
        for i, d in enumerate(dates):
            if i < 1:
                continue
            past = merged[merged["target_day"] < d]
            today = merged[merged["target_day"] == d]
            if len(today) == 0:
                continue
            for g in today[group_col].unique():
                g_past = past[past[group_col] == g]
                if len(g_past) < 5:
                    continue
                smape_a = compute_all_metrics(g_past["y_true"].values, g_past[base_a].values)["sMAPE_floor50"]
                smape_b = compute_all_metrics(g_past["y_true"].values, g_past[base_b].values)["sMAPE_floor50"]
                w_a = 1 / max(smape_a, 0.01)
                w_b = 1 / max(smape_b, 0.01)
                w_total = w_a + w_b
                g_today = today[today[group_col] == g]
                idx = g_today.index
                y_pred[idx] = (w_a * g_today[base_a].values + w_b * g_today[base_b].values) / w_total
        valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
        m = compute_all_metrics(y_true[valid], y_pred[valid])
        results.append((method_name, m["sMAPE_floor50"]))

    # 3. winner_by_period/hour
    for method_name, group_col in [("winner_period", "period"),
                                     ("winner_hour", "hour_business")]:
        y_pred = np.full(n, np.nan)
        dates = sorted(merged["target_day"].unique())
        for d in dates:
            past = merged[merged["target_day"] < d]
            today = merged[merged["target_day"] == d]
            if len(today) == 0:
                continue
            for g in today[group_col].unique():
                g_past = past[past[group_col] == g]
                if len(g_past) < 5:
                    continue
                smape_a = compute_all_metrics(g_past["y_true"].values, g_past[base_a].values)["sMAPE_floor50"]
                smape_b = compute_all_metrics(g_past["y_true"].values, g_past[base_b].values)["sMAPE_floor50"]
                winner = base_a if smape_a < smape_b else base_b
                g_today = today[today[group_col] == g]
                y_pred[g_today.index] = g_today[winner].values
        valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
        m = compute_all_metrics(y_true[valid], y_pred[valid])
        results.append((method_name, m["sMAPE_floor50"]))

    # 4. ridge_stacking
    from sklearn.linear_model import Ridge
    y_pred = np.full(n, np.nan)
    dates = sorted(merged["target_day"].unique())
    for i, d in enumerate(dates):
        if i < 1:
            continue
        past = merged[merged["target_day"] < d]
        today = merged[merged["target_day"] == d]
        if len(past) < 48 or len(today) == 0:
            continue
        X_train = np.column_stack([past[c].values for c in [base_a, base_b]])
        y_train = past["y_true"].values
        X_test = np.column_stack([today[c].values for c in [base_a, base_b]])
        ridger = Ridge(alpha=1.0)
        ridger.fit(X_train, y_train)
        y_pred[today.index] = ridger.predict(X_test)
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    m = compute_all_metrics(y_true[valid], y_pred[valid])
    results.append(("ridge_stacking", m["sMAPE_floor50"]))

    # Save fusion metrics
    fusion_df = pd.DataFrame(results, columns=["method", "sMAPE_floor50"])
    fusion_df = fusion_df.sort_values("sMAPE_floor50").reset_index(drop=True)

    out_dir = os.path.join(OUTPUT, "dayahead_fusion_v2_30d")
    os.makedirs(os.path.join(out_dir, "fusion"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    fusion_df.to_csv(os.path.join(out_dir, "fusion", "fusion_metrics.csv"),
                      index=False, encoding="utf-8-sig")

    logger.info(f"  Fusion results:")
    for _, r in fusion_df.iterrows():
        logger.info(f"    {r['sMAPE_floor50']:.4f}%  {r['method']}")

    return fusion_df


# ── 4. LightGBM correction ─────────────────────────────────────────────────
def run_lgbm_correction(merged, pool_df):
    """Train LightGBM-based residual/spike correctors."""
    logger.info("\n=== LightGBM Correction ===")

    lgbm_col = "lightgbm_90d_high_leaf"
    if lgbm_col not in merged.columns:
        logger.warning("  LightGBM predictions not in merged data")
        return pd.DataFrame(), pd.DataFrame()

    y_true = merged["y_true"].values
    y_pred_lgbm = merged[lgbm_col].values
    residual = y_true - y_pred_lgbm
    abs_residual = np.abs(residual)

    # Selected-hour correction: train CatBoost on residual for hours 11/12/13/17
    from catboost import CatBoostRegressor

    hour_set = {11, 12, 13, 17}
    y_pred_corrected = y_pred_lgbm.copy()

    for h in hour_set:
        h_mask = merged["hour_business"] == h
        if h_mask.sum() < 24:
            continue
        h_indices = merged[h_mask].index

        # Rolling: for each day, train on past days
        dates = sorted(merged["target_day"].unique())
        for d in dates:
            past = merged[(merged["target_day"] < d) & (merged["hour_business"] == h)]
            today = merged[(merged["target_day"] == d) & (merged["hour_business"] == h)]
            if len(past) < 7 or len(today) == 0:
                continue
            X_train = past[["hour_business", "load", "net_load", "bidding_space", "is_weekend"]].fillna(0).values
            y_train = (y_true[past.index] - y_pred_lgbm[past.index])
            X_test = today[["hour_business", "load", "net_load", "bidding_space", "is_weekend"]].fillna(0).values
            if len(X_train) > 0:
                cbr = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05, verbose=0, random_seed=42)
                cbr.fit(X_train, y_train)
                correction = cbr.predict(X_test)
                y_pred_corrected[today.index] = y_pred_lgbm[today.index] + correction

    m_hour = compute_all_metrics(y_true, y_pred_corrected)
    logger.info(f"  Hour-corrected sMAPE: {m_hour['sMAPE_floor50']:.4f}%")

    # Spike correction: train CatBoost on high-price errors
    y_pred_spike = y_pred_lgbm.copy()
    spike_threshold = np.percentile(y_true, 90)
    spike_mask = y_true >= spike_threshold
    if spike_mask.sum() > 10:
        dates = sorted(merged["target_day"].unique())
        for d in dates:
            past = merged[merged["target_day"] < d]
            today = merged[merged["target_day"] == d]
            if len(today) == 0:
                continue
            past_spike = past[y_true[past.index] >= spike_threshold]
            if len(past_spike) < 10:
                continue
            X_train = past_spike[["hour_business", "load", "net_load", "bidding_space", "is_weekend"]].fillna(0).values
            y_train = (y_true[past_spike.index] - y_pred_lgbm[past_spike.index])
            X_test = today[["hour_business", "load", "net_load", "bidding_space", "is_weekend"]].fillna(0).values
            today_spike_mask = y_true[today.index] >= spike_threshold
            if today_spike_mask.sum() > 0 and len(X_train) > 0:
                cbr = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05, verbose=0, random_seed=42)
                cbr.fit(X_train, y_train)
                correction = cbr.predict(X_test)
                y_pred_spike[today.index[today_spike_mask]] = y_pred_lgbm[today.index[today_spike_mask]] + correction[today_spike_mask]

    m_spike = compute_all_metrics(y_true, y_pred_spike)
    logger.info(f"  Spike-corrected sMAPE: {m_spike['sMAPE_floor50']:.4f}%")

    # Save corrections
    out_dir = os.path.join(OUTPUT, "dayahead_lgbm_corrections_30d")
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)

    for name, yp in [("lgbm_selected_hour_corrected", y_pred_corrected),
                      ("lgbm_spike_residual_corrected", y_pred_spike)]:
        out_df = pd.DataFrame({
            "ds": merged["ds"], "y_true": y_true, "y_pred": yp,
            "target_day": merged["target_day"], "hour_business": merged["hour_business"],
            "period": merged["period"], "task": "dayahead", "model_name": name,
        })
        out_df.to_csv(os.path.join(out_dir, "predictions", f"{name}_dayahead.csv"),
                       index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([
        {"model": "lightgbm_90d_high_leaf", "sMAPE_floor50": pool_df[pool_df["model_name"] == "lightgbm_90d_high_leaf"]["sMAPE_floor50"].values[0] if len(pool_df[pool_df["model_name"] == "lightgbm_90d_high_leaf"]) > 0 else np.nan},
        {"model": "lgbm_hour_corrected", "sMAPE_floor50": m_hour["sMAPE_floor50"]},
        {"model": "lgbm_spike_corrected", "sMAPE_floor50": m_spike["sMAPE_floor50"]},
    ])
    summary.to_csv(os.path.join(out_dir, "metrics", "summary.csv"), index=False, encoding="utf-8-sig")

    logger.info(f"  Summary:\n{summary.to_string()}")
    return m_hour, m_spike


# ── 5. Generate report ─────────────────────────────────────────────────────
def gen_report(pool_df, oracle_row, oracle_hour, fusion_df, m_hour, m_spike):
    pool_dir = os.path.join(OUTPUT, "dayahead_model_pool_v2_30d")
    fusion_dir = os.path.join(OUTPUT, "dayahead_fusion_v2_30d")
    corr_dir = os.path.join(OUTPUT, "dayahead_lgbm_corrections_30d")

    best_real = pool_df["sMAPE_floor50"].min()
    best_model = pool_df[pool_df["sMAPE_floor50"] == best_real]["model_name"].values[0]

    lines = [
        "# Day-Ahead Model Pool v2 Report",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Model Pool v2 Ranking",
        "",
        "| Rank | Model | sMAPE | MAE | RMSE |",
        "|---|---|---|---|---|",
    ]
    for i, (_, r) in enumerate(pool_df.iterrows()):
        lines.append(f"| {i+1} | {r['model_name']} | {r['sMAPE_floor50']:.4f}% | {r['MAE']:.2f} | {r['RMSE']:.2f} |")

    lines += [
        "",
        f"| Oracle per-row | {oracle_row:.4f}% |",
        f"| Oracle per-hour | {oracle_hour:.4f}% |",
        "",
        "## 2. Fusion v2 Results",
        "",
        "| Method | sMAPE |",
        "|---|---|",
    ]
    for _, r in fusion_df.iterrows():
        beats = "✅" if r["sMAPE_floor50"] < best_real else "❌"
        lines.append(f"| {r['method']} | {r['sMAPE_floor50']:.4f}% {beats} |")

    lines += [
        "",
        "## 3. LightGBM Corrections",
        "",
        "| Model | sMAPE |",
        "|---|---|",
        f"| lightgbm_90d_high_leaf (baseline) | {best_real:.4f}% |",
        f"| lgbm_hour_corrected | {m_hour['sMAPE_floor50']:.4f}% |" if m_hour else "| lgbm_hour_corrected | N/A |",
        f"| lgbm_spike_corrected | {m_spike['sMAPE_floor50']:.4f}% |" if m_spike else "| lgbm_spike_corrected | N/A |",
        "",
        "## 4. 结论",
        "",
        "| 问题 | 回答 |",
        "|---|---|",
        f"| 当前真实最优 | {best_model} ({best_real:.2f}%) |",
        f"| 模型池 per-row oracle | {oracle_row:.2f}% {'✅ < 10%' if oracle_row < 10 else '❌ >= 10%'} |",
        f"| 模型池 per-hour oracle | {oracle_hour:.2f}% {'✅ < 10%' if oracle_hour < 10 else '❌ >= 10%'} |",
        f"| Fusion 超过 {best_real:.2f}% | {'✅' if fusion_df['sMAPE_floor50'].min() < best_real else '❌'} (最佳: {fusion_df['sMAPE_floor50'].min():.2f}%) |",
        f"| LGBM correction 超过 {best_real:.2f}% | {'✅' if m_hour and m_hour['sMAPE_floor50'] < best_real else '❌'} |",
        f"| 低于 11.5% | {'✅' if best_real < 11.5 else '❌'} |",
        f"| 需要 XGBoost / AutoGluon | {'✅ 目前还有提升空间' if oracle_row < best_real + 2 else '❌ 模型池已趋近上限'} |",
        f"| 需要 N-BEATSx | {'✅ LightGBM 路线仍有潜力' if best_real > 10 else '❌ 已接近极限'} |",
        "",
        "## 5. 下一步建议",
        "",
    ]
    lines.append(f"**当前冠军: {best_model} @ {best_real:.2f}%**")
    lines.append("")
    if oracle_row < best_real + 1:
        lines.append(f"模型池接近 oracle 上限 ({oracle_row:.2f}%)，继续加模型收益递减。")
        lines.append("建议转向: spike correction / holiday model / 新数据源。")
    else:
        lines.append(f"模型池距 oracle ({oracle_row:.2f}%) 还有 {oracle_row - best_real:.2f}pp 空间。")
        lines.append("建议继续: XGBoost 搜索 → 最佳配置2000轮重跑 → 融合。")
    lines.append("")

    # Save reports
    for out_dir, filename in [(pool_dir, "model_pool_v2_report.md"),
                               (fusion_dir, "dayahead_fusion_v2_report.md"),
                               (corr_dir, "lgbm_correction_report.md")]:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "reports", filename), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    logger.info("\n" + "\n".join(lines))


def main():
    # 1. Generate LightGBM predictions
    df_lgbm = gen_lgbm_predictions()
    if len(df_lgbm) == 0:
        logger.error("Failed to generate LightGBM predictions")
        return

    # 2. Model pool + oracle
    pool_df, oracle_row, merged, model_names = run_model_pool(df_lgbm)

    # 3. Fusion
    fusion_df = run_fusion(merged, model_names)

    # 4. LGBM corrections
    m_hour, m_spike = run_lgbm_correction(merged, pool_df)

    # 5. Report
    gen_report(pool_df, oracle_row, 
               pool_df[pool_df["model_name"] == "oracle_per_hour"]["sMAPE_floor50"].values[0] if "oracle_per_hour" in pool_df["model_name"].values else oracle_row,
               fusion_df, m_hour, m_spike)

    logger.info("\n=== All Done ===")


if __name__ == "__main__":
    main()
