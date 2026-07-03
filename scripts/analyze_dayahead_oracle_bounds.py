"""
analyze_dayahead_oracle_bounds.py — Oracle analysis for day-ahead 8% feasibility.

Analyzes theoretical lower bounds by simulating perfect predictions on
specific segments (worst days, worst hours, spikes, holidays) and
combinations thereof. All analyses are pure computation on existing CSVs —
no model training.

Usage:
    python scripts/analyze_dayahead_oracle_bounds.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import compute_all_metrics as _compute_smape

_OUTPUT_ROOT = os.path.join(_PROJECT_DIR, "outputs", "dayahead_oracle")


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute metrics dict from arrays."""
    # Filter out NaN from either array
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[valid]
    yp = y_pred[valid]
    if len(yt) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "sMAPE_floor50": np.nan,
                "peak_MAE_q90": np.nan, "negative_price_hit_rate": np.nan, "n": 0}
    result = _compute_smape(yt, yp)
    return {
        "MAE": float(result.get("MAE", np.nan)),
        "RMSE": float(result.get("RMSE", np.nan)),
        "sMAPE_floor50": float(result.get("sMAPE_floor50", np.nan)),
        "peak_MAE_q90": float(result.get("peak_MAE_q90", np.nan)),
        "negative_price_hit_rate": float(result.get("negative_price_hit_rate", np.nan)),
        "n": int(len(yt)),
    }


def load_pred(path: str) -> pd.DataFrame | None:
    """Load a prediction CSV if it exists."""
    full = os.path.join(_PROJECT_DIR, path) if not os.path.isabs(path) else path
    if os.path.exists(full):
        return pd.read_csv(full)
    return None


def _oracle_replace(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Return y_pred with oracle (set to y_true) where mask is True."""
    y_pred = df["y_pred"].values.copy()
    y_true = df["y_true"].values
    y_pred[mask] = y_true[mask]
    return y_pred


def _compute_segment_smape(df: pd.DataFrame, segment_mask: np.ndarray,
                            label: str, global_smape: float) -> dict:
    """Compute sMAPE with a specific segment oracle-replaced."""
    y_pred_oracle = _oracle_replace(df, segment_mask)
    metrics = compute_all_metrics(df["y_true"].values, y_pred_oracle)
    metrics["label"] = label
    metrics["pct_rows_modified"] = float(segment_mask.mean() * 100)
    metrics["improvement"] = global_smape - metrics["sMAPE_floor50"]
    return metrics


def run_worst_day_oracle(df: pd.DataFrame) -> list[dict]:
    """Oracle on worst days."""
    print("  Running worst-day oracle...")
    daily_smape = df.groupby("target_day").apply(
        lambda g: compute_all_metrics(g["y_true"].values, g["y_pred"].values)["sMAPE_floor50"]
    ).sort_values(ascending=False)
    print(f"    Daily sMAPE range: {daily_smape.min():.2f}% - {daily_smape.max():.2f}%")
    print(f"    Worst 3 days: {daily_smape.head(3).to_dict()}")
    
    global_smape = compute_all_metrics(df["y_true"], df["y_pred"])["sMAPE_floor50"]
    results = []
    
    for n in [1, 3, 5, 10]:
        worst_days = set(daily_smape.head(n).index)
        mask = df["target_day"].isin(worst_days).values
        r = _compute_segment_smape(df, mask, f"worst_{n}_days", global_smape)
        results.append(r)
    
    return results


def run_worst_hour_oracle(df: pd.DataFrame) -> list[dict]:
    """Oracle on worst hours."""
    print("  Running worst-hour oracle...")
    hourly_smape = df.groupby("hour_business").apply(
        lambda g: compute_all_metrics(g["y_true"].values, g["y_pred"].values)["sMAPE_floor50"]
    ).sort_values(ascending=False)
    print(f"    Hourly sMAPE worst 3: {hourly_smape.head(3).to_dict()}")
    print(f"    Hourly sMAPE best 3: {hourly_smape.tail(3).to_dict()}")
    
    global_smape = compute_all_metrics(df["y_true"], df["y_pred"])["sMAPE_floor50"]
    results = []
    
    # Top 1, 3, 5 worst hours
    for n in [1, 3, 5]:
        worst_hours = set(hourly_smape.head(n).index)
        mask = df["hour_business"].isin(worst_hours).values
        r = _compute_segment_smape(df, mask, f"worst_{n}_hours", global_smape)
        results.append(r)
    
    # Specific hour sets
    hour_sets = [
        ("hours_11_12_13_17", [11, 12, 13, 17]),
        ("hours_9_to_16", list(range(9, 17))),
    ]
    for label, hrs in hour_sets:
        mask = df["hour_business"].isin(hrs).values
        r = _compute_segment_smape(df, mask, label, global_smape)
        results.append(r)
    
    return results


def run_spike_oracle(df: pd.DataFrame) -> list[dict]:
    """Oracle on spike hours."""
    print("  Running spike oracle...")
    y_true = df["y_true"].values
    global_smape = compute_all_metrics(df["y_true"], df["y_pred"])["sMAPE_floor50"]
    results = []
    
    # Top 10%
    threshold_p90 = np.percentile(y_true, 90)
    mask_p90 = (y_true >= threshold_p90)
    print(f"    Top 10% threshold: {threshold_p90:.2f}, count={mask_p90.sum()}")
    r = _compute_segment_smape(df, mask_p90, "spike_top10pct", global_smape)
    results.append(r)
    
    # Top 5%
    threshold_p95 = np.percentile(y_true, 95)
    mask_p95 = (y_true >= threshold_p95)
    print(f"    Top 5% threshold: {threshold_p95:.2f}, count={mask_p95.sum()}")
    r = _compute_segment_smape(df, mask_p95, "spike_top5pct", global_smape)
    results.append(r)
    
    # y_true >= 500
    mask_500 = (y_true >= 500)
    print(f"    y_true >= 500: count={mask_500.sum()}")
    r = _compute_segment_smape(df, mask_500, "spike_ge500", global_smape)
    results.append(r)
    
    return results


def run_holiday_oracle(df: pd.DataFrame, holiday_window: tuple = ("2026-02-13", "2026-02-20")) -> list[dict]:
    """Oracle on holiday / abnormal window."""
    print(f"  Running holiday oracle (window: {holiday_window})...")
    start, end = holiday_window
    mask = (df["target_day"] >= start) & (df["target_day"] <= end)
    n_days = df[mask]["target_day"].nunique()
    print(f"    Window covers {n_days} days, {mask.sum()} rows")
    
    global_smape = compute_all_metrics(df["y_true"], df["y_pred"])["sMAPE_floor50"]
    
    r = _compute_segment_smape(df, mask.values, f"holiday_window_{start}_{end}", global_smape)
    return [r]


def run_combined_oracle(df: pd.DataFrame) -> list[dict]:
    """Combined oracle (multiple segments at once)."""
    print("  Running combined oracle...")
    y_true = df["y_true"].values
    global_smape = compute_all_metrics(df["y_true"], df["y_pred"])["sMAPE_floor50"]
    
    # Pre-compute segment masks
    daily_smape = df.groupby("target_day").apply(
        lambda g: compute_all_metrics(g["y_true"].values, g["y_pred"].values)["sMAPE_floor50"]
    ).sort_values(ascending=False)
    
    worst5_days = set(daily_smape.head(5).index)
    worst10_days = set(daily_smape.head(10).index)
    
    hourly_smape = df.groupby("hour_business").apply(
        lambda g: compute_all_metrics(g["y_true"].values, g["y_pred"].values)["sMAPE_floor50"]
    ).sort_values(ascending=False)
    worst3_hours = set(hourly_smape.head(3).index)
    
    threshold_p90 = np.percentile(y_true, 90)
    spike_mask = (y_true >= threshold_p90)
    
    holiday_mask = (df["target_day"] >= "2026-02-13") & (df["target_day"] <= "2026-02-20")
    
    hour_4_13_set = {11, 12, 13, 17}
    
    combos = [
        ("worst5days_worst3hours",
         df["target_day"].isin(worst5_days).values | df["hour_business"].isin(worst3_hours).values),
        ("worst5days_spike_top10pct",
         df["target_day"].isin(worst5_days).values | spike_mask),
        ("holiday_window_hours_11_12_13_17",
         holiday_mask.values | df["hour_business"].isin(hour_4_13_set).values),
        ("holiday_window_spike_top10pct",
         holiday_mask.values | spike_mask),
        ("worst10days_spike_top10pct",
         df["target_day"].isin(worst10_days).values | spike_mask),
    ]
    
    results = []
    for label, mask in combos:
        r = _compute_segment_smape(df, mask, label, global_smape)
        results.append(r)
    
    return results


def run_best_model_oracle(df_base: pd.DataFrame, model_dfs: dict) -> list[dict]:
    """Per-row, per-hour, per-period, per-day best-model oracle."""
    print("  Running best-model oracle...")
    y_true = df_base["y_true"].values
    global_smape = compute_all_metrics(y_true, df_base["y_pred"].values)["sMAPE_floor50"]
    results = []
    
    # Merge all model predictions onto base df
    merged = df_base[["ds", "y_true", "hour_business", "period", "target_day"]].copy()
    merged["y_pred_base"] = df_base["y_pred"].values
    
    for name, df_m in model_dfs.items():
        if name == "catboost_sota":
            continue
        # Align by ds
        m = df_m[["ds", "y_pred"]].rename(columns={"y_pred": f"y_pred_{name}"})
        merged = merged.merge(m, on="ds", how="left")
    
    pred_cols = [c for c in merged.columns if c.startswith("y_pred_")]
    print(f"    Predictions from: {['base'] + [c.replace('y_pred_','') for c in pred_cols]}")
    
    # Per-row: choose min error
    y_t = merged["y_true"].values
    base_pred = merged["y_pred_base"].values
    all_pred_arrays = [base_pred]
    for c in pred_cols:
        all_pred_arrays.append(merged[c].values)
    all_preds = np.column_stack(all_pred_arrays)
    
    # Handle any NaNs from failed merges
    nan_mask = np.isnan(all_preds).any(axis=1)
    if nan_mask.any():
        print(f"    [WARN] {nan_mask.sum()} rows have NaN predictions, falling back to base")
        for i in range(len(all_preds)):
            if np.isnan(all_preds[i]).any():
                all_preds[i] = base_pred[i]  # fallback to base
    
    abs_errors = np.abs(all_preds - y_t.reshape(-1, 1))
    best_idx = np.argmin(abs_errors, axis=1)
    y_pred_best = all_preds[np.arange(len(all_preds)), best_idx]
    
    r = compute_all_metrics(y_t, y_pred_best)
    results.append({
        "label": "oracle_per_row",
        "sMAPE_floor50": r["sMAPE_floor50"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "n": r["n"],
        "pct_rows_modified": float((best_idx > 0).mean() * 100),
        "improvement": global_smape - r["sMAPE_floor50"],
    })
    print(f"      Per-row oracle: {r['sMAPE_floor50']:.4f}%")
    
    # Per-hour: choose best model per hour_business
    merged["best_model_idx"] = best_idx
    for hour in sorted(merged["hour_business"].unique()):
        h_mask = merged["hour_business"] == hour
        h_preds = merged.loc[h_mask, ["y_pred_base"] + list(pred_cols)].values
        h_errors = np.abs(h_preds - merged.loc[h_mask, "y_true"].values.reshape(-1, 1))
        h_best_idx = np.argmin(h_errors, axis=1)
        merged.loc[h_mask, "y_pred_hour_best"] = h_preds[np.arange(h_mask.sum()), h_best_idx]
    
    r = compute_all_metrics(y_t, merged["y_pred_hour_best"].values)
    results.append({
        "label": "oracle_per_hour",
        "sMAPE_floor50": r["sMAPE_floor50"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "n": r["n"],
        "pct_rows_modified": 100.0,
        "improvement": global_smape - r["sMAPE_floor50"],
    })
    print(f"      Per-hour oracle: {r['sMAPE_floor50']:.4f}%")
    
    # Per-period: choose best model per period
    for period in sorted(merged["period"].unique()):
        p_mask = merged["period"] == period
        p_preds = merged.loc[p_mask, ["y_pred_base"] + list(pred_cols)].values
        p_errors = np.abs(p_preds - merged.loc[p_mask, "y_true"].values.reshape(-1, 1))
        p_best_idx = np.argmin(p_errors, axis=1)
        merged.loc[p_mask, "y_pred_period_best"] = p_preds[np.arange(p_mask.sum()), p_best_idx]
    
    r = compute_all_metrics(y_t, merged["y_pred_period_best"].values)
    results.append({
        "label": "oracle_per_period",
        "sMAPE_floor50": r["sMAPE_floor50"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "n": r["n"],
        "pct_rows_modified": 100.0,
        "improvement": global_smape - r["sMAPE_floor50"],
    })
    print(f"      Per-period oracle: {r['sMAPE_floor50']:.4f}%")
    
    # Per-day: choose best model per target_day
    for day in sorted(merged["target_day"].unique()):
        d_mask = merged["target_day"] == day
        d_preds = merged.loc[d_mask, ["y_pred_base"] + list(pred_cols)].values
        d_errors = np.abs(d_preds - merged.loc[d_mask, "y_true"].values.reshape(-1, 1))
        d_best_idx = np.argmin(d_errors, axis=1)
        merged.loc[d_mask, "y_pred_day_best"] = d_preds[np.arange(d_mask.sum()), d_best_idx]
    
    r = compute_all_metrics(y_t, merged["y_pred_day_best"].values)
    results.append({
        "label": "oracle_per_day",
        "sMAPE_floor50": r["sMAPE_floor50"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "n": r["n"],
        "pct_rows_modified": 100.0,
        "improvement": global_smape - r["sMAPE_floor50"],
    })
    print(f"      Per-day oracle: {r['sMAPE_floor50']:.4f}%")
    
    return results


def generate_report(all_results: dict, df: pd.DataFrame) -> str:
    """Generate the oracle bounds markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    global_smape = compute_all_metrics(df["y_true"], df["y_pred"])["sMAPE_floor50"]
    
    lines = [
        "# Day-Ahead Oracle Bounds Analysis",
        f"> Generated: {now}",
        f"> Baseline: catboost_sota @ **{global_smape:.2f}%** sMAPE (30d day-ahead)",
        f"> Target: **8%** (Δ = {global_smape - 8:.2f} pp)",
        "",
        "---",
        "## 1. 最差天 Oracle",
        "",
        "| 修好前 N 天 | 全局 sMAPE | 改进 | 修改行比 |",
        "|---|---|---|---|",
    ]
    
    for r in all_results.get("worst_day", []):
        lines.append(f"| {r['label']} | {r['sMAPE_floor50']:.4f}% | -{r['improvement']:.4f}pp | {r['pct_rows_modified']:.1f}% |")
    
    # Find which N days bring us closest to 8%
    day_results = all_results.get("worst_day", [])
    if day_results:
        best_day_r = day_results[-1]  # worst 10 days
        lines.append("")
        lines.append(f"> **结论：修好最差 10 天 → {best_day_r['sMAPE_floor50']:.2f}%，距 8% 还差 {best_day_r['sMAPE_floor50'] - 8:.2f}pp**")
    
    lines += [
        "",
        "---",
        "## 2. 最差小时 Oracle",
        "",
        "| 修好小时 | 全局 sMAPE | 改进 | 修改行比 |",
        "|---|---|---|---|",
    ]
    
    for r in all_results.get("worst_hour", []):
        lines.append(f"| {r['label']} | {r['sMAPE_floor50']:.4f}% | -{r['improvement']:.4f}pp | {r['pct_rows_modified']:.1f}% |")
    
    hour_results = all_results.get("worst_hour", [])
    if hour_results:
        hours_4_13 = next((r for r in hour_results if r['label'] == 'hours_11_12_13_17'), None)
        if hours_4_13:
            lines.append("")
            lines.append(f"> **结论：只修 hour 11/12/13/17 → {hours_4_13['sMAPE_floor50']:.2f}%，距 8% 还差 {hours_4_13['sMAPE_floor50'] - 8:.2f}pp**")
    
    lines += [
        "",
        "---",
        "## 3. Spike Oracle",
        "",
        "| 定义 | 全局 sMAPE | 改进 | 修改行比 |",
        "|---|---|---|---|",
    ]
    
    for r in all_results.get("spike", []):
        lines.append(f"| {r['label']} | {r['sMAPE_floor50']:.4f}% | -{r['improvement']:.4f}pp | {r['pct_rows_modified']:.1f}% |")
    
    spike_results = all_results.get("spike", [])
    if spike_results:
        best_spike_r = spike_results[0]  # spike top 10%
        lines.append("")
        lines.append(f"> **结论：spike 全修好（top10%）→ {best_spike_r['sMAPE_floor50']:.2f}%，距 8% 还差 {best_spike_r['sMAPE_floor50'] - 8:.2f}pp**")
    
    lines += [
        "",
        "---",
        "## 4. 春节/异常窗口 Oracle",
        "",
        "| 窗口 | 全局 sMAPE | 改进 | 修改行比 |",
        "|---|---|---|---|",
    ]
    
    for r in all_results.get("holiday", []):
        lines.append(f"| {r['label']} | {r['sMAPE_floor50']:.4f}% | -{r['improvement']:.4f}pp | {r['pct_rows_modified']:.1f}% |")
    
    holiday_results = all_results.get("holiday", [])
    if holiday_results:
        h_r = holiday_results[0]
        lines.append("")
        lines.append(f"> **结论：修好春节窗口 → {h_r['sMAPE_floor50']:.2f}%，改进有限 ({h_r['improvement']:.2f}pp)**")
    
    lines += [
        "",
        "---",
        "## 5. 组合 Oracle",
        "",
        "| 组合 | 全局 sMAPE | 改进 | 修改行比 |",
        "|---|---|---|---|",
    ]
    
    for r in all_results.get("combined", []):
        lines.append(f"| {r['label']} | {r['sMAPE_floor50']:.4f}% | -{r['improvement']:.4f}pp | {r['pct_rows_modified']:.1f}% |")
    
    best_combo = None
    if all_results.get("combined"):
        best_combo = min(all_results["combined"], key=lambda x: x["sMAPE_floor50"])
        lines.append("")
        lines.append(f"> **最佳组合：{best_combo['label']} → {best_combo['sMAPE_floor50']:.2f}%**")
    
    lines += [
        "",
        "---",
        "## 6. Best-Model Oracle",
        "",
        "| 方式 | 全局 sMAPE | 改进 | 说明 |",
        "|---|---|---|---|",
    ]
    
    for r in all_results.get("best_model", []):
        lines.append(f"| {r['label']} | {r['sMAPE_floor50']:.4f}% | -{r['improvement']:.4f}pp | 修改 {r['pct_rows_modified']:.1f}% 行 |")
    
    oracles = all_results.get("best_model", [])
    per_row = next((r for r in oracles if r['label'] == 'oracle_per_row'), None)
    if per_row:
        lines.append("")
        lines.append(f"> **现有模型池理论下限 (per-row oracle): {per_row['sMAPE_floor50']:.2f}%**")
        lines.append(f"> 距 8% 还差 {per_row['sMAPE_floor50'] - 8:.2f}pp")
    
    lines += [
        "",
        "---",
        "## 7. 综合结论",
        "",
        f"| 问题 | 回答 |",
        "|---|---|",
        f"| 当前最佳真实模型 | catboost_sota ({global_smape:.2f}%) |",
        f"| 距 8% 还差 | {global_smape - 8:.2f}pp |",
    ]
    
    # Answer the 8 questions
    # 3. Can only fixing worst hours reach 8%?
    worst_hour_vals = [r['sMAPE_floor50'] for r in all_results.get("worst_hour", [])]
    if worst_hour_vals:
        min_hour_oracle = min(worst_hour_vals)
        can_hour = "✅ 能" if min_hour_oracle < 8 else "❌ 不能"
        lines.append(f"| 只修最差小时能否到 8% | {can_hour} (下限: {min_hour_oracle:.2f}%) |")
    
    # 4. Can only fixing spikes reach 8%?
    spike_vals = [r['sMAPE_floor50'] for r in all_results.get("spike", [])]
    if spike_vals:
        min_spike = min(spike_vals)
        can_spike = "✅ 能" if min_spike < 8 else "❌ 不能"
        lines.append(f"| 只修 spike 能否到 8% | {can_spike} (下限: {min_spike:.2f}%) |")
    
    # 5. Can only fixing holiday reach 8%?
    holiday_vals = [r['sMAPE_floor50'] for r in all_results.get("holiday", [])]
    if holiday_vals:
        can_holiday = "✅ 能" if holiday_vals[0] < 8 else "❌ 不能"
        lines.append(f"| 只修春节窗口能否到 8% | {can_holiday} (下限: {holiday_vals[0]:.2f}%) |")
    
    # 6. What combination needed?
    if best_combo:
        needs_more = "需要进一步分析" if best_combo['sMAPE_floor50'] >= 8 else "组合可行"
        lines.append(f"| 达到 8% 需要修哪些 segment | 最佳组合: {best_combo['label']} → {best_combo['sMAPE_floor50']:.2f}%. {needs_more} |")
    
    # 7. Oracle lower bound
    if per_row:
        reachable = "✅ 可达" if per_row['sMAPE_floor50'] <= 8 else "❌ 不可达"
        lines.append(f"| 现有模型池 oracle 下限 | {per_row['sMAPE_floor50']:.2f}% ({reachable}) |")
    
    # 8. Is 8% realistic?
    lowest_oracle = sys.float_info.max
    for category in ["worst_day", "worst_hour", "spike", "holiday", "combined", "best_model"]:
        for r in all_results.get(category, []):
            if r['sMAPE_floor50'] < lowest_oracle:
                lowest_oracle = r['sMAPE_floor50']
    
    realistic = "**现实**" if lowest_oracle <= 8 else "**不现实 — 需要新特征/新模型/新数据**"
    lines.append(f"| 8% 目标是否现实 | {realistic} (理论下限: {lowest_oracle:.2f}%) |")
    
    # 9. Next steps
    lines.append("")
    lines.append("### 下一步建议")
    lines.append("")
    
    # Determine recommendations based on analysis
    if per_row and per_row['sMAPE_floor50'] > 8:
        lines.append("**❌ 现有模型池理论下限仍高于 8%，必须引入新特征/新数据源。**")
        lines.append("")
        lines.append("推荐优先级：")
        lines.append("1. **加新特征** — 当前特征池不足以区分高误差段")
        lines.append("2. **加新数据源** — 考虑燃料价格、碳排放价格、天气更细粒度数据")
        lines.append("3. **做 holiday model** — 春节/异常日期需要独立模型")
        lines.append("4. **做 spike classifier + regressor** — spike 段单独处理")
        lines.append("5. **hour residual correction** — 但仅靠 correction 达不到 8%")
        lines.append("6. **考虑降低目标** — 如果资源有限，12% 比 8% 更现实")
    else:
        lines.append("**✅ 理论可达，建议聚焦修 segment。**")
        lines.append("")
        lines.append("推荐优先级：")
        lines.append("1. **hour residual correction** — 针对最差小时")
        lines.append("2. **spike classifier** — 识别 spike 并修正")
        lines.append("3. **holiday model** — 春节窗口")
        lines.append("4. **继续模型融合** — 在当前 oracle 下限上逼近")
    
    return "\n".join(lines)


def main():
    os.makedirs(os.path.join(_OUTPUT_ROOT, "reports"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_ROOT, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_ROOT, "debug"), exist_ok=True)
    
    print("=" * 60)
    print("Day-Ahead Oracle Bounds Analysis")
    print("=" * 60)
    print()
    
    # Load baseline
    cb_path = os.path.join(_PROJECT_DIR, "outputs", "dayahead_30d_core", "predictions",
                            "catboost_sota_dayahead.csv")
    df = load_pred(cb_path)
    if df is None:
        print("[ERROR] CatBoost baseline not found. Exiting.")
        sys.exit(1)
    
    global_smape = compute_all_metrics(df["y_true"], df["y_pred"])["sMAPE_floor50"]
    print(f"Baseline: catboost_sota {global_smape:.4f}%")
    print()
    
    # Load other models for best-model oracle
    model_paths = {
        "tabpfn_ts_sota": os.path.join(_PROJECT_DIR, "outputs", "dayahead_30d_core",
                                        "predictions", "tabpfn_ts_sota_dayahead.csv"),
        "catboost_dayahead_tuned": os.path.join(_PROJECT_DIR, "outputs", "dayahead_specialists_30d",
                                                 "predictions", "catboost_dayahead_tuned_dayahead.csv"),
        "catboost_period_specialist": os.path.join(_PROJECT_DIR, "outputs", "dayahead_specialists_30d",
                                                    "predictions", "catboost_period_specialist_dayahead.csv"),
    }
    model_dfs = {}
    for name, fp in model_paths.items():
        mdf = load_pred(fp)
        if mdf is not None:
            model_dfs[name] = mdf
    # Also load correction files
    for fname in ["catboost_spike_residual_corrected", "catboost_selected_hour_corrected"]:
        fp = os.path.join(_PROJECT_DIR, "outputs", "dayahead_corrections_30d",
                          "predictions", f"{fname}_dayahead.csv")
        mdf = load_pred(fp)
        if mdf is not None:
            model_dfs[fname] = mdf
    
    print(f"Models available for oracle: {list(model_dfs.keys())}")
    print()
    
    # Run all oracles
    all_results = {}
    
    # 1. Worst-day oracle
    all_results["worst_day"] = run_worst_day_oracle(df)
    print()
    
    # 2. Worst-hour oracle
    all_results["worst_hour"] = run_worst_hour_oracle(df)
    print()
    
    # 3. Spike oracle
    all_results["spike"] = run_spike_oracle(df)
    print()
    
    # 4. Holiday oracle
    all_results["holiday"] = run_holiday_oracle(df)
    print()
    
    # 5. Combined oracle
    all_results["combined"] = run_combined_oracle(df)
    print()
    
    # 6. Best-model oracle
    if model_dfs:
        all_results["best_model"] = run_best_model_oracle(df, model_dfs)
    else:
        all_results["best_model"] = []
        print("  Skipping best-model oracle: no other models available.")
    print()
    
    # Build flat table for CSV export
    csv_rows = []
    for category, results in all_results.items():
        for r in results:
            csv_rows.append({
                "category": category,
                "label": r["label"],
                "sMAPE_floor50": r["sMAPE_floor50"],
                "MAE": r.get("MAE", np.nan),
                "RMSE": r.get("RMSE", np.nan),
                "improvement": r["improvement"],
                "pct_rows_modified": r["pct_rows_modified"],
            })
    
    csv_df = pd.DataFrame(csv_rows)
    csv_df = csv_df.sort_values("sMAPE_floor50").reset_index(drop=True)
    
    # Save metrics
    csv_path = os.path.join(_OUTPUT_ROOT, "metrics", "oracle_bounds.csv")
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Oracle bounds CSV: {csv_path}")
    
    # Save segment details
    seg_path = os.path.join(_OUTPUT_ROOT, "debug", "oracle_segments.csv")
    seg_rows = []
    for category, results in all_results.items():
        for r in results:
            seg_rows.append(r)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(seg_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Oracle segments: {seg_path}")
    
    # Generate report
    report = generate_report(all_results, df)
    report_path = os.path.join(_OUTPUT_ROOT, "reports", "dayahead_oracle_bounds.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Oracle bounds report: {report_path}")
    
    # Summary
    print()
    print("=" * 60)
    print("ORACLE SUMMARY")
    print("=" * 60)
    print(f"Baseline: {global_smape:.2f}%")
    print()
    print("Best oracle per category:")
    for category in ["worst_day", "worst_hour", "spike", "holiday", "combined", "best_model"]:
        if all_results.get(category):
            best = min(all_results[category], key=lambda x: x["sMAPE_floor50"])
            print(f"  {category}: {best['sMAPE_floor50']:.2f}% (label={best['label']})")
    print()


if __name__ == "__main__":
    main()
