# Day-Ahead Model Current State

**Date**: 2026-07-03
**Repository**: `epf-sota-experiment`
**Task**: day-ahead price prediction

---

## Current Best Real Model

| Metric | Value |
|--------|-------|
| Model | `spike_residual_corrector` (CatBoost + residual correction) |
| sMAPE_floor50 | **12.47%** |
| MAE | 37.10 |
| RMSE | 49.50 |
| Training window | 3 months rolling |
| Evaluation period | 2026-02-01 → 2026-03-02 (30 days) |

Architecture: CatBoostRegressor (depth=8, lr=0.03, iter=1500) + spike detection with alpha=0.5, threshold=0.55, max_delta=100.

---

## Full Model Pool Rankings

| Rank | Model | sMAPE_floor50 | Architecture |
|:----:|------|:-------------:|-------------|
| 1 | spike_residual_corrected | **12.47%** | CB + residual correction |
| 2 | selected_hour_corrected | 12.52% | CB + hour residual correction |
| 3 | catboost_sota | 12.58% | CB vanilla |
| 4 | H13-only / H17-only | 12.58% | CB hour specialist |
| 5 | H13+H17 | 12.71% | CB 2-hour specialist |
| 6 | H12+H13+H17 | 13.03% | CB 3-hour specialist |
| 7 | TabPFN-TS | 13.64% | Pre-trained transformer |
| 8 | catboost_tuned | 13.89% | CB tuned |
| 9 | catboost_period | 14.62% | CB per-period specialist |

**Key insight**: No CatBoost derivative model has achieved a breakthrough below 12%. All corrective/specialist approaches yield at most 0.11pp improvement.

---

## Oracle Analysis (Theoretical Lower Bound)

| Oracle Type | sMAPE_floor50 | Implication |
|-------------|:-------------:|-------------|
| Per-row oracle | **10.06%** | Perfect per-row model selection |
| Per-hour oracle (best hour model) | 8.53-17.42% | Per-hour routing limited |
| Per-period oracle | 10.12-11.52% | Per-period routing limited |

**Conclusion**: Even with perfect per-row selection across all 8 validated models, the theoretical sMAPE_floor50 lower bound is 10.06%. The 8% target is **not reachable** with the current model pool.

---

## Oracle Contradiction Resolution

The previous contradiction (20.65% vs 7.26%) was caused by:
- **20.65%**: Buggy per-row sMAPE formula using `max(abs(y_true), 1e-8) + max(abs(y_pred), 1e-8)` without the floor50 mechanism
- **Correct result**: After fixing to use `smape_floor50` from `src/common/metrics.py`, the oracle is **10.06%**

---

## What Was Tried

### Phase 1: Baseline (Day 1-2)
- CatBoost vanilla baseline: 12.58%
- TabPFN-TS: 13.64%
- CatBoost tuned: 13.89%
- CatBoost period specialist: 14.62%

### Phase 2: Residual Correction (Day 2)
- Spike residual corrector: **12.47%** ✅
- Selected hour corrector: 12.52% ✅
- Both produce marginal (0.05-0.11pp) improvement

### Phase 3: Regime Models (Day 3)
- Weighted SMAPE v2: 16.53% ❌
- Midday spike v2 (H13/H17): no change ❌
- Regime MoE v2: 22.58% ❌

### Phase 4: Specialists (Day 3)
- Fair 30-day H13/H17 replacement: identical to baseline ❌
- Same features, less data → same predictions

---

## What Did NOT Work

1. **Hour specialists**: Same features but less data → cannot improve
2. **Weighted training**: Actually hurt performance (16.53%)
3. **Hard-routed MoE**: Regime classifier insufficient data
4. **Period specialists**: Worse than global (14.62%)
5. **Fusion**: All fusion methods (simple avg, inverse weight, winner, ridge) worse than single CatBoost
6. **TabPFN-TS**: Pre-trained transformer underperforms CatBoost on this dataset (13.64%)

---

## Next Phase Recommendations

### Required: New Model Architectures

| Priority | Model Type | Expected Gain |
|:--------:|-----------|:-------------:|
| 1 | LightGBM / XGBoost | Different gradient-boosting family |
| 2 | AutoGluon Tabular | Automated ensemble + stacking |
| 3 | N-BEATSx | Interpretable time series decomposition |
| 4 | TimesNet | 2D time series representation |

### Required: New Features (Already coded in adapters)
- Spring festival window features
- Rolling same-hour price statistics
- Volatility and momentum features
- Ranking features

### Optional: Two-Stage Approach
1. Classify price regime (normal/spike/holiday/midday)
2. Regress price within regime
- Only useful if regime classifier is accurate enough

---

## Files

- Model pool: `outputs/dayahead_model_pool_30d/`
- Oracle audit: `outputs/dayahead_oracle_audit/`
- Reports: `docs/reports/`
