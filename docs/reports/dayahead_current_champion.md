# Day-Ahead Current Champion

> Generated: 2026-07-04 10:15
> Task: dayahead
> Metric: sMAPE_floor50

## Current Champion Models

| Rank | Model | sMAPE_floor50 | Notes |
|:----:|------|:-------------:|-------|
| 🥇 1 | **cfg05 (micro-search LGBM)** | **11.48%** | **New trusted champion! Below 11.5% target.** |
| 🥈 2 | best_two_average (trial_02 + trial_24) | 11.85% | Previous trusted champion |
| 🥉 3 | stage3 baseline (business-fixed) | 11.86% | Baseline on correct business-day mapping |
| 4 | catboost_spike_residual_corrected | 12.47% | Old CatBoost champion |
| 5 | catboost_sota | 12.58% | Original CatBoost baseline |

## cfg05 Configuration

```
window = 90d
objective = mae
num_leaves = 191
min_data_in_leaf = 30
learning_rate = 0.015
lambda_l1 = 0.1
lambda_l2 = 5.0
feature_fraction = 0.85
bagging_fraction = 0.95
bagging_freq = 5
n_estimators = 2000
```

## INVALID Results

| Model | sMAPE | Reason |
|-------|:-----:|--------|
| lgbm_spike_residual_corrected | 11.27% | ❌ Target leakage (y_true in prediction features) |
| Stage3 old (natural day) | 11.64% | ❌ Wrong business-day mapping |
| lightgbm_90d_orig | 11.97% | ⚠️ 690 rows only, missing hour_business=24 |

## Key Audit Findings

| Finding | Detail |
|---------|--------|
| best_two_average reproducible | ✅ Yes. 720 rows, trial_02 + trial_24 simple average |
| lightgbm_90d_orig (11.97%) | ⚠️ Only 690 rows — each day has hours 1-23, missing hour 24 |
| trial_02 y_true vs core | ✅ All trials share same y_true among themselves; different from CatBoost core |
| catboost_spike_residual (12.47%) | ✅ Verified, y_true matches core baseline |
| catboost_sota (12.58%) | ✅ Verified |

## 30-Day Breakdown

| Metric | cfg05 (micro-search) | best_two_average | trial_02 |
|--------|:-------------------:|:----------------:|:--------:|
| sMAPE_floor50 | **11.48%** | 11.85% | 12.07% |
| MAE | — | — | 32.55 |
| RMSE | — | — | 46.98 |
| Hours | 720 (full) | 720 (full) | 720 (full) |

> ⚠️ **Invalid results removed from comparison**:
> - lgbm_spike_residual = 11.27%: ❌ Target leakage (y_true in prediction features)
> - Stage3 old (natural day) = 11.64%: ❌ Wrong business-day mapping

## Target Status

| Target | Status | Gap |
|:------|:------:|:---:|
| Below 12.58% (catboost_sota) | ✅ 11.48% | -1.10pp |
| Below 12.47% (old CatBoost champion) | ✅ 11.48% | -0.99pp |
| Below 12% | ✅ 11.48% | -0.52pp |
| **Below 11.5%** | **✅ 11.48%** | **Done** |
| Below 11% | ❌ | +0.48pp |
| Below 10% | ❌ | +1.48pp |
| Below 8% | ❌ | +3.48pp |

## What Has Been Tried (Stopped Working)

| Approach | Best | Verdict |
|----------|:----:|:-------:|
| CatBoost (sota) | 12.58% | Surpassed by LightGBM |
| CatBoost spike residual correction | 12.47% | 0.11pp gain, but capped |
| CatBoost hour specialist | 12.52% | Marginal, not worth it |
| CatBoost regime v2 | 12.14% (partial) | Feature engineering not breakthrough |
| CatBoost weighted SMAPE | 16.53% | Worsened |
| CatBoost + TabPFN fusion | 12.91% | All fusion below CatBoost alone |
| CatBoost Mixture-of-Experts | 22.58% | Failed |
| Ridge stacking | 14.76% | Overfit |
| TabPFN | 13.64% | Slow, not competitive |
| LightGBM huber/fair objective | >50% | Failed completely |

## Next Directions

**✅ Sprint complete. cfg05 frozen as trusted champion.**

Further improvement below 11.0% requires new architecture (AutoGluon/N-BEATSx) and should be in a new phase, not mixed into this sprint.

### If starting new phase:
1. Consider AutoGluon light preset (not heavy)
2. Consider N-BEATSx with exogenous variables
3. These require longer timelines and are not suitable for current sprint

## Current Configuration (Trusted Champion: cfg05)

```
model = LightGBM
window = 90d
objective = mae
num_leaves = 191
min_data_in_leaf = 30
learning_rate = 0.015
lambda_l1 = 0.1
lambda_l2 = 5.0
feature_fraction = 0.85
bagging_fraction = 0.95
bagging_freq = 5
n_estimators = 2000
```
