# Day-Ahead Trusted Champion Report

> Generated: 2026-07-03 21:55
> Task: dayahead
> Metric: sMAPE_floor50 (only)

## ⚠️ Data Leakage Announcement

**lgbm_spike_residual_corrected (11.27%) has been INVALIDATED due to data leakage.**

Leak: prediction features included `y_true` (line 101 of lgbm_dayahead_corrector.py)
See `docs/reports/dayahead_leakage_audit.md` for full details.

## Current Trusted Champion

**best_two_average = 11.85%**

- Construction: simple average of LightGBM trial_02 + trial_24 predictions
- Fusion: `y_pred = (y_pred_t02 + y_pred_t24) / 2` — pure prediction fusion
- No y_true involved at any stage of fusion
- 720 rows, hours 1-24, 30 days (Feb 1-Mar 2)

## Ranking

| Rank | Model | sMAPE | Leak-free? |
|:----:|------|:-----:|:----------:|
| 🥇 1 | best_two_average (trusted) | **11.85%** | ✅ |
| 2 | catboost_sota (original) | 12.58% | ✅ |
| 3 | catboost_spike_residual (old) | 12.47% | ✅ |
| 4 | lightgbm_trial_02 (single) | 12.07% | ✅ |
| 5 | lgbm_spike_residual (INVALID-leaked) | 11.27% | ⚠️ LEAKED |

## Target Check

| Target | Status |
|:-------|:------:|
| Below 12.58% (CatBoost) | ✅ 11.85% |
| Below 12.47% (old champion) | ✅ 11.85% |
| Below 12% | ✅ 11.85% |
| Below 11.5% | ❌ 11.85% (gap 0.35pp) |

## Anti-Leakage Measures

1. `_validate_prediction_features()` in all corrector prediction paths
2. Denylist: y_true, residual, error, abs_error, future_y, target_actual, oracle, best_model
3. `tests/test_no_target_leakage.py` — static analysis + runtime guard verification
4. All new correction/fusion code must pass before commit

## Recommendation

- Freeze `best_two_average` (11.85%) as current production candidate
- Do NOT use any correction that depends on y_true at prediction time
- Future correction work must pass anti-leakage tests first