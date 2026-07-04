# Day-Ahead cfg05 Champion Freeze Report

> **Generated**: 2026-07-04 11:00
> **Status**: Frozen — no further model runs in this sprint
> **Metric**: sMAPE_floor50

---

## 1. Current Trusted Champion

**cfg05 (LightGBM micro-search) = 11.48%**

This is the current trusted champion, verified to be below the 11.5% target.

### Configuration

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

### Reproduction Script

```bash
python scripts/run_champion_cfg05.py
```

Output: `outputs/dayahead_champion_cfg05_30d/`

---

## 2. Improvement vs Previous Champions

| Model | sMAPE_floor50 | Improvement |
|-------|:-------------:|:-----------:|
| CatBoost baseline | 12.58% | -1.10pp |
| CatBoost spike residual | 12.47% | -0.99pp |
| best_two_average (trial_02 + trial_24) | 11.85% | -0.37pp |
| **cfg05 (champion)** | **11.48%** | — |

**Key**: cfg05 beats all previous champions, including the previous trusted champion best_two_average (11.85%).

---

## 3. Invalid Results (Excluded from Comparison)

### 3.1 lgbm_spike_residual = 11.27% — INVALIDATED

**Reason**: Target leakage in prediction features.

**Details**:
- The `lgbm_spike_residual_corrected` model used `y_true` as a feature during prediction.
- This allowed the model to "see" the future, resulting in an artificially low sMAPE.
- After fixing (removing `y_true` from prediction features), the corrected model achieved 12.47%, not 11.27%.

**Status**: ❌ Invalid, removed from leaderboard.

### 3.2 Stage3 old (natural day) = 11.64% — INVALIDATED

**Reason**: Wrong business-day mapping.

**Details**:
- The old Stage3 implementation used `df["target_day"] = df["ds"].dt.date.astype(str)` (natural day).
- This is incorrect because `hour_business=24` corresponds to `ds = D+1 00:00:00` (business day mapping).
- After fixing (using `business_time_mapping()`), the corrected Stage3 achieved 11.86%, not 11.64%.

**Status**: ❌ Invalid, removed from leaderboard.

---

## 4. Target Status

| Target | Status | Gap |
|:------|:------:|:---:|
| Below 12.58% (CatBoost baseline) | ✅ 11.48% | -1.10pp |
| Below 12.47% (old CatBoost champion) | ✅ 11.48% | -0.99pp |
| Below 12% | ✅ 11.48% | -0.52pp |
| **Below 11.5%** | **✅ 11.48%** | **Done** |
| Below 11% | ❌ | +0.48pp |
| Below 10% | ❌ | +1.48pp |
| Below 8% | ❌ | +3.48pp |

**Conclusion**:
- ✅ cfg05 meets the 11.5% target
- ❌ cfg05 does NOT meet the 11.0% target (would need +0.48pp improvement)

---

## 5. What Has Been Tried (Stopped Working)

| Approach | Best | Verdict |
|----------|:----:|:-------:|
| CatBoost (sota) | 12.58% | Surpassed by LightGBM |
| CatBoost spike residual correction | 12.47% | 0.11pp gain, but capped |
| CatBoost hour specialist | 12.52% | Marginal, not worth it |
| CatBoost regime v2 | 12.14% (partial) | Feature engineering not breakthrough |
| LightGBM huber/fair objective | >50% | Failed completely |
| **LightGBM micro-search (cfg05)** | **11.48%** | **✅ Current champion** |

---

## 6. Why Not AutoGluon / N-BEATSx?

**Recommendation**: Not recommended for this sprint.

**Reasons**:
1. **Time constraint**: AutoGluon and N-BEATSx require longer tuning timelines.
2. **Risk**: These models may not beat cfg05 (11.48%) and could introduce new issues.
3. **Sprint complete**: The 11.5% target has been met. Further improvement below 11.0% should be in a new phase.

**If continuing to target 11.0%**:
- Start a new phase (not mixed into this sprint).
- Consider AutoGluon light preset (not heavy).
- Consider N-BEATSx with exogenous variables.
- These require longer timelines and careful validation.

---

## 7. Next Steps

### For this sprint:
- ✅ **Freeze cfg05 as trusted champion**
- ✅ **No further model runs**
- ✅ **Submit frozen code and reports**

### For next phase (if targeting 11.0%):
1. Start a new phase (separate from this sprint).
2. Consider AutoGluon light preset.
3. Consider N-BEATSx with exogenous variables.
4. These require longer timelines and are not suitable for current sprint.

---

## 8. Validation

### 8.1 Anti-Target-Leakage Test

```bash
python -m pytest tests/test_no_target_leakage.py
```

**Status**: ✅ Passes (all 4 tests)

### 8.2 Business-Day Mapping Check

```bash
python scripts/check_stage3_business_day_mapping.py
```

**Status**: ✅ Passes (all 5 checks)

### 8.3 cfg05 Champion Contract Test

```bash
python -m pytest tests/test_cfg05_champion_contract.py
```

**Status**: ✅ Passes (all 6 tests)

---

## 9. Files

### 9.1 Scripts

- `scripts/run_champion_cfg05.py` — cfg05 reproduction script (day-ahead only)

### 9.2 Tests

- `tests/test_no_target_leakage.py` — Anti-target-leakage tests
- `tests/test_cfg05_champion_contract.py` — cfg05 champion contract tests

### 9.3 Reports

- `docs/reports/dayahead_current_champion.md` — Current champion summary
- `docs/reports/dayahead_cfg05_champion_freeze_report.md` — This report

### 9.4 Outputs (not committed)

- `outputs/dayahead_champion_cfg05_30d/predictions/cfg05_dayahead.csv`
- `outputs/dayahead_champion_cfg05_30d/metrics/summary.csv`
- `outputs/dayahead_champion_cfg05_30d/metrics/hour_metrics.csv`
- `outputs/dayahead_champion_cfg05_30d/metrics/period_metrics.csv`
- `outputs/dayahead_champion_cfg05_30d/reports/cfg05_champion_report.md`

---

## 10. Commit

**Commit message**: `Freeze cfg05 as trusted day-ahead champion`

**Files to commit**:
- `scripts/run_champion_cfg05.py`
- `tests/test_cfg05_champion_contract.py`
- `docs/reports/dayahead_current_champion.md`
- `docs/reports/dayahead_cfg05_champion_freeze_report.md`

**Files NOT to commit**:
- `outputs/` (large files)
- Model files (`.pkl`, `.cbm`)
- Large CSV files
- Cache files

---

## 11. Final Champion

**Current Trusted Champion**: cfg05 = **11.48%**

**Status**:
- ✅ Below 11.5% target
- ❌ Not below 11.0% target

**Sprint complete. Champion frozen.**
