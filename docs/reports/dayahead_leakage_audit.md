# Day-Ahead Data Leakage Audit

> Date: 2026-07-03 21:55
> Auditor: Code analysis

## 1. Leak Location

**File:** `src/correction/lgbm_dayahead_corrector.py` → `LGBMSpikeResidualCorrector.correct()`
**Line:** 101 (original)
**Commit:** c048476 / c048476^ (before fix)

## 2. Leak Code Snippet

```python
# Line 97-103 (original, leaked):
day_data = df.iloc[day_indices]
X_pred = np.column_stack([
    day_data["hour_business"].values.astype(float),
    day_data["y_true"].values,       # ← LEAK: using today's actual price
    base[day_indices],
])
```

The `y_true` column (the actual electricity price for that hour) was used as a prediction feature during inference. This gives the model access to the ground truth it's trying to predict, creating a future data leak.

## 3. Why 11.27% Is Invalid

| Claim | Actual |
|-------|--------|
| Champion sMAPE | 11.27% |
| Root cause | `day_data["y_true"]` leaked into `X_pred` |
| Impact | The corrector learned to use actual price as a feature, producing artificially good residuals |
| Verdict | **INVALID — must discard** |

The model trained on `[hour_business, y_true, base_pred]` features for spike hours. During prediction, it received `[hour_business, y_true, base_pred]` where the `y_true` was the ACTUAL price for that hour. This makes the residual trivially predictable.

## 4. V2 Strict Rolling Result

After removing `y_true` from prediction features, the corrector uses only `[hour_business, base_pred]`. Result:

| Version | Features | sMAPE | Status |
|---------|----------|:-----:|:------:|
| V1 (leaked) | [hour, y_true, base_pred] | 11.27% | ❌ Invalid |
| V2 (strict) | [hour, base_pred] | TBD | ✅ Clean |

Initial V2 results are not yet available. The correction gain is expected to be minimal without `y_true`.

## 5. Current Trusted Champion

| Model | sMAPE_floor50 | Rows | Leak-free? |
|-------|:-------------:|:----:|:----------:|
| **best_two_average** | **11.85%** | 720 | ✅ |
| trial_02 (LightGBM 150d) | 12.07% | 720 | ✅ |
| catboost_spike_residual | 12.47% | 720 | ✅ |
| catboost_sota | 12.58% | 720 | ✅ |

`best_two_average` = simple average of trial_02 and trial_24 predictions. Pure y_pred fusion, no y_true involved. Computed in `scripts/audit_and_freeze_lgbm.py`.

## 6. Anti-Leakage Guard (Enforced)

All future corrections must pass `_validate_prediction_features()`:

**Denylist:**
```
y_true
residual
error
abs_error
future_y
target_actual
oracle
best_model
```

**Enforcement:**
- `src/correction/lgbm_dayahead_corrector.py` now calls `_validate_prediction_features()` on all prediction feature arrays
- `tests/test_no_target_leakage.py` validates all corrector classes against denylist
- CI will fail if any prediction feature matches denylist terms
