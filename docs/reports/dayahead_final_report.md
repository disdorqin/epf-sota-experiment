# Day-Ahead Final Report
Generated: 2026-07-03 11:08

---

## 1. Single Model Rankings (by sMAPE_floor50)

**Base models:**

| model_name | sMAPE_floor50 | MAE | RMSE | n |
|------------|---------------|-----|------|---|
| catboost_sota_dayahead | 12.58% | 37.56 | 50.08 | 720 |
| tabpfn_ts_sota_dayahead | 13.64% | 38.05 | 48.84 | 720 |

*No specialist model results found.*

## 2. Fusion Method Rankings (by sMAPE_floor50)

| model_name | sMAPE_floor50 | MAE | RMSE | n |
|------------|---------------|-----|------|---|
| fused_winner_by_hour_dayahead | 12.91% | 37.37 | 48.39 | 720 |
| fused_inverse_smape_period_dayahead | 12.94% | 37.01 | 47.82 | 720 |
| fused_inverse_smape_hour_dayahead | 12.94% | 37.02 | 47.85 | 720 |
| fused_simple_average_dayahead | 13.01% | 37.10 | 48.10 | 720 |
| fused_winner_by_period_dayahead | 13.10% | 37.57 | 49.10 | 720 |
| fused_ridge_stacking_dayahead | 14.76% | 43.14 | 55.72 | 720 |

## 3. Target Check (sMAPE_floor50)

- **Best model sMAPE:** `12.58%`
- **Below 12%:** ❌ No
- **Below 10%:** ❌ No
- **Below 8%:** ❌ No

## 4. Worst-Case Analysis

- **Overall sMAPE (from diagnosis):** `nan%`
- **Gap to 8%:** `nan` pp
- - Worst period: `17_24` (sMAPE = `nan%`)
- - Worst hour: `hour 1` (sMAPE = `nan%`)

## 5. Spike & Negative Price Performance

*See diagnosis report for detailed spike/negative/SF analysis.*
Refer to the diagnosis report for spike hours, negative hours, and Spring Festival window performance.

## 6. Recommendations

**Recommended day-ahead main model:** `catboost_sota_dayahead` (sMAPE = `12.58%`)
**Recommended fusion method:** `fused_winner_by_hour_dayahead` (sMAPE = `12.91%`)
**Fusion improves over best single model:** ❌ No

❌ **Not ready for integration** — best sMAPE = `12.58%`, still above 12% target.

## 7. Next Steps

1. Tune `catboost_sota_dayahead` further (target transform, more features, hour specialists).
2. Try spike correction for top-decile hours.
3. If Spring Festival window is a major drag, add `is_spring_festival_window` as a feature or do expost correction.
4. Re-run fusion after specialist models are added.
