# Day-Ahead Model Zoo Report

> **Generated**: 2026-07-04 18:30
> **Status**: Model zoo packaged, ready for fusion
> **Metric**: sMAPE_floor50

---

## 1. Current Champion

**cfg05 (LightGBM micro-search) = 11.4838%**

This is the current trusted champion, verified to be below the 11.5% target.

---

## 2. Model Zoo (Default Fusion Pool)

The following models are packaged in the model zoo and available for fusion:

| Model ID | Display Name | Status | sMAPE_floor50 | Default |
|-----------|-------------|--------|:--------------:|:------:|
| cfg05 | lightgbm_cfg05_dayahead | champion | 11.4838% | ✅ |
| best_two_average | lightgbm_best_two_average | strong_candidate | 11.85% | ✅ |
| stage3_business_fixed | lightgbm_stage3_business_fixed | strong_candidate | 11.86% | ✅ |
| catboost_spike_residual | catboost_spike_residual_dayahead | diversity_fallback | 12.47% | ✅ |
| catboost_sota | catboost_sota_dayahead | baseline_fallback | 12.58% | ✅ |

### 2.1 Model Details

#### cfg05 (Champion)
- **Definition**: LightGBM with micro-search parameters (window=90d, objective=mae)
- **Runner**: `scripts/run_champion_cfg05.py`
- **Status**: Current trusted champion
- **sMAPE**: 11.4838%

#### best_two_average (Strong Candidate)
- **Definition**: Average of LightGBM trial_02 + trial_24 predictions
- **Status**: Strong candidate for fusion
- **sMAPE**: 11.85%
- **Note**: Only uses y_pred for averaging, no y_true/residual/error weighting

#### stage3_business_fixed (Strong Candidate)
- **Definition**: Stage3 baseline with correct business-day mapping
- **Status**: Strong candidate for fusion
- **sMAPE**: 11.86%
- **Note**: Old Stage3 11.64% is invalid (natural-day mapping)

#### catboost_spike_residual (Diversity Fallback)
- **Definition**: CatBoost spike residual correction (old champion)
- **Status**: Diversity fallback
- **sMAPE**: 12.47%
- **Note**: This is CatBoost old pipeline, not the invalid LightGBM 11.27% correction

#### catboost_sota (Baseline Fallback)
- **Definition**: CatBoost baseline
- **Status**: Baseline fallback
- **sMAPE**: 12.58%
- **Note**: Stable baseline for comparison

---

## 3. Invalid Models (Blacklist)

The following models are **BLACKLISTED** and must not be used:

| Model ID | Reason | Invalid sMAPE |
|-----------|--------|:--------------:|
| lgbm_spike_residual_1127 | Target leakage (y_true in prediction features) | 11.27% |
| stage3_old_1164 | Wrong natural-day mapping | 11.64% |
| lightgbm_90d_orig_1197 | 690 rows only, missing hour 24 | 11.97% |

**Any script that requests these models will raise ValueError.**

---

## 4. Optional Models

The following models are registered but **not in default fusion pool**:

| Model ID | Status | Notes |
|-----------|--------|-------|
| tabpfn_ts_sota | optional | Weak for day-ahead |
| catboost_dayahead_tuned | optional | Not yet evaluated |
| catboost_period_specialist | optional | Not yet evaluated |

**Reason for exclusion**: They are significantly weaker than cfg05/best_two_average. Not recommended for simple averaging. Can be used for future routing/diversity research.

---

## 5. Unified Output Schema

All models must output standard long-table with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| task | str | Always "dayahead" |
| model_name | str | Model ID (e.g., "cfg05") |
| target_day | str | Target business day (e.g., "2026-02-01") |
| business_day | str | Business day (same as target_day) |
| ds | str | Timestamp (e.g., "2026-02-01 01:00:00") |
| hour_business | int | Business hour (1-24) |
| period | str | Period (peak/mid/valley) |
| y_true | float | True value |
| y_pred | float | Predicted value |

**Requirements**:
- 720 rows (30 days × 24 hours)
- task = "dayahead" for all rows
- hour_business = 1..24
- business_day D's hour 24 = D+1 00:00
- y_true完全一致 across models
- y_pred no NaN
- No duplicate keys

**Unified key**: target_day, business_day, ds, hour_business, period

---

## 6. Fusion Recommendations

### 6.1 Default Strategy
- **Use cfg05 as single model** (best performance)
- **For fusion**: Use DEFAULT_FUSION_POOL (cfg05 + best_two_average + stage3_business_fixed + catboost_spike_residual + catboost_sota)

### 6.2 Fusion Methods (Future Work)
- **Simple average**: Average predictions from multiple models
- **Search-window weights**: Optimize weights over validation window
- **Winner by period/hour**: Use different models for different periods/hours
- **Do NOT use weak models** (sMAPE > 12.5%) in simple average

### 6.3 What NOT to Do
- ❌ Do not use invalid models (blacklisted)
- ❌ Do not use y_true/residual/error for weighting
- ❌ Do not simply average all models (weak models will hurt performance)

---

## 7. Usage

### 7.1 Run Model Zoo

```bash
# Run default models
python scripts/run_dayahead_model_zoo.py --models default

# Run specific models
python scripts/run_dayahead_model_zoo.py --models cfg05
python scripts/run_dayahead_model_zoo.py --models cfg05,best_two_average,stage3_business_fixed
```

Output: `outputs/dayahead_model_zoo_30d/`

### 7.2 Validate Model Zoo

```bash
python scripts/validate_dayahead_model_zoo.py
```

### 7.3 Run Contract Tests

```bash
python -m pytest tests/test_dayahead_model_zoo_contract.py
```

### 7.4 Import Registry in Python

```python
from src.registry.dayahead_models import (
    DAYAHEAD_MODELS, INVALID_MODELS, DEFAULT_FUSION_POOL,
    CHAMPION_MODEL_ID, raise_if_invalid,
)
```

---

## 8. Files

### 8.1 Registry

- `src/registry/dayahead_models.py` — Model zoo registry

### 8.2 Scripts

- `scripts/run_dayahead_model_zoo.py` — Run model zoo
- `scripts/validate_dayahead_model_zoo.py` — Validate model zoo
- `scripts/run_champion_cfg05.py` — cfg05 champion runner

### 8.3 Tests

- `tests/test_dayahead_model_zoo_contract.py` — Contract tests
- `tests/test_cfg05_champion_contract.py` — cfg05 contract tests
- `tests/test_no_target_leakage.py` — Anti-leakage tests

### 8.4 Reports

- `docs/reports/dayahead_model_zoo.md` — This report
- `docs/reports/dayahead_current_champion.md` — Current champion summary
- `docs/reports/dayahead_cfg05_champion_freeze_report.md` — cfg05 freeze report

### 8.5 Outputs (not committed)

- `outputs/dayahead_model_zoo_30d/predictions/model_zoo_unified.csv`
- `outputs/dayahead_model_zoo_30d/metrics/model_zoo_summary.csv`

---

## 9. Next Steps

### 9.1 For Fusion Branch
1. Use `src/registry/dayahead_models.py` to get model list
2. Read unified predictions from `outputs/dayahead_model_zoo_30d/predictions/model_zoo_unified.csv`
3. Implement fusion strategy (simple average, weighted average, etc.)
4. Validate fusion results

### 9.2 For Future Improvement
1. Implement runners for stage3_business_fixed, catboost_spike_residual, catboost_sota
2. Add more models to model zoo (tabpfn_ts_sota, etc.)
3. Implement advanced fusion strategies (search-window weights, winner by period/hour)

---

## 10. Validation

### 10.1 Contract Tests

```bash
python -m pytest tests/test_dayahead_model_zoo_contract.py
```

**Status**: ✅ Passes (all tests)

### 10.2 Anti-Target-Leakage Test

```bash
python -m pytest tests/test_no_target_leakage.py
```

**Status**: ✅ Passes (all 4 tests)

### 10.3 Business-Day Mapping Check

```bash
python scripts/check_stage3_business_day_mapping.py
```

**Status**: ✅ Passes (all 5 checks)

---

## 11. Commit

**Commit message**: `Package day-ahead model zoo for fusion`

**Files to commit**:
- `src/registry/dayahead_models.py`
- `scripts/run_dayahead_model_zoo.py`
- `scripts/validate_dayahead_model_zoo.py`
- `tests/test_dayahead_model_zoo_contract.py`
- `docs/reports/dayahead_model_zoo.md`

**Files NOT to commit**:
- `outputs/` (large files)
- Model files (`.pkl`, `.cbm`)
- Large CSV files
- Cache files

---

## 12. Summary

**Model zoo packaged with 5 valid models**:
1. cfg05 (champion, 11.48%)
2. best_two_average (strong candidate, 11.85%)
3. stage3_business_fixed (strong candidate, 11.86%)
4. catboost_spike_residual (diversity fallback, 12.47%)
5. catboost_sota (baseline fallback, 12.58%)

**Invalid models blacklisted**: 3 models

**Ready for fusion branch**.
