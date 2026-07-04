"""
dayahead_models.py — Model zoo registry for day-ahead forecasting.

This module defines:
  1. DAYAHEAD_MODELS: valid models with metadata
  2. INVALID_MODELS: blacklisted models (must not be used)
  3. DEFAULT_FUSION_POOL: default models for fusion
  4. CHAMPION_MODEL_ID: current trusted champion

Usage:
  from src.registry.dayahead_models import DAYAHEAD_MODELS, INVALID_MODELS
  from src.registry.dayahead_models import DEFAULT_FUSION_POOL, CHAMPION_MODEL_ID
"""

# ── Current trusted champion ──
CHAMPION_MODEL_ID = "cfg05"


# ── Valid day-ahead models ──
DAYAHEAD_MODELS = {
    "cfg05": {
        "display_name": "lightgbm_cfg05_dayahead",
        "status": "champion",
        "smape_floor50": 11.4838,
        "runner": "scripts/run_champion_cfg05.py",
        "default": True,
        "description": "LightGBM micro-search champion (window=90d, objective=mae)",
    },
    "best_two_average": {
        "display_name": "lightgbm_best_two_average",
        "status": "strong_candidate",
        "smape_floor50": 11.85,
        "default": True,
        "description": "Average of LightGBM trial_02 + trial_24 predictions",
    },
    "stage3_business_fixed": {
        "display_name": "lightgbm_stage3_business_fixed",
        "status": "strong_candidate",
        "smape_floor50": 11.86,
        "default": True,
        "description": "Stage3 baseline with correct business_day mapping",
    },
    "catboost_spike_residual": {
        "display_name": "catboost_spike_residual_dayahead",
        "status": "diversity_fallback",
        "smape_floor50": 12.47,
        "default": True,
        "description": "CatBoost spike residual correction (old champion, diversity)",
    },
    "catboost_sota": {
        "display_name": "catboost_sota_dayahead",
        "status": "baseline_fallback",
        "smape_floor50": 12.58,
        "default": True,
        "description": "CatBoost baseline (stable fallback)",
    },
    # ── Optional models (not in default pool) ──
    "tabpfn_ts_sota": {
        "display_name": "tabpfn_ts_sota_dayahead",
        "status": "optional",
        "smape_floor50": None,  # Not yet evaluated
        "default": False,
        "description": "TabPFN-TS (optional, weak for day-ahead)",
    },
    "catboost_dayahead_tuned": {
        "display_name": "catboost_dayahead_tuned",
        "status": "optional",
        "smape_floor50": None,  # Not yet evaluated
        "default": False,
        "description": "CatBoost day-ahead tuned (optional)",
    },
    "catboost_period_specialist": {
        "display_name": "catboost_period_specialist_dayahead",
        "status": "optional",
        "smape_floor50": None,  # Not yet evaluated
        "default": False,
        "description": "CatBoost period specialist (optional)",
    },
}


# ── Invalid models (blacklist) ──
INVALID_MODELS = {
    "lgbm_spike_residual_1127": {
        "reason": "target leakage: y_true used in prediction features",
        "invalid_smape": 11.27,
        "status": "invalid",
    },
    "stage3_old_1164": {
        "reason": "wrong natural-day mapping (hour 24 not mapped to D+1 00:00)",
        "invalid_smape": 11.64,
        "status": "invalid",
    },
    "lightgbm_90d_orig_1197": {
        "reason": "690 rows only, missing hour 24 (incomplete evaluation)",
        "invalid_smape": 11.97,
        "status": "invalid",
    },
}


# ── Default fusion pool ──
DEFAULT_FUSION_POOL = [
    "cfg05",
    "best_two_average",
    "stage3_business_fixed",
    "catboost_spike_residual",
    "catboost_sota",
]


# ── Helper functions ──
def get_valid_model_ids():
    """Return list of valid model IDs."""
    return list(DAYAHEAD_MODELS.keys())


def get_default_model_ids():
    """Return list of default model IDs."""
    return [k for k, v in DAYAHEAD_MODELS.items() if v.get("default", False)]


def get_champion_id():
    """Return champion model ID."""
    return CHAMPION_MODEL_ID


def is_valid_model(model_id):
    """Check if model_id is valid (not blacklisted)."""
    if model_id in INVALID_MODELS:
        return False
    if model_id not in DAYAHEAD_MODELS:
        return False
    return True


def raise_if_invalid(model_id):
    """Raise ValueError if model_id is invalid."""
    if model_id in INVALID_MODELS:
        reason = INVALID_MODELS[model_id]["reason"]
        raise ValueError(
            f"Model '{model_id}' is BLACKLISTED: {reason}"
        )
    if model_id not in DAYAHEAD_MODELS:
        raise ValueError(
            f"Model '{model_id}' not found in DAYAHEAD_MODELS"
        )


def get_model_info(model_id):
    """Get model info dict, or raise if invalid."""
    raise_if_invalid(model_id)
    return DAYAHEAD_MODELS[model_id]


# ── Export ──
__all__ = [
    "DAYAHEAD_MODELS",
    "INVALID_MODELS",
    "DEFAULT_FUSION_POOL",
    "CHAMPION_MODEL_ID",
    "get_valid_model_ids",
    "get_default_model_ids",
    "get_champion_id",
    "is_valid_model",
    "raise_if_invalid",
    "get_model_info",
]
