"""
repo_paths.py — Resolve source repo and data paths using pathlib (safe with Chinese/spaces).

Usage:
    from src.common.repo_paths import get_source_repo, get_data_path
"""

from __future__ import annotations

from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

# ── Default locations ──────────────────────────────────────────────
_DEFAULT_REPO = (
    Path("D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0")
)
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
_PATHS_YAML = _CONFIGS_DIR / "paths.yaml"


def _resolve_paths_yaml() -> dict:
    """Load paths from paths.yaml if it exists."""
    if _PATHS_YAML.exists():
        with open(str(_PATHS_YAML), encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_source_repo() -> Path:
    """Return the absolute Path to the source repository."""
    cfg = _resolve_paths_yaml()
    repo_str = cfg.get("source_repo")
    if repo_str:
        p = Path(repo_str)
        if p.exists():
            return p.resolve()
    # fallback: default location
    if _DEFAULT_REPO.exists():
        return _DEFAULT_REPO.resolve()
    raise FileNotFoundError(
        f"Source repo not found. Check configs/paths.yaml or supply --source-repo."
    )


def get_data_path(custom: str | Path | None = None) -> Path:
    """Return data file path.  Prefer custom > yaml > default location."""
    if custom is not None:
        p = Path(custom)
        if p.exists():
            return p.resolve()
    cfg = _resolve_paths_yaml()
    data_str = cfg.get("default_data")
    if data_str:
        p = Path(data_str)
        if p.exists():
            return p.resolve()
    # fallback: default
    p = _DEFAULT_REPO / "data" / "shandong_pmos_hourly.csv"
    if p.exists():
        return p.resolve()
    raise FileNotFoundError("Data file not found anywhere.")
