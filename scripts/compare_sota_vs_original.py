"""
compare_sota_vs_original.py — Generate the sota_comparison_report.md.

Usage:
    python scripts/compare_sota_vs_original.py ^
        --walkforward-dir outputs/sota_walkforward ^
        --start 2026-02-01 ^
        --end 2026-02-28 ^
        --models catboost_sota,chronos2_zero_shot ^
        --source-repo "D:\...\electricity_forecast_model2.0"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Ensure src on path (absolute via os.path) ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.reports.build_report import build_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SOTA comparison report")
    parser.add_argument("--walkforward-dir", type=str, default="outputs/sota_walkforward")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--models", type=str, default="catboost_sota,chronos2_zero_shot")
    parser.add_argument("--source-repo", type=str, default=None, help="Original repo path")
    return parser.parse_args()


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",")]

    # Check if original baseline outputs exist
    baseline_found = False
    if args.source_repo:
        repo = Path(args.source_repo)
        for task in ["dayahead", "realtime"]:
            for model in ["lightgbm", "timesfm"]:
                candidates = [
                    repo / "fusion_runs" / task / f"{model}_output.csv",
                    repo / "lightGBM" / "outputs" / f"lightgbm_{task}.csv",
                    repo / "TimesFM" / "output" / f"timesfm_{task}.csv",
                ]
                for c in candidates:
                    if c.exists():
                        baseline_found = True
                        break

    # Check Chronos fallback
    chronos_fallback = {}
    manifest_path = Path(args.walkforward_dir) / "debug" / "run_manifest.json"
    if manifest_path.exists():
        with open(str(manifest_path), encoding="utf-8") as f:
            manifest = json.load(f)
        if "chronos_fallback" in manifest:
            chronos_fallback = manifest["chronos_fallback"]

    # Build report
    report = build_report(
        output_root=args.walkforward_dir,
        start_date=args.start,
        end_date=args.end,
        models_used=models,
        original_baseline_found=baseline_found,
        chronos_fallback=chronos_fallback,
    )

    # Write report
    report_dir = Path(args.walkforward_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "sota_comparison_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
