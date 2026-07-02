"""
test_metrics.py — Verify sMAPE_floor50 matches the original repo's implementation.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import numpy as np

from src.common.metrics import smape_floor50, mae, compute_all_metrics


def test_smape_floor50_identical_to_original():
    """
    Verify our smape_floor50 matches the original fusion/metrics.py logic.

    Original (from fusion/metrics.py):
        def smape_floor50(y_true, y_pred, eps=1e-6):
            true_clip = np.where(y_true < 50.0, 50.0, y_true)
            pred_clip = np.where(y_pred < 50.0, 50.0, y_pred)
            denom = (np.abs(true_clip) + np.abs(pred_clip)) / 2.0
            denom = np.where(denom < eps, eps, denom)
            return float(np.mean(np.abs(pred_clip - true_clip) / denom) * 100.0)
    """

    # ── Test case 1: All values above 50 ──
    y_true = np.array([100.0, 200.0, 150.0])
    y_pred = np.array([110.0, 190.0, 160.0])
    result = smape_floor50(y_true, y_pred)
    expected = float(np.mean(np.abs(np.array([110., 190., 160.]) - np.array([100., 200., 150.]))
                             / ((np.abs(np.array([100., 200., 150.])) + np.abs(np.array([110., 190., 160.]))) / 2.0)) * 100.0)
    assert abs(result - expected) < 1e-10, f"{result} != {expected}"
    print(f"  Test 1 passed: {result:.4f}")

    # ── Test case 2: Values below 50 (floor) ──
    y_true = np.array([20.0, 30.0, 200.0])
    y_pred = np.array([25.0, 200.0, 180.0])
    result = smape_floor50(y_true, y_pred)
    # Manual: true_clip = [50, 50, 200], pred_clip = [50, 200, 180]
    # denom = (|50|+|50|)/2=50, (|50|+|200|)/2=125, (|200|+|180|)/2=190
    # errors = |50-50|/50=0, |200-50|/125=1.2, |180-200|/190≈0.10526
    # mean = (0 + 1.2 + 0.10526)/3 * 100 ≈ 43.509
    expected = 100.0 * (0.0 + 1.2 + 20.0 / 190.0) / 3.0
    assert abs(result - expected) < 1e-10, f"{result} != {expected}"
    print(f"  Test 2 passed: {result:.4f}")

    # ── Test case 3: Mixed with perfect predictions ──
    y_true = np.array([100.0, -10.0, 300.0])
    y_pred = np.array([100.0, -10.0, 300.0])
    result = smape_floor50(y_true, y_pred)
    assert abs(result) < 1e-10, f"Expected 0, got {result}"
    print(f"  Test 3 passed: {result:.4f}")

    # ── Test case 4: Very small values (eps protection) ──
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, -1.0])
    # true_clip = [50, 50], pred_clip = [50, 50]
    # All clip to 50 → all errors 0
    result = smape_floor50(y_true, y_pred)
    assert abs(result) < 1e-10, f"Expected 0, got {result}"
    print(f"  Test 4 passed: {result:.4f}")

    print("✓ test_smape_floor50_identical_to_original")


def test_mae():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 290.0])
    result = mae(y_true, y_pred)
    expected = (10.0 + 10.0 + 10.0) / 3.0
    assert abs(result - expected) < 1e-10, f"{result} != {expected}"
    print("✓ test_mae")


def test_compute_all_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 290.0, 410.0])
    metrics = compute_all_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "sMAPE_floor50" in metrics
    assert "peak_MAE_q90" in metrics
    assert "negative_price_hit_rate" in metrics
    assert "high_spike_MAE_q90" in metrics
    print(f"  Metrics: {metrics}")
    print("✓ test_compute_all_metrics")


if __name__ == "__main__":
    test_smape_floor50_identical_to_original()
    test_mae()
    test_compute_all_metrics()
    print("\n✅ All metrics tests passed!")
