#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
watch_gatefix.py — continuous supervisor for the two P1.1 recompute tasks.
Polls filesystem for completion of both metrics.json outputs and prints a
progress heartbeat from the run logs. Exits 0 when both are ready.
"""
import os, sys, json, time

MODELS_ROOT = r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\models"
M_V1 = os.path.join(MODELS_ROOT, "outputs/p1_dayahead/run_gatefix_v1/metrics/metrics.json")
M_V1_180 = os.path.join(MODELS_ROOT, "outputs/p1_dayahead/run_gatefix_v1_180/metrics/metrics.json")
LOG_V1 = os.path.join(MODELS_ROOT, "logs/gatefix_v1.log")
LOG_V1_180 = os.path.join(MODELS_ROOT, "logs/gatefix_v1_180.log")

V1_MODELS = ["baseline_lgbm25", "cfg05", "xgboost_rich", "ensemble_rich"]

def last_line(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        return lines[-1] if lines else "(empty)"
    except Exception:
        return "(no log yet)"

def v1_ready():
    if not os.path.exists(M_V1):
        return False
    try:
        d = json.load(open(M_V1, encoding="utf-8"))
        names = {o["model_name"] for o in d.get("overall", [])}
        return all(m in names for m in V1_MODELS)
    except Exception:
        return False

def v180_ready():
    if not os.path.exists(M_V1_180):
        return False
    try:
        d = json.load(open(M_V1_180, encoding="utf-8"))
        names = {o["model_name"] for o in d.get("overall", [])}
        return "cfg05" in names
    except Exception:
        return False

def failed(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        return ("Traceback (most recent call last)" in txt) and ("Error" in txt)
    except Exception:
        return False

def main():
    max_wait = int(sys.argv[1]) if len(sys.argv) > 1 else 590
    interval = 20
    t0 = time.time()
    print("[supervisor] watching gatefix_v1 (4 models) + gatefix_v1_180 (cfg05 180d)")
    while True:
        r1, r180 = v1_ready(), v180_ready()
        elapsed = int(time.time() - t0)
        print("[%3ds] v1=%s  v180=%s" % (elapsed, "READY" if r1 else "....", "READY" if r180 else "...."))
        print("        v1  : " + last_line(LOG_V1)[:140])
        print("        v180: " + last_line(LOG_V1_180)[:140])
        if failed(LOG_V1):
            print("[supervisor] ERROR detected in gatefix_v1 log!"); sys.exit(2)
        if failed(LOG_V1_180):
            print("[supervisor] ERROR detected in gatefix_v1_180 log!"); sys.exit(2)
        if r1 and r180:
            print("[supervisor] BOTH READY -> proceed to build")
            sys.exit(0)
        if elapsed >= max_wait:
            print("[supervisor] timeout (max_wait=%ds); re-invoke to keep watching" % max_wait)
            sys.exit(1)
        time.sleep(interval)

if __name__ == "__main__":
    main()
