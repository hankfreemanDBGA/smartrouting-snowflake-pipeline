#!/usr/bin/env python3
"""
Retrain ALL models on 22k.csv, then run the full evaluation suite on validatemarch.csv.

Steps:
  1. Build training splits from 22k.csv -> training_tables/
  2. Train 5-tower baseline -> exports/multitower_sale_5towers
  3. Train CatBoost replica (uses training_tables when tuappend.csv absent) -> exports/catboost_model_replica
  4. Run KS optimization with CatBoost configs -> ks_with_catboost_results.csv
  5. Build best config (replace housing with CatBoost tower) -> exports/multitower_sale_5towers_best
  6. Train 4-tower custom -> exports/multitower_sale_4towers_custom
  7. Build 3-tower drop-lead from 4-tower -> exports/multitower_sale_3towers_custom_drop_lead
  8. Prepare validatemarch (extract routing_transunion_raw)
  9. Run comprehensive_model_comparison --dataset validatemarch

So all six models (5tower_baseline, 5tower_catboost, 3tower_drop_lead, catboost_standalone,
2tower_age_bps, 4tower_drop_investments) are retrained on 22k and evaluated on validatemarch.

Usage:
  python run_retrain_all_22k_validate_march.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIONAL = Path(__file__).resolve().parent
TUAPPEND = REPO_ROOT / "tuappend.csv"
TUAPPEND_BAK = REPO_ROOT / "tuappend.csv.bak"


def run(cmd, description):
    print(f"\n{'='*70}\n{description}\n{'='*70}")
    r = subprocess.run(cmd, cwd=REPO_ROOT, shell=False)
    if r.returncode != 0:
        print(f"Failed: {description}", file=sys.stderr)
        sys.exit(r.returncode)


def main():
    # Ensure CatBoost replica trains from 22k (training_tables) not tuappend
    if TUAPPEND.exists():
        print("Temporarily renaming tuappend.csv so CatBoost replica trains from 22k splits...")
        shutil.move(str(TUAPPEND), str(TUAPPEND_BAK))
        restore_tuappend = True
    else:
        restore_tuappend = False

    try:
        run(
            [sys.executable, str(OPTIONAL / "build_training_splits_from_22k.py")],
            "Step 1: Build training splits from 22k.csv",
        )
        run(
            [sys.executable, str(REPO_ROOT / "train_multitower_sale_5towers.py")],
            "Step 2: Train 5-tower baseline on 22k",
        )
        run(
            [sys.executable, str(REPO_ROOT / "train_catboost_model_replica.py")],
            "Step 3: Train CatBoost replica on 22k",
        )
        run(
            [sys.executable, str(REPO_ROOT / "compare_ks_with_catboost.py")],
            "Step 4: KS optimization with CatBoost configs (22k val/test/holdout)",
        )
        run(
            [sys.executable, str(REPO_ROOT / "build_best_config_model.py")],
            "Step 5: Build best config (replace housing with CatBoost) -> multitower_sale_5towers_best",
        )
        run(
            [sys.executable, str(OPTIONAL / "train_multitower_sale_4towers_custom.py")],
            "Step 6: Train 4-tower custom on 22k",
        )
        run(
            [sys.executable, str(OPTIONAL / "build_3tower_from_4tower_custom.py")],
            "Step 7: Build 3-tower variants (includes drop_lead)",
        )
        run(
            [sys.executable, str(OPTIONAL / "prepare_validatemarch_for_eval.py")],
            "Step 8: Prepare validatemarch (routing_transunion_raw)",
        )
        run(
            [sys.executable, str(OPTIONAL / "comprehensive_model_comparison.py"), "--dataset", "validatemarch"],
            "Step 9: Full evaluation suite on validatemarch",
        )
    finally:
        if restore_tuappend and TUAPPEND_BAK.exists():
            print("Restoring tuappend.csv...")
            shutil.move(str(TUAPPEND_BAK), str(TUAPPEND))

    print("\nDone. All models retrained on 22k; validatemarch evals in comprehensive_model_comparison/")
    print("Summary: comprehensive_model_comparison/summary_results.csv")


if __name__ == "__main__":
    main()
