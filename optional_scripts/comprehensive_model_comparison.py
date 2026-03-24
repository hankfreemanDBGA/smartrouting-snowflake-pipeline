#!/usr/bin/env python3
"""
Comprehensive analysis comparing three models across feb6, feb10, and feb11 datasets:
1. 5-tower baseline (no CatBoost tower)
2. 5-tower with CatBoost replacing housing
3. 3-tower drop lead

For each dataset and model, runs:
- Rescore using routing_transunion_raw
- Confusion matrix analysis
- Tier close rate analysis
- Decile analysis
- Comparison visualizations

Usage:
  python comprehensive_model_comparison.py
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Model configurations
MODELS = {
    "5tower_baseline": {
        "name": "5-Tower Baseline",
        "model_dir": REPO_ROOT / "exports" / "multitower_sale_5towers",
        "proba_col": "PREDICTION_PROBA_5tower",
        "tier_col": "tier_5tower",
    },
    "5tower_catboost": {
        "name": "5-Tower + CatBoost",
        "model_dir": REPO_ROOT / "exports" / "multitower_sale_5towers_best",
        "proba_col": "PREDICTION_PROBA_5tower_catboost",
        "tier_col": "tier_5tower_catboost",
    },
    "3tower_drop_lead": {
        "name": "3-Tower Drop Lead",
        "model_dir": REPO_ROOT / "exports" / "multitower_sale_3towers_custom_drop_lead",
        "proba_col": "PREDICTION_PROBA_3tower",
        "tier_col": "tier_3tower",
    },
    "catboost_standalone": {
        "name": "CatBoost standalone",
        "model_dir": REPO_ROOT / "exports" / "catboost_model_replica",
        "proba_col": "PREDICTION_PROBA_catboost",
        "tier_col": "tier_catboost",
        "is_catboost_standalone": True,
    },
    "2tower_age_bps": {
        "name": "2-Tower (Age/Gender + BPS/Income)",
        "model_dir": REPO_ROOT / "exports" / "multitower_sale_4towers_custom",  # Extract from 4-tower
        "proba_col": "PREDICTION_PROBA_2tower",
        "tier_col": "tier_2tower",
        "tower_subset": ["age_gender", "bps_income"],  # Extract these towers
    },
    "4tower_drop_investments": {
        "name": "4-Tower Drop Investments",
        "model_dir": REPO_ROOT / "exports" / "multitower_sale_5towers_best",
        "proba_col": "PREDICTION_PROBA_4tower_drop_inv",
        "tier_col": "tier_4tower_drop_inv",
    },
}

# Datasets
DATASETS = {
    "feb6": {
        "input": REPO_ROOT / "5towertraining_feb6_api_results.csv",
        "name": "Feb 6",
    },
    "feb10": {
        "input": REPO_ROOT / "feb10testapi_api_results.csv",
        "name": "Feb 10",
    },
    "feb11": {
        "input": REPO_ROOT / "feb11apitest_api_results.csv",
        "name": "Feb 11",
    },
    "feb1011": {
        "input": REPO_ROOT / "feb1011testing.csv",
        "name": "Feb 10-11 Testing",
    },
    "feb13": {
        "input": REPO_ROOT / "feb13testing_api_results.csv",
        "name": "Feb 13 Testing",
    },
    "feb9": {
        "input": REPO_ROOT / "feb9testing_api_results.csv",
        "name": "Feb 9 Testing",
    },
    "validatemarch": {
        "input": REPO_ROOT / "validatemarch_api_results.csv",
        "name": "Validate March",
    },
}


def rescore_dataset_catboost_standalone(input_csv, model_key, model_config):
    """Rescore using CatBoost model standalone."""
    import joblib
    
    cb_model_path = model_config["model_dir"] / "model.pkl"
    cb_meta_path = model_config["model_dir"] / "metadata.pkl"
    
    if not cb_model_path.exists():
        print(f"  Error: CatBoost model not found at {cb_model_path}")
        return None
    
    print(f"  Loading CatBoost model from {cb_model_path}")
    cb_model, cb_features, _ = joblib.load(cb_model_path)
    
    # Load categorical features
    cb_cat_features = []
    if cb_meta_path.exists():
        try:
            cb_meta = joblib.load(cb_meta_path)
            if isinstance(cb_meta, dict) and 'cat_cols' in cb_meta:
                cb_cat_features = [f for f in cb_features if f in cb_meta['cat_cols']]
        except:
            pass
    
    # Fallback: try original metadata
    if not cb_cat_features:
        orig_meta_path = REPO_ROOT / "catboost_metadata" / "close_rate_model_v4_metadata.pkl"
        if orig_meta_path.exists():
            try:
                orig_meta = joblib.load(orig_meta_path)
                if isinstance(orig_meta, dict) and 'cat_cols' in orig_meta:
                    cb_cat_features = [f for f in cb_features if f in orig_meta['cat_cols']]
            except:
                pass
    
    from inference_server.app_5tower_unified import _feature_row_from_tu_response
    
    df = pd.read_csv(input_csv)
    n = len(df)
    print(f"  Processing {n} rows")
    
    proba_col = model_config["proba_col"]
    tier_col = model_config["tier_col"]
    pred_col = f"PREDICTION_{model_key}"
    
    proba_list = []
    tier_list = []
    pred_list = []
    errors = 0
    
    for i, row in df.iterrows():
        raw_val = row.get("routing_transunion_raw")
        if pd.isna(raw_val) or raw_val == "" or (isinstance(raw_val, str) and raw_val.strip() == ""):
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            continue
        
        try:
            tu_response = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
        except Exception as e:
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            errors += 1
            if errors <= 3:
                print(f"    Row {i}: parse error: {e}")
            continue
        
        try:
            feature_row = _feature_row_from_tu_response(tu_response)
            
            # Prepare CatBoost features
            cb_cats = set(cb_cat_features)
            cb_df = pd.DataFrame(index=[0])
            for f in cb_features:
                if f in feature_row:
                    val = feature_row[f]
                    if f in cb_cats:
                        cb_df[f] = str(val) if val is not None and pd.notna(val) else 'MISSING'
                    else:
                        cb_df[f] = val
                else:
                    cb_df[f] = 'MISSING' if f in cb_cats else 0
            
            for col in cb_df.columns:
                if col in cb_cats:
                    cb_df[col] = cb_df[col].astype(str).replace('nan', 'MISSING').fillna('MISSING')
                else:
                    cb_df[col] = pd.to_numeric(cb_df[col], errors='coerce').fillna(0)
            
            proba = cb_model.predict_proba(cb_df)[0, 1]
            pred_val = 1 if proba >= 0.5 else 0
            
            # Assign tier
            if proba >= 0.75:
                tier = "Gold"
            elif proba >= 0.5:
                tier = "Silver"
            elif proba >= 0.25:
                tier = "Bronze"
            else:
                tier = "Tin"
            
            proba_list.append(round(proba, 6))
            tier_list.append(tier)
            pred_list.append(pred_val)
        except Exception as e:
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            errors += 1
            if errors <= 5:
                print(f"    Row {i}: predict error: {e}")
    
    df[proba_col] = proba_list
    df[tier_col] = tier_list
    df[pred_col] = pred_list
    
    n_scored = sum(1 for p in proba_list if p is not None)
    print(f"  Scored: {n_scored}/{n} rows, {errors} errors")
    
    return df


def rescore_dataset_2tower(input_csv, model_key, model_config):
    """Rescore using 2-tower model (extract subset from 4-tower)."""
    import joblib
    
    # Load 4-tower model
    model_path = model_config["model_dir"] / "model.pkl"
    if not model_path.exists():
        print(f"  Error: Model not found at {model_path}")
        return None
    
    print(f"  Loading 4-tower model from {model_path}")
    towers, meta, _ = joblib.load(model_path)
    
    # Extract subset towers
    tower_subset = model_config.get("tower_subset", [])
    tower_indices = []
    subset_towers = []
    for i, (name, model, feat_list) in enumerate(towers):
        if name in tower_subset:
            tower_indices.append(i)
            subset_towers.append((name, model, feat_list))
    
    if len(subset_towers) != len(tower_subset):
        print(f"  Error: Could not find all towers {tower_subset} in model")
        return None
    
    # Load lookups
    lookups_dir = model_config["model_dir"] / "lookups"
    lookups = {}
    if lookups_dir.exists():
        for lookup_file in lookups_dir.glob("*.json"):
            with open(lookup_file, 'r') as f:
                lookups[lookup_file.stem] = json.load(f)
    
    from inference_server.app_5tower_unified import _feature_row_from_tu_response
    from ml.train_router import encode_df
    
    df = pd.read_csv(input_csv)
    n = len(df)
    print(f"  Processing {n} rows")
    
    proba_col = model_config["proba_col"]
    tier_col = model_config["tier_col"]
    pred_col = f"PREDICTION_{model_key}"
    
    proba_list = []
    tier_list = []
    pred_list = []
    errors = 0
    
    for i, row in df.iterrows():
        raw_val = row.get("routing_transunion_raw")
        if pd.isna(raw_val) or raw_val == "" or (isinstance(raw_val, str) and raw_val.strip() == ""):
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            continue
        
        try:
            tu_response = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
        except Exception as e:
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            errors += 1
            if errors <= 3:
                print(f"    Row {i}: parse error: {e}")
            continue
        
        try:
            feature_row = _feature_row_from_tu_response(tu_response)
            feature_df = pd.DataFrame([feature_row])
            if "SALE_MADE_FLAG" not in feature_df.columns:
                feature_df["SALE_MADE_FLAG"] = 0
            
            # Get tower predictions
            P_list = []
            for name, model, feat_list in subset_towers:
                X, _ = encode_df(feature_df, lookups, feature_list=feat_list)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
                p = model.predict_proba(X)[0, 1]
                P_list.append(p)
            
            # For 2-tower, use simple average (meta-model expects 4 inputs)
            # Alternatively, could train a 2-tower meta-model, but average is simpler for comparison
            proba = np.mean(P_list)
            pred_val = 1 if proba >= 0.5 else 0
            
            # Assign tier
            if proba >= 0.75:
                tier = "Gold"
            elif proba >= 0.5:
                tier = "Silver"
            elif proba >= 0.25:
                tier = "Bronze"
            else:
                tier = "Tin"
            
            proba_list.append(round(proba, 6))
            tier_list.append(tier)
            pred_list.append(pred_val)
        except Exception as e:
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            errors += 1
            if errors <= 5:
                print(f"    Row {i}: predict error: {e}")
    
    df[proba_col] = proba_list
    df[tier_col] = tier_list
    df[pred_col] = pred_list
    
    n_scored = sum(1 for p in proba_list if p is not None)
    print(f"  Scored: {n_scored}/{n} rows, {errors} errors")
    
    return df


def rescore_dataset(input_csv, model_key, model_config):
    """Rescore a dataset with a model using routing_transunion_raw."""
    print(f"\n{'='*70}")
    print(f"Rescoring {input_csv.name} with {model_config['name']}")
    print(f"{'='*70}")
    
    if not input_csv.exists():
        print(f"  Skipping: {input_csv} not found")
        return None
    
    # Check for routing_transunion_raw
    df_check = pd.read_csv(input_csv, nrows=1)
    if "routing_transunion_raw" not in df_check.columns:
        print(f"  Skipping: {input_csv} missing routing_transunion_raw")
        return None
    
    # Handle special cases
    if model_config.get("is_catboost_standalone"):
        return rescore_dataset_catboost_standalone(input_csv, model_key, model_config)
    
    if model_config.get("tower_subset"):
        return rescore_dataset_2tower(input_csv, model_key, model_config)
    
    # Standard multitower model
    os.environ["MODEL_PATH"] = str(model_config["model_dir"].resolve())
    from inference_server.app_5tower_unified import (
        _load_model,
        _feature_row_from_tu_response,
        _run_predict,
    )
    
    print(f"  Loading model from {model_config['model_dir']}")
    try:
        _load_model()
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None
    
    df = pd.read_csv(input_csv)
    n = len(df)
    print(f"  Processing {n} rows")
    
    proba_col = model_config["proba_col"]
    tier_col = model_config["tier_col"]
    pred_col = f"PREDICTION_{model_key}"
    
    proba_list = []
    tier_list = []
    pred_list = []
    errors = 0
    
    for i, row in df.iterrows():
        raw_val = row.get("routing_transunion_raw")
        if pd.isna(raw_val) or raw_val == "" or (isinstance(raw_val, str) and raw_val.strip() == ""):
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            continue
        
        try:
            tu_response = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
        except Exception as e:
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            errors += 1
            if errors <= 3:
                print(f"    Row {i}: parse error: {e}")
            continue
        
        try:
            feature_row = _feature_row_from_tu_response(tu_response)
            pred_val, proba, tier = _run_predict(feature_row)
            proba_list.append(round(proba, 6))
            tier_list.append(tier)
            pred_list.append(pred_val)
        except Exception as e:
            proba_list.append(None)
            tier_list.append(None)
            pred_list.append(None)
            errors += 1
            if errors <= 5:
                print(f"    Row {i}: predict error: {e}")
    
    df[proba_col] = proba_list
    df[tier_col] = tier_list
    df[pred_col] = pred_list
    
    n_scored = sum(1 for p in proba_list if p is not None)
    print(f"  Scored: {n_scored}/{n} rows, {errors} errors")
    
    return df


def run_analysis(df, dataset_key, model_key, model_config, output_dir, dataset_config):
    """Run confusion, tier, and decile analysis on scored data."""
    proba_col = model_config["proba_col"]
    tier_col = model_config["tier_col"]
    
    # Filter to scored rows
    scored = df[df[proba_col].notna()].copy()
    if len(scored) == 0:
        print(f"  No scored rows for analysis")
        return None
    
    # Create analysis-ready CSV: set PREDICTION_PROBA and tier from model-specific columns
    # Also set api_action='model' and api_model for compatibility with existing analysis scripts
    analysis_df = scored.copy()
    analysis_df["PREDICTION_PROBA"] = analysis_df[proba_col]
    analysis_df["tier"] = analysis_df[tier_col]
    analysis_df["api_action"] = "model"
    analysis_df["api_model"] = model_key
    analysis_df["PREDICTION"] = (analysis_df[proba_col].astype(float) >= 0.5).astype(int)
    
    # Save analysis-ready CSV
    analysis_csv = output_dir / f"{dataset_key}_{model_key}_for_analysis.csv"
    analysis_df.to_csv(analysis_csv, index=False)
    
    # Prepare outcome for summary stats
    if "SALE_MADE_FLAG" in scored.columns:
        sale_flag = scored["SALE_MADE_FLAG"].astype(str).str.upper()
        scored["y_true"] = ((sale_flag == "Y") | (sale_flag == "1")).astype(int)
    elif "SALE_MADE_BINARY" in scored.columns:
        scored["y_true"] = pd.to_numeric(scored["SALE_MADE_BINARY"], errors="coerce").fillna(0).astype(int)
    else:
        scored["y_true"] = 0
    
    scored["y_pred"] = (scored[proba_col].astype(float) >= 0.5).astype(int)
    scored["proba"] = scored[proba_col].astype(float)
    
    results = {
        "dataset": dataset_key,
        "model": model_key,
        "model_name": model_config["name"],
        "n_scored": len(scored),
        "n_sales": int(scored["y_true"].sum()),
    }
    
    # Confusion matrix
    cm = sk_confusion_matrix(scored["y_true"], scored["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    results.update({
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round((tp + tn) / len(scored), 4) if len(scored) > 0 else 0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0,
    })
    
    # KS statistic
    if scored["y_true"].sum() > 0 and (scored["y_true"] == 0).sum() > 0:
        proba_sale = scored[scored["y_true"] == 1]["proba"]
        proba_no_sale = scored[scored["y_true"] == 0]["proba"]
        ks_stat, _ = ks_2samp(proba_sale, proba_no_sale)
        results["KS"] = round(ks_stat, 4)
    else:
        results["KS"] = None
    
    # Tier close rates
    tier_order = ["Gold", "Silver", "Bronze", "Tin"]
    tier_stats = []
    for tier in tier_order:
        tier_rows = scored[scored[tier_col] == tier]
        n_tier = len(tier_rows)
        sales_tier = int(tier_rows["y_true"].sum())
        close_rate = round(sales_tier / n_tier, 4) if n_tier > 0 else None
        tier_stats.append({
            "tier": tier,
            "n": n_tier,
            "sales": sales_tier,
            "close_rate": close_rate,
        })
        results[f"{tier.lower()}_n"] = n_tier
        results[f"{tier.lower()}_close_rate"] = close_rate
    
    overall_close_rate = round(scored["y_true"].sum() / len(scored), 4) if len(scored) > 0 else 0
    results["overall_close_rate"] = overall_close_rate
    
    # Save individual analysis outputs
    prefix = f"{dataset_key}_{model_key}_"
    
    # Save tier close rates CSV
    tier_df = pd.DataFrame(tier_stats)
    tier_csv = output_dir / f"{prefix}tier_close_rates.csv"
    tier_df.to_csv(tier_csv, index=False)
    print(f"    Saved: {tier_csv.name}")
    
    # Save confusion matrix stats CSV
    confusion_csv = output_dir / f"{prefix}confusion_matrix_stats.csv"
    confusion_df = pd.DataFrame([{
        "model": model_key,
        "TN": results["TN"], "FP": results["FP"], "FN": results["FN"], "TP": results["TP"],
        "accuracy": results["accuracy"], "precision": results["precision"],
        "recall": results["recall"], "specificity": round(results["TN"] / (results["TN"] + results["FP"]), 4) if (results["TN"] + results["FP"]) > 0 else 0,
        "f1": results["f1"], "n": results["n_scored"],
    }])
    confusion_df.to_csv(confusion_csv, index=False)
    print(f"    Saved: {confusion_csv.name}")
    
    # Save decile analysis CSV
    try:
        scored["decile"] = pd.qcut(scored["proba"], q=10, labels=False, duplicates="drop") + 1
        decile_rows = []
        for d in sorted(scored["decile"].dropna().unique()):
            decile_sub = scored[scored["decile"] == d]
            y_t = decile_sub["y_true"].values
            y_p = decile_sub["y_pred"].values
            tp = int(((y_t == 1) & (y_p == 1)).sum())
            tn = int(((y_t == 0) & (y_p == 0)).sum())
            fp = int(((y_t == 0) & (y_p == 1)).sum())
            fn = int(((y_t == 1) & (y_p == 0)).sum())
            decile_rows.append({
                "decile": int(d),
                "n": len(decile_sub),
                "proba_min": round(decile_sub["proba"].min(), 4),
                "proba_max": round(decile_sub["proba"].max(), 4),
                "proba_mean": round(decile_sub["proba"].mean(), 4),
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                "sales": int(decile_sub["y_true"].sum()),
                "close_rate": round(decile_sub["y_true"].mean(), 4),
                "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
                "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0,
            })
        decile_df = pd.DataFrame(decile_rows)
        decile_csv = output_dir / f"{prefix}decile_analysis.csv"
        decile_df.to_csv(decile_csv, index=False)
        print(f"    Saved: {decile_csv.name}")
    except Exception as e:
        print(f"    Decile analysis error: {e}")
    
    # Create visualizations
    try:
        # Probability distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        proba_vals = scored["proba"]
        if scored["y_true"].sum() > 0 and (scored["y_true"] == 0).sum() > 0:
            p0 = proba_vals[scored["y_true"] == 0]
            p1 = proba_vals[scored["y_true"] == 1]
            ax.hist(p0, bins=30, alpha=0.5, color="steelblue", density=True, label="No sale")
            ax.hist(p1, bins=30, alpha=0.5, color="coral", density=True, label="Sale")
            ax.legend()
        else:
            ax.hist(proba_vals, bins=40, color="steelblue", alpha=0.7, density=True)
        ax.set_xlabel("P(sale)")
        ax.set_ylabel("Density")
        ax.set_title(f"{dataset_config['name']} - {model_config['name']} - Probability Distribution")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}proba_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {prefix}proba_distribution.png")
        
        # Tier distribution bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        tier_counts = [tier_stats[i]["n"] for i in range(4)]
        colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#808080"]
        ax.bar(tier_order, tier_counts, color=colors, edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_xlabel("Tier")
        ax.set_title(f"{dataset_config['name']} - {model_config['name']} - Tier Distribution")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}tier_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {prefix}tier_distribution.png")
    except Exception as e:
        print(f"    Visualization error: {e}")
    
    return results


def create_comparison_visualizations(all_results, output_dir):
    """Create comparison charts across datasets and models."""
    df_results = pd.DataFrame(all_results)
    
    model_list = [("5tower_baseline", "5-Tower Baseline"),
                  ("5tower_catboost", "5-Tower + CatBoost"),
                  ("3tower_drop_lead", "3-Tower Drop Lead"),
                  ("catboost_standalone", "CatBoost standalone"),
                  ("2tower_age_bps", "2-Tower Age/BPS")]
    
    # 1. KS comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    datasets = df_results["dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.15
    
    for i, (model_key, model_name) in enumerate(model_list):
        model_data = df_results[df_results["model"] == model_key]
        ks_vals = []
        for ds in datasets:
            ds_data = model_data[model_data["dataset"] == ds]
            if len(ds_data) > 0 and pd.notna(ds_data["KS"].iloc[0]):
                ks_vals.append(ds_data["KS"].iloc[0])
            else:
                ks_vals.append(0)
        ax.bar(x + i * width, ks_vals, width, label=model_name)
    
    ax.set_xlabel("Dataset")
    ax.set_ylabel("KS Statistic")
    ax.set_title("KS Statistic Comparison Across Models and Datasets")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([DATASETS[d]["name"] for d in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_ks_statistic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: comparison_ks_statistic.png")
    
    # 2. Overall close rate comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model_key, model_name) in enumerate(model_list):
        model_data = df_results[df_results["model"] == model_key]
        close_rates = []
        for ds in datasets:
            ds_data = model_data[model_data["dataset"] == ds]
            if len(ds_data) > 0:
                close_rates.append(ds_data["overall_close_rate"].iloc[0] * 100)
            else:
                close_rates.append(0)
        ax.bar(x + i * width, close_rates, width, label=model_name)
    
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Close Rate (%)")
    ax.set_title("Overall Close Rate Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([DATASETS[d]["name"] for d in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_close_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: comparison_close_rate.png")
    
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Close Rate (%)")
    ax.set_title("Overall Close Rate Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASETS[d]["name"] for d in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_close_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: comparison_close_rate.png")
    
    # 3. Tier distribution comparison (heatmap)
    tier_order = ["Gold", "Silver", "Bronze", "Tin"]
    heatmap_data = []
    for dataset in datasets:
        row = []
        for model_key in ["5tower_baseline", "5tower_catboost", "3tower_drop_lead", "catboost_standalone", "2tower_age_bps"]:
            model_data = df_results[(df_results["dataset"] == dataset) & 
                                    (df_results["model"] == model_key)]
            if len(model_data) > 0:
                total = model_data["n_scored"].iloc[0]
                if total > 0:
                    gold_pct = (model_data["gold_n"].iloc[0] / total * 100) if pd.notna(model_data["gold_n"].iloc[0]) else 0
                    silver_pct = (model_data["silver_n"].iloc[0] / total * 100) if pd.notna(model_data["silver_n"].iloc[0]) else 0
                    bronze_pct = (model_data["bronze_n"].iloc[0] / total * 100) if pd.notna(model_data["bronze_n"].iloc[0]) else 0
                    tin_pct = (model_data["tin_n"].iloc[0] / total * 100) if pd.notna(model_data["tin_n"].iloc[0]) else 0
                    row.extend([gold_pct, silver_pct, bronze_pct, tin_pct])
                else:
                    row.extend([0, 0, 0, 0])
            else:
                row.extend([0, 0, 0, 0])
        heatmap_data.append(row)
    
    model_labels = ["5T-Base", "5T-CatBoost", "3T", "CatBoost", "2T"]
    # Ensure we have the right number of columns (5 models × 4 tiers = 20)
    expected_cols = len(model_labels) * len(tier_order)
    if len(heatmap_data[0]) != expected_cols:
        print(f"  Warning: Heatmap data has {len(heatmap_data[0])} columns, expected {expected_cols}")
        # Pad or truncate rows if needed
        for i, row in enumerate(heatmap_data):
            if len(row) < expected_cols:
                heatmap_data[i] = row + [0] * (expected_cols - len(row))
            elif len(row) > expected_cols:
                heatmap_data[i] = row[:expected_cols]
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                               index=[DATASETS[d]["name"] for d in datasets],
                               columns=[f"{m}_{t}" for m in model_labels for t in tier_order])
    
    fig, ax = plt.subplots(figsize=(16, 6))
    if sns is not None:
        sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "% of Scored Rows"})
    else:
        im = ax.imshow(heatmap_df.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(heatmap_df.columns)))
        ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index)
        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                ax.text(j, i, f"{heatmap_df.iloc[i, j]:.1f}", ha="center", va="center")
        plt.colorbar(im, ax=ax, label="% of Scored Rows")
    ax.set_title("Tier Distribution Comparison (% of Scored Rows)")
    ax.set_xlabel("Model-Tier")
    ax.set_ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_tier_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: comparison_tier_distribution.png")


def create_combined_proba_distribution(all_scored_data, output_dir):
    """Create combined probability distribution plot for all models/datasets, filtered to answered calls + linkage score >= 75000 + age not null."""
    # Calculate grid size: 2 datasets × 5 models = 10 plots, use 2 rows × 5 cols
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for dataset_key, dataset_config in DATASETS.items():
        for model_key, model_config in MODELS.items():
            # Find matching scored data
            scored_item = None
            for item in all_scored_data:
                if item["dataset"] == dataset_key and item["model"] == model_key:
                    scored_item = item
                    break
            
            if scored_item is None:
                continue
            
            df = scored_item["data"]
            proba_col = model_config["proba_col"]
            
            # Filter to answered calls only
            if "ANSWERED_FLAG" in df.columns:
                answered = df[df["ANSWERED_FLAG"].notna()].copy()
                # Check if ANSWERED_FLAG is Y/1/True
                answered_flag = answered["ANSWERED_FLAG"].astype(str).str.upper()
                answered = answered[(answered_flag == "Y") | (answered_flag == "1") | (answered_flag == "TRUE")]
            else:
                answered = df.copy()
            
            # Extract linkage score and age for filtering
            # Check if PHONE_1_LINKAGE_SCORE exists in original columns (from input CSV)
            linkage_scores = []
            ages = []
            
            for _, row in answered.iterrows():
                linkage_score = None
                age_val = None
                
                # Try original PHONE_1_LINKAGE_SCORE column first (if exists in input data)
                if "PHONE_1_LINKAGE_SCORE" in answered.columns:
                    linkage_score = row.get("PHONE_1_LINKAGE_SCORE")
                
                # Try to get age from routing_age or tu_age columns first
                if "routing_age" in answered.columns:
                    age_val = row.get("routing_age")
                elif "tu_age" in answered.columns:
                    age_val = row.get("tu_age")
                
                # Extract from raw TU response if needed
                if linkage_score is None or age_val is None:
                    raw_val = row.get("routing_transunion_raw")
                    if pd.notna(raw_val) and raw_val != "":
                        try:
                            tu_response = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                            if isinstance(tu_response, dict) and "response" in tu_response:
                                resp_data = tu_response["response"]
                                individuals = resp_data.get("individuals", [])
                                if individuals:
                                    indiv = individuals[0]
                                    
                                    # Get linkage score from phones array (TU API uses phone_linkage_score, 0-100 scale)
                                    if linkage_score is None:
                                        phones = indiv.get("phones", [])
                                        if phones and len(phones) > 0:
                                            phone_data = phones[0]
                                            linkage_score = phone_data.get("linkage_score") or phone_data.get("phone_linkage_score")
                                    
                                    # Get age if not already found
                                    if age_val is None:
                                        attrs = indiv.get("attributes", {})
                                        age_val = attrs.get("age")
                        except:
                            pass
                
                linkage_scores.append(linkage_score)
                ages.append(age_val)
            
            answered["_linkage_score"] = linkage_scores
            answered["_age"] = ages
            
            # Apply filters
            filtered = answered.copy()
            
            # Linkage score filter: >= 75000 if from PHONE_1_LINKAGE_SCORE, or >= 75 if from TU API (0-100 scale)
            if "_linkage_score" in filtered.columns:
                linkage_numeric = pd.to_numeric(filtered["_linkage_score"], errors="coerce")
                # Check if values are in 0-100 range (TU API) or larger (training data)
                if linkage_numeric.notna().any():
                    max_val = linkage_numeric.max()
                    if max_val <= 100:
                        # TU API scale: use >= 75 (equivalent to 75000/1000)
                        filtered = filtered[linkage_numeric.notna() & (linkage_numeric >= 75)].copy()
                    else:
                        # Training data scale: use >= 75000
                        filtered = filtered[linkage_numeric.notna() & (linkage_numeric >= 75000)].copy()
            
            # Age filter: not null
            if "_age" in filtered.columns:
                age_numeric = pd.to_numeric(filtered["_age"], errors="coerce")
                filtered = filtered[age_numeric.notna()].copy()
            
            # Filter to scored rows
            answered_scored = filtered[filtered[proba_col].notna()].copy()
            
            if len(answered_scored) == 0:
                continue
            
            ax = axes[plot_idx]
            proba_vals = answered_scored[proba_col].astype(float)
            
            # Prepare outcome
            if "SALE_MADE_FLAG" in answered_scored.columns:
                sale_flag = answered_scored["SALE_MADE_FLAG"].astype(str).str.upper()
                y_true = ((sale_flag == "Y") | (sale_flag == "1")).astype(int)
            elif "SALE_MADE_BINARY" in answered_scored.columns:
                y_true = pd.to_numeric(answered_scored["SALE_MADE_BINARY"], errors="coerce").fillna(0).astype(int)
            else:
                y_true = pd.Series([0] * len(answered_scored))
            
            if y_true.sum() > 0 and (y_true == 0).sum() > 0:
                p0 = proba_vals[y_true == 0]
                p1 = proba_vals[y_true == 1]
                ax.hist(p0, bins=30, alpha=0.5, color="steelblue", density=True, label="No sale")
                ax.hist(p1, bins=30, alpha=0.5, color="coral", density=True, label="Sale")
                ax.legend()
            else:
                ax.hist(proba_vals, bins=40, color="steelblue", alpha=0.7, density=True)
            
            ax.set_xlabel("P(sale)")
            ax.set_ylabel("Density")
            ax.set_title(f"{dataset_config['name']} - {model_config['name']}\n(n={len(answered_scored)}, answered + linkage≥75 + age)")
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
            if plot_idx >= 10:
                break
    
    # Hide unused subplots
    for i in range(plot_idx, 10):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_proba_distributions_answered_only.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: combined_proba_distributions_answered_only.png")


def create_threshold_analysis_table(all_scored_data, output_dir):
    """Create table showing close rates for different gold/silver/bronze threshold combinations. Filtered to answered calls + linkage score >= 75000 + age not null."""
    # Define threshold combinations to test
    threshold_combos = [
        {"gold": 0.75, "silver": 0.5, "bronze": 0.25, "name": "Standard (0.75/0.5/0.25)"},
        {"gold": 0.6, "silver": 0.4, "bronze": 0.2, "name": "Lower (0.6/0.4/0.2)"},
        {"gold": 0.7, "silver": 0.45, "bronze": 0.25, "name": "Medium (0.7/0.45/0.25)"},
        {"gold": 0.65, "silver": 0.4, "bronze": 0.2, "name": "Medium-Low (0.65/0.4/0.2)"},
        {"gold": 0.8, "silver": 0.55, "bronze": 0.3, "name": "Higher (0.8/0.55/0.3)"},
        {"gold": 0.6, "silver": 0.3, "bronze": 0.15, "name": "Custom 3-tower style (0.6/0.3/0.15)"},
    ]
    
    results_rows = []
    
    for dataset_key, dataset_config in DATASETS.items():
        for model_key, model_config in MODELS.items():
            # Find matching scored data
            scored_item = None
            for item in all_scored_data:
                if item["dataset"] == dataset_key and item["model"] == model_key:
                    scored_item = item
                    break
            
            if scored_item is None:
                continue
            
            df = scored_item["data"]
            proba_col = model_config["proba_col"]
            
            # Filter to answered calls only
            if "ANSWERED_FLAG" in df.columns:
                answered = df[df["ANSWERED_FLAG"].notna()].copy()
                answered_flag = answered["ANSWERED_FLAG"].astype(str).str.upper()
                answered = answered[(answered_flag == "Y") | (answered_flag == "1") | (answered_flag == "TRUE")]
            else:
                answered = df.copy()
            
            # Extract linkage score and age for filtering
            linkage_scores = []
            ages = []
            
            for _, row in answered.iterrows():
                linkage_score = None
                age_val = None
                
                # Try original PHONE_1_LINKAGE_SCORE column first (if exists in input data)
                if "PHONE_1_LINKAGE_SCORE" in answered.columns:
                    linkage_score = row.get("PHONE_1_LINKAGE_SCORE")
                
                # Try to get age from routing_age or tu_age columns first
                if "routing_age" in answered.columns:
                    age_val = row.get("routing_age")
                elif "tu_age" in answered.columns:
                    age_val = row.get("tu_age")
                
                # Extract from raw TU response if needed
                if linkage_score is None or age_val is None:
                    raw_val = row.get("routing_transunion_raw")
                    if pd.notna(raw_val) and raw_val != "":
                        try:
                            tu_response = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                            if isinstance(tu_response, dict) and "response" in tu_response:
                                resp_data = tu_response["response"]
                                individuals = resp_data.get("individuals", [])
                                if individuals:
                                    indiv = individuals[0]
                                    
                                    # Get linkage score from phones array (TU API uses phone_linkage_score, 0-100 scale)
                                    if linkage_score is None:
                                        phones = indiv.get("phones", [])
                                        if phones and len(phones) > 0:
                                            phone_data = phones[0]
                                            linkage_score = phone_data.get("linkage_score") or phone_data.get("phone_linkage_score")
                                    
                                    # Get age if not already found
                                    if age_val is None:
                                        attrs = indiv.get("attributes", {})
                                        age_val = attrs.get("age")
                        except:
                            pass
                
                linkage_scores.append(linkage_score)
                ages.append(age_val)
            
            answered["_linkage_score"] = linkage_scores
            answered["_age"] = ages
            
            # Apply filters
            filtered = answered.copy()
            
            # Linkage score filter: >= 75000 if from PHONE_1_LINKAGE_SCORE, or >= 75 if from TU API (0-100 scale)
            if "_linkage_score" in filtered.columns:
                linkage_numeric = pd.to_numeric(filtered["_linkage_score"], errors="coerce")
                # Check if values are in 0-100 range (TU API) or larger (training data)
                if linkage_numeric.notna().any():
                    max_val = linkage_numeric.max()
                    if max_val <= 100:
                        # TU API scale: use >= 75 (equivalent to 75000/1000)
                        filtered = filtered[linkage_numeric.notna() & (linkage_numeric >= 75)].copy()
                    else:
                        # Training data scale: use >= 75000
                        filtered = filtered[linkage_numeric.notna() & (linkage_numeric >= 75000)].copy()
            
            # Age filter: not null
            if "_age" in filtered.columns:
                age_numeric = pd.to_numeric(filtered["_age"], errors="coerce")
                filtered = filtered[age_numeric.notna()].copy()
            
            # Filter to scored rows
            answered_scored = filtered[filtered[proba_col].notna()].copy()
            
            if len(answered_scored) == 0:
                continue
            
            # Prepare outcome
            if "SALE_MADE_FLAG" in answered_scored.columns:
                sale_flag = answered_scored["SALE_MADE_FLAG"].astype(str).str.upper()
                y_true = ((sale_flag == "Y") | (sale_flag == "1")).astype(int)
            elif "SALE_MADE_BINARY" in answered_scored.columns:
                y_true = pd.to_numeric(answered_scored["SALE_MADE_BINARY"], errors="coerce").fillna(0).astype(int)
            else:
                y_true = pd.Series([0] * len(answered_scored))
            
            proba_vals = answered_scored[proba_col].astype(float)
            
            # Test each threshold combination
            for thresh in threshold_combos:
                # Assign tiers based on thresholds
                def assign_tier(p):
                    if p >= thresh["gold"]:
                        return "Gold"
                    elif p >= thresh["silver"]:
                        return "Silver"
                    elif p >= thresh["bronze"]:
                        return "Bronze"
                    else:
                        return "Tin"
                
                answered_scored["tier_test"] = proba_vals.apply(assign_tier)
                
                # Calculate close rates per tier
                tier_stats = {}
                for tier in ["Gold", "Silver", "Bronze", "Tin"]:
                    tier_mask = answered_scored["tier_test"] == tier
                    tier_rows = answered_scored[tier_mask]
                    n_tier = len(tier_rows)
                    if n_tier > 0:
                        tier_y = y_true[tier_mask]
                        sales_tier = int(tier_y.sum())
                        close_rate = round(sales_tier / n_tier, 4)
                    else:
                        sales_tier = 0
                        close_rate = None
                    tier_stats[tier.lower()] = {
                        "n": n_tier,
                        "sales": sales_tier,
                        "close_rate": close_rate,
                    }
                
                overall_close_rate = round(y_true.sum() / len(answered_scored), 4) if len(answered_scored) > 0 else 0
                
                results_rows.append({
                    "dataset": dataset_key,
                    "model": model_config["name"],
                    "threshold_name": thresh["name"],
                    "gold_threshold": thresh["gold"],
                    "silver_threshold": thresh["silver"],
                    "bronze_threshold": thresh["bronze"],
                    "gold_n": tier_stats["gold"]["n"],
                    "gold_sales": tier_stats["gold"]["sales"],
                    "gold_close_rate": tier_stats["gold"]["close_rate"],
                    "silver_n": tier_stats["silver"]["n"],
                    "silver_sales": tier_stats["silver"]["sales"],
                    "silver_close_rate": tier_stats["silver"]["close_rate"],
                    "bronze_n": tier_stats["bronze"]["n"],
                    "bronze_sales": tier_stats["bronze"]["sales"],
                    "bronze_close_rate": tier_stats["bronze"]["close_rate"],
                    "tin_n": tier_stats["tin"]["n"],
                    "tin_sales": tier_stats["tin"]["sales"],
                    "tin_close_rate": tier_stats["tin"]["close_rate"],
                    "overall_n": len(answered_scored),
                    "overall_sales": int(y_true.sum()),
                    "overall_close_rate": overall_close_rate,
                })
    
    if results_rows:
        df_thresholds = pd.DataFrame(results_rows)
        threshold_csv = output_dir / "threshold_analysis_table.csv"
        df_thresholds.to_csv(threshold_csv, index=False)
        print(f"  Saved: {threshold_csv.name}")
        
        # Print summary
        print("\n  Threshold Analysis Summary (answered calls only):")
        print("  " + "="*100)
        for dataset in df_thresholds["dataset"].unique():
            print(f"\n  {DATASETS[dataset]['name']}:")
            dataset_df = df_thresholds[df_thresholds["dataset"] == dataset]
            for model in dataset_df["model"].unique():
                print(f"    {model}:")
                model_df = dataset_df[dataset_df["model"] == model]
                for _, row in model_df.iterrows():
                    print(f"      {row['threshold_name']}:")
                    print(f"        Gold:   n={row['gold_n']:4} sales={row['gold_sales']:3} rate={row['gold_close_rate']:.2%}" if row['gold_close_rate'] else f"        Gold:   n={row['gold_n']:4} sales={row['gold_sales']:3} rate=N/A")
                    print(f"        Silver: n={row['silver_n']:4} sales={row['silver_sales']:3} rate={row['silver_close_rate']:.2%}" if row['silver_close_rate'] else f"        Silver: n={row['silver_n']:4} sales={row['silver_sales']:3} rate=N/A")
                    print(f"        Bronze: n={row['bronze_n']:4} sales={row['bronze_sales']:3} rate={row['bronze_close_rate']:.2%}" if row['bronze_close_rate'] else f"        Bronze: n={row['bronze_n']:4} sales={row['bronze_sales']:3} rate=N/A")
                    print(f"        Tin:     n={row['tin_n']:4} sales={row['tin_sales']:3} rate={row['tin_close_rate']:.2%}" if row['tin_close_rate'] else f"        Tin:     n={row['tin_n']:4} sales={row['tin_sales']:3} rate=N/A")
                    print(f"        Overall: n={row['overall_n']:4} sales={row['overall_sales']:3} rate={row['overall_close_rate']:.2%}")
    else:
        print("  No threshold analysis data generated.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive model comparison: rescore and analyze.")
    parser.add_argument("--dataset", type=str, default=None, help="Run only this dataset key (e.g. feb1011)")
    parser.add_argument("--model", type=str, default=None, help="Run only this model key (e.g. 4tower_drop_investments)")
    args = parser.parse_args()

    output_dir = REPO_ROOT / "comprehensive_model_comparison"
    output_dir.mkdir(exist_ok=True)

    datasets_to_run = DATASETS
    models_to_run = MODELS
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Error: Unknown dataset '{args.dataset}'. Choose from: {list(DATASETS.keys())}", file=sys.stderr)
            sys.exit(1)
        datasets_to_run = {args.dataset: DATASETS[args.dataset]}
    if args.model:
        if args.model not in MODELS:
            print(f"Error: Unknown model '{args.model}'. Choose from: {list(MODELS.keys())}", file=sys.stderr)
            sys.exit(1)
        models_to_run = {args.model: MODELS[args.model]}
    
    print("="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    print(f"Output directory: {output_dir}")
    if args.dataset or args.model:
        print(f"Filter: dataset={args.dataset or 'all'}, model={args.model or 'all'}")
    
    all_results = []
    all_scored_data = []
    
    # Rescore and analyze each dataset with each model
    for dataset_key, dataset_config in datasets_to_run.items():
        input_csv = dataset_config["input"]
        
        for model_key, model_config in models_to_run.items():
            # Rescore
            df_scored = rescore_dataset(input_csv, model_key, model_config)
            if df_scored is None:
                continue
            
            # Save rescored CSV
            output_csv = output_dir / f"{dataset_key}_{model_key}_scored.csv"
            df_scored.to_csv(output_csv, index=False)
            print(f"  Saved: {output_csv.name}")
            
            # Run analysis
            results = run_analysis(df_scored, dataset_key, model_key, model_config, output_dir, dataset_config)
            if results:
                all_results.append(results)
                all_scored_data.append({
                    "dataset": dataset_key,
                    "model": model_key,
                    "data": df_scored,
                })
    
    # Save summary results CSV
    if all_results:
        df_summary = pd.DataFrame(all_results)
        summary_csv = output_dir / "summary_results.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"\nSaved summary: {summary_csv}")
        
        # Create comparison visualizations
        print("\nCreating comparison visualizations...")
        create_comparison_visualizations(all_results, output_dir)
        
        # Create combined probability distribution (answered calls only)
        print("\nCreating combined probability distribution (answered calls only)...")
        create_combined_proba_distribution(all_scored_data, output_dir)
        
        # Create threshold analysis table
        print("\nCreating threshold analysis table...")
        create_threshold_analysis_table(all_scored_data, output_dir)
        
        # Print summary table
        print("\n" + "="*70)
        print("SUMMARY RESULTS")
        print("="*70)
        print(df_summary[["dataset", "model_name", "n_scored", "n_sales", "KS", "overall_close_rate", 
                          "precision", "recall", "f1"]].to_string(index=False))
    else:
        print("\nNo results to summarize.")


if __name__ == "__main__":
    main()
