#!/usr/bin/env python3
"""
Build the best configuration model export:
- Replace housing tower with CatBoost replica tower
- Meta-model: saga solver, l1 penalty, C=0.050
- Export to exports/multitower_sale_5towers_best/ for deployment
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from ml.config import RANDOM_SEED
from ml.splits import load_train_val_test_holdout
from ml.train_router import encode_df, build_lookups_from_df

import train_multitower_sale_5towers as mt5

TARGET = "SALE_MADE_FLAG"
EXPORT_DIR = mt5.EXPORT_DIR
BEST_EXPORT_DIR = REPO_ROOT / "exports" / "multitower_sale_5towers_best"

# Best configuration parameters
BEST_CONFIG = {
    'replace_tower': 'housing',
    'solver': 'saga',
    'penalty': 'l1',
    'C': 0.050
}


def get_catboost_predictions(train_df, val_df, test_df):
    """Get predictions from the trained CatBoost replica (exports/catboost_model_replica)."""
    cb_path = REPO_ROOT / "exports" / "catboost_model_replica" / "model.pkl"
    cb_meta_path = REPO_ROOT / "exports" / "catboost_model_replica" / "metadata.pkl"

    if not cb_path.exists():
        print(f"CatBoost replica not found: {cb_path}")
        return None, None, None

    cb_model, cb_features, _ = joblib.load(cb_path)

    cat_feature_names = []
    if cb_meta_path.exists():
        try:
            meta = joblib.load(cb_meta_path)
            if isinstance(meta, dict) and 'cat_cols' in meta:
                cat_feature_names = [f for f in cb_features if f in meta['cat_cols']]
        except Exception:
            pass
    if not cat_feature_names:
        orig_path = REPO_ROOT / "catboost_metadata" / "close_rate_model_v4_metadata.pkl"
        if orig_path.exists():
            try:
                orig_meta = joblib.load(orig_path)
                cat_cols = orig_meta.get('cat_cols', [])
                cat_feature_names = [f for f in cb_features if f in cat_cols]
            except Exception:
                pass

    for f in cb_features:
        if f not in train_df.columns:
            if f in cat_feature_names:
                train_df[f] = 'MISSING'
                val_df[f] = 'MISSING'
                if test_df is not None:
                    test_df[f] = 'MISSING'
            else:
                train_df[f] = 0
                val_df[f] = 0
                if test_df is not None:
                    test_df[f] = 0

    def prepare_df(df, features, cat_features_list):
        X = pd.DataFrame(index=df.index)
        for f in features:
            if f in df.columns:
                X[f] = df[f]
            else:
                X[f] = 'MISSING' if f in cat_features_list else 0

        for col in X.columns:
            if col in cat_features_list:
                X[col] = X[col].astype(str).fillna('MISSING')
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        return X

    try:
        X_train = prepare_df(train_df, cb_features, cat_feature_names)
        X_val = prepare_df(val_df, cb_features, cat_feature_names)

        train_pred = cb_model.predict_proba(X_train)[:, 1]
        val_pred = cb_model.predict_proba(X_val)[:, 1]

        test_pred = None
        if test_df is not None:
            X_test = prepare_df(test_df, cb_features, cat_feature_names)
            test_pred = cb_model.predict_proba(X_test)[:, 1]

        return train_pred, val_pred, test_pred
    except Exception as e:
        print(f"Error generating CatBoost replica predictions: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    print("="*70)
    print("BUILDING BEST CONFIGURATION MODEL EXPORT")
    print("="*70)
    print(f"Configuration: Replace {BEST_CONFIG['replace_tower']} with CatBoost replica tower")
    print(f"Meta-model: {BEST_CONFIG['solver']}, {BEST_CONFIG['penalty']}, C={BEST_CONFIG['C']}")
    print("="*70)

    print("\nLoading data...")
    train_df, val_df, test_df, _ = load_train_val_test_holdout()
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    model_path = EXPORT_DIR / "model.pkl"
    if not model_path.exists():
        print(f"5-tower model not found: {model_path}")
        return 1

    towers, meta_5tower, use_product_meta = joblib.load(model_path)
    tower_names = [t[0] for t in towers]
    print(f"\nLoaded 5-tower model: {tower_names}")

    replace_idx = None
    for i, name in enumerate(tower_names):
        if name == BEST_CONFIG['replace_tower']:
            replace_idx = i
            break

    if replace_idx is None:
        print(f"Error: Tower '{BEST_CONFIG['replace_tower']}' not found in {tower_names}")
        return 1

    print("\nGetting CatBoost replica predictions...")
    cb_train, cb_val, cb_test = get_catboost_predictions(
        train_df.copy(), val_df.copy(), test_df.copy()
    )
    if cb_train is None:
        print("  Failed to get CatBoost replica predictions")
        return 1

    lookups = build_lookups_from_df(train_df)

    def get_tower_probas(df, lookups):
        P = []
        for name, model, feat_list in towers:
            X, _ = encode_df(df, lookups, feature_list=feat_list)
            P.append(model.predict_proba(X)[:, 1])
        return P

    print("\nGenerating tower predictions...")
    P_train = get_tower_probas(train_df, lookups)
    P_val = get_tower_probas(val_df, lookups)

    P_train[replace_idx] = cb_train
    P_val[replace_idx] = cb_val

    print("\nFitting meta-model with best parameters...")
    X_train_meta = np.column_stack(P_train)
    X_val_meta = np.column_stack(P_val)

    meta_params = {
        'max_iter': 2000,
        'random_state': RANDOM_SEED,
        'class_weight': 'balanced',
        'C': BEST_CONFIG['C'],
        'solver': BEST_CONFIG['solver'],
        'penalty': BEST_CONFIG['penalty'],
    }

    meta = LogisticRegression(**meta_params)
    meta.fit(X_train_meta, train_df[TARGET].astype(int).values)

    proba_val = meta.predict_proba(X_val_meta)[:, 1]
    y_val = val_df[TARGET].astype(int).values
    from scipy.stats import ks_2samp
    val_ks, _ = ks_2samp(proba_val[y_val == 1], proba_val[y_val == 0])
    print(f"  Validation KS: {val_ks:.4f}")

    new_towers = []
    for i, (name, model, feat_list) in enumerate(towers):
        if i == replace_idx:
            new_towers.append(('catboost', model, feat_list))
        else:
            new_towers.append((name, model, feat_list))

    print(f"\nSaving best configuration model to {BEST_EXPORT_DIR}...")
    BEST_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_save_path = BEST_EXPORT_DIR / "model.pkl"
    joblib.dump((new_towers, meta, False), model_save_path)
    print(f"  Saved: {model_save_path}")

    lookups_dir = BEST_EXPORT_DIR / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)

    source_lookups_dir = EXPORT_DIR / "lookups"
    if source_lookups_dir.exists():
        import shutil
        for lookup_file in source_lookups_dir.glob("*.json"):
            shutil.copy2(lookup_file, lookups_dir / lookup_file.name)
        print(f"  Copied {len(list(lookups_dir.glob('*.json')))} lookup files")

    threshold_source = EXPORT_DIR / "threshold.json"
    if threshold_source.exists():
        import shutil
        shutil.copy2(threshold_source, BEST_EXPORT_DIR / "threshold.json")
        print(f"  Copied threshold.json")
    else:
        import json
        with open(BEST_EXPORT_DIR / "threshold.json", "w") as f:
            json.dump({"threshold": 0.5}, f)
        print(f"  Created threshold.json")

    import shutil
    cb_src = REPO_ROOT / "exports" / "catboost_model_replica"
    for fname, dest_name in (("model.pkl", "catboost_model.pkl"), ("metadata.pkl", "catboost_metadata.pkl")):
        src = cb_src / fname
        dest = BEST_EXPORT_DIR / dest_name
        if src.exists():
            shutil.copy2(src, dest)
            print(f"  Copied {fname} -> {dest_name}")
        else:
            print(f"  Warning: {src} not found; inference needs {dest_name}")

    config_metadata = {
        "config": "replace_housing_with_catboost",
        "meta_solver": BEST_CONFIG['solver'],
        "meta_penalty": BEST_CONFIG['penalty'],
        "meta_C": BEST_CONFIG['C'],
        "validation_KS": float(val_ks),
        "replaced_tower": BEST_CONFIG['replace_tower'],
        "tower_names": [t[0] for t in new_towers]
    }
    import json
    with open(BEST_EXPORT_DIR / "config_metadata.json", "w") as f:
        json.dump(config_metadata, f, indent=2)
    print(f"  Saved config_metadata.json")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Best configuration model exported to: {BEST_EXPORT_DIR}")
    print(f"  Validation KS: {val_ks:.4f}")
    print(f"  Meta-model: {BEST_CONFIG['solver']}, {BEST_CONFIG['penalty']}, C={BEST_CONFIG['C']}")
    print(f"  Towers: {[t[0] for t in new_towers]}")
    print(f"\nReady for deployment!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
