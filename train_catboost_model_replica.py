#!/usr/bin/env python3
"""
Train a CatBoost replica tower using CatBoost with RFE to maximize validation KS.
Uses the reference feature list from metadata and trains with CatBoost, selecting features via RFE.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from ml.splits import load_train_val_test_holdout
from ml.config import RANDOM_SEED
from ml.train_router import encode_df, build_lookups_from_df

try:
    import catboost as cb
except ImportError:
    print("Error: catboost not installed. Run: pip install catboost")
    sys.exit(1)

TARGET = "SALE_MADE_FLAG"
EXPORT_DIR = REPO_ROOT / "exports" / "catboost_model_replica"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _load_catboost_features():
    """Load feature list from reference metadata (catboost_metadata/)."""
    metadata_path = REPO_ROOT / "catboost_metadata" / "close_rate_model_v4_metadata.pkl"
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return None, None
    
    metadata = joblib.load(str(metadata_path))
    if isinstance(metadata, dict):
        features = metadata.get("feature_names")
        cat_features = metadata.get("cat_cols")
        return features, cat_features
    return None, None


def _fit_catboost(X_train, y_train, cat_features=None, verbose=True):
    """Fit CatBoost model. X_train can be DataFrame or numpy array."""
    n_train = len(y_train)
    n_pos, n_neg = int((y_train == 1).sum()), n_train - int((y_train == 1).sum())
    
    if verbose:
        print(f"    CatBoost (n={n_train}, pos={n_pos}, neg={n_neg})")
    
    # CatBoost parameters tuned for replica tower
    model = cb.CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=RANDOM_SEED,
        verbose=False,
        cat_features=cat_features,
    )
    
    # If DataFrame, CatBoost handles categoricals automatically
    # If numpy array, need to pass cat_features indices
    model.fit(X_train, y_train)
    return model


def _rfe_by_val_ks(X_train_array, y_train, X_val_array, y_val, feature_list, 
                    X_train_df, X_val_df, cat_indices=None, 
                    rfe_min=10, rfe_max=150, step=10, verbose=True):
    """
    RFE to maximize validation KS. Uses RandomForest for RFE, then trains CatBoost with selected features.
    X_train_array/X_val_array: encoded numpy arrays for RFE
    X_train_df/X_val_df: DataFrames for CatBoost training
    Returns (best_features, best_ks, best_model, rfe_results).
    """
    from scipy.stats import ks_2samp
    
    n_features_range = list(range(rfe_min, min(rfe_max, X_train_array.shape[1]) + 1, step))
    if X_train_array.shape[1] not in n_features_range:
        n_features_range.append(X_train_array.shape[1])
    
    best_ks = -1.0
    best_n = rfe_min
    best_features = None
    best_model = None
    rfe_results = []
    
    if verbose:
        print(f"\nRFE: Testing {len(n_features_range)} feature counts to maximize validation KS...")
        print(f"  Range: {n_features_range[0]} to {n_features_range[-1]} features")
        print(f"  Using RandomForest for RFE, then CatBoost for final model")
    
    for n in n_features_range:
        if verbose:
            print(f"  Testing n_features={n}...", end="", flush=True)
        
        # Use RandomForest for RFE (handles numeric arrays)
        rfe_estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight='balanced',
        )
        
        rfe = RFE(
            estimator=rfe_estimator,
            n_features_to_select=n,
            step=1,
            verbose=0,
        )
        
        try:
            rfe.fit(X_train_array, y_train)
            selected_mask = rfe.support_
            selected_idx = [i for i in range(len(feature_list)) if selected_mask[i]]
            
            if len(selected_idx) == 0:
                continue
            
            # Get selected features
            selected_features = [feature_list[i] for i in selected_idx]
            
            # Use DataFrames for CatBoost
            X_train_sub_df = X_train_df[selected_features].copy()
            X_val_sub_df = X_val_df[selected_features].copy()
            
            # Update categorical indices for selected features
            cat_indices_sub = None
            if cat_indices:
                cat_indices_sub = [i for i, f in enumerate(selected_features) 
                                 if feature_list.index(f) in cat_indices]
                
                # Fill NaN in categorical features with 'MISSING' string
                for idx in cat_indices_sub:
                    col_name = selected_features[idx]
                    X_train_sub_df[col_name] = X_train_sub_df[col_name].fillna('MISSING').astype(str)
                    X_val_sub_df[col_name] = X_val_sub_df[col_name].fillna('MISSING').astype(str)
            
            # Train CatBoost with selected features
            model = cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=RANDOM_SEED,
                verbose=False,
                cat_features=cat_indices_sub,
            )
            model.fit(X_train_sub_df, y_train)
            
            val_proba = model.predict_proba(X_val_sub_df)[:, 1]
            ks, _ = ks_2samp(val_proba[y_val == 1], val_proba[y_val == 0])
            
            rfe_results.append({
                "n_features": n,
                "features": selected_features,
                "val_ks": float(ks),
            })
            
            if verbose:
                print(f"  KS={ks:.4f}")
            
            if ks > best_ks:
                best_ks = ks
                best_n = n
                best_features = selected_features
                best_model = model
                
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            continue
    
    if best_features is None:
        if verbose:
            print("  Warning: RFE failed, using all features")
        best_features = feature_list
        cat_indices_best = cat_indices
        best_model = _fit_catboost(X_train_df, y_train, cat_indices_best, verbose=False)
        val_proba = best_model.predict_proba(X_val_df)[:, 1]
        best_ks, _ = ks_2samp(val_proba[y_val == 1], val_proba[y_val == 0])
    
    if verbose:
        print(f"\nBest: n_features={best_n}, validation KS={best_ks:.4f}")
    
    return best_features, best_ks, best_model, rfe_results


def _rfe_find_best_100_features(X_train_array, y_train, X_val_array, y_val, feature_list,
                                 X_train_df, X_val_df, cat_indices=None, verbose=True):
    """
    Find the optimal 100 features using multiple RFE runs and selecting the best.
    Returns (best_features, best_ks, best_model, rfe_results).
    """
    from scipy.stats import ks_2samp
    
    target_n = 100
    best_ks = -1.0
    best_features = None
    best_model = None
    rfe_results = []
    
    if verbose:
        print(f"\nFinding optimal {target_n} features using multiple RFE approaches...")
    
    # Approach 1: Direct RFE to 100 features
    if verbose:
        print(f"\nApproach 1: Direct RFE to {target_n} features...")
    
    rfe_estimator = RandomForestClassifier(
        n_estimators=100,  # More trees for better feature importance
        max_depth=15,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced',
    )
    
    rfe = RFE(
        estimator=rfe_estimator,
        n_features_to_select=target_n,
        step=1,
        verbose=0,
    )
    
    try:
        rfe.fit(X_train_array, y_train)
        selected_mask = rfe.support_
        selected_idx = [i for i in range(len(feature_list)) if selected_mask[i]]
        selected_features = [feature_list[i] for i in selected_idx]
        
        # Evaluate with CatBoost
        X_train_sub_df = X_train_df[selected_features].copy()
        X_val_sub_df = X_val_df[selected_features].copy()
        
        cat_indices_sub = None
        if cat_indices:
            cat_indices_sub = [i for i, f in enumerate(selected_features) 
                             if feature_list.index(f) in cat_indices]
            for idx in cat_indices_sub:
                col_name = selected_features[idx]
                X_train_sub_df[col_name] = X_train_sub_df[col_name].fillna('MISSING').astype(str)
                X_val_sub_df[col_name] = X_val_sub_df[col_name].fillna('MISSING').astype(str)
        
        model = cb.CatBoostClassifier(
            iterations=200,  # More iterations for better model
            depth=6,
            learning_rate=0.1,
            random_seed=RANDOM_SEED,
            verbose=False,
            cat_features=cat_indices_sub,
        )
        model.fit(X_train_sub_df, y_train)
        
        val_proba = model.predict_proba(X_val_sub_df)[:, 1]
        ks, _ = ks_2samp(val_proba[y_val == 1], val_proba[y_val == 0])
        
        if verbose:
            print(f"  RFE direct: KS={ks:.4f}")
        
        rfe_results.append({
            "method": "RFE_direct",
            "n_features": len(selected_features),
            "features": selected_features,
            "val_ks": float(ks),
        })
        
        if ks > best_ks:
            best_ks = ks
            best_features = selected_features
            best_model = model
    
    except Exception as e:
        if verbose:
            print(f"  Error in direct RFE: {e}")
    
    # Approach 2: Try RFE with different step sizes and pick best
    if verbose:
        print(f"\nApproach 2: RFE with different step sizes...")
    
    for step_size in [1, 2, 5]:
        try:
            rfe_estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                class_weight='balanced',
            )
            
            rfe = RFE(
                estimator=rfe_estimator,
                n_features_to_select=target_n,
                step=step_size,
                verbose=0,
            )
            
            rfe.fit(X_train_array, y_train)
            selected_mask = rfe.support_
            selected_idx = [i for i in range(len(feature_list)) if selected_mask[i]]
            selected_features = [feature_list[i] for i in selected_idx]
            
            if len(selected_features) != target_n:
                continue
            
            # Evaluate with CatBoost
            X_train_sub_df = X_train_df[selected_features].copy()
            X_val_sub_df = X_val_df[selected_features].copy()
            
            cat_indices_sub = None
            if cat_indices:
                cat_indices_sub = [i for i, f in enumerate(selected_features) 
                                 if feature_list.index(f) in cat_indices]
                for idx in cat_indices_sub:
                    col_name = selected_features[idx]
                    X_train_sub_df[col_name] = X_train_sub_df[col_name].fillna('MISSING').astype(str)
                    X_val_sub_df[col_name] = X_val_sub_df[col_name].fillna('MISSING').astype(str)
            
            model = cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=RANDOM_SEED,
                verbose=False,
                cat_features=cat_indices_sub,
            )
            model.fit(X_train_sub_df, y_train)
            
            val_proba = model.predict_proba(X_val_sub_df)[:, 1]
            ks, _ = ks_2samp(val_proba[y_val == 1], val_proba[y_val == 0])
            
            if verbose:
                print(f"  RFE step={step_size}: KS={ks:.4f}")
            
            rfe_results.append({
                "method": f"RFE_step_{step_size}",
                "n_features": len(selected_features),
                "features": selected_features,
                "val_ks": float(ks),
            })
            
            if ks > best_ks:
                best_ks = ks
                best_features = selected_features
                best_model = model
                
        except Exception as e:
            if verbose:
                print(f"  Error with step={step_size}: {e}")
            continue
    
    # Approach 3: Multiple RFE runs with different random seeds to find best
    if verbose:
        print(f"\nApproach 3: Multiple RFE runs with different seeds...")
    
    for seed_offset in [0, 1, 2]:
        try:
            rfe_estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=RANDOM_SEED + seed_offset,
                n_jobs=-1,
                class_weight='balanced',
            )
            
            rfe = RFE(
                estimator=rfe_estimator,
                n_features_to_select=target_n,
                step=5,
                verbose=0,
            )
            
            rfe.fit(X_train_array, y_train)
            selected_mask = rfe.support_
            selected_idx = [i for i in range(len(feature_list)) if selected_mask[i]]
            selected_features = [feature_list[i] for i in selected_idx]
            
            if len(selected_features) != target_n:
                continue
            
            # Evaluate with CatBoost
            X_train_sub_df = X_train_df[selected_features].copy()
            X_val_sub_df = X_val_df[selected_features].copy()
            
            cat_indices_sub = None
            if cat_indices:
                cat_indices_sub = [i for i, f in enumerate(selected_features) 
                                 if feature_list.index(f) in cat_indices]
                for idx in cat_indices_sub:
                    col_name = selected_features[idx]
                    X_train_sub_df[col_name] = X_train_sub_df[col_name].fillna('MISSING').astype(str)
                    X_val_sub_df[col_name] = X_val_sub_df[col_name].fillna('MISSING').astype(str)
            
            model = cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=RANDOM_SEED,
                verbose=False,
                cat_features=cat_indices_sub,
            )
            model.fit(X_train_sub_df, y_train)
            
            val_proba = model.predict_proba(X_val_sub_df)[:, 1]
            ks, _ = ks_2samp(val_proba[y_val == 1], val_proba[y_val == 0])
            
            if verbose:
                print(f"  RFE seed_offset={seed_offset}: KS={ks:.4f}")
            
            rfe_results.append({
                "method": f"RFE_seed_{seed_offset}",
                "n_features": len(selected_features),
                "features": selected_features,
                "val_ks": float(ks),
            })
            
            if ks > best_ks:
                best_ks = ks
                best_features = selected_features
                best_model = model
                
        except Exception as e:
            if verbose:
                print(f"  Error with seed_offset={seed_offset}: {e}")
            continue
    
    # Approach 4: Use feature importance from full model and select top 100
    if verbose:
        print(f"\nApproach 4: Feature importance from full model...")
    
    try:
        # Train full RandomForest to get feature importance
        rf_full = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight='balanced',
        )
        rf_full.fit(X_train_array, y_train)
        
        # Get top 100 features by importance
        importances = rf_full.feature_importances_
        top_100_idx = np.argsort(importances)[-target_n:][::-1]
        selected_features = [feature_list[i] for i in top_100_idx]
        
        # Evaluate with CatBoost
        X_train_sub_df = X_train_df[selected_features].copy()
        X_val_sub_df = X_val_df[selected_features].copy()
        
        cat_indices_sub = None
        if cat_indices:
            cat_indices_sub = [i for i, f in enumerate(selected_features) 
                             if feature_list.index(f) in cat_indices]
            for idx in cat_indices_sub:
                col_name = selected_features[idx]
                X_train_sub_df[col_name] = X_train_sub_df[col_name].fillna('MISSING').astype(str)
                X_val_sub_df[col_name] = X_val_sub_df[col_name].fillna('MISSING').astype(str)
        
        model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=RANDOM_SEED,
            verbose=False,
            cat_features=cat_indices_sub,
        )
        model.fit(X_train_sub_df, y_train)
        
        val_proba = model.predict_proba(X_val_sub_df)[:, 1]
        ks, _ = ks_2samp(val_proba[y_val == 1], val_proba[y_val == 0])
        
        if verbose:
            print(f"  Feature importance: KS={ks:.4f}")
        
        rfe_results.append({
            "method": "feature_importance",
            "n_features": len(selected_features),
            "features": selected_features,
            "val_ks": float(ks),
        })
        
        if ks > best_ks:
            best_ks = ks
            best_features = selected_features
            best_model = model
            
    except Exception as e:
        if verbose:
            print(f"  Error in feature importance approach: {e}")
    
    if best_features is None:
        if verbose:
            print("  Warning: All approaches failed, using all features")
        best_features = feature_list
        cat_indices_best = cat_indices
        if cat_indices_best:
            for idx in cat_indices_best:
                col_name = feature_list[idx]
                X_train_df[col_name] = X_train_df[col_name].fillna('MISSING').astype(str)
                X_val_df[col_name] = X_val_df[col_name].fillna('MISSING').astype(str)
        best_model = _fit_catboost(X_train_df, y_train, cat_indices_best, verbose=False)
        val_proba = best_model.predict_proba(X_val_df)[:, 1]
        best_ks, _ = ks_2samp(val_proba[y_val == 1], val_proba[y_val == 0])
    
    if verbose:
        print(f"\nBest approach: {len(best_features)} features, validation KS={best_ks:.4f}")
        # Show which method won
        best_result = max(rfe_results, key=lambda x: x['val_ks'])
        print(f"  Winning method: {best_result['method']}")
    
    return best_features, best_ks, best_model, rfe_results


def main():
    print("Loading reference feature list from metadata...")
    cb_features, cb_cat_features = _load_catboost_features()
    if cb_features is None:
        print("Failed to load features from catboost_metadata")
        return 1
    
    print(f"Found {len(cb_features)} features ({len(cb_cat_features) if cb_cat_features else 0} categorical)")
    
    print("\nLoading data from tuappend.csv...")
    # Load directly from tuappend.csv to get all features
    tuappend_path = REPO_ROOT / "tuappend.csv"
    if not tuappend_path.exists():
        print(f"tuappend.csv not found: {tuappend_path}")
        print("Falling back to training tables...")
        train_df, val_df, test_df, _ = load_train_val_test_holdout()
    else:
        print(f"Reading {tuappend_path}...")
        df_full = pd.read_csv(tuappend_path, low_memory=False)
        print(f"  Loaded {len(df_full):,} rows")
        
        # Apply same filters as build_training_splits.py
        # PHONE_1 not null
        df_full = df_full[df_full["PHONE_1"].notna()].copy()
        
        # PHONE_1_LINKAGE_SCORE >= 75000
        linkage = pd.to_numeric(df_full.get("PHONE_1_LINKAGE_SCORE", pd.Series()), errors="coerce")
        if "PHONE_1_LINKAGE_SCORE" in df_full.columns:
            df_full = df_full[linkage.notna() & (linkage >= 75000)].copy()
        
        # GENDER filtering
        gender_raw = df_full.get("GENDER", pd.Series()).astype(str).str.strip().str.upper()
        gender_map = {"MALE": "M", "FEMALE": "F", "M": "M", "F": "F"}
        df_full["GENDER"] = gender_raw.map(gender_map)
        if "TU_GENDER" in df_full.columns:
            tu_g = df_full["TU_GENDER"].astype(str).str.strip().str.upper()
            df_full.loc[df_full["GENDER"].isna(), "GENDER"] = tu_g.map(gender_map)
        df_full = df_full[df_full["GENDER"].isin(["M", "F"])].copy()
        
        # AGE
        age_tu = pd.to_numeric(df_full.get("AGE"), errors="coerce")
        age_years = pd.to_numeric(df_full.get("AGE_YEARS"), errors="coerce")
        df_full["AGE"] = age_tu.fillna(age_years)
        df_full = df_full[df_full["AGE"].notna()].copy()
        
        # SALE_MADE_FLAG
        if "SALE_MADE_FLAG" in df_full.columns:
            df_full = df_full[df_full["SALE_MADE_FLAG"].notna()].copy()
            sale_raw = df_full["SALE_MADE_FLAG"].astype(str).str.strip().str.upper()
            df_full[TARGET] = (sale_raw.isin(["Y", "TRUE", "1", "YES"]) | (df_full["SALE_MADE_FLAG"] == True)).astype(int)
        
        # ANSWERED_FLAG
        if "ANSWERED_FLAG" in df_full.columns:
            ans = df_full["ANSWERED_FLAG"].astype(str).str.strip().str.upper()
            df_full = df_full[ans.isin(["Y", "TRUE", "1", "YES"]) | (df_full["ANSWERED_FLAG"] == True)].copy()
        
        print(f"  After filters: {len(df_full):,} rows")
        
        # Split into train/val/test using same logic as build_training_splits.py
        from sklearn.model_selection import train_test_split
        np.random.seed(RANDOM_SEED)
        
        # First split: holdout (10%)
        df_trainvaltest, df_holdout = train_test_split(
            df_full, test_size=0.10, random_state=RANDOM_SEED, stratify=df_full[TARGET] if TARGET in df_full.columns else None
        )
        
        # Second split: train (60%), val (20%), test (20%) of remaining 90%
        df_train, df_valtest = train_test_split(
            df_trainvaltest, test_size=0.4, random_state=RANDOM_SEED, stratify=df_trainvaltest[TARGET] if TARGET in df_trainvaltest.columns else None
        )
        df_val, df_test = train_test_split(
            df_valtest, test_size=0.5, random_state=RANDOM_SEED, stratify=df_valtest[TARGET] if TARGET in df_valtest.columns else None
        )
        
        train_df = df_train
        val_df = df_val
        test_df = df_test
        print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    # Map common aliases (metadata feature names -> our column names)
    feature_alias_map = {
        "TU_GENDER": "GENDER",
        "TU_STATE": "STATE",
        "TU_ZIP": "ZIP",  # if exists
    }
    
    # Filter to features that exist in our data (with aliases)
    available_features = []
    feature_mapping = {}  # metadata feature -> actual column name
    missing_features = []
    
    for feat in cb_features:
        if feat in train_df.columns:
            available_features.append(feat)
            feature_mapping[feat] = feat
        elif feat in feature_alias_map:
            alias = feature_alias_map[feat]
            if alias in train_df.columns:
                available_features.append(alias)
                feature_mapping[feat] = alias
                print(f"  Using alias: {feat} -> {alias}")
            else:
                missing_features.append(feat)
        else:
            missing_features.append(feat)
    
    print(f"\nAvailable features: {len(available_features)}/{len(cb_features)}")
    if missing_features:
        print(f"Missing features ({len(missing_features)}): {list(missing_features)[:10]}...")
    
    # Use the actual column names (may be aliases)
    actual_feature_cols = [feature_mapping.get(f, f) for f in available_features if f in feature_mapping]
    # Also include features that were direct matches
    for f in available_features:
        if f not in actual_feature_cols:
            actual_feature_cols.append(f)
    
    if len(available_features) == 0:
        print("No features available!")
        return 1
    
    # Build lookups for encoding (needed for RFE with RandomForest)
    print("\nBuilding lookups...")
    lookups = build_lookups_from_df(train_df)
    
    # Encode for RFE (RandomForest needs numeric arrays)
    print(f"\nEncoding data for RFE with {len(actual_feature_cols)} features...")
    X_train_encoded, y_train = encode_df(train_df, lookups, feature_list=actual_feature_cols)
    X_val_encoded, y_val = encode_df(val_df, lookups, feature_list=actual_feature_cols)
    X_test_encoded, y_test = encode_df(test_df, lookups, feature_list=actual_feature_cols)
    
    # Prepare DataFrames for CatBoost (keep original types)
    X_train_df = train_df[actual_feature_cols].copy()
    X_val_df = val_df[actual_feature_cols].copy()
    X_test_df = test_df[actual_feature_cols].copy()
    
    print(f"Training set: {X_train_df.shape}, {y_train.sum()} positive")
    print(f"Validation set: {X_val_df.shape}, {y_val.sum()} positive")
    print(f"Test set: {X_test_df.shape}, {y_test.sum()} positive")
    
    # Identify categorical feature indices for CatBoost
    cat_indices = None
    if cb_cat_features:
        cat_feature_names = [feature_mapping.get(f, f) for f in cb_cat_features if feature_mapping.get(f, f) in actual_feature_cols]
        cat_indices = [i for i, f in enumerate(actual_feature_cols) if f in cat_feature_names]
        print(f"\nCategorical features: {len(cat_indices)}/{len(actual_feature_cols)}")
    
    # Perform RFE to find optimal 100 features
    print("\n" + "="*60)
    print("RECURSIVE FEATURE ELIMINATION (target: 100 features)")
    print("="*60)
    print("Finding optimal 100 features to maximize validation KS...")
    
    # Use RFE to select exactly 100 features
    best_features, best_val_ks, model, rfe_results = _rfe_find_best_100_features(
        X_train_encoded, y_train, X_val_encoded, y_val, 
        actual_feature_cols,
        X_train_df, X_val_df,  # DataFrames for CatBoost
        cat_indices=cat_indices,
        verbose=True
    )
    
    # Get DataFrames with best features
    X_train_best = X_train_df[best_features].copy()
    X_val_best = X_val_df[best_features].copy()
    X_test_best = X_test_df[best_features].copy()
    
    # Update categorical indices for best features and handle NaNs
    cat_indices_best = None
    if cat_indices:
        cat_indices_best = [i for i, f in enumerate(best_features) if actual_feature_cols.index(f) in cat_indices]
        # Fill NaN in categorical features
        for idx in cat_indices_best:
            col_name = best_features[idx]
            X_train_best[col_name] = X_train_best[col_name].fillna('MISSING').astype(str)
            X_val_best[col_name] = X_val_best[col_name].fillna('MISSING').astype(str)
            X_test_best[col_name] = X_test_best[col_name].fillna('MISSING').astype(str)
    
    # Retrain final model with best features (if RFE didn't already train it)
    if model is None:
        print(f"\nTraining final CatBoost model with {len(best_features)} features...")
        model = _fit_catboost(X_train_best, y_train, cat_indices_best, verbose=True)
    
    # Evaluate
    train_proba = model.predict_proba(X_train_best)[:, 1]
    val_proba = model.predict_proba(X_val_best)[:, 1]
    test_proba = model.predict_proba(X_test_best)[:, 1]
    
    train_ks, _ = ks_2samp(train_proba[y_train == 1], train_proba[y_train == 0])
    val_ks, _ = ks_2samp(val_proba[y_val == 1], val_proba[y_val == 0])
    test_ks, _ = ks_2samp(test_proba[y_test == 1], test_proba[y_test == 0])
    
    print(f"\n" + "="*60)
    print("FINAL KS SCORES")
    print("="*60)
    print(f"  Train: {train_ks:.4f}")
    print(f"  Val:   {val_ks:.4f}")
    print(f"  Test:  {test_ks:.4f}")
    print(f"\nSelected {len(best_features)} features (from {len(actual_feature_cols)} available)")
    
    # Save model
    print(f"\nSaving model to {EXPORT_DIR}...")
    model_path = EXPORT_DIR / "model.pkl"
    joblib.dump((model, best_features, lookups), model_path)
    print(f"  Saved: {model_path}")
    
    # Save metadata
    metadata = {
        "feature_names": best_features,
        "n_features_selected": len(best_features),
        "catboost_feature_mapping": feature_mapping,
        "original_features": cb_features,
        "missing_features": list(missing_features),
        "rfe_results": rfe_results,
        "train_ks": float(train_ks),
        "val_ks": float(val_ks),
        "test_ks": float(test_ks),
        "best_val_ks": float(best_val_ks),
    }
    metadata_path = EXPORT_DIR / "metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"  Saved: {metadata_path}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
