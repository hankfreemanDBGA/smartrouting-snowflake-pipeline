#!/usr/bin/env python3
"""
Maximize validation KS by testing configurations with CatBoost model included:
1. 5-tower baseline
2. 6-tower (5 towers + CatBoost)
3. Replace each tower with CatBoost (5 configurations)
4. Optimize meta-model hyperparameters for each configuration
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from ml.config import RANDOM_SEED
from ml.splits import load_train_val_test_holdout
from ml.train_router import encode_df, build_lookups_from_df

import train_multitower_sale_5towers as mt5

TARGET = "SALE_MADE_FLAG"
EXPORT_DIR = mt5.EXPORT_DIR


def get_catboost_predictions(train_df, val_df, test_df=None):
    """Get CatBoost model predictions - try loading saved predictions first, else generate."""
    import catboost as cb
    
    # Try loading saved predictions first
    preds_path = REPO_ROOT / "exports" / "catboost_model_replica" / "predictions_val.pkl"
    if preds_path.exists():
        print("Loading saved CatBoost predictions...")
        try:
            preds = joblib.load(preds_path)
            # Check if they match our data size
            if len(preds.get('train', [])) == len(train_df) and len(preds.get('val', [])) == len(val_df):
                print(f"  Loaded saved predictions: train={len(preds['train'])}, val={len(preds['val'])}")
                if test_df is not None and 'test' in preds and len(preds['test']) == len(test_df):
                    return preds['train'], preds['val'], preds['test']
                return preds['train'], preds['val'], None
        except:
            pass
    
    # Generate predictions
    cb_model_path = REPO_ROOT / "exports" / "catboost_model_replica" / "model.pkl"
    cb_meta_path = REPO_ROOT / "exports" / "catboost_model_replica" / "metadata.pkl"
    
    if not cb_model_path.exists():
        print(f"CatBoost model not found: {cb_model_path}")
        return None, None, None
    
    print("Loading CatBoost model and generating predictions...")
    cb_model, cb_features, _ = joblib.load(cb_model_path)
    print(f"  Model has {len(cb_features)} features")
    
    # Load metadata to get categorical feature info
    cat_feature_names = []
    if cb_meta_path.exists():
        try:
            meta = joblib.load(cb_meta_path)
            # Try to infer categoricals from original metadata
            cb_orig_meta = REPO_ROOT / "catboost_metadata" / "close_rate_model_v4_metadata.pkl"
            if cb_orig_meta.exists():
                orig_meta = joblib.load(cb_orig_meta)
                cat_cols = orig_meta.get('cat_cols', [])
                cat_feature_names = [f for f in cb_features if f in cat_cols]
                print(f"  Identified {len(cat_feature_names)} categorical features")
        except:
            pass
    
    # Check which features are available
    available = [f for f in cb_features if f in train_df.columns]
    missing = [f for f in cb_features if f not in train_df.columns]
    print(f"  Available: {len(available)}/{len(cb_features)}")
    if missing:
        print(f"  Missing: {len(missing)} features - adding with defaults")
        # Add missing features with defaults
        for f in missing:
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
    
    # Prepare dataframes with exact features in order
    def prepare_df(df, features, cat_features_list):
        X = pd.DataFrame(index=df.index)
        for f in features:
            if f in df.columns:
                X[f] = df[f]
            else:
                X[f] = 'MISSING' if f in cat_features_list else 0
        
        # Handle NaN and types - convert categoricals to strings
        for col in X.columns:
            if col in cat_features_list:
                # Categorical: convert to string
                X[col] = X[col].astype(str).fillna('MISSING')
            else:
                # Numeric: fill NaN with 0
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        return X
    
    print("  Preparing data...")
    try:
        X_train = prepare_df(train_df, cb_features, cat_feature_names)
        X_val = prepare_df(val_df, cb_features, cat_feature_names)
        
        print("  Predicting...")
        train_pred = cb_model.predict_proba(X_train)[:, 1]
        val_pred = cb_model.predict_proba(X_val)[:, 1]
        
        test_pred = None
        if test_df is not None:
            X_test = prepare_df(test_df, cb_features, cat_feature_names)
            test_pred = cb_model.predict_proba(X_test)[:, 1]
        
        # Save predictions for next time
        try:
            save_dict = {'train': train_pred, 'val': val_pred}
            if test_pred is not None:
                save_dict['test'] = test_pred
            joblib.dump(save_dict, preds_path)
            print(f"  Saved predictions to {preds_path}")
        except:
            pass
        
        print(f"  Success! Train shape: {train_pred.shape}, Val shape: {val_pred.shape}")
        if test_pred is not None:
            print(f"  Test shape: {test_pred.shape}")
        return train_pred, val_pred, test_pred
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_meta_model_config(X_train_meta, y_train, X_val_meta, y_val, C, solver='lbfgs', 
                           penalty='l2', l1_ratio=None, max_iter=1000):
    """Test a meta-model configuration and return validation KS."""
    try:
        params = {
            'max_iter': max_iter,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced',
            'C': C,
            'solver': solver,
            'penalty': penalty,
        }
        if penalty == 'elasticnet':
            if l1_ratio is None:
                return None
            params['l1_ratio'] = l1_ratio
        elif penalty == 'l1':
            if solver not in ['liblinear', 'saga']:
                return None
        
        meta = LogisticRegression(**params)
        meta.fit(X_train_meta, y_train)
        proba_val = meta.predict_proba(X_val_meta)[:, 1]
        ks_val, _ = ks_2samp(proba_val[y_val == 1], proba_val[y_val == 0])
        return ks_val
    except Exception as e:
        return None


def optimize_meta_model(X_train_meta, y_train, X_val_meta, y_val, config_name, verbose=True):
    """Optimize meta-model hyperparameters for a given tower configuration."""
    C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    
    best_ks = -1
    best_params = None
    results = []
    
    # Test L2 penalty with different solvers
    for solver in ['lbfgs', 'liblinear', 'saga']:
        for C in C_VALUES:
            ks_val = test_meta_model_config(
                X_train_meta, y_train, X_val_meta, y_val,
                C=C, solver=solver, penalty='l2'
            )
            if ks_val is not None:
                results.append({
                    'config': config_name,
                    'solver': solver,
                    'penalty': 'l2',
                    'C': C,
                    'l1_ratio': None,
                    'val_KS': ks_val,
                })
                if ks_val > best_ks:
                    best_ks = ks_val
                    best_params = {'solver': solver, 'penalty': 'l2', 'C': C, 'l1_ratio': None}
                if verbose:
                    print(f"    {solver:8s} l2  C={C:6.3f}: Val KS={ks_val:.4f}")
    
    # Test L1 penalty (only with liblinear and saga)
    for solver in ['liblinear', 'saga']:
        for C in C_VALUES:
            ks_val = test_meta_model_config(
                X_train_meta, y_train, X_val_meta, y_val,
                C=C, solver=solver, penalty='l1'
            )
            if ks_val is not None:
                results.append({
                    'config': config_name,
                    'solver': solver,
                    'penalty': 'l1',
                    'C': C,
                    'l1_ratio': None,
                    'val_KS': ks_val,
                })
                if ks_val > best_ks:
                    best_ks = ks_val
                    best_params = {'solver': solver, 'penalty': 'l1', 'C': C, 'l1_ratio': None}
                if verbose:
                    print(f"    {solver:8s} l1  C={C:6.3f}: Val KS={ks_val:.4f}")
    
    # Test ElasticNet (only with saga, fewer C values)
    for C in C_VALUES[:6]:  # Test fewer C values for elasticnet
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            ks_val = test_meta_model_config(
                X_train_meta, y_train, X_val_meta, y_val,
                C=C, solver='saga', penalty='elasticnet', l1_ratio=l1_ratio
            )
            if ks_val is not None:
                results.append({
                    'config': config_name,
                    'solver': 'saga',
                    'penalty': 'elasticnet',
                    'C': C,
                    'l1_ratio': l1_ratio,
                    'val_KS': ks_val,
                })
                if ks_val > best_ks:
                    best_ks = ks_val
                    best_params = {'solver': 'saga', 'penalty': 'elasticnet', 'C': C, 'l1_ratio': l1_ratio}
                if verbose:
                    print(f"    saga    elasticnet C={C:6.3f} l1_ratio={l1_ratio:.1f}: Val KS={ks_val:.4f}")
    
    return best_ks, best_params, results


def main():
    print("="*70)
    print("MAXIMIZING VALIDATION KS - WITH ALEC MODEL")
    print("="*70)
    
    # Load 5-tower model
    model_path = EXPORT_DIR / "model.pkl"
    if not model_path.exists():
        print(f"5-tower model not found: {model_path}")
        return 1
    
    towers, meta_5tower, _ = joblib.load(model_path)
    tower_names = [t[0] for t in towers]
    print(f"\nLoaded 5-tower model: {tower_names}")
    
    # Load pre-split data
    print("\nLoading data...")
    train_df, val_df, test_df, _ = load_train_val_test_holdout()
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    lookups = build_lookups_from_df(train_df)
    
    def get_tower_probas(df, lookups):
        P = []
        for name, model, feat_list in towers:
            X, _ = encode_df(df, lookups, feature_list=feat_list)
            P.append(model.predict_proba(X)[:, 1])
        return P
    
    # Get 5-tower predictions
    print("\nGenerating 5-tower predictions...")
    P_train_5tower = get_tower_probas(train_df, lookups)
    P_val_5tower = get_tower_probas(val_df, lookups)
    P_test_5tower = get_tower_probas(test_df, lookups)
    
    y_train = train_df[TARGET].astype(int).values
    y_val = val_df[TARGET].astype(int).values
    y_test = test_df[TARGET].astype(int).values
    
    # Get CatBoost model predictions
    print("\nGetting CatBoost model predictions...")
    cb_train, cb_val, cb_test = get_catboost_predictions(train_df.copy(), val_df.copy(), test_df.copy())
    if cb_train is None:
        print("  Failed to get CatBoost predictions. Skipping CatBoost configurations.")
        cb_train = None
        cb_val = None
        cb_test = None
    else:
        # Calculate standalone CatBoost KS
        cb_ks_val, _ = ks_2samp(cb_val[y_val == 1], cb_val[y_val == 0])
        print(f"  CatBoost standalone Val KS: {cb_ks_val:.4f}")
    
    all_results = []
    
    # ===== CONFIGURATION 1: 5-tower baseline =====
    print("\n" + "="*70)
    print("CONFIGURATION 1: 5-tower baseline")
    print("="*70)
    X_train_meta = np.column_stack(P_train_5tower)
    X_val_meta = np.column_stack(P_val_5tower)
    
    print("\nOptimizing meta-model hyperparameters...")
    best_ks, best_params, results = optimize_meta_model(
        X_train_meta, y_train, X_val_meta, y_val, 
        config_name='5tower_baseline', verbose=True
    )
    all_results.extend(results)
    print(f"\nBest: Val KS={best_ks:.4f}, Params: {best_params}")
    
    # ===== CONFIGURATION 2: 6-tower (5 towers + CatBoost) =====
    if cb_train is not None:
        print("\n" + "="*70)
        print("CONFIGURATION 2: 6-tower (5 towers + CatBoost)")
        print("="*70)
        P_train_6tower = P_train_5tower + [cb_train]
        P_val_6tower = P_val_5tower + [cb_val]
        
        X_train_meta = np.column_stack(P_train_6tower)
        X_val_meta = np.column_stack(P_val_6tower)
        
        print("\nOptimizing meta-model hyperparameters...")
        best_ks, best_params, results = optimize_meta_model(
            X_train_meta, y_train, X_val_meta, y_val,
            config_name='6tower_with_catboost', verbose=True
        )
        all_results.extend(results)
        print(f"\nBest: Val KS={best_ks:.4f}, Params: {best_params}")
        
        # ===== CONFIGURATIONS 3-7: Replace each tower with CatBoost =====
        for i, tower_name in enumerate(tower_names):
            print("\n" + "="*70)
            print(f"CONFIGURATION {i+3}: Replace {tower_name} with CatBoost")
            print("="*70)
            
            P_train_replace = P_train_5tower.copy()
            P_val_replace = P_val_5tower.copy()
            P_train_replace[i] = cb_train
            P_val_replace[i] = cb_val
            
            X_train_meta = np.column_stack(P_train_replace)
            X_val_meta = np.column_stack(P_val_replace)
            
            print("\nOptimizing meta-model hyperparameters...")
            best_ks, best_params, results = optimize_meta_model(
                X_train_meta, y_train, X_val_meta, y_val,
                config_name=f'replace_{tower_name}_with_catboost', verbose=True
            )
            all_results.extend(results)
            print(f"\nBest: Val KS={best_ks:.4f}, Params: {best_params}")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("SUMMARY - BEST CONFIGURATIONS")
    print("="*70)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Find best for each configuration
        for config_name in results_df['config'].unique():
            config_results = results_df[results_df['config'] == config_name]
            best_idx = config_results['val_KS'].idxmax()
            best = config_results.loc[best_idx]
            
            print(f"\n{config_name.upper()}:")
            print(f"  Best Val KS: {best['val_KS']:.4f}")
            print(f"  Solver: {best['solver']}")
            print(f"  Penalty: {best['penalty']}")
            print(f"  C: {best['C']:.3f}")
            if pd.notna(best['l1_ratio']):
                print(f"  L1 Ratio: {best['l1_ratio']:.2f}")
        
        # Overall best
        best_overall_idx = results_df['val_KS'].idxmax()
        best_overall = results_df.loc[best_overall_idx]
        
        print(f"\n{'='*70}")
        print("OVERALL BEST CONFIGURATION:")
        print("="*70)
        print(f"  Validation KS: {best_overall['val_KS']:.4f}")
        print(f"  Config: {best_overall['config']}")
        print(f"  Solver: {best_overall['solver']}")
        print(f"  Penalty: {best_overall['penalty']}")
        print(f"  C: {best_overall['C']:.3f}")
        if pd.notna(best_overall['l1_ratio']):
            print(f"  L1 Ratio: {best_overall['l1_ratio']:.2f}")
        
        # Save results
        output_path = REPO_ROOT / "ks_with_catboost_results.csv"
        results_df = results_df.sort_values('val_KS', ascending=False)
        results_df.to_csv(output_path, index=False)
        print(f"\n  Results saved to: {output_path}")
        print(f"  Total configurations tested: {len(results_df)}")
        
        # Show top 10
        print(f"\n  Top 10 configurations:")
        print("  " + "-"*66)
        for idx, row in results_df.head(10).iterrows():
            penalty_str = f"{row['penalty']}"
            if pd.notna(row['l1_ratio']):
                penalty_str += f" (l1_ratio={row['l1_ratio']:.1f})"
            print(f"  {row['val_KS']:.4f} | {row['config']:30s} | {row['solver']:8s} | {penalty_str:20s} | C={row['C']:6.3f}")
    else:
        print("  No results to summarize.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
