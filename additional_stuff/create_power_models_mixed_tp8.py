#!/usr/bin/env python3
"""
Unified power model for prefill and decode with separate feature columns (TP8 only).

Features:
- Trains a single model for both prefill and decode power consumption
- Uses separate feature columns: prefill_*, decode_*
- Three data modes:
  - prefill: decode features set to 0
  - decode: prefill features set to 0  
  - mixed: both prefill and decode features present (from combined_profiling_results.csv)
- Filters out TP2 and TP4 data to create separate TP8 model
- Saves as .joblib and optionally converts to ONNX
"""
import os
import time
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from lightgbm import LGBMRegressor

def load_and_prepare_prefill(path, model_name, tp=None):
    """Load prefill power data and convert to unified format with decode features = 0"""
    df = pd.read_csv(path)
    if tp is not None:
        df['tp_degree'] = tp
    
    # Map to unified format with prefill-specific columns
    df_unified = pd.DataFrame({
        'prefill_batch_size': df['batch_size'],
        'prefill_input_len_sum': df['input_len_sum'],
        'prefill_input_len_mean': df['input_len_mean'],
        'prefill_input_len_std': df['input_len_std'],
        'decode_batch_size': 0,
        'decode_input_len_sum': 0,
        'decode_input_len_mean': 0,
        'decode_input_len_std': 0,
        'tp_degree': df['tp_degree'],
        'freq_mhz': df['freq_mhz'],
        'power_w': df['power_w']
    })
    
    return df_unified.dropna()


def load_and_prepare_decode(path, model_name, tp=None):
    """Load decode power data and convert to unified format with prefill features = 0"""
    df = pd.read_csv(path)
    if tp is not None:
        df['tp_degree'] = tp
    
    # Map to unified format with decode-specific columns
    df_unified = pd.DataFrame({
        'prefill_batch_size': 0,
        'prefill_input_len_sum': 0,
        'prefill_input_len_mean': 0,
        'prefill_input_len_std': 0,
        'decode_batch_size': df['batch_size'],
        'decode_input_len_sum': df['input_len_sum'],
        'decode_input_len_mean': df['input_len_mean'],
        'decode_input_len_std': df['input_len_std'],
        'tp_degree': df['tp_degree'],
        'freq_mhz': df['freq_mhz'],
        'power_w': df['power_w']
    })
    
    return df_unified.dropna()


def load_and_prepare_mixed(path, model_name):
    """Load mixed power data from combined_profiling_results.csv"""
    df = pd.read_csv(path)
    
    # Map combined_profiling_results columns to unified format
    # For mixed data, we need to estimate prefill features from the mixed batch
    df_unified = pd.DataFrame({
        'test_name': df['test_name'],  # keep for reference
        'prefill_batch_size': df['num_prefill_reqs'],
        'prefill_input_len_sum': df['sum_ctx_len'],
        'prefill_input_len_mean': df['mean_ctx_len'],
        'prefill_input_len_std': df['std_ctx_len'],
        'decode_batch_size': df['num_decode_reqs'],
        'decode_input_len_sum': df['sum_decode_len'],
        'decode_input_len_mean': df['mean_decode_len'],
        'decode_input_len_std': df['std_decode_len'],
        'tp_degree': df['tp_size'],
        'freq_mhz': df['frequency'],
        'power_w': df['avg_power_w']
    })
    
    return df_unified.dropna()


def filter_inputs(df):
    """Filter to keep only median power for duplicate configurations"""
    print("before filtering:", df.shape)
    group_cols = ['prefill_batch_size', 'prefill_input_len_sum', 'prefill_input_len_mean', 
                  'prefill_input_len_std', 'decode_batch_size', 'decode_input_len_sum', 
                  'decode_input_len_mean', 'decode_input_len_std', 'tp_degree', 'freq_mhz']
    df_filtered = df.groupby(group_cols).median().reset_index()
    print("after filtering:", df_filtered.shape)
    return df_filtered


def build_model(role='unified'):
    """Build just the estimator (no pipeline).

    Parameters:
    - role: 'unified' for combined prefill/decode model
    """
    est = LGBMRegressor(
        random_state=42, 
        linear_tree=True,
        device_type='gpu',
        monotone_constraints_method='intermediate',
        force_col_wise=True,
        verbose=-1
    )

    return est


def get_param_grid():
    """Define parameter grid for hyperparameter search"""
    param_grid = {
        'boosting_type': ['gbdt'],
        'learning_rate': [0.08],
        'linear_lambda': [1e-3],
        'min_child_samples': [30],
        'num_iterations': [300],
        'num_leaves': [90,],
        'reg_lambda': [1e-1],
    }
    
    return param_grid


def custom_scorer(y_true, y_pred):
    """
    Custom scorer that denormalizes predictions and ignores errors <= 0.001.
    
    Parameters:
    - y_true: ground truth in log space
    - y_pred: predictions in log space
    
    Returns:
    - Negative MAE (for GridSearchCV maximization)
    """
    # Denormalize by applying exp
    y_true_denorm = np.exp(y_true)
    y_pred_denorm = np.exp(y_pred)
    
    # Calculate absolute errors
    abs_errors = np.abs(y_true_denorm - y_pred_denorm)
    
    # Set errors <= 0.001 to 0
    abs_errors[abs_errors <= 0.001] = 0
    
    # Return negative MAE (GridSearchCV maximizes, so negate for minimization)
    return -np.mean(abs_errors)


def train_and_save(df, feature_cols, target_col, out_name, model_dir, role: str = 'unified', use_grid_search=True):
    results = {}
    df_t = df.dropna(subset=[target_col])
    if df_t.shape[0] < 10:
        print(f"not enough rows to train {target_col} (found {df_t.shape[0]}). skipping.")
        return None

    X = df_t[feature_cols]
    print(f"features used for {target_col}:", feature_cols)
    y = df_t[target_col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Log normalization using numpy (log1p handles zeros safely)
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Apply log1p normalization to all features except tp_degree
    feature_cols_to_normalize = [col for col in feature_cols if col != 'tp_degree']
    X_train_scaled[feature_cols_to_normalize] = np.log1p(X_train[feature_cols_to_normalize])
    X_test_scaled[feature_cols_to_normalize] = np.log1p(X_test[feature_cols_to_normalize])
    
    # Apply log transform to target
    y_train_log = np.log(y_train)
    
    model = build_model(role=role)
    
    if use_grid_search:
        print("\n=== Performing hyperparameter search with GridSearchCV ===")
        param_grid = get_param_grid()
        
        # Use custom scorer
        scorer = make_scorer(custom_scorer)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring=scorer,
            n_jobs=2,
            return_train_score=True,
            verbose=2
        )
        
        grid_search.fit(X_train_scaled, y_train_log)
        
        print("\n=== Best parameters found ===")
        print(grid_search.best_params_)
        print(f"Best cross-validation MAE: {-grid_search.best_score_:.4f}")
        
        # Use the best estimator
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(os.path.join(model_dir, f"{target_col}_grid_search_results.csv"), index=False)
        print(f"Grid search results saved to: {os.path.join(model_dir, f'{target_col}_grid_search_results.csv')}")

        # Re-train on full training set with best parameters
        if use_grid_search and best_params:
            print("\n=== Re-training on full training set with best parameters ===")
            model.fit(X_train_scaled, y_train_log)
    else:
        print("\n=== Training with default parameters (no grid search) ===")
        model.fit(X_train_scaled, y_train_log)
        best_params = None

    # Predict and apply exp transform
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.exp(y_pred_log)
    
    mae = mean_absolute_error(y_test, y_pred)
    mape = (abs(y_test - y_pred) / y_test).mean() * 100

    # Calculate metrics for each category separately
    X_test_with_pred = X_test.copy()
    X_test_with_pred['ground_truth'] = y_test.values
    X_test_with_pred['predicted'] = y_pred
    
    # Identify categories based on batch sizes
    is_prefill_only = (X_test_with_pred['prefill_batch_size'] > 0) & (X_test_with_pred['decode_batch_size'] == 0)
    is_decode_only = (X_test_with_pred['prefill_batch_size'] == 0) & (X_test_with_pred['decode_batch_size'] > 0)
    is_mixed = (X_test_with_pred['prefill_batch_size'] > 0) & (X_test_with_pred['decode_batch_size'] > 0)
    
    category_metrics = {}
    for category_name, mask in [('prefill_only', is_prefill_only), 
                                 ('decode_only', is_decode_only), 
                                 ('mixed', is_mixed)]:
        if mask.sum() > 0:
            y_true_cat = X_test_with_pred.loc[mask, 'ground_truth']
            y_pred_cat = X_test_with_pred.loc[mask, 'predicted']
            mae_cat = mean_absolute_error(y_true_cat, y_pred_cat)
            mape_cat = (abs(y_true_cat - y_pred_cat) / y_true_cat).mean() * 100
            category_metrics[category_name] = {
                'mae': mae_cat,
                'mape': mape_cat,
                'n_samples': mask.sum()
            }
            print(f"  {category_name}: MAE={mae_cat:.4f}, MAPE={mape_cat:.4f}, n={mask.sum()}")
        else:
            category_metrics[category_name] = {'mae': None, 'mape': None, 'n_samples': 0}
            print(f"  {category_name}: No samples in test set")

    os.makedirs(model_dir, exist_ok=True)
    joblib_path = os.path.join(model_dir, out_name)
    
    # Save model with normalization info
    model_data = {
        'model': model,
        'normalization': 'log1p',
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, joblib_path)

    results = {
        'model_path': joblib_path,
        'mae': mae,
        'mape': mape,
        'n_train': len(X_train),
        'category_metrics': category_metrics,
        'best_params': best_params
    }
    print(f"trained {target_col}: saved -> {joblib_path}  MAE={mae:.4f}  MAPE={mape:.4f}  train_rows={len(X_train)}")

    # Save detailed predictions file
    results_df = X_test_with_pred.copy()
    results_df['error'] = results_df['predicted'] - results_df['ground_truth']
    results_df['category'] = 'unknown'
    results_df.loc[is_prefill_only, 'category'] = 'prefill_only'
    results_df.loc[is_decode_only, 'category'] = 'decode_only'
    results_df.loc[is_mixed, 'category'] = 'mixed'
    results_df = results_df.sort_values(by='error', key=abs, ascending=False).reset_index(drop=True)
    results_csv_path = os.path.join(model_dir, f"{target_col}_pred_vs_true.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    # Save category-specific metrics to CSV
    category_metrics_df = pd.DataFrame([
        {
            'category': cat_name,
            'mae': metrics['mae'],
            'mape': metrics['mape'],
            'n_samples': metrics['n_samples']
        }
        for cat_name, metrics in category_metrics.items()
    ])
    category_metrics_csv_path = os.path.join(model_dir, f"{target_col}_category_metrics.csv")
    category_metrics_df.to_csv(category_metrics_csv_path, index=False)
    print(f"  Category metrics saved to: {category_metrics_csv_path}")

    return results


def predict_with_model(inputs, model_data, feature_cols):
    """
    inputs: list/tuple single-row matching feature_cols or pandas.DataFrame
    model_data: dict with 'model', 'normalization' keys
    returns: float prediction
    """
    if isinstance(inputs, (list, tuple)):
        X = pd.DataFrame([inputs], columns=feature_cols)
    elif isinstance(inputs, pd.DataFrame):
        X = inputs.copy()
    else:
        raise ValueError('inputs must be list/tuple or pandas.DataFrame')

    for c in feature_cols:
        try:
            X[c] = pd.to_numeric(X[c], errors='coerce')
        except Exception:
            pass

    model = model_data['model']
    
    # Apply log normalization
    X_scaled = np.log1p(X)
    
    # Predict (returns log-transformed values)
    y_pred_log = model.predict(X_scaled)
    
    # Apply exp transform
    y_pred = np.exp(y_pred_log)
    
    return float(y_pred[0])


def main():
    parser = argparse.ArgumentParser(description='Train unified power model for prefill and decode (TP8 only)')
    parser.add_argument('--model-dir', default=('/export2/obasit/ClusterLevelServing/vllm_profiler_logs/power_model/tp8'), help='Directory to store trained models')
    parser.add_argument('--no-grid-search', action='store_true', help='Disable hyperparameter grid search')
    args = parser.parse_args()

    # Data paths for decode
    CSV_PATH_TP2_D = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_decode/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/decode_powers.csv"
    CSV_PATH_TP2_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_Nov12_TP2_decode_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/decode_powers.csv"
    CSV_PATH_TP4_D = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_decode/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_powers.csv"
    CSV_PATH_TP4_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_Nov12_TP4_decode_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_powers.csv"
    CSV_PATH_TP4_D2 = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_Nov19_TP4_decode_latency_largebatch/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_powers.csv"
    CSV_PATH_TP4_D3 = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_Nov21_TP4_decoder_latency_batchgt512/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_powers.csv"

    # Data paths for prefill
    CSV_PATH_TP2_P = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_prefill_back_to_back/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/prefill_powers.csv"
    CSV_PATH_TP4_P = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_prefill_back_to_back/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4_back_to_back/prefill_powers.csv"
    CSV_PATH_TP2_P_1 = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_prefill_real_arrival/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/prefill_powers.csv"
    CSV_PATH_TP4_P_1 = "/export2/obasit/ClusterLevelServing/vllm_logs/energy_profiler_logs/profiler_logs_prefill_real_arrival/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_powers.csv"

    # Data path for mixed
    CSV_PATH_MIXED = "/export2/obasit/ClusterLevelServing/vllm_profiler_logs/combined_profiling_results.csv"

    MODEL_DIR = os.path.join(args.model_dir)

    # Unified feature columns (both prefill and decode features)
    UNIFIED_FEATURE_COLS = [
        'prefill_batch_size', 'prefill_input_len_sum', 'prefill_input_len_mean', 'prefill_input_len_std',
        'decode_batch_size', 'decode_input_len_sum', 'decode_input_len_mean', 'decode_input_len_std',
        'tp_degree', 'freq_mhz'
    ]
    UNIFIED_TARGET = 'power_w'
    UNIFIED_OUT = 'unified_power_model_tp8.joblib'

    print('\n=== Loading and preprocessing all data (prefill, decode, mixed) ===')
    
    # Load prefill data
    print('\n--- Loading prefill data ---')
    df_p2 = load_and_prepare_prefill(CSV_PATH_TP2_P, 'Llama-3.3-70B-Instruct', tp=2)
    df_p2_1 = load_and_prepare_prefill(CSV_PATH_TP2_P_1, 'Llama-3.3-70B-Instruct', tp=2)
    print(f"Prefill TP2 samples: {len(df_p2)} + {len(df_p2_1)}")
    df_p2_all = pd.concat([df_p2, df_p2_1], ignore_index=True)
    
    df_p4 = load_and_prepare_prefill(CSV_PATH_TP4_P, 'Llama-3.3-70B-Instruct', tp=4)
    df_p4_1 = load_and_prepare_prefill(CSV_PATH_TP4_P_1, 'Llama-3.3-70B-Instruct', tp=4)
    print(f"Prefill TP4 samples: {len(df_p4)} + {len(df_p4_1)}")
    df_p4_all = pd.concat([df_p4, df_p4_1], ignore_index=True)
    
    df_prefill_all = pd.concat([df_p2_all, df_p4_all], ignore_index=True)
    print(f"Total prefill samples: {len(df_prefill_all)}")

    # Load decode data
    print('\n--- Loading decode data ---')
    df_d2 = load_and_prepare_decode(CSV_PATH_TP2_D, 'Llama-3.3-70B-Instruct', tp=2)
    df_d21 = load_and_prepare_decode(CSV_PATH_TP2_D1, 'Llama-3.3-70B-Instruct', tp=2)
    print(f"Decode TP2 samples: {len(df_d2)} + {len(df_d21)}")
    
    df_d4 = load_and_prepare_decode(CSV_PATH_TP4_D, 'Llama-3.3-70B-Instruct', tp=4)
    df_d41 = load_and_prepare_decode(CSV_PATH_TP4_D1, 'Llama-3.3-70B-Instruct', tp=4)
    df_d42 = load_and_prepare_decode(CSV_PATH_TP4_D2, 'Llama-3.3-70B-Instruct', tp=4)
    df_d43 = load_and_prepare_decode(CSV_PATH_TP4_D3, 'Llama-3.3-70B-Instruct', tp=4)
    print(f"Decode TP4 samples: {len(df_d4)} + {len(df_d41)} + {len(df_d42)} + {len(df_d43)}")
    
    df_decode_all = pd.concat([df_d2, df_d21, df_d4, df_d41, df_d42, df_d43], ignore_index=True)
    print(f"Total decode samples: {len(df_decode_all)}")

    # Load mixed data
    print('\n--- Loading data from profiler ---')
    df_mixed = load_and_prepare_mixed(CSV_PATH_MIXED, 'Llama-3.3-70B-Instruct')
    print(f"Total mixed samples: {len(df_mixed)}")
    print(f' TP2 mixed samples: {len(df_mixed[(df_mixed["tp_degree"]==2) & (df_mixed["test_name"].str.contains("mixed"))])}, prefill samples: {len(df_mixed[(df_mixed["tp_degree"]==2) & (df_mixed["test_name"].str.contains("prefill"))])}, decode samples: {len(df_mixed[(df_mixed["tp_degree"]==2) & (df_mixed["test_name"].str.contains("decode"))])}')
    print(f' TP4 mixed samples: {len(df_mixed[(df_mixed["tp_degree"]==4) & (df_mixed["test_name"].str.contains("mixed"))])}, prefill samples: {len(df_mixed[(df_mixed["tp_degree"]==4) & (df_mixed["test_name"].str.contains("prefill"))])}, decode samples: {len(df_mixed[(df_mixed["tp_degree"]==4) & (df_mixed["test_name"].str.contains("decode"))])}')
    print(f' TP8 mixed samples: {len(df_mixed[(df_mixed["tp_degree"]==8) & (df_mixed["test_name"].str.contains("mixed"))])}, prefill samples: {len(df_mixed[(df_mixed["tp_degree"]==8) & (df_mixed["test_name"].str.contains("prefill"))])}, decode samples: {len(df_mixed[(df_mixed["tp_degree"]==8) & (df_mixed["test_name"].str.contains("decode"))])}')
    df_mixed.drop(columns=['test_name'], inplace=True)

    # Combine all data
    print('\n--- Combining all data ---')
    df_combined = pd.concat([df_prefill_all, df_decode_all, df_mixed], ignore_index=True)
    df_combined = df_combined[df_combined['tp_degree'] == 8].copy()
    print(f"Total combined samples before filtering: {len(df_combined)}")
    
    df_combined = filter_inputs(df_combined)
    print(f"Total combined samples after filtering: {len(df_combined)}")
    
    # Save combined data
    df_combined.to_csv(os.path.join(MODEL_DIR, 'unified_power_data_cleaned_tp8.csv'), index=False)
    
    # Train unified model
    print('\n=== Training unified power model (TP8) ===')
    stats_unified = train_and_save(
        df_combined, 
        UNIFIED_FEATURE_COLS, 
        UNIFIED_TARGET, 
        UNIFIED_OUT, 
        MODEL_DIR, 
        role='unified',
        use_grid_search=not args.no_grid_search
    )

    # Load model for sample inference
    print('\n=== Sample inference with unified model ===')
    model_joblib = os.path.join(MODEL_DIR, UNIFIED_OUT)
    model_data = joblib.load(model_joblib) if os.path.exists(model_joblib) else None

    if model_data is not None:
        # Example 1: Pure prefill (decode features = 0)
        sample_prefill = [
            10,  # prefill_batch_size
            5000,  # prefill_input_len_sum
            500,  # prefill_input_len_mean
            100,  # prefill_input_len_std
            0,  # decode_batch_size
            0,  # decode_input_len_sum
            0,  # decode_input_len_mean
            0,  # decode_input_len_std
            8,  # tp_degree
            1830,  # freq_mhz
        ]

        # Example 2: Pure decode (prefill features = 0)
        sample_decode = [
            0,  # prefill_batch_size
            0,  # prefill_input_len_sum
            0,  # prefill_input_len_mean
            0,  # prefill_input_len_std
            4,  # decode_batch_size
            1200,  # decode_input_len_sum
            300,  # decode_input_len_mean
            50,  # decode_input_len_std
            2,  # tp_degree
            1830,  # freq_mhz
        ]

        # Example 3: Mixed (both prefill and decode)
        sample_mixed = [
            1,  # prefill_batch_size
            354,  # prefill_input_len_sum
            354,  # prefill_input_len_mean
            0,  # prefill_input_len_std
            6,  # decode_batch_size
            1589,  # decode_input_len_sum
            266.5,  # decode_input_len_mean
            231.13,  # decode_input_len_std
            2,  # tp_degree
            360,  # freq_mhz
        ]

        t0 = time.time()
        try:
            r_prefill = predict_with_model(sample_prefill, model_data, UNIFIED_FEATURE_COLS)
            print(f'Sample prefill prediction: {r_prefill:.2f}W')
        except Exception as e:
            print(f'Prefill sample predict failed: {e}')
        print(f'Prefill inference time: {time.time()-t0:.6f}s')

        t0 = time.time()
        try:
            r_decode = predict_with_model(sample_decode, model_data, UNIFIED_FEATURE_COLS)
            print(f'Sample decode prediction: {r_decode:.2f}W')
        except Exception as e:
            print(f'Decode sample predict failed: {e}')
        print(f'Decode inference time: {time.time()-t0:.6f}s')

        t0 = time.time()
        try:
            r_mixed = predict_with_model(sample_mixed, model_data, UNIFIED_FEATURE_COLS)
            print(f'Sample mixed prediction: {r_mixed:.2f}W')
        except Exception as e:
            print(f'Mixed sample predict failed: {e}')
        print(f'Mixed inference time: {time.time()-t0:.6f}s')

    print('\nDone.')


if __name__ == '__main__':
    main()
