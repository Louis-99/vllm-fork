#!/usr/bin/env python3
"""
Unified latency model for prefill and decode with separate feature columns.

Features:
- Trains a single model for both prefill and decode latency
- Uses separate feature columns: prefill_batch_size, decode_batch_size, etc.
- Three data modes:
  - prefill: decode features set to 0
  - decode: prefill features set to 0  
  - mixed: both prefill and decode features present (from combined_profiling_results.csv)
- Saves as .joblib and optionally converts to ONNX
"""
import os
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

import onnxruntime as ort
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from onnxmltools import convert_lightgbm

def load_and_prepare(path, model_name, tp=None, numeric_cols=None):
    df = pd.read_csv(path)
    if tp is not None:
        df['tp_degree'] = tp
    if numeric_cols is None:
        numeric_cols = []
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['model'] = model_name
    df = df.dropna(subset=numeric_cols + ['model'])
    return df


def load_and_prepare_prefill(path, model_name, tp=None):
    """Load prefill data and convert to unified format with decode features = 0"""
    df = pd.read_csv(path)
    if tp is not None:
        df['tp_degree'] = tp
    
    # Rename to prefill-specific columns
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
        'latency_s': df['latency_prefill_s']
    })
    
    return df_unified.dropna()


def load_and_prepare_decode(path, model_name, tp=None):
    """Load decode data and convert to unified format with prefill features = 0"""
    df = pd.read_csv(path)
    if tp is not None:
        df['tp_degree'] = tp
    
    # Rename to decode-specific columns
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
        'latency_s': df['latency_decode_s']
    })
    
    return df_unified.dropna()


def load_and_prepare_mixed(path, model_name):
    """Load mixed data from combined_profiling_results.csv"""
    df = pd.read_csv(path)
    
    # Map combined_profiling_results columns to unified format
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
        'latency_s': df['mean_latency']
    })
    
    return df_unified.dropna()


def filter_inputs(df):
    """
    Filter to keep only median latency for duplicate configurations.
    """

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
    est = lgb.LGBMRegressor(
        random_state=42, 
        linear_tree=True,
        device_type='gpu',
        monotone_constraints_method='intermediate',
        force_col_wise=True,
        verbose=-1,
    )

    return est


def get_param_grid():
    """Define parameter grid for hyperparameter search"""
    param_grid = {
        'boosting_type': ['gbdt'],
        'learning_rate': [0.1],
        'linear_lambda': [1e-3, 5e-3],
        'min_child_samples': [30, 40, ],
        'num_iterations': [400],
        'num_leaves': [80],
        'reg_lambda': [1e-2, 1e-3],
        'monotonic_cst': [[0, 0, 0, 0, 0, 0, 0, 0, 0, -1]]
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
    
    # underprediction is slightly rewarded
    abs_errors[(y_true_denorm - y_pred_denorm <= 0.001) & (y_true_denorm > y_pred_denorm)] = 0
    
    # Return negative MAE (GridSearchCV maximizes, so negate for minimization)
    return -np.mean(abs_errors)


def train_and_save(df, feature_cols, target_col, out_name, model_dir, role: str = 'unified', use_grid_search=True):
    results = {}
    df_t = df.dropna(subset=[target_col])
    if df_t.shape[0] < 10:
        print(f"not enough rows to train {target_col} (found {df_t.shape[0]}). skipping.")
        return None

    X = df_t[feature_cols].copy()
    y = df_t[target_col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    # Ensure column names are preserved as strings
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.columns = [str(c) for c in feature_cols]
    X_test.columns = [str(c) for c in feature_cols]
    
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
            n_jobs=1,
            verbose=2,
            return_train_score=True
        )
        
        grid_search.fit(X_train_scaled, y_train_log)
        
        print("\n=== Best parameters found ===")
        print(grid_search.best_params_)
        print(f"Best cross-validation MAE: {-grid_search.best_score_:.6f}")
        
        # Use the best estimator
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(os.path.join(model_dir, f"{target_col}_grid_search_results.csv"), index=False)
        print(f"Grid search results saved to: {os.path.join(model_dir, f'{target_col}_grid_search_results.csv')}")
    else:
        print("\n=== Training with default parameters (no grid search) ===")
        model.fit(X_train_scaled, y_train_log)
        best_params = None

    print("Training completed.")
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

    # Save native LightGBM txt model (compatible with C++ inference via lightgbm C API)
    model.booster_.save_model(joblib_path)
    print(f"  Native LightGBM model saved -> {joblib_path}")

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


def load_onnx(path):
    if path and os.path.exists(path):
        return ort.InferenceSession(path)
    return None


def predict_with_model(inputs, model):
    """
    inputs: list/tuple single-row matching feature_cols or pandas.DataFrame
    returns: float prediction
    """
    # Convert to 2D numpy array if needed
    if isinstance(inputs, list):
        inputs = np.array(inputs)
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    
    # Apply log normalization to all features except tp_degree (index 8)
    X_scaled = np.log1p(inputs)
    X_scaled[:, 8] = inputs[:, 8]
    
    # Predict (returns log-transformed values)
    y_pred_log = model.predict(X_scaled)

    # Apply exp transform
    y_pred = np.exp(y_pred_log)
    
    return float(y_pred[0])


def main():
    parser = argparse.ArgumentParser(description='Train unified latency model for prefill and decode')
    parser.add_argument('--model-dir', default=('/export2/obasit/ClusterLevelServing/vllm_profiler_logs/latency_model/lightgbm_tp2_tp4'), help='Directory to store trained models')
    parser.add_argument('--no-grid-search', action='store_true', help='Disable hyperparameter grid search')
    args = parser.parse_args()

    # Data paths for prefill
    CSV_PATH_TP2_P = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/default_log_path/prefill_latencies.csv"
    CSV_PATH_TP2_P1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP2_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/prefill_latencies.csv"
    CSV_PATH_TP2_P2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov13_TP24_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/prefill_latencies.csv"
    CSV_PATH_TP2_P3 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov25_prefill_chunked2k_profiling/tp2/prefill_latencies.csv"
    CSV_PATH_TP4_P = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"
    CSV_PATH_TP4_P1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP4_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"
    CSV_PATH_TP4_P2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov13_TP24_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"
    CSV_PATH_TP4_P3 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov25_prefill_chunked2k_profiling/tp4/prefill_latencies.csv"

    # Data paths for decode
    CSV_PATH_TP2_D = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/default_log_path/decode_latencies.csv"
    CSV_PATH_TP2_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP2_decode_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/decode_latencies.csv"
    CSV_PATH_TP4_D = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_TP4_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP4_decode_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_TP4_D2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov19_TP4_decode_latency_largebatch/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_TP4_D3 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov21_TP4_decoder_latency_batchgt512/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"

    # Data path for mixed
    CSV_PATH_MIXED = "/export2/obasit/ClusterLevelServing/vllm_profiler_logs/combined_profiling_results.csv"
    CSV_PATH_MIXED_2 = "/export2/obasit/ClusterLevelServing/vllm_profiler_logs/older_style_latency_only_profiling/mixed_inference_results_tp2.csv"
    CSV_PATH_MIXED_4 = "/export2/obasit/ClusterLevelServing/vllm_profiler_logs/older_style_latency_only_profiling/mixed_inference_results_tp4.csv"
    CSV_PATH_MIXED_8 = "/export2/obasit/ClusterLevelServing/vllm_profiler_logs/older_style_latency_only_profiling/mixed_inference_results_tp8.csv"
    CSV_PATH_DECODE_2 = "/export2/obasit/ClusterLevelServing/vllm_profiler_logs/older_style_latency_only_profiling/decode_inference_results_tp2.csv"
    CSV_PATH_DECODE_4 = "/export2/obasit/ClusterLevelServing/vllm_profiler_logs/older_style_latency_only_profiling/decode_inference_results_tp4.csv"
    CSV_PATH_DECODE_8 = "/export2/obasit/ClusterLevelServing/vllm_profiler_logs/older_style_latency_only_profiling/decode_inference_results_tp8.csv"

    # together ai logs
    CSV_PATH_MIXED1 = "/export2/obasit/ClusterLevelServing/vllm_logs/4_nodes_logs_together_ai/kube_results/profiler_logs_0/placeonly_collo/mixed_latencies.csv"
    CSV_PATH_MIXED2 = "/export2/obasit/ClusterLevelServing/vllm_logs/4_nodes_logs_together_ai/kube_results/profiler_logs_0/dynamo/mixed_latencies.csv"
    CSV_PATH_P1 = "/export2/obasit/ClusterLevelServing/vllm_logs/4_nodes_logs_together_ai/kube_results/profiler_logs_0/placeonly_disag/prefill_latencies.csv"
    CSV_PATH_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/4_nodes_logs_together_ai/kube_results/profiler_logs_0/placeonly_disag/decode_latencies.csv"

    MODEL_DIR = os.path.join(args.model_dir)

    # Unified feature columns (both prefill and decode features)
    UNIFIED_FEATURE_COLS = [
        'prefill_batch_size', 'prefill_input_len_sum', 'prefill_input_len_mean', 'prefill_input_len_std',
        'decode_batch_size', 'decode_input_len_sum', 'decode_input_len_mean', 'decode_input_len_std',
        'tp_degree', 'freq_mhz'
    ]
    UNIFIED_TARGET = 'latency_s'
    UNIFIED_OUT = 'unified_latency_model_tp2_tp4.txt'

    print('\n=== Loading and preprocessing all data (prefill, decode, mixed) ===')
    
    # Load prefill data
    print('\n--- Loading prefill data ---')
    df_p2 = load_and_prepare_prefill(CSV_PATH_TP2_P, 'Llama-3.3-70B-Instruct', tp=2)
    df_p21 = load_and_prepare_prefill(CSV_PATH_TP2_P1, 'Llama-3.3-70B-Instruct', tp=2)
    df_p22 = load_and_prepare_prefill(CSV_PATH_TP2_P2, 'Llama-3.3-70B-Instruct', tp=2)
    df_p23 = load_and_prepare_prefill(CSV_PATH_TP2_P3, 'Llama-3.3-70B-Instruct', tp=2)
    print(f"Prefill TP2 samples: {len(df_p2)} + {len(df_p21)} + {len(df_p22)} + {len(df_p23)}")
    
    df_p4 = load_and_prepare_prefill(CSV_PATH_TP4_P, 'Llama-3.3-70B-Instruct', tp=4)
    df_p41 = load_and_prepare_prefill(CSV_PATH_TP4_P1, 'Llama-3.3-70B-Instruct', tp=4)
    df_p42 = load_and_prepare_prefill(CSV_PATH_TP4_P2, 'Llama-3.3-70B-Instruct', tp=4)
    df_p43 = load_and_prepare_prefill(CSV_PATH_TP4_P3, 'Llama-3.3-70B-Instruct', tp=4)
    print(f"Prefill TP4 samples: {len(df_p4)} + {len(df_p41)} + {len(df_p42)} + {len(df_p43)}")

    df_p_t = load_and_prepare_prefill(CSV_PATH_P1, 'Llama-3.3-70B-Instruct')
    
    df_prefill_all = pd.concat([df_p2, df_p21, df_p22, df_p23, df_p4, df_p41, df_p42, df_p43, df_p_t], ignore_index=True)
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

    df_d_t = load_and_prepare_decode(CSV_PATH_D1, 'Llama-3.3-70B-Instruct')
    
    df_decode_all = pd.concat([df_d2, df_d21, df_d4, df_d41, df_d42, df_d43, df_d_t], ignore_index=True)
    print(f"Total decode samples: {len(df_decode_all)}")

    # Load mixed data
    print('\n--- Loading data from profiler ---')
    df_mixed = load_and_prepare_mixed(CSV_PATH_MIXED, 'Llama-3.3-70B-Instruct')
    df_mixed1 = load_and_prepare_mixed(CSV_PATH_MIXED1, 'Llama-3.3-70B-Instruct')
    df_mixed2 = load_and_prepare_mixed(CSV_PATH_MIXED2, 'Llama-3.3-70B-Instruct')
    df_mixed_2 = load_and_prepare_mixed(CSV_PATH_MIXED_2, 'Llama-3.3-70B-Instruct')
    df_mixed_4 = load_and_prepare_mixed(CSV_PATH_MIXED_4, 'Llama-3.3-70B-Instruct')
    df_mixed_8 = load_and_prepare_mixed(CSV_PATH_MIXED_8, 'Llama-3.3-70B-Instruct')
    df_decode_2 = load_and_prepare_mixed(CSV_PATH_DECODE_2, 'Llama-3.3-70B-Instruct')
    df_decode_4 = load_and_prepare_mixed(CSV_PATH_DECODE_4, 'Llama-3.3-70B-Instruct')
    df_decode_8 = load_and_prepare_mixed(CSV_PATH_DECODE_8, 'Llama-3.3-70B-Instruct')
    df_mixed = pd.concat([df_mixed, df_mixed1, df_mixed2, df_mixed_2, df_mixed_4, df_mixed_8, df_decode_2, df_decode_4, df_decode_8], ignore_index=True)
    print(f"Total mixed samples: {len(df_mixed)}")
    print(f' TP2 mixed samples: {len(df_mixed[(df_mixed["tp_degree"]==2) & (df_mixed["test_name"].str.contains("mixed"))])}, prefill samples: {len(df_mixed[(df_mixed["tp_degree"]==2) & (df_mixed["test_name"].str.contains("prefill"))])}, decode samples: {len(df_mixed[(df_mixed["tp_degree"]==2) & (df_mixed["test_name"].str.contains("decode"))])}')
    print(f' TP4 mixed samples: {len(df_mixed[(df_mixed["tp_degree"]==4) & (df_mixed["test_name"].str.contains("mixed"))])}, prefill samples: {len(df_mixed[(df_mixed["tp_degree"]==4) & (df_mixed["test_name"].str.contains("prefill"))])}, decode samples: {len(df_mixed[(df_mixed["tp_degree"]==4) & (df_mixed["test_name"].str.contains("decode"))])}')
    print(f' TP8 mixed samples: {len(df_mixed[(df_mixed["tp_degree"]==8) & (df_mixed["test_name"].str.contains("mixed"))])}, prefill samples: {len(df_mixed[(df_mixed["tp_degree"]==8) & (df_mixed["test_name"].str.contains("prefill"))])}, decode samples: {len(df_mixed[(df_mixed["tp_degree"]==8) & (df_mixed["test_name"].str.contains("decode"))])}')
    df_mixed.drop(columns=['test_name'], inplace=True)

    # Combine all data
    print('\n--- Combining all data ---')
    df_combined = pd.concat([df_prefill_all, df_decode_all, df_mixed], ignore_index=True)
    df_combined = df_combined[df_combined['tp_degree'] != 8].copy()
    print(f"Total combined samples before filtering: {len(df_combined)}")
    
    df_combined = filter_inputs(df_combined)
    print(f"Total combined samples after filtering: {len(df_combined)}")
    
    # Save combined data
    df_combined.to_csv(os.path.join(MODEL_DIR, 'unified_data_cleaned.csv'), index=False)
    
    # Train unified model
    print('\n=== Training unified latency model ===')
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
    model_path = os.path.join(MODEL_DIR, UNIFIED_OUT)
    if not os.path.exists(model_path):
        print(f'Model not found at {model_path}, skipping inference.')
    else:
        booster = lgb.Booster(model_file=model_path)

        # prefill_batch_size, prefill_input_len_sum, prefill_input_len_mean, prefill_input_len_std,
        # decode_batch_size, decode_input_len_sum, decode_input_len_mean, decode_input_len_std,
        # tp_degree, freq_mhz

        samples = {
            'prefill only (tp2, 1830MHz)': [2, 512, 256, 44,        0, 0, 0, 0,                 2, 1830],
            'prefill only (tp4, 1080MHz)': [4, 2030, 169.1, 312.1,  0, 0, 0, 0,                 4, 1080],
            'decode only  (tp2, 1830MHz)': [0, 0, 0, 0,             32, 9984, 312.0, 195.5,     2, 1830],
            'decode only  (tp4, 1080MHz)': [0, 0, 0, 0,             200, 55755, 278.7, 178.9,   4, 1080],
            'mixed        (tp2, 1830MHz)': [2, 512, 256, 44,        32, 9984, 312.0, 195.5,     2, 1830],
            'mixed        (tp4, 1080MHz)': [4, 2030, 169.1, 312.1,  200, 55755, 278.7, 178.9,   4, 1080],
        }

        for label, sample in samples.items():
            t0 = time.time()
            try:
                pred = predict_with_model(sample, booster)
                print(f'  {label}: {pred*1e3:.3f}ms  ({(time.time()-t0)*1e3:.3f}ms)')
            except Exception as e:
                print(f'  {label}: FAILED - {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
