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
import joblib
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
    df['model'] = model_name
    
    # Rename to prefill-specific columns
    df_unified = pd.DataFrame({
        'model': df['model'],
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
    df['model'] = model_name
    
    # Rename to decode-specific columns
    df_unified = pd.DataFrame({
        'model': df['model'],
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
    df['model'] = model_name
    
    # Map combined_profiling_results columns to unified format
    df_unified = pd.DataFrame({
        'test_name': df['test_name'],  # keep for reference
        'model': df['model'],
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
    group_cols = ['model', 'prefill_batch_size', 'prefill_input_len_sum', 'prefill_input_len_mean', 
                  'prefill_input_len_std', 'decode_batch_size', 'decode_input_len_sum', 
                  'decode_input_len_mean', 'decode_input_len_std', 'tp_degree', 'freq_mhz']
    df_filtered = df.groupby(group_cols).median().reset_index()
    print("after filtering:", df_filtered.shape)
    return df_filtered


def build_pipeline(role='unified'):
    """Build a preprocessing + estimator pipeline.

    Parameters:
    - role: 'unified' for combined prefill/decode model
    """
    # OneHotEncoder for `model` and StandardScaler for remainder
    cat_cols = ['model']
    pre = ColumnTransformer([
        ('ohe_model', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder=StandardScaler())
    pre.set_output(transform="pandas")  # Preserve DataFrame structure and column names

    # est = GradientBoostingRegressor(
    #     n_estimators=500,
    #     random_state=42,
    #     verbose=0,
    #     n_iter_no_change=20,
    #     validation_fraction=0.05,
    #     tol=1e-5
    # )

    # est = RandomForestRegressor(
    #     random_state=42,
    #     n_jobs=30
    # )

    # est = HistGradientBoostingRegressor(
    #     random_state=42,
    #     verbose=0,
    #     n_iter_no_change=20,
    #     validation_fraction=0.05,
    # )

    est = lgb.LGBMRegressor(
        random_state=42, 
        linear_tree=True,
        device_type='cpu',
        monotone_constraints_method='intermediate',
        n_jobs=32,
        force_col_wise=True,
        verbose=-1
    )

    pipe = Pipeline([
        ('pre', pre), 
        ('est', est)
    ])

    # Wrap pipeline to apply log transform to target and exp to predictions
    pipe_with_exp = TransformedTargetRegressor(
        regressor=pipe,
        func=np.log,
        inverse_func=np.exp
    )

    return pipe_with_exp


def get_param_grid():
    """Define parameter grid for hyperparameter search"""
    # param_grid = {
    #     'est__max_depth': [15, 18],
    #     'est__learning_rate': [0.15, 0.18],
    #     'est__min_samples_split': [2, 3],
    #     'est__min_samples_leaf': [2, 3],
    #     'est__subsample': [0.95]
    # }

    # param_grid = {
    #     'est__max_depth': [18, 20, 22],
    #     'est__min_samples_split': [2, 3],
    #     'est__min_samples_leaf': [2, 3],
    #     'est__n_estimators': [200, 300]
    # }

    # param_grid = {
    #     'est__max_depth': [15, 20],
    #     'est__learning_rate': [0.1, 0.2],
    #     'est__max_leaf_nodes': [20, 25, 30],
    #     'est__max_depth': [10, 15, 20],
    #     'est__min_samples_leaf': [10, 20, 30],
    #     'est__monotonic_cst': [ (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # no monotonic constraints
    #                             (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1),  # decreasing with freq_mhz
    #                             (0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1)]  # decreasing with tp_degree and freq_mhz
    # }

    param_grid = {
        'regressor__est__boosting_type': ['gbdt'],
        'regressor__est__num_leaves': [70, 80],
        'regressor__est__num_iterations': [300, 400],
        'regressor__est__learning_rate': [0.1],
        'regressor__est__learning_rate': [1e-1],
        'regressor__est__min_child_samples': [30, 40],
        'regressor__est__monotonic_cst': [ 
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),      # no monotonic constraints
        ],   
        'regressor__est__linear_lambda': [0, 1e-3],
        'regressor__est__reg_lambda': [1e-2],
    }
    
    return param_grid


def train_and_save(df, feature_cols, target_col, out_name, model_dir, convert_onnx=True, role: str = 'unified', use_grid_search=True):
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

    pipe = build_pipeline(role=role)
    
    if use_grid_search:
        print("\n=== Performing hyperparameter search with GridSearchCV ===")
        param_grid = get_param_grid()
        
        # Custom scorer: 0 if abs(error) <= 0.001, else MSE
        def custom_scorer(y_true, y_pred):
            errors = np.abs(y_true - y_pred)
            mse = errors ** 2
            # Return negative MSE for errors > threshold, 0 for errors within threshold
            return -np.where(errors <= 0.001, 0, mse).mean()
        
        scorer = make_scorer(custom_scorer, greater_is_better=True)
        
        grid_search = GridSearchCV(
            pipe,
            param_grid,
            cv=5,
            scoring=scorer,
            n_jobs=1,
            verbose=2,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        print("\n=== Best parameters found ===")
        print(grid_search.best_params_)
        print(f"Best cross-validation MAE: {-grid_search.best_score_:.6f}")
        
        # Use the best estimator
        pipe = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(os.path.join(model_dir, f"{target_col}_grid_search_results.csv"), index=False)
        print(f"Grid search results saved to: {os.path.join(model_dir, f'{target_col}_grid_search_results.csv')}")
    else:
        print("\n=== Training with default parameters (no grid search) ===")
        pipe.fit(X_train, y_train)
        best_params = None
    
    # Train with best parameters from grid search
    if use_grid_search and best_params:
        print("\n=== Retraining on full training set with best parameters ===")
        pipe.fit(X_train, y_train)

    print("Training completed.")
    y_pred = pipe.predict(X_test)
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
    joblib.dump(pipe, joblib_path)

    onnx_path = joblib_path.replace('.joblib', '.onnx')
    if convert_onnx:
        try:
            # build initial types: for convert_sklearn we declare each input column
            initial_type = []
            for name in feature_cols:
                if name == 'model':
                    initial_type.append((name, StringTensorType([None, 1])))
                else:
                    initial_type.append((name, FloatTensorType([None, 1])))
            
            if isinstance(pipe.named_steps['est'], lgb.LGBMRegressor):
                onnx_model = convert_lightgbm(pipe.named_steps['est'], initial_types=initial_type)
            else:
                onnx_model = convert_sklearn(pipe, initial_types=initial_type)

            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            converted = True
        except Exception as e:
            print(f"ONNX conversion failed for {out_name}: {e}")
            converted = False
    else:
        converted = False

    results = {
        'model_path': joblib_path,
        'onnx_path': onnx_path if converted else None,
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


def predict_with_model(inputs, model, feature_cols):
    """
    inputs: list/tuple single-row matching feature_cols or pandas.DataFrame
    model: sklearn Pipeline (joblib) or onnxruntime InferenceSession
    returns: float prediction
    """
    if isinstance(inputs, (list, tuple)):
        X = pd.DataFrame([inputs], columns=feature_cols)
    elif isinstance(inputs, pd.DataFrame):
        X = inputs[feature_cols].copy()  # Ensure column order matches
    else:
        raise ValueError('inputs must be list/tuple or pandas.DataFrame')

    # Ensure proper dtypes and column names
    X.columns = [str(c) for c in feature_cols]
    for c in feature_cols:
        if c == 'model':
            X[c] = X[c].astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')

    # if ONNX session
    if hasattr(model, 'get_inputs'):
        input_feed = {}
        X_enc = X.copy()
        X_enc['model'] = X_enc['model'].astype(str)
        for col in feature_cols:
            val = X_enc[col].values.reshape(-1, 1)
            if X_enc[col].dtype == object or col == 'model':
                input_feed[col] = val.astype(str)
            else:
                input_feed[col] = val.astype('float32')
        out = model.run(None, input_feed)[0]
        return float(out[0][0].item()) if hasattr(out[0][0], 'item') else float(out[0][0])

    # otherwise assume sklearn pipeline
    pred = model.predict(X)
    return float(pred[0])


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

    MODEL_DIR = os.path.join(args.model_dir)

    # Unified feature columns (both prefill and decode features)
    UNIFIED_FEATURE_COLS = [
        'model',
        'prefill_batch_size', 'prefill_input_len_sum', 'prefill_input_len_mean', 'prefill_input_len_std',
        'decode_batch_size', 'decode_input_len_sum', 'decode_input_len_mean', 'decode_input_len_std',
        'tp_degree', 'freq_mhz'
    ]
    UNIFIED_TARGET = 'latency_s'
    UNIFIED_OUT = 'unified_latency_model_tp2_tp4.joblib'

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
    
    df_prefill_all = pd.concat([df_p2, df_p21, df_p22, df_p23, df_p4, df_p41, df_p42, df_p43], ignore_index=True)
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
        convert_onnx=True, 
        role='unified',
        use_grid_search=not args.no_grid_search
    )

    # Load model for sample inference
    print('\n=== Sample inference with unified model ===')
    if stats_unified and stats_unified.get('onnx_path'):
        model = load_onnx(stats_unified['onnx_path'])
    else:
        model_joblib = os.path.join(MODEL_DIR, UNIFIED_OUT)
        model = joblib.load(model_joblib) if os.path.exists(model_joblib) else None

    if model is not None:
        # Example 1: Pure prefill (decode features = 0)
        pre_sm = np.array([590, 267, 476, 10, 47, 102])
        sample_prefill = [
            'Llama-3.3-70B-Instruct',
            len(pre_sm),  # prefill_batch_size
            np.sum(pre_sm),  # prefill_input_len_sum
            np.mean(pre_sm),  # prefill_input_len_mean
            np.std(pre_sm),  # prefill_input_len_std
            0,  # decode_batch_size
            0,  # decode_input_len_sum
            0,  # decode_input_len_mean
            0,  # decode_input_len_std
            2,  # tp_degree
            360,  # freq_mhz
        ]

        # Example 2: Pure decode (prefill features = 0)
        dec_samp = np.array([34, 17, 75, 11, 7, 648, 117, 20, 17, 63, 10, 30, 26, 13, 19, 31, 15, 11, 40, 264]) + np.array([115, 104, 104, 99, 98, 93, 92, 87, 87, 87, 84, 83, 83, 81, 80, 79, 77, 76, 75, 74])
        sample_decode = [
            'Llama-3.3-70B-Instruct',
            0,  # prefill_batch_size
            0,  # prefill_input_len_sum
            0,  # prefill_input_len_mean
            0,  # prefill_input_len_std
            len(dec_samp),  # decode_batch_size
            np.sum(dec_samp),  # decode_input_len_sum
            np.mean(dec_samp),  # decode_input_len_mean
            np.std(dec_samp),  # decode_input_len_std
            4,  # tp_degree
            1830,  # freq_mhz
        ]

        # Example 3: Mixed (both prefill and decode)
        sample_mixed = [
            'Llama-3.3-70B-Instruct',
            1,  # prefill_batch_size
            354,  # prefill_input_len_sum
            354.0,  # prefill_input_len_mean
            0.0,  # prefill_input_len_std
            6,  # decode_batch_size
            1589,  # decode_input_len_sum
            266.5,  # decode_input_len_mean
            231.13,  # decode_input_len_std
            2,  # tp_degree
            360,  # freq_mhz
        ]

        t0 = time.time()
        try:
            r_prefill = predict_with_model(sample_prefill, model, UNIFIED_FEATURE_COLS)
            print(f'Sample prefill prediction: {r_prefill:.6f}s')
        except Exception as e:
            print(f'Prefill sample predict failed: {e}')
        print(f'Prefill inference time: {time.time()-t0:.6f}s')

        t0 = time.time()
        try:
            r_decode = predict_with_model(sample_decode, model, UNIFIED_FEATURE_COLS)
            print(f'Sample decode prediction: {r_decode:.6f}s')
        except Exception as e:
            print(f'Decode sample predict failed: {e}')
        print(f'Decode inference time: {time.time()-t0:.6f}s')

        t0 = time.time()
        try:
            r_mixed = predict_with_model(sample_mixed, model, UNIFIED_FEATURE_COLS)
            print(f'Sample mixed prediction: {r_mixed:.6f}s')
        except Exception as e:
            print(f'Mixed sample predict failed: {e}')
        print(f'Mixed inference time: {time.time()-t0:.6f}s')

    print('\nDone.')


if __name__ == '__main__':
    main()
