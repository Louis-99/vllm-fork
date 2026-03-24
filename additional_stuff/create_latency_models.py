#!/usr/bin/env python3
"""
Combined script to train/save latency models for prefill and decode.

Features:
- Trains prefill and/or decode LightGBM models and saves as native .txt
- Separate FEATURE_COLS, targets and CSV inputs for prefill and decode so they
  can be changed independently.
- Simple CLI to enable/disable each model and tune paths.
"""
import os
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor, Booster

def load_and_prepare(path, model_name, tp=None, numeric_cols=None):
    df = pd.read_csv(path)
    if tp is not None:
        df['tp_degree'] = tp
    if numeric_cols is None:
        numeric_cols = []
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols)
    return df

def filter_inputs(df):
    # for rows with same batch size, input_len_sum, input_len_mean, input len std, tp_degree keep only median latency
    print("before filtering:", df.shape)
    group_cols = ['batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std', 'tp_degree', 'freq_mhz']
    df_filtered = df.groupby(group_cols).median().reset_index()
    print("after filtering:", df_filtered.shape)
    return df_filtered


def build_model(role, **kwargs):
    """Build a LightGBM estimator with optional parameter overrides.

    Parameters:
    - role: 'prefill' or 'decode'
    - **kwargs: hyperparameter overrides
    """
    # Default hyperparameters by role
    if role == 'prefill':
        params = {
            'objective': 'regression',
            'linear_tree': True,
            'random_state': 42,
            'monotonic_cst': [0, 0, 0, 0, 0, 0],
            'n_jobs': 1,
            'verbosity': -1,
            'device_type': 'gpu',
        }
    elif role == 'decode':
        params = {
            'objective': 'regression',
            'linear_tree': True,
            'random_state': 42,
            'monotonic_cst': [0, 0, 0, 0, 0, 0],
            'n_jobs': 1,
            'verbosity': -1,
            'device_type': 'gpu',
        }
    else:
        raise ValueError(f"Unknown role {role}. Expected 'prefill' or 'decode'.")
    
    # Apply overrides
    params.update(kwargs)
    return LGBMRegressor(**params)


def grid_search_lgbm(X_train, y_train, X_test, y_test, role, feature_cols):
    """Grid search over LightGBM hyperparameters using GridSearchCV.
    
    Returns: (best_model, best_params, best_mae, best_mape)
    """
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.08, 0.1, 0.15],
        'linear_lambda': [1e-3],
        'min_child_samples': [30, 40],
        'num_iterations': [300, 400],
        'num_leaves': [70, 80, 90],
        'reg_lambda': [1e-2],
    }
    
    # Create base model
    base_model = build_model(role)
    
    # Perform grid search with 5-fold cross-validation
    print(f"Grid search: testing {len(param_grid['learning_rate']) * len(param_grid['num_leaves']) * len(param_grid['reg_lambda'])} parameter combinations for {role} (5-fold CV)")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=16,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = (abs(y_test - y_pred) / y_test).mean() * 100
    
    print(f"  Best params: {best_params}  MAE={mae:.6f}  MAPE={mape:.2f}%\n")
    return best_model, best_params, mae, mape


def apply_log_transform(X, y, feature_cols):
    """Apply log1p to features and log to target."""
    X_scaled = X.copy()
    feature_cols_to_normalize = [col for col in feature_cols if col != 'tp_degree']
    X_scaled[feature_cols_to_normalize] = np.log1p(X[feature_cols_to_normalize])
    y_log = np.log(y)
    return X_scaled, y_log


def train_and_save(df, feature_cols, target_col, out_name, model_dir, role: str = 'default', use_grid_search=True):
    results = {}
    df_t = df.dropna(subset=[target_col])
    if df_t.shape[0] < 10:
        print(f"not enough rows to train {target_col} (found {df_t.shape[0]}). skipping.")
        return None

    X = df_t[feature_cols].copy()
    print(f"features used for {target_col}:", feature_cols)
    y = df_t[target_col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply log transforms
    X_train_scaled, y_train_log = apply_log_transform(X_train, y_train, feature_cols)
    X_test_scaled, y_test_log = apply_log_transform(X_test, y_test, feature_cols)
    
    # Train model with grid search
    if use_grid_search:
        model, best_params, _, _ = grid_search_lgbm(X_train_scaled, y_train_log, X_test_scaled, y_test_log, role, feature_cols)
    else:
        model = build_model(role=role)
        model.fit(X_train_scaled, y_train_log)
    
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.exp(y_pred_log)

    mae = mean_absolute_error(y_test, y_pred)
    mape = (abs(y_test - y_pred) / y_test).mean() * 100

    os.makedirs(model_dir, exist_ok=True)
    model_txt_path = os.path.join(model_dir, out_name)
    model.booster_.save_model(model_txt_path)

    results = {
        'model_path': model_txt_path,
        'mae': mae,
        'mape': mape,
        'n_train': len(X_train),
    }
    print(f"trained {target_col}: saved -> {model_txt_path}  MAE={mae:.4f}  MAPE={mape:.4f}  train_rows={len(X_train)}")

    # Save detailed predictions file
    results_df = X_test.copy()
    results_df['ground_truth'] = y_test.values
    results_df['predicted'] = y_pred
    results_df['error'] = results_df['predicted'] - results_df['ground_truth']
    results_df = results_df.sort_values(by='error', key=abs, ascending=False).reset_index(drop=True)
    results_csv_path = os.path.join(model_dir, f"{target_col}_pred_vs_true.csv")
    results_df.to_csv(results_csv_path, index=False)

    return results


def load_lightgbm_txt(path):
    if path and os.path.exists(path):
        return Booster(model_file=path)
    return None


def predict_with_model(inputs, model, feature_cols):
    """
    inputs: list/tuple single-row matching feature_cols or pandas.DataFrame
    model: trained LightGBM model (LGBMRegressor or Booster)
    returns: float prediction
    """
    if isinstance(inputs, (list, tuple)):
        X = pd.DataFrame([inputs], columns=feature_cols)
    elif isinstance(inputs, pd.DataFrame):
        X = inputs.copy()
    else:
        raise ValueError('inputs must be list/tuple or pandas.DataFrame')

    for c in feature_cols:
        if c != 'model':
            # keep model as string
            try:
                X[c] = pd.to_numeric(X[c], errors='coerce')
            except Exception:
                pass

    # LightGBM expects categorical columns to be marked as category dtype.
    print("Input features for prediction:", X.to_dict(orient='records')[0])
    pred = model.predict(X)
    return float(pred[0] if np.ndim(pred) > 0 else pred)


def main():
    parser = argparse.ArgumentParser(description='Train prefill and decode latency models')
    parser.add_argument('--no-prefill', action='store_true', help='Disable training prefill model')
    parser.add_argument('--no-decode', action='store_true', help='Disable training decode model')
    parser.add_argument('--model-dir', default=('/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/models_tree_disag/H100'), help='Directory to store trained models')
    args = parser.parse_args()

    CSV_PATH_TP2_D = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/default_log_path/decode_latencies.csv"
    CSV_PATH_TP2_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP2_decode_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/decode_latencies.csv"
    CSV_PATH_TP4_D = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_TP4_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP4_decode_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_TP4_D2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov19_TP4_decode_latency_largebatch/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_TP4_D3 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov21_TP4_decoder_latency_batchgt512/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_D4 = "/export2/obasit/ClusterLevelServing/vllm_logs/4_nodes_logs_together_ai/kube_results/profiler_logs_0/placeonly_disag/decode_latencies.csv"


    CSV_PATH_TP2_P = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/default_log_path/prefill_latencies.csv"
    CSV_PATH_TP2_P1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP2_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/prefill_latencies.csv"
    CSV_PATH_TP2_P2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov13_TP24_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/prefill_latencies.csv"
    CSV_PATH_TP2_P3 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov25_prefill_chunked2k_profiling/tp2/prefill_latencies.csv"
    CSV_PATH_TP4_P = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"
    CSV_PATH_TP4_P1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP4_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"
    CSV_PATH_TP4_P2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov13_TP24_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"
    CSV_PATH_TP4_P3 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov25_prefill_chunked2k_profiling/tp4/prefill_latencies.csv"
    CSV_PATH_P4 = "/export2/obasit/ClusterLevelServing/vllm_logs/4_nodes_logs_together_ai/kube_results/profiler_logs_0/placeonly_disag/prefill_latencies.csv"

    MODEL_DIR = os.path.join(args.model_dir)

    # Prefill configuration
    PREFILL_FEATURE_COLS = [
        'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std', 'tp_degree', 'freq_mhz'
    ]
    PREFILL_TARGET = 'latency_prefill_s'

    # Decode configuration
    DECODE_FEATURE_COLS = [
        'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std', 'tp_degree', 'freq_mhz'
    ]
    DECODE_TARGET = 'latency_decode_s'

    # Train prefill TP2 and TP4 models
    stats_prefill_tp2 = None
    stats_prefill_tp4 = None
    if not args.no_prefill:
        print('\n=== Preprocessing prefill CSVs and training prefill models (TP2 and TP4) ===')
        # Load all prefill data at once
        df_p2 = load_and_prepare(CSV_PATH_TP2_P, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p21 = load_and_prepare(CSV_PATH_TP2_P1, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p22 = load_and_prepare(CSV_PATH_TP2_P2, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p23 = load_and_prepare(CSV_PATH_TP2_P3, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        print(f"prefill samples {len(df_p2.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_P}")
        print(f"prefill samples {len(df_p21.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_P1}")
        print(f"prefill samples {len(df_p22.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_P2}")
        print(f"prefill samples {len(df_p23.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_P3}")
        df_p4 = load_and_prepare(CSV_PATH_TP4_P, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p41 = load_and_prepare(CSV_PATH_TP4_P1, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p42 = load_and_prepare(CSV_PATH_TP4_P2, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p43 = load_and_prepare(CSV_PATH_TP4_P3, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        print(f"prefill samples {len(df_p4.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_P}")
        print(f"prefill samples {len(df_p41.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_P1}")
        print(f"prefill samples {len(df_p42.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_P2}")
        print(f"prefill samples {len(df_p43.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_P3}")
        df_p = load_and_prepare(CSV_PATH_P4, 'Llama-3.3-70B-Instruct', numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        print(f"prefill samples {len(df_p.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_P4}")
        df_prefill_all = pd.concat([df_p2, df_p21, df_p22, df_p23, df_p4, df_p41, df_p42, df_p43, df_p], ignore_index=True).dropna(subset=[PREFILL_TARGET]+PREFILL_FEATURE_COLS)

        df_prefill_all = filter_inputs(df_prefill_all)

        # Split filtered data by TP degree
        df_prefill_tp2 = df_prefill_all[df_prefill_all['tp_degree'] == 2].copy()
        df_prefill_tp4 = df_prefill_all[df_prefill_all['tp_degree'] == 4].copy()

        df_prefill_tp2.to_csv(os.path.join(MODEL_DIR, 'prefill_tp2_cleaned.csv'), index=False)
        print(f"\nTraining prefill TP2 model...")
        stats_prefill_tp2 = train_and_save(df_prefill_tp2, PREFILL_FEATURE_COLS, PREFILL_TARGET, 'prefill_tp2_model.txt', MODEL_DIR, role='prefill', use_grid_search=True)

        df_prefill_tp4.to_csv(os.path.join(MODEL_DIR, 'prefill_tp4_cleaned.csv'), index=False)
        print(f"\nTraining prefill TP4 model...")
        stats_prefill_tp4 = train_and_save(df_prefill_tp4, PREFILL_FEATURE_COLS, PREFILL_TARGET, 'prefill_tp4_model.txt', MODEL_DIR, role='prefill', use_grid_search=True)

    # Train decode TP2 and TP4 models
    stats_decode_tp2 = None
    stats_decode_tp4 = None
    if not args.no_decode:
        print('\n=== Preprocessing decode CSVs and training decode models (TP2 and TP4) ===')
        # Load all decode data at once
        df_d2 = load_and_prepare(CSV_PATH_TP2_D, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        df_d21 = load_and_prepare(CSV_PATH_TP2_D1, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        print(f"Decode samples {len(df_d2.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_D}")
        print(f"Decode samples {len(df_d21.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_D1}")
        df_d4 = load_and_prepare(CSV_PATH_TP4_D, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        df_d41 = load_and_prepare(CSV_PATH_TP4_D1, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        df_d42 = load_and_prepare(CSV_PATH_TP4_D2, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        df_d43 = load_and_prepare(CSV_PATH_TP4_D3, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        print(f"Decode samples {len(df_d4.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_D}")
        print(f"Decode samples {len(df_d41.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_D1}")
        print(f"Decode samples {len(df_d42.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_D2}")
        print(f"Decode samples {len(df_d43.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_D3}")
        df_d = load_and_prepare(CSV_PATH_D4, 'Llama-3.3-70B-Instruct', numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        print(f"Decode samples {len(df_d.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_D4}")
        df_decode_all = pd.concat([df_d2, df_d21, df_d4, df_d41, df_d42, df_d43, df_d], ignore_index=True).dropna(subset=[DECODE_TARGET]+DECODE_FEATURE_COLS)
        print(f"Total decode samples after concat: {len(df_decode_all)}")
        df_decode_all = filter_inputs(df_decode_all)

        # Split filtered data by TP degree
        df_decode_tp2 = df_decode_all[df_decode_all['tp_degree'] == 2].copy()
        df_decode_tp4 = df_decode_all[df_decode_all['tp_degree'] == 4].copy()

        df_decode_tp2.to_csv(os.path.join(MODEL_DIR, 'decode_tp2_cleaned.csv'), index=False)
        print(f"\nTraining decode TP2 model...")
        stats_decode_tp2 = train_and_save(df_decode_tp2, DECODE_FEATURE_COLS, DECODE_TARGET, 'decode_tp2_model.txt', MODEL_DIR, role='decode', use_grid_search=True)

        df_decode_tp4.to_csv(os.path.join(MODEL_DIR, 'decode_tp4_cleaned.csv'), index=False)
        print(f"\nTraining decode TP4 model...")
        stats_decode_tp4 = train_and_save(df_decode_tp4, DECODE_FEATURE_COLS, DECODE_TARGET, 'decode_tp4_model.txt', MODEL_DIR, role='decode', use_grid_search=True)

    # Load models for a quick sample inference demonstration
    print('\n=== Quick sample inference (LightGBM .txt models) ===')
    pre_model_tp2 = load_lightgbm_txt(os.path.join(MODEL_DIR, 'prefill_tp2_model.txt'))
    pre_model_tp4 = load_lightgbm_txt(os.path.join(MODEL_DIR, 'prefill_tp4_model.txt'))
    dec_model_tp2 = load_lightgbm_txt(os.path.join(MODEL_DIR, 'decode_tp2_model.txt'))
    dec_model_tp4 = load_lightgbm_txt(os.path.join(MODEL_DIR, 'decode_tp4_model.txt'))

    # Example inputs
    pre_sm = np.array([590, 267, 476, 10, 47, 102])
    sample_prefill = [
        np.log1p(len(pre_sm)),  # batch_size
        np.log1p(np.sum(pre_sm)),  # input_len_sum
        np.log1p(np.mean(pre_sm)),  # input_len_mean
        np.log1p(np.std(pre_sm)),  # input_len_std
        2,  # tp_degree
        np.log1p(360),  # freq_mhz
    ]

    dec_samp = np.array([34, 17, 75, 11, 7, 648, 117, 20, 17, 63, 10, 30, 26, 13, 19, 31, 15, 11, 40, 264, 419, 254, 30, 37, 270, 6, 24, 58, 28, 14, 14, 10, 38, 9, 12, 504, 32, 7, 28, 24, 173, 162, 199, 26, 8, 43, 18, 80, 49, 8, 44, 52, 272, 152, 13, 12, 81, 537, 515, 186, 53, 325, 7, 67, 11, 350, 78, 6, 13, 9, 22, 105, 11, 15, 474, 53, 6, 106, 21, 351, 450]) + np.array([115, 104, 104, 99, 98, 93, 92, 87, 87, 87, 84, 83, 83, 81, 80, 79, 77, 76, 75, 74, 73, 69, 69, 67, 65, 62, 61, 60, 59, 57, 56, 56, 55, 53, 53, 53, 53, 51, 48, 48, 46, 45, 44, 44, 43, 43, 42, 41, 41, 40, 40, 38, 34, 31, 30, 30, 28, 28, 27, 25, 25, 25, 24, 23, 23, 22, 15, 15, 15, 7, 6, 5, 5, 4, 3, 3, 3, 3, 2, 2, 1])
    sample_decode = [
        np.log1p(len(dec_samp)),  # batch_size
        np.log1p(np.sum(dec_samp)),  # input_len_sum
        np.log1p(np.mean(dec_samp)),  # input_len_mean
        np.log1p(np.std(dec_samp)),  # input_len_std
        4,  # tp_degree
        np.log1p(1830),  # freq_mhz
    ]

    if pre_model_tp2 is not None:
        t0 = time.time()
        try:
            r = predict_with_model(sample_prefill, pre_model_tp2, PREFILL_FEATURE_COLS)
            print('Sample prefill TP2 prediction:', r)
        except Exception as e:
            print('prefill TP2 sample predict failed:', e)
        print(f'prefill TP2 inference time: {time.time()-t0:.6f}s')

    if pre_model_tp4 is not None:
        t0 = time.time()
        try:
            r = predict_with_model(sample_prefill, pre_model_tp4, PREFILL_FEATURE_COLS)
            print('Sample prefill TP4 prediction:', r)
        except Exception as e:
            print('prefill TP4 sample predict failed:', e)
        print(f'prefill TP4 inference time: {time.time()-t0:.6f}s')

    if dec_model_tp2 is not None:
        t0 = time.time()
        try:
            r = predict_with_model(sample_decode, dec_model_tp2, DECODE_FEATURE_COLS)
            print('Sample decode TP2 prediction:', r)
        except Exception as e:
            print('decode TP2 sample predict failed:', e)
        print(f'decode TP2 inference time: {time.time()-t0:.6f}s')

    if dec_model_tp4 is not None:
        t0 = time.time()
        try:
            r = predict_with_model(sample_decode, dec_model_tp4, DECODE_FEATURE_COLS)
            print('Sample decode TP4 prediction:', r)
        except Exception as e:
            print('decode TP4 sample predict failed:', e)
        print(f'decode TP4 inference time: {time.time()-t0:.6f}s')

    print('\nDone.')


if __name__ == '__main__':
    main()
