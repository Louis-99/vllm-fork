#!/usr/bin/env python3
"""
Combined script to train/save latency models for prefill and decode.

Features:
- Trains prefill and/or decode RandomForest pipelines and saves as .joblib
- Optionally converts to ONNX and saves .onnx files
- Separate FEATURE_COLS, targets and CSV inputs for prefill and decode so they
  can be changed independently.
- Simple CLI to enable/disable each model and tune paths.
"""
import os
import time
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import onnxruntime as ort

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
    return df


def build_pipeline(role):
    """Build a preprocessing + estimator pipeline.

    Parameters:
    - role: 'prefill' or 'decode' 
    """
    # Both scripts used OneHotEncoder for `model` and StandardScaler for remainder
    cat_cols = ['model']
    pre = ColumnTransformer([
        ('ohe_model', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder=StandardScaler())

    # Choose defaults by role but allow overrides
    if role == 'prefill':
        est = RandomForestRegressor(n_estimators=6, random_state=42, n_jobs=-1, max_depth=20)
    elif role == 'decode':
        est = RandomForestRegressor(n_estimators=6, random_state=42, n_jobs=-1, max_depth=20)

    return Pipeline([('pre', pre), ('est', est)])


def train_and_save(df, feature_cols, target_col, out_name, model_dir, convert_onnx=True, role: str = 'default'):
    results = {}
    df_t = df.dropna(subset=[target_col])
    if df_t.shape[0] < 10:
        print(f"not enough rows to train {target_col} (found {df_t.shape[0]}). skipping.")
        return None

    X = df_t[feature_cols]
    y = df_t[target_col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline(role=role)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = (abs(y_test - y_pred) / y_test).mean() * 100

    os.makedirs(model_dir, exist_ok=True)
    joblib_path = os.path.join(model_dir, out_name)
    joblib.dump(pipe, joblib_path)

    onnx_path = joblib_path.replace('.joblib', '.onnx')
    if convert_onnx:
        # build initial types: for convert_sklearn we declare each input column
        initial_type = []
        for name in feature_cols:
            if name == 'model':
                initial_type.append((name, StringTensorType([None, 1])))
            else:
                initial_type.append((name, FloatTensorType([None, 1])))
        try:
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
    }
    print(f"trained {target_col}: saved -> {joblib_path}  MAE={mae:.4f}  MAPE={mape:.4f}  train_rows={len(X_train)}")

    # Save detailed predictions file
    results_df = X_test.copy()
    results_df['ground_truth'] = y_test.values
    results_df['predicted'] = y_pred
    results_df['error'] = results_df['predicted'] - results_df['ground_truth']
    results_df = results_df.sort_values(by='error', key=abs, ascending=False).reset_index(drop=True)
    results_csv_path = os.path.join(model_dir, f"{target_col}_pred_vs_true.csv")
    results_df.to_csv(results_csv_path, index=False)

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
    parser = argparse.ArgumentParser(description='Train prefill and decode latency models')
    parser.add_argument('--no-prefill', action='store_true', help='Disable training prefill model')
    parser.add_argument('--no-decode', action='store_true', help='Disable training decode model')
    parser.add_argument('--model-dir', default=('/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/models_tree/H100'), help='Directory to store trained models')
    args = parser.parse_args()

    CSV_PATH_TP2_D = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/default_log_path/decode_latencies.csv"
    CSV_PATH_TP2_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP2_decode_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/decode_latencies.csv"
    CSV_PATH_TP4_D = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_TP4_D1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP4_decode_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"
    CSV_PATH_TP4_D2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov19_TP4_decode_latency_largebatch/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/decode_latencies.csv"


    CSV_PATH_TP2_P = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/default_log_path/prefill_latencies.csv"
    CSV_PATH_TP2_P1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP2_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/prefill_latencies.csv"
    CSV_PATH_TP2_P2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov13_TP24_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP2Prefill_1xTP2/prefill_latencies.csv"
    CSV_PATH_TP4_P = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"
    CSV_PATH_TP4_P1 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov12_TP4_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"
    CSV_PATH_TP4_P2 = "/export2/obasit/ClusterLevelServing/vllm_logs/latency_profiler_logs/profiler_logs_Nov13_TP24_prefill_latency_smallfreq/meta-llama/Llama-3.3-70B-Instruct/H100/1xTP4Prefill_1xTP4/prefill_latencies.csv"

    MODEL_DIR = os.path.join(args.model_dir)

    # Prefill configuration
    PREFILL_FEATURE_COLS = [
        'model', 'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std', 'tp_degree', 'freq_mhz'
    ]
    PREFILL_TARGET = 'latency_prefill_s'
    PREFILL_OUT = 'prefill_model.joblib'

    # Decode configuration
    DECODE_FEATURE_COLS = [
        'model', 'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std', 'tp_degree', 'freq_mhz'
    ]
    DECODE_TARGET = 'latency_decode_s'
    DECODE_OUT = 'decode_model.joblib'

    # Train prefill
    if not args.no_prefill:
        print('\n=== Preprocessing prefill CSVs and training prefill model ===')
        df_p2 = load_and_prepare(CSV_PATH_TP2_P, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p21 = load_and_prepare(CSV_PATH_TP2_P1, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p22 = load_and_prepare(CSV_PATH_TP2_P2, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        print(f"prefill samples {len(df_p2.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_P}")
        print(f"prefill samples {len(df_p21.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_P1}")
        print(f"prefill samples {len(df_p22.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_P2}")
        df_p4 = load_and_prepare(CSV_PATH_TP4_P, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p41 = load_and_prepare(CSV_PATH_TP4_P1, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        df_p42 = load_and_prepare(CSV_PATH_TP4_P2, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=PREFILL_FEATURE_COLS + [PREFILL_TARGET])
        print(f"prefill samples {len(df_p4.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_P}")
        print(f"prefill samples {len(df_p41.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_P1}")
        print(f"prefill samples {len(df_p42.get(PREFILL_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_P2}")
        df_prefill = pd.concat([df_p2, df_p21, df_p22, df_p4, df_p41, df_p42], ignore_index=True)
        df_prefill.to_csv(os.path.join(MODEL_DIR, 'prefill_cleaned.csv'), index=False)
        stats_prefill = train_and_save(df_prefill, PREFILL_FEATURE_COLS, PREFILL_TARGET, PREFILL_OUT, MODEL_DIR, convert_onnx=True, role='prefill')
    else:
        stats_prefill = None

    # Train decode
    if not args.no_decode:
        print('\n=== Preprocessing decode CSVs and training decode model ===')
        df_d2 = load_and_prepare(CSV_PATH_TP2_D, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        df_d21 = load_and_prepare(CSV_PATH_TP2_D1, 'Llama-3.3-70B-Instruct', tp=2, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        print(f"Decode samples {len(df_d2.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_D}")
        print(f"Decode samples {len(df_d21.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP2_D1}")
        df_d4 = load_and_prepare(CSV_PATH_TP4_D, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        df_d41 = load_and_prepare(CSV_PATH_TP4_D1, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        df_d42 = load_and_prepare(CSV_PATH_TP4_D2, 'Llama-3.3-70B-Instruct', tp=4, numeric_cols=DECODE_FEATURE_COLS + [DECODE_TARGET])
        print(f"Decode samples {len(df_d4.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_D}")
        print(f"Decode samples {len(df_d41.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_D1}")
        print(f"Decode samples {len(df_d42.get(DECODE_TARGET, pd.Series()).dropna())} from {CSV_PATH_TP4_D2}")
        df_decode = pd.concat([df_d2, df_d21, df_d4, df_d41, df_d42], ignore_index=True)
        df_decode.to_csv(os.path.join(MODEL_DIR, 'decode_cleaned.csv'), index=False)
        stats_decode = train_and_save(df_decode, DECODE_FEATURE_COLS, DECODE_TARGET, DECODE_OUT, MODEL_DIR, convert_onnx=True, role='decode')
    else:
        stats_decode = None

    # Load models for a quick sample inference demonstration
    print('\n=== Quick sample inference (if ONNX created or joblib available) ===')
    # Try ONNX first if requested, otherwise joblib
    if stats_prefill and stats_prefill.get('onnx_path'):
        pre_model = load_onnx(stats_prefill['onnx_path'])
    else:
        pre_joblib = os.path.join(MODEL_DIR, PREFILL_OUT)
        pre_model = joblib.load(pre_joblib) if os.path.exists(pre_joblib) else None

    if stats_decode and stats_decode.get('onnx_path'):
        dec_model = load_onnx(stats_decode['onnx_path'])
    else:
        dec_joblib = os.path.join(MODEL_DIR, DECODE_OUT)
        dec_model = joblib.load(dec_joblib) if os.path.exists(dec_joblib) else None

    # Example inputs
    sample_prefill = [
        'Llama-3.3-70B-Instruct',
        6,  # batch_size
        1472,  # input_len_sum
        245.33,  # input_len_mean
        247.41,  # input_len_std
        4,  # tp_degree
        360,  # freq_mhz
    ]

    sample_decode = [
        'Llama-3.3-70B-Instruct',
        274,  # batch_size
        93821,  # input_len_sum
        171.2,  # input_len_mean
        202.19,  # input_len_std
        4,  # tp_degree
        360,  # freq_mhz
    ]

    if pre_model is not None:
        t0 = time.time()
        try:
            r = predict_with_model(sample_prefill, pre_model, PREFILL_FEATURE_COLS)
            print('Sample prefill prediction:', r)
        except Exception as e:
            print('prefill sample predict failed:', e)
        print(f'prefill inference time: {time.time()-t0:.6f}s')

    if dec_model is not None:
        t0 = time.time()
        try:
            r = predict_with_model(sample_decode, dec_model, DECODE_FEATURE_COLS)
            print('Sample decode prediction:', r)
        except Exception as e:
            print('decode sample predict failed:', e)
        print(f'decode inference time: {time.time()-t0:.6f}s')

    print('\nDone.')


if __name__ == '__main__':
    main()
