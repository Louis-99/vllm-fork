import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import time

import joblib
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

def load_and_prepare(path, model, tp=None):
    df = pd.read_csv(path)
    if tp is not None:
        df['tp_degree'] = tp
    df = df[df["freq_mhz"] > 1250]
    # ensure numeric columns are numeric
    for c in ["batch_size", "input_len_sum", "input_len_mean", "input_len_std", "latency_prefill_s", "latency_decode_s", "tp_degree", "freq_mhz"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].dtype == float:
                df[c] = df[c].apply(lambda x: int(x * 1000) / 1000 if pd.notnull(x) else x)
    
    df["model"] = model

    # ensure monotonic increase of both input_len_mean and latency columns per batch size
    df = df.sort_values(by=["model", 'batch_size', 'input_len_mean', 'input_len_std']).reset_index(drop=True)
    for bs in df['batch_size'].unique():
        mask = df['batch_size'] == bs
        df_bs = df[mask]
        # Ensure monotonicity of latency_prefill_s
        last_val = None
        middle_val = None
        for i, row in df_bs.iterrows():
            val = row["latency_prefill_s"]
            if pd.isna(val):
                last_val = middle_val
                middle_val = None
                continue
            if last_val is not None and middle_val is not None and (val < middle_val or val < last_val) and (abs(val - middle_val) < abs(val - last_val)):
                df_bs.at[i, "latency_prefill_s"] = pd.NA
            elif last_val is not None and middle_val is not None and (val < middle_val or val < last_val) and (abs(val - middle_val) > abs(val - last_val)):
                df_bs.at[i-1, "latency_prefill_s"] = pd.NA
            last_val = middle_val
            middle_val = val
        # Ensure monotonicity of latency_prefill_s
        last_val = None
        middle_val = None
        for i, row in df_bs.iterrows():
            val = row["latency_decode_s"]
            if pd.isna(val):
                last_val = middle_val
                middle_val = None
                continue
            if last_val is not None and middle_val is not None and (val < middle_val or val < last_val) and (abs(val - middle_val) < abs(val - last_val)):
                df_bs.at[i, "latency_decode_s"] = pd.NA
            elif last_val is not None and middle_val is not None and (val < middle_val or val < last_val) and (abs(val - middle_val) > abs(val - last_val)):
                df_bs.at[i-1, "latency_decode_s"] = pd.NA
            last_val = middle_val
            middle_val = val
        # put back
        df.loc[mask, :] = df_bs

    return df

def build_pipeline(target_col):
    # onehot encode model, scale numeric features
    cat_cols = ["model"]
    num_cols = ["batch_size", "input_len_sum", "input_len_mean", "input_len_std", "tp_degree", "freq_mhz"]
    pre = ColumnTransformer([
        ("ohe_model", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="passthrough")
    if target_col == "latency_decode_s":
        est = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        est = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("est", est)])
    # return Pipeline([("est", est)])


def train_and_save(df):
    # Train decode model (rows with latency_decode_s) and prefill model (rows with latency_prefill_s)
    results = {}
    for target_col, out_name in [("latency_decode_s", "decode_model.joblib"), ("latency_prefill_s", "prefill_model.joblib")]:
        df_t = df.dropna(subset=[target_col])
        if df_t.shape[0] < 10:
            print(f"not enough rows to train {target_col} (found {df_t.shape[0]}). skipping.")
            continue
        X = df_t[FEATURE_COLS]
        y = df_t[target_col].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = build_pipeline(target_col)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = (abs(y_test - y_pred) / y_test).mean() * 100
        r2 = r2_score(y_test, y_pred)
        path = os.path.join(MODEL_DIR, out_name)

        joblib.dump(pipe, path)
        # Save as ONNX model
        onnx_path = os.path.join(MODEL_DIR, out_name.replace(".joblib", ".onnx"))
        from skl2onnx.common.data_types import StringTensorType
        initial_type = []
        for name in FEATURE_COLS:
            if name == "model":
                initial_type.append((name, StringTensorType([None, 1])))
            else:
                initial_type.append((name, FloatTensorType([None, 1])))
        onnx_model = convert_sklearn(pipe, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        results[target_col] = {"model_path": path, "mae": mae, "mape": mape, "r2": r2, "n_train": len(X_train)}
        print(f"trained {target_col}: saved -> {path}  MAE={mae:.4f}  MAPE={mape:.4f}  R2={r2:.4f}  train_rows={len(X_train)}")
        # Save ground truth, predicted values, and input features to CSV
        results_df = X_test.copy()
        results_df["ground_truth"] = y_test.values
        results_df["predicted"] = y_pred
        results_df["error"] = results_df["predicted"] - results_df["ground_truth"]
        results_csv_path = os.path.join(MODEL_DIR, f"{target_col}_pred_vs_true.csv")
        results_df.to_csv(results_csv_path, index=False)
    return results

def load_models():
    dec = None
    pre = None
    dec_path = os.path.join(MODEL_DIR, "decode_model.onnx")
    pre_path = os.path.join(MODEL_DIR, "prefill_model.onnx")
    if os.path.exists(dec_path):
        dec = ort.InferenceSession(dec_path)
    if os.path.exists(pre_path):
        pre = ort.InferenceSession(pre_path)
    return dec, pre

def predict_latencies(inputs, decode_model, prefill_model):
    """
    inputs: single row list/tuple in order [model,batch_size,total_tokens,tp_degree,freq_mhz]
            or pandas.DataFrame with columns FEATURE_COLS
    returns: dict with keys decode_time and prefill_time (None if corresponding model missing)
    """
    if isinstance(inputs, (list, tuple)):
        X = pd.DataFrame([inputs], columns=FEATURE_COLS)
    elif isinstance(inputs, pd.DataFrame):
        X = inputs.copy()
    else:
        raise ValueError("inputs must be list/tuple or pandas.DataFrame")
    # ensure numeric dtypes
    for c in ["batch_size", "input_len_sum", "input_len_mean", "input_len_std", "tp_degree", "freq_mhz"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    out = {}
    out["decode_time"] = None
    out["prefill_time"] = None
    # ONNX model expects a single input (e.g., 'input') with the full feature array
    input_name = decode_model.get_inputs()[0].name
    # One-hot encode the "model" column (which is a float here) for ONNX input
    X_enc = X.copy()
    # Convert model to string for one-hot encoding
    X_enc["model"] = X_enc["model"].astype(str)
    # Prepare input_feed as a dict of {feature_name: np.array([[value]])}
    input_feed = {}
    for col in FEATURE_COLS:
        val = X_enc[col].values.reshape(-1, 1)
        if X_enc[col].dtype == object or col == "model":
            input_feed[col] = val.astype(str)
        else:
            input_feed[col] = val.astype("float32")
    out["decode_time"] = float(decode_model.run(None, input_feed)[0][0])

    out["prefill_time"] = float(prefill_model.run(None, input_feed)[0][0])
    return out

if __name__ == '__main__':
    CSV_PATH_TP2_D = "/export2/obasit/ClusterLevelServing/vllm_logs/profiler_logs/google/gemma-2-27b-it/A100_40GB/batch_controlled_runs/decode/1xTP2Prefill_1xTP2/metrics.csv"
    CSV_PATH_TP2_P = "/export2/obasit/ClusterLevelServing/vllm_logs/profiler_logs/google/gemma-2-27b-it/A100_40GB/batch_controlled_runs/prefill/1xTP2Prefill_1xTP2/metrics.csv"
    # CSV_PATH_TP4 = "/export2/obasit/ClusterLevelServing/vllm_logs/profiler_logs/google/gemma-2-27b-it/A100_40GB/1xTP4Prefill_1xTP4/metrics.csv"
    MODEL_DIR = os.path.join("/export2/obasit/ClusterLevelServing/vllm_logs/profiler_logs/google/gemma-2-27b-it/A100_40GB/batch_controlled_runs", "models_tree")
    os.makedirs(MODEL_DIR, exist_ok=True)

    FEATURE_COLS = ["model", "batch_size", "input_len_sum", "input_len_mean", "input_len_std", "tp_degree", "freq_mhz"]

    df_tp2_P = load_and_prepare(CSV_PATH_TP2_P, "gemma-2-27b-it", 2)
    print(f"prefill samples {len(df_tp2_P['latency_prefill_s'].dropna())}, Decode samples {len(df_tp2_P['latency_decode_s'].dropna())} from {CSV_PATH_TP2_P}")
    df_tp2_D = load_and_prepare(CSV_PATH_TP2_D, "gemma-2-27b-it", 2)
    print(f"prefill samples {len(df_tp2_D['latency_prefill_s'].dropna())}, Decode samples {len(df_tp2_D['latency_decode_s'].dropna())} from {CSV_PATH_TP2_D}")
    df = pd.concat([df_tp2_P, df_tp2_D], ignore_index=True)

    df[["model", "batch_size", "input_len_sum", "input_len_mean", "input_len_std", "latency_prefill_s", "tp_degree", "freq_mhz"]].dropna().to_csv("prefill_cleaned.csv", index=False)

    stats = train_and_save(df)
    dec_model, pre_model = load_models()
    # example prediction
    example = ["gemma-2-27b-it", 1, 100, 100, 0, 2, 1410]  # model,batch_size,total_tokens,tp_degree,freq_mhz
    start = time.time()

    pred = predict_latencies(example, decode_model=dec_model, prefill_model=pre_model)
    elapsed = time.time() - start
    print(f"Prediction took {elapsed:.6f} seconds")
    print("example input:", example)
    print("predicted latencies:", pred)