import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from skl2onnx.common.data_types import StringTensorType
from sklearn import tree

def load_and_prepare(path, model, tp=None):
    df = pd.read_csv(path)
    if tp is not None:
        df['tp_degree'] = tp
    df = df[df["freq_mhz"] > 1250]
    for c in ["batch_size", "input_len_sum", "input_len_mean", "input_len_std", "latency_prefill_s", "tp_degree", "freq_mhz", "since_last_batch_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["model"] = model
    return df

def build_pipeline(target_col):
    cat_cols = ["model"]
    num_cols = ["batch_size", "input_len_sum", "input_len_mean", "input_len_std", "tp_degree", "freq_mhz", "since_last_batch_s"]
    pre = ColumnTransformer([
        ("ohe_model", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder=StandardScaler())
    est = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("est", est)])

def train_and_save(df):
    results = {}
    for target_col, out_name in [("latency_prefill_s", "prefill_model.joblib")]:
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

        path = os.path.join(MODEL_DIR, out_name)
        joblib.dump(pipe, path)

        onnx_path = os.path.join(MODEL_DIR, out_name.replace(".joblib", ".onnx"))
        initial_type = []
        for name in FEATURE_COLS:
            if name == "model":
                initial_type.append((name, StringTensorType([None, 1])))
            else:
                initial_type.append((name, FloatTensorType([None, 1])))
        onnx_model = convert_sklearn(pipe, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        results[target_col] = {"model_path": path, "mae": mae, "mape": mape, "n_train": len(X_train)}
        print(f"trained {target_col}: saved -> {path}  MAE={mae:.4f}  MAPE={mape:.4f}  train_rows={len(X_train)}")

        results_df = X_test.copy()
        results_df["ground_truth"] = y_test.values
        results_df["predicted"] = y_pred
        results_df["error"] = results_df["predicted"] - results_df["ground_truth"]
        results_df = results_df.sort_values(by="error", key=abs, ascending=False).reset_index(drop=True)
        results_csv_path = os.path.join(MODEL_DIR, f"{target_col}_pred_vs_true.csv")
        results_df.to_csv(results_csv_path, index=False)

        

        # Plot and save the first tree in the RandomForest
        estimator = pipe.named_steps["est"].estimators_[0]
        fig, ax = plt.subplots(figsize=(20, 10))
        tree.plot_tree(
            estimator,
            feature_names=pipe.named_steps["pre"].get_feature_names_out(FEATURE_COLS),
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=3
        )
        plot_path = os.path.join(MODEL_DIR, f"{target_col}_tree_plot.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
    return results

def load_prefill_model():
    pre = None
    pre_path = os.path.join(MODEL_DIR, "prefill_model.onnx")
    if os.path.exists(pre_path):
        pre = ort.InferenceSession(pre_path)
    return pre

def predict_prefill_latency(inputs, prefill_model):
    """
    inputs: single row list/tuple in order [model,batch_size,total_tokens,tp_degree,freq_mhz]
            or pandas.DataFrame with columns FEATURE_COLS
    returns: dict with key prefill_time (None if model missing)
    """
    if isinstance(inputs, (list, tuple)):
        X = pd.DataFrame([inputs], columns=FEATURE_COLS)
    elif isinstance(inputs, pd.DataFrame):
        X = inputs.copy()
    else:
        raise ValueError("inputs must be list/tuple or pandas.DataFrame")
    for c in ["batch_size", "input_len_sum", "input_len_mean", "input_len_std", "tp_degree", "freq_mhz", "since_last_batch_s"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    out = {}
    out["prefill_time"] = None
    input_name = prefill_model.get_inputs()[0].name
    X_enc = X.copy()
    X_enc["model"] = X_enc["model"].astype(str)
    input_feed = {}
    for col in FEATURE_COLS:
        val = X_enc[col].values.reshape(-1, 1)
        if X_enc[col].dtype == object or col == "model":
            input_feed[col] = val.astype(str)
        else:
            input_feed[col] = val.astype("float32")
    out["prefill_time"] = float(prefill_model.run(None, input_feed)[0][0].item())
    return out

if __name__ == '__main__':
    CSV_PATH_TP2_P = "/export2/obasit/ClusterLevelServing/vllm_logs/profiler_logs/google/gemma-2-27b-it/A100_40GB/batch_controlled_runs/prefill/1xTP2Prefill_1xTP2/metrics.csv"
    MODEL_DIR = os.path.join("/export2/obasit/ClusterLevelServing/vllm_logs/profiler_logs/google/gemma-2-27b-it/A100_40GB/batch_controlled_runs", "models_tree")
    os.makedirs(MODEL_DIR, exist_ok=True)

    FEATURE_COLS = ["model", "batch_size", "input_len_sum", "input_len_mean", "input_len_std", "tp_degree", "freq_mhz", "since_last_batch_s"]

    df_tp2_P = load_and_prepare(CSV_PATH_TP2_P, "gemma-2-27b-it", 2)
    print(f"prefill samples {len(df_tp2_P['latency_prefill_s'].dropna())} from {CSV_PATH_TP2_P}")
    df = df_tp2_P
    df[["model", "batch_size", "input_len_sum", "input_len_mean", "input_len_std", "latency_prefill_s", "tp_degree", "freq_mhz", "since_last_batch_s"]].dropna().to_csv("prefill_cleaned.csv", index=False)

    stats = train_and_save(df)
    pre_model = load_prefill_model()

