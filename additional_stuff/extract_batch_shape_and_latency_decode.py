import itertools
import json
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parse_vllm_output import load_logs_prefill_decode_power_logs


@dataclass
class LatencyAndShape:
    batch_size: int
    input_len_sum: int
    input_len_mean: int
    input_len_std: float
    latency_prefill_s: float
    latency_decode_s: float
    power_w: float
    freq_mhz: float

# get power for this window
def compute_power_w(df_power_sub):
    energy_j = 0.0
    freqs = []
    gpu_power_cols = [col for col in df_power_sub.columns if col.startswith("GPU_") and col.endswith("_power_w")]
    gpu_freq_cols = [col for col in df_power_sub.columns if col.startswith("GPU_") and col.endswith("_freq_mhz")]
    for col in gpu_power_cols:
        energy_j += np.trapezoid(df_power_sub[col], df_power_sub['Timestamp'])
    for col in gpu_freq_cols:
        freqs.append(np.mean(df_power_sub[col]))
    duration = df_power_sub['Timestamp'].max() - df_power_sub['Timestamp'].min()
    if duration > 0.2:
        return (energy_j / duration, np.mean(freqs))  # return average power in W and average freq in MHz
    else:
        return np.nan, np.nan

def calc_stats(expr_dir: Path) -> LatencyAndShape:
    logs_dict = load_logs(expr_dir)

    stats_list = []
    for k, v in logs_dict.items():
        decode, prefill, power = v
        stats_l = calc_stats_single_instance_decode(decode, power)
        stats_list.extend(stats_l)

    return stats_list

def calc_stats_single_instance_decode(df_perf_metric_decode_steady: pd.DataFrame, df_power: pd.DataFrame) -> list[LatencyAndShape]:
    df_perf_metric_decode_steady['request_ids_iter_tbt_evald'] = df_perf_metric_decode_steady['request_ids_iter_tbt'].apply(eval)
    df_perf_metric_decode_steady['inter_token_latencies_iter_evald'] = df_perf_metric_decode_steady['inter_token_latencies_iter'].apply(eval)
    df_perf_metric_decode_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_decode_steady['num_prompt_tokens_reqs'].apply(eval)

    num_computed_dict = {}
    #first get num_computed_tokens for each req in each row
    for row in df_perf_metric_decode_steady.itertuples():
        if len(row.request_ids_iter_tbt_evald) == 0:
            continue
        for ID, in_len in zip(row.request_ids_iter_tbt_evald, row.num_prompt_tokens_reqs_evald):
            if ID not in num_computed_dict:
                num_computed_dict[ID] = in_len + 1
            else:
                num_computed_dict[ID] = num_computed_dict[ID] + 1
    df_perf_metric_decode_steady['num_computed_tokens_reqs_evald'] = df_perf_metric_decode_steady['request_ids_iter_tbt_evald'].apply(
        lambda reqs: [num_computed_dict[ID] for ID in reqs] if reqs else []
    )

    lat_and_shape_list = []

    start_time = df_perf_metric_decode_steady['now'].min()
    end_time = df_perf_metric_decode_steady['now'].max()
    df_power = df_power[(df_power['Timestamp'] >= start_time) & (df_power['Timestamp'] <= end_time)]
    
    window = 0.5  # 0.5 second windows
    start_subsection = df_power['Timestamp'].min()
    end_subsection = start_subsection + window  # 0.5 second windows
    while end_subsection <= df_power['Timestamp'].max():
        df_power_sub = df_power[(df_power['Timestamp'] >= start_subsection) & (df_power['Timestamp'] < end_subsection)]
        df_decode_sub = df_perf_metric_decode_steady[(df_perf_metric_decode_steady['now'] >= start_subsection) & (df_perf_metric_decode_steady['now'] < end_subsection)]
        if df_decode_sub["now"].max() - df_decode_sub["now"].min() < window-0.1:  # skip sections with very little decode activity
            start_subsection = end_subsection
            end_subsection = start_subsection + window
            continue

        power_w, gpu_freq = compute_power_w(df_power_sub)
        if np.isnan(power_w):
            start_subsection = end_subsection
            end_subsection = start_subsection + window
            continue

        # get batch shapes and latencies for this window
        batch_size = df_decode_sub['num_running_reqs'].median()
        input_lens = df_decode_sub['num_computed_tokens_reqs_evald'].dropna().tolist()
        input_lens = [lens for lens in input_lens if len(lens) > 0]
        if len(input_lens) == 0:
            start_subsection = end_subsection
            end_subsection = start_subsection + window
            continue
        input_len_sum = int(np.median([sum(lens) for lens in input_lens]))
        input_len_mean = float(np.median([np.mean(lens) for lens in input_lens]))
        input_len_std = float(np.std([np.mean(lens) for lens in input_lens]))
        latencies = df_decode_sub['inter_token_latencies_iter_evald'].dropna().tolist()
        latency_decode_s = np.median([np.median(lats) for lats in latencies])

        lat_and_shape_list.append(LatencyAndShape(
            batch_size=batch_size,
            input_len_sum=input_len_sum,
            input_len_mean=input_len_mean,
            input_len_std=input_len_std,
            latency_prefill_s=np.nan,  # no prefill in decode logs
            latency_decode_s=latency_decode_s,
            power_w=power_w,
            freq_mhz=gpu_freq
        ))

        start_subsection = end_subsection
        end_subsection = start_subsection + window
        

    return lat_and_shape_list


def percentile_or_nan(a, q):
    if len(a) > 0:
        return np.percentile(a, q)
    else:
        return np.nan

def load_logs(expr_dir: Path) -> dict:
    logs = {}
    for subfolder in sorted(expr_dir.iterdir()):
        if subfolder.is_dir():
            try:
                (df_perf_metric_decode, df_perf_metric_prefill, df_power) = load_logs_prefill_decode_power_logs(subfolder)
                if (not df_perf_metric_decode.empty and not df_power.empty):
                    logs[subfolder.name] = (df_perf_metric_decode, df_perf_metric_prefill, df_power)
            except Exception as e:
                print(f"Skipping {subfolder} due to error: {e}")
    return logs


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        expr_root = Path(sys.argv[1])
    else:
        expr_root = Path('/export2/obasit/ClusterLevelServing/vllm_logs') / \
            'test_logs' 

    # structure of log files should be like this:
    # |-> expr_root
    # |  |-> disag_1P1D_test
    # |  |  |-> prefill_1
    # |  |  |  |-> engine_*.csv
    # |  |  |  |-> power_log_*.csv
    # |  |  |-> decode_1
    # |  |  |  |-> engine_*.csv
    # |  |  |  |-> power_log_*.csv
    # |  |
    # |  |-> disag_2P1D_test
    # |  |  |-> prefill_1
    # |  |  |  |-> engine_*.csv
    # |  |  |  |-> power_log_*.csv
    # |  |  |-> prefill_2
    # |  |  |  |-> engine_*.csv
    # |  |  |  |-> power_log_*.csv
    # |  |  |-> decode_1
    # |  |  |  |-> engine_*.csv
    # |  |  |  |-> power_log_*.csv
    # ...

    df_stats = []
    stats_list = []
    for expr_dir in sorted(expr_root.glob('*')):
        if not expr_dir.is_dir():
            continue
        if not any(child.is_dir() for child in expr_dir.iterdir()):
            continue
        print('expr_dir: ', expr_dir)
        stats_list.append(calc_stats(expr_dir))
    stats_list = list(itertools.chain.from_iterable(stats_list))

    # merge prefill and decode for same batch shape, input lengths
    merged_stats_decode = {}
    merged_stats_power = {}
    for stats in stats_list:
        freq = int(round(stats.freq_mhz / 10.0) * 10) if not np.isnan(stats.freq_mhz) else np.nan
        key = (stats.batch_size, stats.input_len_sum, stats.input_len_mean, stats.input_len_std, freq)
        if key not in merged_stats_decode:
            merged_stats_decode[key] = []
        merged_stats_decode[key].append(stats.latency_decode_s)
        if key not in merged_stats_power:
            merged_stats_power[key] = []
        merged_stats_power[key].append(stats.power_w)

    merged_stats_rows = []
    for key in set(merged_stats_power.keys()).union(set(merged_stats_decode.keys())):
        power = merged_stats_power.get(key, [])
        latencies_decode = merged_stats_decode.get(key, [])
        merged_stats_rows.append({
            'batch_size': key[0],
            'input_len_sum': key[1],
            'input_len_mean': key[2],
            'input_len_std': key[3],
            'latency_prefill_s': np.nan,
            'latency_decode_s': np.median(latencies_decode) if latencies_decode else np.nan,
            'power_w': np.median(power) if power else np.nan,
            'freq_mhz': key[4]
        })
    merged_stats_df = pd.DataFrame(merged_stats_rows, columns=[
        'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std',
        'latency_prefill_s', 'latency_decode_s', 'power_w', 'freq_mhz'])

    df_stats = pd.DataFrame(merged_stats_df)
    df_stats = df_stats.sort_values(by=['batch_size', 'input_len_mean']).reset_index(drop=True)
    df_stats.to_csv(expr_root / 'metrics.csv', index=False)