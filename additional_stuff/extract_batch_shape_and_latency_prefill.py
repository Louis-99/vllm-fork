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
from extract_batch_shape_and_latency_decode import LatencyAndShape
from extract_batch_shape_and_latency_decode import compute_power_w



def calc_stats(expr_dir: Path) -> LatencyAndShape:
    logs_dict = load_logs(expr_dir)

    stats_list = []
    for k, v in logs_dict.items():
        decode, prefill, power = v
        stats_l = calc_stats_single_instance_prefill(prefill, power)

        stats_list.extend(stats_l)

    return stats_list


def calc_stats_single_instance_prefill(df_perf_metric_prefill_steady: pd.DataFrame, df_power: pd.DataFrame) -> list[LatencyAndShape]:
    df_perf_metric_prefill_steady['request_ids_iter_ttft_evald'] = df_perf_metric_prefill_steady['request_ids_iter_ttft'].apply(eval)
    df_perf_metric_prefill_steady['time_to_first_tokens_iter_evald'] = df_perf_metric_prefill_steady['time_to_first_tokens_iter'].apply(eval)
    df_perf_metric_prefill_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_prefill_steady['num_prompt_tokens_reqs'].apply(eval)

    lat_and_shape_list = []

    batch_size_input_len_sum_mean_std_under_obs = (0, 0, 0, 0)
    latencies = []
    obs_start_time = 0
    obs_end_time = 0
    for row in df_perf_metric_prefill_steady.itertuples():
        if len(row.num_prompt_tokens_reqs_evald) == 0:
            continue
        if batch_size_input_len_sum_mean_std_under_obs != (len(row.num_prompt_tokens_reqs_evald), 
                                                  np.sum(row.num_prompt_tokens_reqs_evald),
                                                  np.mean(row.num_prompt_tokens_reqs_evald),
                                                  np.std(row.num_prompt_tokens_reqs_evald)):
            # save older data if we have enough samples
            if obs_end_time - obs_start_time > 0.4:
                df_power_obs = df_power[(df_power['Timestamp'] >= obs_start_time) & (df_power['Timestamp'] <= obs_end_time)]
                power_w, freq = compute_power_w(df_power_obs)

                lat_and_shape_list.append(LatencyAndShape(
                    batch_size=batch_size_input_len_sum_mean_std_under_obs[0],
                    input_len_sum=(batch_size_input_len_sum_mean_std_under_obs[1]),
                    input_len_mean=(batch_size_input_len_sum_mean_std_under_obs[2]),
                    input_len_std=(batch_size_input_len_sum_mean_std_under_obs[3]),
                    latency_prefill_s=np.median(latencies),
                    latency_decode_s=np.nan,
                    power_w=power_w,
                    freq_mhz=freq
                ))
            # else:
            #     print(f"Skipping observation with insufficient duration: {obs_end_time} - {obs_start_time} = {obs_end_time - obs_start_time}s, len(latencies): {len(latencies)}")

            batch_size_input_len_sum_mean_std_under_obs = (len(row.num_prompt_tokens_reqs_evald), 
                                                            np.sum(row.num_prompt_tokens_reqs_evald),
                                                            np.mean(row.num_prompt_tokens_reqs_evald),
                                                            np.std(row.num_prompt_tokens_reqs_evald))
            obs_start_time = row.now
            obs_end_time = row.now
            latencies = [row.step_with_batch_queue_time_ms/ 1000.0]
        else:
            obs_end_time = row.now
            latencies.append(row.step_with_batch_queue_time_ms/ 1000.0)
    # save the last one
    if obs_end_time - obs_start_time > 0.4:
        df_power_obs = df_power[(df_power['Timestamp'] >= obs_start_time) & (df_power['Timestamp'] <= obs_end_time)]
        power_w, freq = compute_power_w(df_power_obs)

        lat_and_shape_list.append(LatencyAndShape(
            batch_size=batch_size_input_len_sum_mean_std_under_obs[0],
            input_len_sum=(batch_size_input_len_sum_mean_std_under_obs[1]),
            input_len_mean=(batch_size_input_len_sum_mean_std_under_obs[2]),
            input_len_std=(batch_size_input_len_sum_mean_std_under_obs[3]),
            latency_prefill_s=np.median(latencies),
            latency_decode_s=np.nan,
            power_w=power_w,
            freq_mhz=freq
        ))

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
                if (not df_perf_metric_prefill.empty and not df_power.empty):
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
    merged_stats_prefill = {}
    merged_stats_power = {}
    for stats in stats_list:
        freq = int(round(stats.freq_mhz / 10.0) * 10) if not np.isnan(stats.freq_mhz) else np.nan
        key = (stats.batch_size, stats.input_len_sum, stats.input_len_mean, stats.input_len_std, freq)
        if not np.isnan(stats.latency_prefill_s):
            if key not in merged_stats_prefill:
                merged_stats_prefill[key] = []
            merged_stats_prefill[key].append(stats.latency_prefill_s)
        if key not in merged_stats_power:
            merged_stats_power[key] = []
        merged_stats_power[key].append(stats.power_w)
    
    merged_stats_rows = []
    for key in set(merged_stats_prefill.keys()).union(set(merged_stats_power.keys())):
        latencies_prefill = merged_stats_prefill.get(key, [])
        powers = merged_stats_power.get(key, [])
        merged_stats_rows.append({
            'batch_size': key[0],
            'input_len_sum': key[1],
            'input_len_mean': key[2],
            'input_len_std': key[3],
            'latency_prefill_s': np.median(latencies_prefill) if latencies_prefill else np.nan,
            'latency_decode_s': np.nan,
            'power_w': np.median(powers) if powers else np.nan,
            'freq_mhz': key[4]
        })
    merged_stats_df = pd.DataFrame(merged_stats_rows, columns=[
        'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std',
        'latency_prefill_s', 'latency_decode_s', 'power_w', 'freq_mhz'])

    df_stats = pd.DataFrame(merged_stats_df)
    df_stats = df_stats.sort_values(by=['batch_size', 'input_len_mean']).reset_index(drop=True)
    df_stats.to_csv(expr_root / 'metrics.csv', index=False)