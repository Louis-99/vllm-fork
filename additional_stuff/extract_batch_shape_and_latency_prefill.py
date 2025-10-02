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
    df_perf_metric_prefill_steady['time_since_last_iter'] = df_perf_metric_prefill_steady['now'].diff().fillna(0)

    if "step_with_batch_queue_time_ms" not in df_perf_metric_prefill_steady.columns:
        df_perf_metric_prefill_steady["step_with_batch_queue_time_ms"] = df_perf_metric_prefill_steady["step_with_batch_queue_time_ms_1_iters_delay"].shift(-1)

    lat_and_shape_list = []

    # batch_size_input_len_sum_mean_std_under_obs = (0, 0, 0, 0)
    # latencies = []
    # obs_start_time = 0
    # obs_end_time = 0
    # for row injtinue
        # if batch_size_input_len_sum_mean_std_under_obs != (len(row.num_prompt_tokens_reqs_evald), 
        #                                           np.sum(row.num_prompt_tokens_reqs_evald),
        #                                           np.mean(row.num_prompt_tokens_reqs_evald),
        #                                           np.std(row.num_prompt_tokens_reqs_evald)):
        #     # save older data if we have enough samples
        #     if obs_end_time - obs_start_time > 0.4:
        #         df_power_obs = df_power[(df_power['Timestamp'] >= obs_start_time) & (df_power['Timestamp'] <= obs_end_time)]
        #         power_w, freq = compute_power_w(df_power_obs)

        #         lat_and_shape_list.append(LatencyAndShape(
        #             batch_size=batch_size_input_len_sum_mean_std_under_obs[0],
        #             input_len_sum=(batch_size_input_len_sum_mean_std_under_obs[1]),
        #             input_len_mean=(batch_size_input_len_sum_mean_std_under_obs[2]),
        #             input_len_std=(batch_size_input_len_sum_mean_std_under_obs[3]),
        #             latency_prefill_s=np.median(latencies),
        #             latency_decode_s=np.nan,
        #             power_w=power_w,
        #             freq_mhz=freq
        #         ))
        #     # else:
        #     #     print(f"Skipping observation with insufficient duration: {obs_end_time} - {obs_start_time} = {obs_end_time - obs_start_time}s, len(latencies): {len(latencies)}")

        #     batch_size_input_len_sum_mean_std_under_obs = (len(row.num_prompt_tokens_reqs_evald), 
        #                                                     np.sum(row.num_prompt_tokens_reqs_evald),
        #                                                     np.mean(row.num_prompt_tokens_reqs_evald),
        #                                                     np.std(row.num_prompt_tokens_reqs_evald))
        #     obs_start_time = row.now
        #     obs_end_time = row.now
        #     latencies = [row.step_with_batch_queue_time_ms/ 1000.0]
        # else:
        #     obs_end_time = row.now
        #     latencies.append(row.step_with_batch_queue_time_ms/ 1000.0)
    # save the last one
    # if obs_end_time - obs_start_time > 0.4:
    #     df_power_obs = df_power[(df_power['Timestamp'] >= obs_start_time) & (df_power['Timestamp'] <= obs_end_time)]
    #     power_w, freq = compute_power_w(df_power_obs)

    #     lat_and_shape_list.append(LatencyAndShape(
    #         batch_size=batch_size_input_len_sum_mean_std_under_obs[0],
    #         input_len_sum=(batch_size_input_len_sum_mean_std_under_obs[1]),
    #         input_len_mean=(batch_size_input_len_sum_mean_std_under_obs[2]),
    #         input_len_std=(batch_size_input_len_sum_mean_std_under_obs[3]),
    #         latency_prefill_s=np.median(latencies),
    #         latency_decode_s=np.nan,
    #         power_w=power_w,
    #         freq_mhz=freq
    #     ))
    for row in df_perf_metric_prefill_steady.itertuples():
        if len(row.request_ids_iter_ttft_evald) == 0:
            continue
        batch_size = len(row.num_prompt_tokens_reqs_evald)
        input_lens = row.num_prompt_tokens_reqs_evald
        input_lens = [lens for lens in input_lens if lens > 0]
        input_len_sum = int(np.sum(input_lens))
        input_len_mean = float(np.mean(input_lens))
        input_len_std = float(np.std(input_lens))
        latencies = [row.step_with_batch_queue_time_ms/ 1000.0]
        latency_prefill_s = np.median(latencies) if len(latencies) > 0 else np.nan

        if latency_prefill_s > row.time_since_last_iter:
            continue
        if latency_prefill_s <= 0.002:  # if less than 2ms
            continue

        lat_and_shape_list.append(LatencyAndShape(
            batch_size=batch_size,
            input_len_sum=input_len_sum,
            input_len_mean=input_len_mean,
            input_len_std=input_len_std,
            latency_prefill_s=latency_prefill_s,  # no prefill in decode logs
            latency_decode_s=np.nan,
            power_w=0,
            freq_mhz=1410,
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
                # if (not df_perf_metric_prefill.empty and not df_power.empty):
                if (not df_perf_metric_prefill.empty):
                    logs[subfolder.name] = (df_perf_metric_decode, df_perf_metric_prefill, df_power)
            except Exception as e:
                print(f"Skipping {subfolder} due to error: {e}")
    return logs


def load_logs_prefill_decode_power_logs(expr_dir: Path) -> Tuple[
    pd.DataFrame,   # decode
    pd.DataFrame,   # prefill
    pd.DataFrame,   # power
]:
    decode_csv_paths = None
    # Read decode CSV if it exists
    if "prefill" not in str(expr_dir):
        decode_csv_paths = list(expr_dir.glob('engine_*.csv'))
    
    if decode_csv_paths is not None:
        if len(decode_csv_paths) > 1:
            raise FileNotFoundError("More than one engine_*.csv file found in the directory")
        df_perf_metric_decode = pd.read_csv(decode_csv_paths[0])
    else:
        df_perf_metric_decode = pd.DataFrame()

    prefill_csv_paths = None
    # Read prefill CSV if it exists
    if "decode" not in str(expr_dir) or "prefill_and_decode" in str(expr_dir):
        prefill_csv_paths = list(expr_dir.glob('engine_*.csv'))
    if prefill_csv_paths is not None:
        if len(prefill_csv_paths) > 1:
            raise FileNotFoundError("More than one engine_*.csv file found in the directory")
        df_perf_metric_prefill = pd.read_csv(prefill_csv_paths[0])
    else:
        df_perf_metric_prefill = pd.DataFrame()

    # Read the single power log CSV
    df_power = pd.DataFrame()

    return df_perf_metric_decode, df_perf_metric_prefill, df_power

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
            'freq_mhz': key[4],
        })
    merged_stats_df = pd.DataFrame(merged_stats_rows, columns=[
        'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std',
        'latency_prefill_s', 'latency_decode_s', 'power_w', 'freq_mhz'])

    print(f'len of unmerged stats: {len(stats_list)}, merged stats: {len(merged_stats_df)}')

    df_stats = pd.DataFrame(merged_stats_df)
    df_stats = df_stats.sort_values(by=['batch_size', 'input_len_mean']).reset_index(drop=True)
    df_stats.to_csv(expr_root / 'metrics.csv', index=False)