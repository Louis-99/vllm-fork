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
import argparse


@dataclass
class LatencyAndShape:
    batch_size: int
    input_len_sum: int
    input_len_mean: int
    input_len_std: float
    latency_prefill_s: float
    latency_decode_s: float
    freq_mhz: float
    tp_degree: int

def calc_stats_decode(expr_dir: Path) -> list[LatencyAndShape]:
    """Collect decode stats from subfolders under expr_dir."""
    logs_dict = load_logs(expr_dir, require="decode")

    stats_list: list[LatencyAndShape] = []
    import concurrent.futures
    
    def process_logs_entry(k, v):
        df_perf_metric_decode, df_perf_metric_prefill, df_power = v
        if df_perf_metric_decode is None or df_power is None:
            return []
        return calc_stats_single_instance_decode(df_perf_metric_decode, df_power)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda item: process_logs_entry(item[0], item[1]), logs_dict.items()))
    
    for stats_l in results:
        stats_list.extend(stats_l)

    return stats_list

def calc_stats_prefill(expr_dir: Path) -> list[LatencyAndShape]:
    """Collect prefill stats from subfolders under expr_dir."""
    logs_dict = load_logs(expr_dir, require="prefill")

    stats_list: list[LatencyAndShape] = []
    import concurrent.futures
    
    def process_logs_entry(k, v):
        df_perf_metric_decode, df_perf_metric_prefill, df_power = v
        if df_perf_metric_prefill is None or df_power is None:
            return []
        return calc_stats_single_instance_prefill(df_perf_metric_prefill, df_power)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda item: process_logs_entry(item[0], item[1]), logs_dict.items()))
    
    for stats_l in results:
        stats_list.extend(stats_l)

    return stats_list

def calc_stats_single_instance_decode(df_perf_metric_decode_steady: pd.DataFrame, df_power: pd.DataFrame) -> list[LatencyAndShape]:
    # get single freq_mhz for all gpus
    df_power['freq_mhz'] = df_power[[col for col in df_power.columns if col.startswith("GPU_") and col.endswith("_freq_mhz")]].max(axis=1)

    df_perf_metric_decode_steady['request_ids_iter_tbt_evald'] = df_perf_metric_decode_steady['request_ids_iter_tbt'].apply(eval)
    df_perf_metric_decode_steady['inter_token_latencies_iter_evald'] = df_perf_metric_decode_steady['inter_token_latencies_iter'].apply(eval)
    df_perf_metric_decode_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_decode_steady['num_prompt_tokens_reqs'].apply(eval)
    df_perf_metric_decode_steady['time_since_last_iter'] = df_perf_metric_decode_steady['now'].diff().fillna(0)
    num_computed_dict = {}
    num_computed_tokens_list = [[] for _ in range(len(df_perf_metric_decode_steady))]
    #first get num_computed_tokens for each req in each row
    for row in df_perf_metric_decode_steady.itertuples():
        if len(row.request_ids_iter_tbt_evald) == 0:
            continue
        else:
            for ID, in_len in zip(row.request_ids_iter_tbt_evald, row.num_prompt_tokens_reqs_evald):
                if ID not in num_computed_dict:
                    num_computed_dict[ID] = in_len + 1
                else:
                    num_computed_dict[ID] = num_computed_dict[ID] + 1
            num_computed_tokens_list[row.Index] = [num_computed_dict[ID] for ID in row.request_ids_iter_tbt_evald]
    df_perf_metric_decode_steady['num_computed_tokens_reqs_evald'] = num_computed_tokens_list

    # drop rows with KV cache greater than 95%
    df_perf_metric_decode_steady = df_perf_metric_decode_steady[df_perf_metric_decode_steady['KV_usage_perc'] < 0.95].copy()

    lat_and_shape_list = []
    import concurrent.futures

    def process_row(row, df_power):
        if len(row.request_ids_iter_tbt_evald) == 0:
            return None
        batch_size = len(row.num_prompt_tokens_reqs_evald)
        input_lens = row.num_computed_tokens_reqs_evald
        input_lens = [lens for lens in input_lens if lens > 0]
        if len(input_lens) == 0:
            return None
        input_len_sum = int(np.sum(input_lens))
        input_len_mean = float(np.mean(input_lens))
        input_len_std = float(np.std(input_lens))
        latencies = row.inter_token_latencies_iter_evald
        latency_decode_s = np.median(latencies) if len(latencies) > 0 else np.nan

        freq_mhz = np.max(df_power[(df_power['Timestamp'] >= row.now - 0.05) & (df_power['Timestamp'] <= row.now + 0.05)]['freq_mhz'])
        tp = len([col for col in df_power.columns if col.startswith("GPU_") and col.endswith("_freq_mhz")])

        return LatencyAndShape(
            batch_size=batch_size,
            input_len_sum=input_len_sum,
            input_len_mean=input_len_mean,
            input_len_std=input_len_std,
            latency_prefill_s=np.nan,  # no prefill in decode logs
            latency_decode_s=latency_decode_s,
            freq_mhz=freq_mhz,
            tp_degree=tp,
        )

    rows = list(df_perf_metric_decode_steady.itertuples())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda row: process_row(row, df_power), rows))

    lat_and_shape_list = [res for res in results if res is not None]

    return lat_and_shape_list

def calc_stats_single_instance_prefill(df_perf_metric_prefill_steady: pd.DataFrame, df_power: pd.DataFrame) -> list[LatencyAndShape]:
    # get single freq_mhz for all gpus
    df_power['freq_mhz'] = df_power[[col for col in df_power.columns if col.startswith("GPU_") and col.endswith("_freq_mhz")]].max(axis=1)

    df_perf_metric_prefill_steady['request_ids_iter_ttft_evald'] = df_perf_metric_prefill_steady['request_ids_iter_ttft'].apply(eval)
    df_perf_metric_prefill_steady['time_to_first_tokens_iter_evald'] = df_perf_metric_prefill_steady['time_to_first_tokens_iter'].apply(eval)
    df_perf_metric_prefill_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_prefill_steady['num_prompt_tokens_reqs'].apply(eval)

    # drop empty rows first
    df_perf_metric_prefill_steady = df_perf_metric_prefill_steady[df_perf_metric_prefill_steady['request_ids_iter_ttft_evald'].apply(lambda x: len(x) > 0)].copy()
    # then do shift of gpu times
    df_perf_metric_prefill_steady.loc[:, "step_with_batch_queue_time_ms"] = df_perf_metric_prefill_steady["step_with_batch_queue_time_ms_1_iters_delay"].shift(-1)

    # drop rows with KV cache greater than 95%
    df_perf_metric_prefill_steady = df_perf_metric_prefill_steady[df_perf_metric_prefill_steady['KV_usage_perc'] < 0.95].copy()    

    lat_and_shape_list = []

    import concurrent.futures

    def process_row(row, df_power):
        if len(row.request_ids_iter_ttft_evald) == 0:
            return None
        batch_size = len(row.num_prompt_tokens_reqs_evald)
        input_lens = row.num_prompt_tokens_reqs_evald
        input_lens = [lens for lens in input_lens if lens > 0]
        if len(input_lens) == 0:
            return None
        input_len_sum = int(np.sum(input_lens))
        input_len_mean = float(np.mean(input_lens))
        input_len_std = float(np.std(input_lens))
        latencies = [row.step_with_batch_queue_time_ms / 1000.0]
        latency_prefill_s = np.median(latencies) if len(latencies) > 0 else np.nan

        freq_mhz = np.max(df_power[(df_power['Timestamp'] >= row.now - 0.05) & (df_power['Timestamp'] <= row.now + 0.05)]['freq_mhz'])
        tp = len([col for col in df_power.columns if col.startswith("GPU_") and col.endswith("_freq_mhz")])

        return LatencyAndShape(
            batch_size=batch_size,
            input_len_sum=input_len_sum,
            input_len_mean=input_len_mean,
            input_len_std=input_len_std,
            latency_prefill_s=latency_prefill_s,
            latency_decode_s=np.nan,
            freq_mhz=freq_mhz,
            tp_degree=tp,
        )

    rows = list(df_perf_metric_prefill_steady.itertuples())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda row: process_row(row, df_power), rows))

    lat_and_shape_list = [res for res in results if res is not None]

    return lat_and_shape_list


def percentile_or_nan(a, q):
    if len(a) > 0:
        return np.percentile(a, q)
    else:
        return np.nan


def load_logs(expr_dir: Path, require: str = "prefill") -> dict:
    """Load logs for subfolders. `require` can be 'decode' or 'prefill'.
    When 'decode' we include folders that have non-empty decode and power logs.
    When 'prefill' we include folders that have non-empty prefill and power logs.
    Returns dict mapping subfolder.name -> (df_decode, df_prefill, df_power) where any missing dataframe is None.
    """
    logs = {}
    for subfolder in sorted(expr_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        try:
            (df_perf_metric_decode, df_perf_metric_prefill, df_power) = load_logs_prefill_decode_power_logs(subfolder)
            has_power = df_power is not None and (not df_power.empty)
            has_decode = df_perf_metric_decode is not None and (not df_perf_metric_decode.empty)
            has_prefill = df_perf_metric_prefill is not None and (not df_perf_metric_prefill.empty)

            include = False
            if require == "decode" and has_decode and has_power:
                include = True
            elif require == "prefill" and has_prefill and has_power:
                include = True

            if include:
                # replace empty dfs with None for clarity
                df_perf_metric_decode = df_perf_metric_decode if has_decode else None
                df_perf_metric_prefill = df_perf_metric_prefill if has_prefill else None
                logs[subfolder.name] = (df_perf_metric_decode, df_perf_metric_prefill, df_power)
        except Exception as e:
            print(f"Skipping {subfolder} due to error: {e}")
    return logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract batch shape and latency (prefill/decode) from vllm logs")
    parser.add_argument('expr_root', nargs='?', default=str(Path('/export2/obasit/ClusterLevelServing/vllm_logs') / 'test_logs'),
                        help='root folder containing experiment folders (default: /export2/.../vllm_logs/test_logs)')
    parser.add_argument('--mode', choices=['decode', 'prefill', 'both'], default='both',
                        help='which metrics to extract')
    args = parser.parse_args()
    expr_root = Path(args.expr_root)
    mode = args.mode

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

    # depending on mode, collect stats and write appropriate CSV(s)
    if mode in ('decode', 'both'):
        decode_stats_all = []
        for expr_dir in sorted(expr_root.glob('*')):
            if not expr_dir.is_dir():
                continue
            if not any(child.is_dir() for child in expr_dir.iterdir()):
                continue
            print('expr_dir (decode):', expr_dir)
            decode_stats_all.append(calc_stats_decode(expr_dir))
        decode_stats_all = list(itertools.chain.from_iterable(decode_stats_all))
        
        merged_stats_df = pd.DataFrame(decode_stats_all, columns=[
            'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std',
            'latency_prefill_s', 'latency_decode_s', 'freq_mhz', 'tp_degree'])

        print(f'len of decode stats: {len(decode_stats_all)}')

        df_stats = pd.DataFrame(merged_stats_df)
        df_stats = df_stats.sort_values(by=['batch_size', 'input_len_sum', 'input_len_mean']).reset_index(drop=True)
        df_stats.to_csv(expr_root / 'decode_latencies.csv', index=False)

    if mode in ('prefill', 'both'):
        prefill_stats_all = []
        for expr_dir in sorted(expr_root.glob('*')):
            if not expr_dir.is_dir():
                continue
            if not any(child.is_dir() for child in expr_dir.iterdir()):
                continue
            print('expr_dir (prefill):', expr_dir)
            prefill_stats_all.append(calc_stats_prefill(expr_dir))
            print(f'Collected {len(prefill_stats_all[-1])} prefill stats from {expr_dir}')
        prefill_stats_all = list(itertools.chain.from_iterable(prefill_stats_all))
        
        merged_stats_df = pd.DataFrame(prefill_stats_all, columns=[
            'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std',
            'latency_prefill_s', 'latency_decode_s', 'freq_mhz', 'tp_degree'])

        merged_stats_df = merged_stats_df[merged_stats_df['input_len_sum'] < 8001]

        print(f'len of prefill stats: {len(prefill_stats_all)}')

        df_stats = pd.DataFrame(merged_stats_df)
        df_stats = df_stats.sort_values(by=['batch_size', 'input_len_mean']).reset_index(drop=True)
        df_stats.to_csv(expr_root / 'prefill_latencies.csv', index=False)