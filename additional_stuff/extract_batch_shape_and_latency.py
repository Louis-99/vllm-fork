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

def calc_stats(expr_dir: Path) -> LatencyAndShape:
    logs_dict = load_logs(expr_dir)

    stats_list = []
    for k, v in logs_dict.items():
        decode, prefill, power = v
        if decode.empty:
            stats_l = calc_stats_single_instance_prefill(prefill)
        elif prefill.empty:
            stats_l = calc_stats_single_instance_decode(decode)
        stats_list.extend(stats_l)

    return stats_list


def calc_stats_single_instance_prefill(df_perf_metric_prefill_steady: pd.DataFrame) -> list[LatencyAndShape]:

    df_perf_metric_prefill_steady['request_ids_iter_ttft_evald'] = df_perf_metric_prefill_steady['request_ids_iter_ttft'].apply(eval)
    df_perf_metric_prefill_steady['time_to_first_tokens_iter_evald'] = df_perf_metric_prefill_steady['time_to_first_tokens_iter'].apply(eval)
    df_perf_metric_prefill_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_prefill_steady['num_prompt_tokens_reqs'].apply(eval)

    lat_and_shape_list = []

    for row in df_perf_metric_prefill_steady.itertuples():
        assert len(row.request_ids_iter_ttft_evald) == len(row.time_to_first_tokens_iter_evald) == len(row.num_prompt_tokens_reqs_evald)

        # Compute statistics
        batch_size = len(row.request_ids_iter_ttft_evald)
        input_len_mean = np.mean(row.num_prompt_tokens_reqs_evald)
        input_len_std = np.std(row.num_prompt_tokens_reqs_evald)
        latency_prefill_s = np.mean(row.time_to_first_tokens_iter_evald)

        lat_and_shape = LatencyAndShape(
            input_len_sum=sum(row.num_prompt_tokens_reqs_evald),
            batch_size=batch_size,
            input_len_mean=input_len_mean,
            input_len_std=input_len_std,
            latency_prefill_s=latency_prefill_s,
            latency_decode_s=np.nan
        )
        lat_and_shape_list.append(lat_and_shape)

    return lat_and_shape_list

def calc_stats_single_instance_decode(df_perf_metric_decode_steady: pd.DataFrame) -> list[LatencyAndShape]:
    df_perf_metric_decode_steady['request_ids_iter_tbt_evald'] = df_perf_metric_decode_steady['request_ids_iter_tbt'].apply(eval)
    df_perf_metric_decode_steady['inter_token_latencies_iter_evald'] = df_perf_metric_decode_steady['inter_token_latencies_iter'].apply(eval)
    df_perf_metric_decode_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_decode_steady['num_prompt_tokens_reqs'].apply(eval)

    lat_and_shape_list = []

    ID_len_dict = {}

    for row in df_perf_metric_decode_steady.itertuples():
        if len(row.request_ids_iter_tbt_evald) == 0:
            continue
        for ID, in_len in zip(row.request_ids_iter_tbt_evald, row.num_prompt_tokens_reqs_evald):
            if ID not in ID_len_dict:
                ID_len_dict[ID] = in_len
            else:
                ID_len_dict[ID] = ID_len_dict[ID] + 1

        inputs = [ID_len_dict[ID] for ID in row.request_ids_iter_tbt_evald]

        # Compute statistics
        batch_size = len(row.num_prompt_tokens_reqs_evald)
        input_len_sum = sum(inputs)
        input_len_mean = np.mean(inputs)
        input_len_std = np.std(inputs)
        latency_decode_s = np.mean(row.inter_token_latencies_iter_evald)

        lat_and_shape = LatencyAndShape(
            batch_size=batch_size,
            input_len_sum=input_len_sum,
            input_len_mean=input_len_mean,
            input_len_std=input_len_std,
            latency_prefill_s=np.nan,
            latency_decode_s=latency_decode_s
        )
        lat_and_shape_list.append(lat_and_shape)

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
                if (df_perf_metric_decode.empty and not df_perf_metric_prefill.empty) or (not df_perf_metric_decode.empty and df_perf_metric_prefill.empty):
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
    for expr_dir in sorted(expr_root.glob('*')):
        if not expr_dir.is_dir():
            continue
        if not any(child.is_dir() for child in expr_dir.iterdir()):
            continue
        print('expr_dir: ', expr_dir)
        stats_list = calc_stats(expr_dir)

        # merge prefill and decode for same batch shape, input lengths
        merged_stats_prefill = {}
        merged_stats_decode = {}
        for stats in stats_list:
            key = (stats.batch_size, stats.input_len_sum, stats.input_len_mean, stats.input_len_std)
            if not np.isnan(stats.latency_prefill_s):
                if key not in merged_stats_prefill:
                    merged_stats_prefill[key] = []
                merged_stats_prefill[key].append(stats.latency_prefill_s)
            if not np.isnan(stats.latency_decode_s):
                if key not in merged_stats_decode:
                    merged_stats_decode[key] = []
                merged_stats_decode[key].append(stats.latency_decode_s)
        merged_stats_rows = []
        for key in set(merged_stats_prefill.keys()).union(set(merged_stats_decode.keys())):
            latencies_prefill = merged_stats_prefill.get(key, [])
            latencies_decode = merged_stats_decode.get(key, [])
            merged_stats_rows.append({
                'batch_size': key[0],
                'input_len_sum': key[1],
                'input_len_mean': key[2],
                'input_len_std': key[3],
                'latency_prefill_s': np.median(latencies_prefill) if latencies_prefill else np.nan,
                'latency_decode_s': np.median(latencies_decode) if latencies_decode else np.nan
            })
        merged_stats_df = pd.DataFrame(merged_stats_rows, columns=[
            'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std',
            'latency_prefill_s', 'latency_decode_s'])

    df_stats = pd.DataFrame(merged_stats_df)
    df_stats = df_stats.sort_values(by=['batch_size', 'input_len_mean']).reset_index(drop=True)
    df_stats.to_csv(expr_root / 'metrics.csv', index=False)