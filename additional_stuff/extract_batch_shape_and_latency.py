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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


@dataclass
class LatencyAndShape:
    test_name: str
    num_prefill_reqs: int
    sum_ctx_len: int
    mean_ctx_len: float
    std_ctx_len: float
    num_decode_reqs: int
    sum_decode_len: int
    mean_decode_len: float
    std_decode_len: float
    frequency: float
    tp_size: int
    mean_latency: float  


def calc_stats_mixed(expr_dir: Path) -> list[LatencyAndShape]:
    """Collect mixed (prefill + decode) stats from subfolders under expr_dir.
    
    For each iteration that has both prefill and decode activity, extract:
    - Number of prefill and decode requests
    - Their respective input lengths and latencies
    - Combined metrics
    """
    logs_dict = load_logs(expr_dir)

    stats_list: list[LatencyAndShape] = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [
            executor.submit(calc_stats_single_instance_mixed, expr_dir, df_prefill, df_power)
            for k, (df_decode, df_prefill, df_power) in logs_dict.items()
            if df_decode is not None and df_prefill is not None and df_power is not None
        ]
        for future in futures:
            stats_list.extend(future.result())

    return stats_list

def calc_stats_single_instance_mixed(
    expr_dir: Path,
    df_perf_metric_prefill: pd.DataFrame,
    df_power: pd.DataFrame
) -> list[LatencyAndShape]:
    """Extract stats from mixed prefill/decode workloads by aligning on timestamps.
    
    For each prefill iteration, find the closest decode iteration in time and
    combine them into mixed workload records.
    """
    # get single freq_mhz for all gpus
    df_power['freq_mhz'] = df_power[[col for col in df_power.columns if col.startswith("GPU_") and col.endswith("_freq_mhz")]].mean(axis=1)
    tp_degree = len([col for col in df_power.columns if col.startswith("GPU_") and col.endswith("_freq_mhz")])
    # Prepare prefill data
    df_perf_metric_prefill_steady = df_perf_metric_prefill[df_perf_metric_prefill['KV_usage_perc'] < 0.95].copy()
    df_perf_metric_prefill_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_prefill_steady['num_prompt_tokens_reqs'].apply(eval)
    df_perf_metric_prefill_steady['max_num_generation_tokens_iter_evald'] = df_perf_metric_prefill_steady['max_num_generation_tokens_iter'].apply(eval)
    df_perf_metric_prefill_steady['inter_token_latencies_iter_evald'] = df_perf_metric_prefill_steady['inter_token_latencies_iter'].apply(eval)
    
    lat_and_shape_list = []
    for row in df_perf_metric_prefill_steady.itertuples():
        if len(row.num_prompt_tokens_reqs_evald) == 0 and len(row.max_num_generation_tokens_iter_evald) == 0:
            continue

        prefill_lens = []
        decode_lens = []
        prompt_lens = row.num_prompt_tokens_reqs_evald
        gen_tokens = row.max_num_generation_tokens_iter_evald
        for i, (prompt_len, gen_tok) in enumerate(zip(prompt_lens, gen_tokens)):
            if gen_tok == 1:
                # This is a prefill request
                prefill_lens.append(prompt_len)
            else:
                # This is a decode request
                # Total KV cache size = prompt_len + gen_tokens
                decode_lens.append(prompt_len + gen_tok)

        lat_and_shape_list.append(LatencyAndShape(
            test_name=expr_dir.name,
            num_prefill_reqs=len(prefill_lens),
            sum_ctx_len=sum(prefill_lens),
            mean_ctx_len=np.mean(prefill_lens) if prefill_lens else 0,
            std_ctx_len=np.std(prefill_lens) if prefill_lens else 0,
            num_decode_reqs=len(decode_lens),
            sum_decode_len=sum(decode_lens),
            mean_decode_len=np.mean(decode_lens) if decode_lens else 0,
            std_decode_len=np.std(decode_lens) if decode_lens else 0,
            frequency=row.freq_mhz,
            tp_size=tp_degree,
            mean_latency=row.inter_token_latencies_iter_evald[-1] if row.inter_token_latencies_iter_evald else np.nan
        ))
    return lat_and_shape_list




def percentile_or_nan(a, q):
    if len(a) > 0:
        return np.percentile(a, q)
    else:
        return np.nan


def load_logs(expr_dir: Path) -> dict:
    """Load logs for subfolders that have prefill, decode, AND power logs.
    
    Returns dict mapping subfolder.name -> (df_decode, df_prefill, df_power) where 
    any missing dataframe is None.
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

            if has_decode and has_prefill and has_power:
                logs[subfolder.name] = (df_perf_metric_decode, df_perf_metric_prefill, df_power)
        except Exception as e:
            print(f"Skipping {subfolder} due to error: {e}")
    return logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract mixed batch shape and latency from vllm logs")
    parser.add_argument('expr_root', nargs='?', default=str(Path('/export2/obasit/ClusterLevelServing/vllm_logs') / 'test_logs'),
                        help='root folder containing experiment folders (default: /export2/.../vllm_logs/test_logs)')
    args = parser.parse_args()
    expr_root = Path(args.expr_root)

    # structure of log files should be like this:
    # |-> expr_root
    # |  |-> disag_1P1D_test
    # |  |  |-> prefill_1
    # |  |  |  |-> engine_*.csv
    # |  |  |  |-> power_log_*.csv
    # |  |  |-> decode_1
    # |  |  |  |-> engine_*.csv
    # |  |  |  |-> power_log_*.csv
    # ...

    mixed_stats_all = []
    for expr_dir in sorted(expr_root.glob('*')):
        if not expr_dir.is_dir():
            continue
        if not any(child.is_dir() for child in expr_dir.iterdir()):
            continue
        print('expr_dir (mixed):', expr_dir)
        mixed_stats_all.append(calc_stats_mixed(expr_dir))
        if mixed_stats_all[-1]:
            print(f'Collected {len(mixed_stats_all[-1])} mixed stats from {expr_dir}')
    mixed_stats_all = list(itertools.chain.from_iterable(mixed_stats_all))
    
    if mixed_stats_all:
        # Create dataframe with all columns including mixed-specific ones
        df_stats = pd.DataFrame([asdict(s) for s in mixed_stats_all])
        
        print(f'len of mixed stats: {len(mixed_stats_all)}')
        df_stats.to_csv(expr_root / 'mixed_latencies.csv', index=False)
        print(f'Mixed latency data saved to {expr_root / "mixed_latencies.csv"}')
    else:
        print("No mixed workload stats collected")