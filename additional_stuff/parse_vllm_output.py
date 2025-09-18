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


@dataclass
class PerfStats:
    throughput_rps: float   # Requests per second
    ttft_mean: float
    ttft_p90: float
    ttft_p99: float
    tpot_mean: float
    tpot_p90: float
    tpot_p99: float
    power_w: float
    energy_j: float
    energy_per_token: float
    avg_running_q: float
    avg_waiting_q: float
    kv_usage_mean: float
    kv_usage_p99: float
    freq_mhz_mean: float
    freq_mhz_p10: float
    freq_mhz_p50: float
    freq_mhz_p90: float
    expr_duration_s: float
    num_requests: int
    num_tokens_decoded: int
    num_tokens_prefilled: int

def calc_perf_stats(expr_dir: Path) -> PerfStats:
    raw_logs_dict = load_logs(expr_dir)
    raw_logs_dict_steady = extract_steady_region(raw_logs_dict)

    perfstats_list = []
    for k, v in raw_logs_dict_steady.items():
        decode, prefill, power = v
        perfstats = calc_perf_stats_single_instance(k, decode, prefill, power)
        perfstats_list.append((k, perfstats))

    total_requests = sum(p.num_requests for k, p in perfstats_list if "prefill" in k)
    total_duration_prefill = max(p.expr_duration_s for k, p in perfstats_list if "prefill" in k)
    total_duration = max(p.expr_duration_s for _, p in perfstats_list)
    total_energy = sum(p.energy_j for _, p in perfstats_list)
    total_decode = sum(p.num_tokens_decoded for k, p in perfstats_list if "decode" in k)
    total_prefill = sum(p.num_tokens_prefilled for k, p in perfstats_list if "decode" in k)

    total_perfstats = PerfStats(
        throughput_rps=total_requests / total_duration_prefill,
        ttft_mean=0,
        ttft_p90=0,
        ttft_p99=0,
        tpot_mean=0,
        tpot_p90=0,
        tpot_p99=0,
        power_w=total_energy / total_duration,
        energy_j=total_energy,
        energy_per_token=total_energy / (total_decode + total_prefill),
        avg_running_q=0,
        avg_waiting_q=0,
        kv_usage_mean=0,
        kv_usage_p99=0,
        freq_mhz_mean=0,
        freq_mhz_p10=0,
        freq_mhz_p50=0,
        freq_mhz_p90=0,
        expr_duration_s=total_duration,
        num_requests=total_requests,
        num_tokens_decoded=total_decode,
        num_tokens_prefilled=total_prefill
    )
    perfstats_list.append(('total', total_perfstats))
    return perfstats_list


def calc_perf_stats_single_instance(root_name: str,
                                    df_perf_metric_decode_steady: pd.DataFrame,
                                    df_perf_metric_prefill_steady: pd.DataFrame,
                                    df_power_steady: pd.DataFrame) -> PerfStats:
    if not df_perf_metric_decode_steady.empty:
        df_perf_metric_decode_steady['request_ids_iter_ttft_evald'] = df_perf_metric_decode_steady['request_ids_iter_ttft'].apply(eval)
        df_perf_metric_decode_steady['request_ids_iter_tbt_evald'] = df_perf_metric_decode_steady['request_ids_iter_tbt'].apply(eval)
        df_perf_metric_decode_steady['time_to_first_tokens_iter_evald'] = df_perf_metric_decode_steady['time_to_first_tokens_iter'].apply(eval)
        df_perf_metric_decode_steady['inter_token_latencies_iter_evald'] = df_perf_metric_decode_steady['inter_token_latencies_iter'].apply(eval)
    if not df_perf_metric_prefill_steady.empty:
        df_perf_metric_prefill_steady['request_ids_iter_ttft_evald'] = df_perf_metric_prefill_steady['request_ids_iter_ttft'].apply(eval)
        df_perf_metric_prefill_steady['request_ids_iter_tbt_evald'] = df_perf_metric_prefill_steady['request_ids_iter_tbt'].apply(eval)
        df_perf_metric_prefill_steady['time_to_first_tokens_iter_evald'] = df_perf_metric_prefill_steady['time_to_first_tokens_iter'].apply(eval)
        df_perf_metric_prefill_steady['inter_token_latencies_iter_evald'] = df_perf_metric_prefill_steady['inter_token_latencies_iter'].apply(eval)

    # Calculate duration using min and max from both decode and prefill dfs
    decode_min = df_perf_metric_decode_steady['now'].min() if not df_perf_metric_decode_steady.empty else None
    decode_max = df_perf_metric_decode_steady['now'].max() if not df_perf_metric_decode_steady.empty else None
    prefill_min = df_perf_metric_prefill_steady['now'].min() if not df_perf_metric_prefill_steady.empty else None
    prefill_max = df_perf_metric_prefill_steady['now'].max() if not df_perf_metric_prefill_steady.empty else None

    min_time = min([t for t in [decode_min, prefill_min] if t is not None])
    max_time = max([t for t in [decode_max, prefill_max] if t is not None])
    duration = max_time - min_time

    # Calculate power/energy/freq within only the steady region
    freq_arr_list = []
    # Sum energy across all GPU_i_power_w columns
    energy_j_steady = 0.0
    for col in df_power_steady.columns:
        if col.startswith('GPU_') and col.endswith('_power_w'):
            energy_j_steady += np.trapezoid(
                df_power_steady[col], df_power_steady['Timestamp'])
        if col.startswith('GPU_') and col.endswith('_freq_mhz'):
            freq_arr_list.append(df_power_steady[col].to_numpy())
    power_w = energy_j_steady / duration

    # unique request IDs = num requests served
    # prefer prefill as prefill df is filled when chunked prefill is used
    unique_req_ids = set()
    if "prefill" in root_name:
        unique_req_ids.update(itertools.chain.from_iterable(
            df_perf_metric_prefill_steady['request_ids_iter_ttft_evald']))
        unique_req_ids.update(itertools.chain.from_iterable(
            df_perf_metric_prefill_steady['request_ids_iter_tbt_evald']))
    else:
        unique_req_ids.update(itertools.chain.from_iterable(
            df_perf_metric_decode_steady['request_ids_iter_ttft_evald']))
        unique_req_ids.update(itertools.chain.from_iterable(
            df_perf_metric_decode_steady['request_ids_iter_tbt_evald']))

    ttft_list = []
    tpot_list = []
    # ttft
    if "prefill" in root_name:
        ttft_list = [item for sublist in df_perf_metric_prefill_steady['time_to_first_tokens_iter_evald'] for item in sublist]

    # tpot calculations, create dict of req_id to list of tbts
    tbts_dict = dict()
    tbts_dict.update({id: [] for id in unique_req_ids})
    if "prefill_and_decode" in root_name:
        for req_id_row, tbts_row in df_perf_metric_prefill_steady[['request_ids_iter_tbt_evald', 'inter_token_latencies_iter_evald']].itertuples(index=False, name=None):
            for id, tbts in zip(req_id_row, tbts_row):
                tbts_dict[id].append(tbts)
    elif "decode" in root_name:
        for req_id_tbt_row, tbts_row, req_id_ttft_row, ttft_row in df_perf_metric_decode_steady[['request_ids_iter_tbt_evald', 'inter_token_latencies_iter_evald', 'request_ids_iter_ttft_evald', 'time_to_first_tokens_iter_evald']].itertuples(index=False, name=None):
            for id_tbt, tbts in zip(req_id_tbt_row, tbts_row):
                tbts_dict[id_tbt].append(tbts)
            # add ttft as well if you want to include queueing time in tpot
            # for id_ttft, ttft in zip(req_id_ttft_row, ttft_row):
            #     tbts_dict[id_ttft].append(ttft)
    tpot_list = [sum(tbts)/len(tbts) for tbts in tbts_dict.values() if len(tbts) > 0]

    if 'prefill' in root_name:
        total_prefilled = sum(df_perf_metric_prefill_steady['num_prompt_tokens'].to_list())
        total_decoded = sum(df_perf_metric_prefill_steady['num_generation_tokens'].to_list())
    else:
        total_prefilled = sum(df_perf_metric_decode_steady['num_prompt_tokens'].to_list())
        total_decoded = sum(df_perf_metric_decode_steady['num_generation_tokens'].to_list())

    
    running_list = []
    waiting_list = []
    kv_usage_list = []
    if "prefill" in root_name:
        running_list = df_perf_metric_prefill_steady['num_running_reqs'].to_list()
        waiting_list = df_perf_metric_prefill_steady['num_waiting_reqs'].to_list()
        kv_usage_list = df_perf_metric_prefill_steady['KV_usage_perc'].to_list()
    elif "decode" in root_name:
        running_list = df_perf_metric_decode_steady['num_running_reqs'].to_list()
        waiting_list = df_perf_metric_decode_steady['num_waiting_reqs'].to_list()
        kv_usage_list = df_perf_metric_decode_steady['KV_usage_perc'].to_list()

    return PerfStats(
        num_requests=len(unique_req_ids),
        throughput_rps=len(unique_req_ids) / duration,
        ttft_mean=float(np.mean(ttft_list)),
        ttft_p90=float(percentile_or_nan(ttft_list, q=90)),
        ttft_p99=float(percentile_or_nan(ttft_list, q=99)),
        tpot_mean=float(np.mean(tpot_list)),
        tpot_p90=float(percentile_or_nan(tpot_list, q=90)),
        tpot_p99=float(percentile_or_nan(tpot_list, q=99)),
        avg_running_q=np.mean(running_list),
        avg_waiting_q=np.mean(waiting_list),
        kv_usage_mean=float(np.mean(kv_usage_list)),
        kv_usage_p99=float(percentile_or_nan(kv_usage_list, q=99)),
        power_w=power_w,
        energy_j=energy_j_steady,
        freq_mhz_mean=float(np.mean(freq_arr_list)),
        freq_mhz_p10=float(percentile_or_nan(
            freq_arr_list, q=10)),
        freq_mhz_p50=float(percentile_or_nan(
            freq_arr_list, q=50)),
        freq_mhz_p90=float(percentile_or_nan(
            freq_arr_list, q=90)),
        expr_duration_s=duration,
        num_tokens_decoded= total_decoded,
        num_tokens_prefilled=total_prefilled,
        energy_per_token=energy_j_steady / (total_decoded + total_prefilled),
    )

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
                logs[subfolder.name] = load_logs_prefill_decode_power_logs(subfolder)
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
    power_log_files = list(expr_dir.glob('power_log.csv'))
    if len(power_log_files) != 1:
        raise FileNotFoundError("There should be exactly one power_log.csv file in the directory")
    df_power = pd.read_csv(power_log_files[0])

    df_perf_metric_decode = df_perf_metric_decode.dropna()
    df_perf_metric_prefill = df_perf_metric_prefill.dropna()

    return df_perf_metric_decode, df_perf_metric_prefill, df_power


def extract_steady_region(
    raw_logs_dict: dict,
    start_clip_minutes: float = 0.0,
    end_clip_minutes: float = 0.0
) -> dict:
    """
    Drop the first and last clip_minutes of data from df_perf_metric_*
    """
    # Gather all decode and prefill logs
    decode_dfs = []
    prefill_dfs = []
    power_dfs = []

    for logs in raw_logs_dict.values():
        if isinstance(logs, tuple) and len(logs) == 3:
            decode_df, prefill_df, power_df = logs
            if not decode_df.empty:
                decode_dfs.append(decode_df)
            if not prefill_df.empty:
                prefill_dfs.append(prefill_df)
            if not power_df.empty:
                power_dfs.append(power_df)

    # Concatenate all logs
    df_perf_metric_decode_all = pd.concat(decode_dfs, ignore_index=True) if decode_dfs else pd.DataFrame()
    df_perf_metric_prefill_all = pd.concat(prefill_dfs, ignore_index=True) if prefill_dfs else pd.DataFrame()

    # Find min and max times
    decode_min = df_perf_metric_decode_all['now'].min() if not df_perf_metric_decode_all.empty else None
    decode_max = df_perf_metric_decode_all['now'].max() if not df_perf_metric_decode_all.empty else None
    prefill_min = df_perf_metric_prefill_all['now'].min() if not df_perf_metric_prefill_all.empty else None
    prefill_max = df_perf_metric_prefill_all['now'].max() if not df_perf_metric_prefill_all.empty else None

    # Use the earliest start and latest end
    global_min = min([t for t in [decode_min, prefill_min] if t is not None])
    global_max = max([t for t in [decode_max, prefill_max] if t is not None])

    # Clip minutes from start/end
    steady_start = global_min + (start_clip_minutes * 60)
    steady_end = global_max - (end_clip_minutes * 60)

    # Filter steady region
    raw_logs_dict_steady = {}
    for key, logs in raw_logs_dict.items():
        if isinstance(logs, tuple) and len(logs) == 3:
            decode_df, prefill_df, power_df = logs
            decode_df_steady = decode_df[(decode_df['now'] >= steady_start) & (decode_df['now'] <= steady_end)] if not decode_df.empty else pd.DataFrame()
            prefill_df_steady = prefill_df[(prefill_df['now'] >= steady_start) & (prefill_df['now'] <= steady_end)] if not prefill_df.empty else pd.DataFrame()
            power_df_steady = power_df[(power_df['Timestamp'] >= steady_start) & (power_df['Timestamp'] <= steady_end)] if not power_df.empty else pd.DataFrame()
            raw_logs_dict_steady[key] = (decode_df_steady, prefill_df_steady, power_df_steady)
    return raw_logs_dict_steady


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        expr_root = Path(sys.argv[1])
    else:
        expr_root = Path('/export2/obasit/ClusterLevelServing/vllm_logs') / \
            'test_logs' 

    # structure of log files should be like this:
    # |-> expr_root
    # |  |-> mixed_logs_test
    # |  |  |-> prefill_and_decode
    # |  |  |  |-> engine_*.csv
    # |  |  |  |-> power_log_*.csv
    # |  |
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
        perfstats_list = calc_perf_stats(expr_dir)
        for key, perfstats in perfstats_list:
            df_stats.append({
                'expr_dir': expr_dir.name,
                'instance': key,
                **asdict(perfstats)
            })
    df_stats = pd.DataFrame(df_stats)
    df_stats.to_csv(expr_root / 'metrics.csv', index=False)