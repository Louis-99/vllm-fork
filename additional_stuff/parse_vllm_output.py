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
    power_mean: float
    power_p99: float
    energy_j: float
    energy_per_token: float
    energy_prefill: float
    prefill_energy_per_token: float
    energy_decode: float
    decode_energy_per_token: float
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
    avg_req_len: float
    p99_req_len: float

def calc_perf_stats(expr_dir: Path) -> PerfStats:
    raw_logs_dict = load_logs(expr_dir)
    raw_logs_dict_steady = extract_steady_region(raw_logs_dict)

    perfstats_list = []
    ttft_list = []
    tpot_list = []
    total_power_samples = []
    for k, v in raw_logs_dict_steady.items():
        decode, prefill, power = v
        perfstats, ttft, tpot = calc_perf_stats_single_instance(k, decode, prefill, power)
        perfstats_list.append((k, perfstats))
        ttft_list = ttft + ttft_list
        tpot_list = tpot + tpot_list
        power_samples = summed_power_by_timestamp(power)
        if not power_samples.empty:
            total_power_samples.append(power_samples)
        
    total_xput = sum(p.throughput_rps for k, p in perfstats_list if "prefill" in k)
    total_requests = sum(p.num_requests for k, p in perfstats_list if "prefill" in k)
    total_duration = np.median([p.expr_duration_s for _, p in perfstats_list])
    total_energy = sum(p.energy_j for _, p in perfstats_list)
    total_decode = sum(p.num_tokens_decoded for _, p in perfstats_list)
    total_prefill = sum(p.num_tokens_prefilled for _, p in perfstats_list)

    prefill_decodes = sum(p.num_tokens_decoded for k, p in perfstats_list if "prefill" in k)
    prefill_energy = sum(p.energy_j for k, p in perfstats_list if "prefill" in k)
    decode_decodes = sum(p.num_tokens_decoded for k, p in perfstats_list if "decode" in k)
    decode_energy = sum(p.energy_j for k, p in perfstats_list if "decode" in k)

    total_power_df = build_cluster_power_series(total_power_samples)
    if not total_power_df.empty:
        total_power_mean = float(total_power_df['power_w'].mean())
        total_power_p99 = float(percentile_or_nan(total_power_df['power_w'].to_numpy(), q=99))
    else:
        total_power_mean = np.nan
        total_power_p99 = np.nan

    total_perfstats = PerfStats(
        throughput_rps=total_xput,
        ttft_mean=np.mean(ttft_list),
        ttft_p90=np.percentile(ttft_list, 90),
        ttft_p99=np.percentile(ttft_list, 99),
        tpot_mean=np.mean(tpot_list),
        tpot_p90=np.percentile(tpot_list, 90),
        tpot_p99=np.percentile(tpot_list, 99),
        power_w=total_energy / total_duration,
        power_mean=total_power_mean,
        power_p99=total_power_p99,
        energy_j=total_energy,
        energy_prefill=prefill_energy,
        energy_decode=decode_energy,
        energy_per_token=total_energy / (total_decode),
        prefill_energy_per_token=prefill_energy / prefill_decodes,
        decode_energy_per_token=decode_energy / decode_decodes,
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
        num_tokens_prefilled=total_prefill,
        avg_req_len=0,
        p99_req_len=0,
    )
    perfstats_list.append(('total', total_perfstats))

    # save tpot_list and ttft_list to csv
    pd.DataFrame({'ttft_s': ttft_list}).to_csv(expr_dir / f'ttft.csv', index=False)
    pd.DataFrame({'tpot_s': tpot_list}).to_csv(expr_dir / f'tpot.csv', index=False)

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
        df_perf_metric_decode_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_decode_steady['num_prompt_tokens_reqs'].apply(eval)
    if not df_perf_metric_prefill_steady.empty:
        df_perf_metric_prefill_steady['request_ids_iter_ttft_evald'] = df_perf_metric_prefill_steady['request_ids_iter_ttft'].apply(eval)
        df_perf_metric_prefill_steady['request_ids_iter_tbt_evald'] = df_perf_metric_prefill_steady['request_ids_iter_tbt'].apply(eval)
        df_perf_metric_prefill_steady['time_to_first_tokens_iter_evald'] = df_perf_metric_prefill_steady['time_to_first_tokens_iter'].apply(eval)
        df_perf_metric_prefill_steady['inter_token_latencies_iter_evald'] = df_perf_metric_prefill_steady['inter_token_latencies_iter'].apply(eval)
        df_perf_metric_prefill_steady['num_prompt_tokens_reqs_evald'] = df_perf_metric_prefill_steady['num_prompt_tokens_reqs'].apply(eval)

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
    power_samples = summed_power_by_timestamp(df_power_steady)
    for col in df_power_steady.columns:
        if col.startswith('GPU_') and col.endswith('_power_w'):
            energy_j_steady += np.trapz(
                df_power_steady[col], df_power_steady['Timestamp'])
        if col.startswith('GPU_') and col.endswith('_freq_mhz'):
            freq_arr_list.append(df_power_steady[col].to_numpy())
    power_w = energy_j_steady / duration
    power_mean = float(power_samples['power_w'].mean()) if not power_samples.empty else np.nan
    power_p99 = float(percentile_or_nan(power_samples['power_w'].to_numpy(), q=99)) if not power_samples.empty else np.nan

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
                # if tbts < 0.3:
                tbts_dict[id_tbt].append(tbts)
            # add ttft as well if you want to include queueing time in tpot
            # for id_ttft, ttft in zip(req_id_ttft_row, ttft_row):
            #     tbts_dict[id_ttft].append(ttft)
    tpot_list = [sum(tbts)/len(tbts) for tbts in tbts_dict.values() if len(tbts) > 0]

    if 'prefill' in root_name:
        total_prefilled = sum(df_perf_metric_prefill_steady['num_prompt_tokens_reqs_evald'].sum())    # num prompt tokens
        total_decoded = sum(df_perf_metric_prefill_steady['num_generation_tokens'].to_list())       # one token created in prefill
        req_lens = list(itertools.chain.from_iterable(df_perf_metric_prefill_steady['num_prompt_tokens_reqs_evald']))
        avg_req_len = np.mean(req_lens)
        p99_req_len = np.percentile(req_lens, 99)
    else:
        total_prefilled = 0
        total_decoded = sum(df_perf_metric_decode_steady['num_generation_tokens'].to_list())        # number of tokens generated
        req_lens = list(itertools.chain.from_iterable(df_perf_metric_decode_steady['num_prompt_tokens_reqs_evald']))
        avg_req_len = np.mean(req_lens)
        p99_req_len = np.percentile(req_lens, 99)

    
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
        energy_prefill=0,
        energy_decode=0,
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
        energy_per_token=energy_j_steady / (total_decoded),
        prefill_energy_per_token=0,
        decode_energy_per_token=0,
        avg_req_len=avg_req_len,
        p99_req_len=p99_req_len,
        power_mean=power_mean,
        power_p99=power_p99,
    ), ttft_list, tpot_list

def percentile_or_nan(a, q):
    if len(a) > 0:
        return np.percentile(a, q)
    else:
        return np.nan


def summed_power_by_timestamp(df_power: pd.DataFrame) -> pd.DataFrame:
    if df_power.empty:
        return pd.DataFrame(columns=['Timestamp', 'power_w'])

    power_cols = [
        col for col in df_power.columns
        if col.startswith('GPU_') and col.endswith('_power_w')
    ]
    if not power_cols:
        return pd.DataFrame(columns=['Timestamp', 'power_w'])

    total_power = df_power[['Timestamp'] + power_cols].copy()
    total_power['power_w'] = total_power[power_cols].sum(axis=1)
    return total_power[['Timestamp', 'power_w']]


def build_cluster_power_series(power_samples_list: list[pd.DataFrame]) -> pd.DataFrame:
    if not power_samples_list:
        return pd.DataFrame(columns=['Timestamp', 'power_w'])

    prepared_series = []
    for power_samples in power_samples_list:
        if power_samples.empty:
            continue
        series = power_samples.sort_values('Timestamp').groupby('Timestamp', as_index=False)['power_w'].mean()
        if not series.empty:
            prepared_series.append(series)

    if not prepared_series:
        return pd.DataFrame(columns=['Timestamp', 'power_w'])

    base_index = max(range(len(prepared_series)), key=lambda idx: len(prepared_series[idx]))
    aligned_power = prepared_series[base_index].rename(columns={'power_w': 'power_w_0'}).copy()

    next_col_idx = 1
    for idx, series in enumerate(prepared_series):
        if idx == base_index:
            continue
        aligned_power = pd.merge_asof(
            aligned_power.sort_values('Timestamp'),
            series.sort_values('Timestamp').rename(columns={'power_w': f'power_w_{next_col_idx}'}),
            on='Timestamp',
            direction='nearest',
        )
        next_col_idx += 1

    power_cols = [col for col in aligned_power.columns if col.startswith('power_w_')]
    aligned_power['power_w'] = aligned_power[power_cols].sum(axis=1)
    return aligned_power[['Timestamp', 'power_w']]

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
    if "prefill" not in str(expr_dir.name):
        decode_csv_paths = list(expr_dir.glob('engine_*.csv'))
    
    if decode_csv_paths is not None:
        if len(decode_csv_paths) > 1:
            raise FileNotFoundError("More than one engine_*.csv file found in the directory")
        df_perf_metric_decode = pd.read_csv(decode_csv_paths[0])
    else:
        df_perf_metric_decode = pd.DataFrame()

    prefill_csv_paths = None
    # Read prefill CSV if it exists
    if "decode" not in str(expr_dir.name) or "prefill_and_decode" in str(expr_dir.name):
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
    start_clip_minutes: float = 0.5,
    end_clip_minutes: float = 0.25,
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
            print(f"Decode df shape: {decode_df.shape}, Prefill df shape: {prefill_df.shape}, Power df shape: {power_df.shape}")
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
    steady_end = global_min + (4.5 * 60)
    assert steady_end > steady_start, "Steady end time must be greater than steady start time"
    assert steady_end < global_max, "Steady end time must be less than global max time"

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