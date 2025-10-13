#!/usr/bin/env python3
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from parse_vllm_output import load_logs_prefill_decode_power_logs


@dataclass
class LatencyAndShape:
    batch_size: int
    input_len_sum: int
    input_len_mean: float
    input_len_std: float
    power_w: float
    freq_mhz: float


def compute_power_w(df_power_sub: pd.DataFrame):
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
        return (energy_j / duration, np.mean(freqs))
    else:
        return np.nan, np.nan


def calc_stats(expr_dir: Path, mode: str) -> List[LatencyAndShape]:
    logs_dict = load_logs(expr_dir, mode)

    stats_list: List[LatencyAndShape] = []
    for k, v in logs_dict.items():
        decode_df, prefill_df, power_df = v
        if mode == 'decode':
            stats_l = calc_stats_single_instance_decode(decode_df, power_df)
        else:
            stats_l = calc_stats_single_instance_prefill(prefill_df, power_df)

        stats_list.extend(stats_l)

    return stats_list


def calc_stats_single_instance_decode(df_perf_metric_decode_steady: pd.DataFrame, df_power: pd.DataFrame) -> List[LatencyAndShape]:

    df = df_perf_metric_decode_steady.copy()
    df['request_ids_iter_tbt_evald'] = df['request_ids_iter_tbt'].apply(eval)
    df['inter_token_latencies_iter_evald'] = df['inter_token_latencies_iter'].apply(eval)
    df['num_prompt_tokens_reqs_evald'] = df['num_prompt_tokens_reqs'].apply(eval)
    df['time_since_last_iter'] = df['now'].diff().fillna(0)
    num_computed_dict = {}
    num_computed_tokens_list = [[] for _ in range(len(df))]
    # first get num_computed_tokens for each req in each row
    for row in df.itertuples():
        if len(row.request_ids_iter_tbt_evald) == 0:
            continue
        else:
            for ID, in_len in zip(row.request_ids_iter_tbt_evald, row.num_prompt_tokens_reqs_evald):
                if ID not in num_computed_dict:
                    num_computed_dict[ID] = in_len + 1
                else:
                    num_computed_dict[ID] = num_computed_dict[ID] + 1
            num_computed_tokens_list[row.Index] = [num_computed_dict[ID] for ID in row.request_ids_iter_tbt_evald]
    df['num_computed_tokens_reqs_evald'] = num_computed_tokens_list

    lat_and_shape_list: List[LatencyAndShape] = []

    if df.empty or df['now'].isnull().all():
        return lat_and_shape_list

    start_time = df['now'].min()
    end_time = df['now'].max()
    df_power = df_power[(df_power['Timestamp'] >= start_time) & (df_power['Timestamp'] <= end_time)]

    window = 0.25  # seconds
    start_subsection = df_power['Timestamp'].min() if not df_power.empty else None
    if start_subsection is None:
        return lat_and_shape_list
    end_subsection = start_subsection + window
    while end_subsection <= df_power['Timestamp'].max():
        df_power_sub = df_power[(df_power['Timestamp'] >= start_subsection) & (df_power['Timestamp'] < end_subsection)]
        df_decode_sub = df[(df['now'] >= start_subsection) & (df['now'] < end_subsection)]
        if df_decode_sub.empty or (df_decode_sub['now'].max() - df_decode_sub['now'].min() < 0.2):
            start_subsection = end_subsection
            end_subsection = start_subsection + window
            continue

        power_w, gpu_freq = compute_power_w(df_power_sub)
        if np.isnan(power_w):
            start_subsection = end_subsection
            end_subsection = start_subsection + window
            continue

        # get batch shapes and latencies for this window
        batch_size = df_decode_sub['num_prompt_tokens_reqs_evald'].apply(len).median()
        input_lens = df_decode_sub['num_computed_tokens_reqs_evald'].dropna().tolist()
        input_lens = [lens for lens in input_lens if len(lens) > 0]
        if len(input_lens) == 0:
            start_subsection = end_subsection
            end_subsection = start_subsection + window
            continue
        input_len_sum = int(np.max([sum(lens) for lens in input_lens]))
        input_len_mean = float(np.max([np.mean(lens) for lens in input_lens]))
        input_len_std = float(np.median([np.std(lens) for lens in input_lens]))

        lat_and_shape_list.append(LatencyAndShape(
            batch_size=batch_size,
            input_len_sum=input_len_sum,
            input_len_mean=input_len_mean,
            input_len_std=input_len_std,
            power_w=power_w,
            freq_mhz=gpu_freq
        ))

        start_subsection = end_subsection
        end_subsection = start_subsection + window

    return lat_and_shape_list


def calc_stats_single_instance_prefill(df_perf_metric_prefill_steady: pd.DataFrame, df_power: pd.DataFrame) -> List[LatencyAndShape]:

    df = df_perf_metric_prefill_steady.copy()
    df['time_to_first_tokens_iter_evald'] = df['time_to_first_tokens_iter'].apply(eval)
    df['num_prompt_tokens_reqs_evald'] = df['num_prompt_tokens_reqs'].apply(eval)

    lat_and_shape_list: List[LatencyAndShape] = []

    batch_size_input_len_sum_mean_std_under_obs = (0, 0, 0, 0)
    obs_start_time = 0
    obs_end_time = 0
    for row in df.itertuples():
        current_tuple = (len(row.num_prompt_tokens_reqs_evald),
                         np.sum(row.num_prompt_tokens_reqs_evald),
                         np.mean(row.num_prompt_tokens_reqs_evald),
                         np.std(row.num_prompt_tokens_reqs_evald))
        if batch_size_input_len_sum_mean_std_under_obs != current_tuple:
            # save older data if we have enough samples
            if obs_end_time - obs_start_time >= 0.25:
                df_power_obs = df_power[(df_power['Timestamp'] >= obs_start_time) & (df_power['Timestamp'] <= obs_end_time)]
                power_w, freq = compute_power_w(df_power_obs)

                lat_and_shape_list.append(LatencyAndShape(
                    batch_size=batch_size_input_len_sum_mean_std_under_obs[0],
                    input_len_sum=(batch_size_input_len_sum_mean_std_under_obs[1]),
                    input_len_mean=(batch_size_input_len_sum_mean_std_under_obs[2]),
                    input_len_std=(batch_size_input_len_sum_mean_std_under_obs[3]),
                    power_w=power_w,
                    freq_mhz=freq
                ))

            batch_size_input_len_sum_mean_std_under_obs = current_tuple
            obs_start_time = row.now
            obs_end_time = row.now
        else:
            obs_end_time = row.now
    # save the last one
    if obs_end_time - obs_start_time >= 0.25:
        df_power_obs = df_power[(df_power['Timestamp'] >= obs_start_time) & (df_power['Timestamp'] <= obs_end_time)]
        power_w, freq = compute_power_w(df_power_obs)

        lat_and_shape_list.append(LatencyAndShape(
            batch_size=batch_size_input_len_sum_mean_std_under_obs[0],
            input_len_sum=(batch_size_input_len_sum_mean_std_under_obs[1]),
            input_len_mean=(batch_size_input_len_sum_mean_std_under_obs[2]),
            input_len_std=(batch_size_input_len_sum_mean_std_under_obs[3]),
            power_w=power_w,
            freq_mhz=freq
        ))

    return lat_and_shape_list


def percentile_or_nan(a, q):
    if len(a) > 0:
        return np.percentile(a, q)
    else:
        return np.nan


def load_logs(expr_dir: Path, mode: str) -> dict:
    logs = {}
    for subfolder in sorted(expr_dir.iterdir()):
        if subfolder.is_dir():
            try:
                (df_perf_metric_decode, df_perf_metric_prefill, df_power) = load_logs_prefill_decode_power_logs(subfolder)
                if mode == 'decode':
                    if (not df_perf_metric_decode.empty and not df_power.empty):
                        logs[subfolder.name] = (df_perf_metric_decode, df_perf_metric_prefill, df_power)
                else:
                    if (not df_perf_metric_prefill.empty and not df_power.empty):
                        logs[subfolder.name] = (df_perf_metric_decode, df_perf_metric_prefill, df_power)
            except Exception as e:
                print(f"Skipping {subfolder} due to error: {e}")
    return logs


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(description='Extract batch shapes and power for prefill or decode logs')
    parser.add_argument('expr_root', nargs='?', default=str(Path('/export2/obasit/ClusterLevelServing/vllm_logs') / 'test_logs'),
                        help='Root folder containing experiment folders')
    parser.add_argument('--mode', choices=['prefill', 'decode'], required=True, help='Which logs to extract: prefill or decode')
    args = parser.parse_args(argv)

    expr_root = Path(args.expr_root)
    mode = args.mode

    df_stats = []
    stats_list = []
    for expr_dir in sorted(expr_root.glob('*')):
        if not expr_dir.is_dir():
            continue
        if not any(child.is_dir() for child in expr_dir.iterdir()):
            continue
        print('expr_dir: ', expr_dir)
        stats_list.append(calc_stats(expr_dir, mode))
    stats_list = list(itertools.chain.from_iterable(stats_list))

    merged_stats_df = pd.DataFrame(stats_list, columns=[
        'batch_size', 'input_len_sum', 'input_len_mean', 'input_len_std',
        'power_w', 'freq_mhz'])

    print(f'len of unmerged stats: {len(stats_list)}, merged stats: {len(merged_stats_df)}')

    df_stats = pd.DataFrame(merged_stats_df)
    if mode == 'decode':
        df_stats = df_stats.sort_values(by=['batch_size', 'input_len_sum', 'input_len_mean']).reset_index(drop=True)
        out_name = 'decode_powers.csv'
    else:
        df_stats = df_stats.sort_values(by=['batch_size', 'input_len_mean']).reset_index(drop=True)
        out_name = 'prefill_powers.csv'

    out_path = expr_root / out_name
    df_stats.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main(sys.argv[1:])
