import os
import shutil
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
# from parse_vllm_output import load_logs
# from plot_latency_and_power_model_discrepency import load_logs_and_calc_difference
# from utils import get_cdf_data

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_qps_timeline(ax, trace_path, window_sz=1.0, start_time=None, end_time=None):
    """
    Plots a timeline of QPS (queries per second) over time from a given LLM
    request trace.
    """
    df = pd.read_csv(trace_path)

    if start_time is not None:
        df = df[df['arrived_at'] >= start_time]
    if end_time is not None:
        df = df[df['arrived_at'] <= end_time]

    max_time = df['arrived_at'].max()
    bins = np.arange(0, max_time + window_sz, window_sz)
    df['time_bin'] = pd.cut(df['arrived_at'], bins=bins, right=False)

    grouped = df.groupby('time_bin', observed=True)
    qps = grouped.size() / window_sz
    midpoints = bins[:-1] + window_sz / 2
    time_labels = midpoints[:len(qps)]

    ax.plot(time_labels, qps.values)


def plot_qps_timelines(output_path: Path = Path('figs/qps_timelines.pdf')):
    trace_path = Path(
        '/export2/kong102/energy_efficient_serving_results/datasets/processed/azure_2024_code_qps-default_req-cnt16803695.csv')
    params = [
        [0, 36000, 30],
        [0, 600, 1],
        [0, 60, 0.2],
    ]
    fig, axs = plt.subplots(3, 1, figsize=(5, 3.5))
    for ax, (start_time, end_time, window_sz) in zip(axs, params):
        plot_qps_timeline(ax, trace_path, window_sz, start_time, end_time)
        ax.set_ylim(ymin=0)
        ax.set_ylabel('RPS')
    axs[-1].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    pdfcrop(output_path)


def plot_avg_lengths_timeline(trace_path, window_sz=1.0, start_time=None, end_time=None,
                              output_path='figs/avg_lengths_timeline.pdf'):
    """
    Plots the average number of prefill and decode tokens over time from a given LLM request trace.
    """
    df = pd.read_csv(trace_path)

    if start_time is not None:
        df = df[df['arrived_at'] >= start_time]
    if end_time is not None:
        df = df[df['arrived_at'] <= end_time]

    max_time = df['arrived_at'].max()
    bins = np.arange(0, max_time + window_sz, window_sz)
    df['time_bin'] = pd.cut(df['arrived_at'], bins=bins, right=False)

    grouped = df.groupby('time_bin', observed=True)
    avg_input_len = grouped['num_prefill_tokens'].mean()
    avg_output_len = grouped['num_decode_tokens'].mean()
    midpoints = bins[:-1] + window_sz / 2
    time_labels = midpoints[:len(avg_input_len)]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(time_labels, avg_input_len.values, label='Prefill Tokens')
    ax.plot(time_labels, avg_output_len.values, label='Decode Tokens')
    ax.set_title('Average Input/Output Length Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average Tokens')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    pdfcrop(output_path)


def plot_execution_timeline(axs,
                            expr_dir: Path,
                            t_start: Optional[float] = None,
                            t_end: Optional[float] = None,
                            plot_ylabels: bool = True):
    """
    Plot vertically-stacked timeline figures showing QPS and sys metrics, to
    show that low QPS leads to under-utilization.
    """
    df_perf_metric, df_power, _ = load_logs(expr_dir)

    t_min = df_perf_metric['now'].min()
    df_perf_metric['now'] = df_perf_metric['now'] - t_min
    df_power['Timestamp'] = df_power['Timestamp'] - t_min
    if t_start:
        df_perf_metric = df_perf_metric[df_perf_metric['now'] >= t_start]
        df_power = df_power[df_power['Timestamp'] >= t_start]
    if t_end:
        df_perf_metric = df_perf_metric[df_perf_metric['now'] < t_end]
        df_power = df_power[df_power['Timestamp'] < t_end]

    df_perf_metric['now'] = df_perf_metric['now'] - t_start
    df_power['Timestamp'] = df_power['Timestamp'] - t_start

    # QPS
    ax = axs[0]
    window_sz = 1.0
    bins = np.arange(df_perf_metric['now'].min(),
                     df_perf_metric['now'].max() + window_sz,
                     window_sz)
    df_perf_metric['time_bin'] = pd.cut(df_perf_metric['now'], bins=bins, right=False)
    df_perf_metric['num_accepted_reqs_diff'] = df_perf_metric['num_accepted_reqs'].diff()
    grouped = df_perf_metric.groupby('time_bin')
    qps_arr = grouped['num_accepted_reqs_diff'].sum() / window_sz
    ax.plot(bins[:-1], qps_arr, label='QPS', color='C0')
    if plot_ylabels:
        ax.set_ylabel('QPS')

    # GPU mem util
    ax = axs[1]
    ax.plot(df_perf_metric['now'], df_perf_metric['gpu_cache_usage_sys'],
            label='GPU mem util', color='C1')
    if plot_ylabels:
        ax.set_ylabel('GPU mem\nutil (%)')

    # Waiting queue len
    ax = axs[2]
    ax.plot(df_perf_metric['now'], df_perf_metric['num_waiting_sys'],
            label='Waiting queue len', color='C1')
    if plot_ylabels:
        ax.set_ylabel('Waiting\nqueue len')

    # TTFT
    ax = axs[3]
    t_arr = []
    ttft_arr = []
    for _, row in df_perf_metric.iterrows():
        ttft_arr_ = eval(row['time_to_first_tokens_iter'])
        if len(ttft_arr_):
            t_arr.append(row['now'])
            ttft_arr.append(np.mean(ttft_arr_))
    ax.plot(t_arr, moving_average(ttft_arr, win=20), color='C2')
    if plot_ylabels:
        ax.set_ylabel('TTFT mov.\navg. (s)')
    ax.axhline(y=1.0, color='r', label='SLO', linestyle='--')
    ax.legend(loc='upper left')

    # TBT
    ax = axs[4]
    ax.plot(df_perf_metric['now'].iloc[1:],
            moving_average(df_perf_metric['now'].diff().iloc[1:], win=20),
            color='C2')
    ax.set_xlabel('Time (s)')
    if plot_ylabels:
        ax.set_ylabel('TBT mov.\navg. (s)')
    ax.axhline(y=0.25, color='r', label='SLO', linestyle='--')
    ax.legend(loc='upper left')

    # Plot a vertical line in the middle
    if t_start and t_end:
        for ax in axs:
            ax.axvline(x=(t_end - t_start) / 2, color='gray', linestyle='--', linewidth=0.8)

    # Compute power per token
    num_tokens = (df_perf_metric['num_prompt_tokens_iter'].sum() +
                  df_perf_metric['num_generation_tokens_iter'].sum())
    total_time = df_power['Timestamp'].max() - df_power['Timestamp'].min()
    # Assumes power is logged uniformly
    total_energy = df_power['GPU_0_power_w'].mean() * total_time
    energy_per_token = total_energy / num_tokens
    print('energy_per_token: ', energy_per_token)


def plot_execution_timelines(output_path: Path = Path('figs/execution_timeline.pdf')):
    fig, axs = plt.subplots(5, 2, figsize=(6, 5.5), sharex='col', sharey='row')

    plot_execution_timeline(
        axs[:, 0],
        Path('/export2/kong102/energy_efficient_serving_results/request_timing/2025-05-13_simulate-autoscaling/A40_Llama-3.1-8B-Instruct_qps9.7_reqs6000_fixed1740'),
        t_start=300, t_end=540, plot_ylabels=True)
    plot_execution_timeline(
        axs[:, 1],
        Path('/export2/kong102/energy_efficient_serving_results/request_timing/2025-05-13_simulate-autoscaling/A40_Llama-3.1-8B-Instruct_qps9.0_reqs5000_fixed1740'),
        t_start=300, t_end=540, plot_ylabels=False)

    axs[0, 0].set_title('Slight under-provisioning')
    axs[0, 1].set_title('Heavy under-provisioning')
    axs[1, 0].set_ylim(22, 100)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    pdfcrop(output_path)


def plot_qps_variance_time_curve(csv_path,
                                 time_scales=[0.1, 0.2, 0.5, 1, 2, 5, 10, 100, 1000, 10000],
                                 output_path='figs/variance_time_plot.pdf'):
    """
    Reads LLM request arrival trace from a CSV and plots a normalized variance-time curve of QPS
    at specified time scales. Normalized variance is variance divided by mean squared.
    """
    # Load the trace
    df = pd.read_csv(csv_path)
    df = df.sort_values('arrived_at')

    # Get the total duration
    total_duration = df['arrived_at'].max() - df['arrived_at'].min()

    # Store normalized variances for each time scale
    normalized_variances = []

    for scale in time_scales:
        # Compute QPS at this scale
        bins = np.arange(0, total_duration + scale, scale)
        qps_counts, _ = np.histogram(df['arrived_at'], bins=bins)
        qps = qps_counts / scale

        # Compute normalized variance: var / mean^2
        mean_qps = np.mean(qps)
        if mean_qps > 0:
            normalized_var = np.var(qps) / (mean_qps ** 2)
        else:
            normalized_var = 0
        normalized_variances.append(normalized_var)

    # Plot log-log normalized variance-time curve
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    ax.plot(time_scales, normalized_variances, marker='o')

    ax.set_xscale('log')
    ax.set_ylim(0.65, 1.55)

    # Set X and Y to display 2 significant digits
    def two_sig_figs_plain(x, _):
        if x == 1000:
            return '1k'
        elif x == 10000:
            return '10k'
        elif x < 1e-2 or x >= 1e4:
            return f'{x:.0f}' if x >= 100 else f'{x:.2f}'.rstrip('0').rstrip('.')
        else:
            return f'{x:.2f}'.rstrip('0').rstrip('.')  # strip trailing .0 if not needed
    ax.xaxis.set_major_formatter(FuncFormatter(two_sig_figs_plain))
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)

    # Set X ticks at data point locations
    ax.set_xticks(time_scales)
    ax.tick_params(axis='x', rotation=45)

    ax.set_xlabel('Time scale (s)')
    ax.set_ylabel('Normalized\nVariance')

    fig.tight_layout()
    fig.savefig(output_path)
    pdfcrop(output_path)


def plot_clock_switch_latency_cdf(output_path=Path('figs/clock_switch_latency_cdf.pdf')):
    result_root = Path(
        '/export2/kong102/energy_efficient_serving_results/request_timing/2025-05-02_test-freq-apply-latency')

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))

    gpu_dir_linestyle = [
        ['T4', 't4_phi', '-'],
        ['A40', 'a40_llama', '--'],
        ['A100', 'a100-80gb_llama', '-.'],
        ['H100', 'h100_llama', ':'],
    ]
    for gpu, dir, linestyle in gpu_dir_linestyle:
        df = pd.read_csv(result_root / dir / 'freq_mod_log.csv')
        lat_arr = (df['freq_mod_end'] - df['freq_mod_start']).to_numpy() * 1000.0
        x, y = get_cdf_data(lat_arr)
        ax.plot(x, y, label=gpu, linestyle=linestyle)
    ax.set_xlabel('Clock switch latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    pdfcrop(output_path)


def plot_microscopic_study(output_path=Path('figs/microscopic_study.pdf')):
    expr_dir = Path(
        '/export2/kong102/energy_efficient_serving_results/request_timing/2025-05-13_microscopic-study/logs')
    df_perf_metric, df_power, df_freq_mod = load_logs(expr_dir)

    t_start = 1171300
    t_end = t_start + 3
    df_perf_metric = df_perf_metric[(df_perf_metric['now'] >= t_start - 1)
                                    & (df_perf_metric['now'] < t_end + 1)].reset_index(drop=True)
    df_power = df_power[(df_power['Timestamp'] >= t_start - 1) & (
        df_power['Timestamp'] < t_end + 1)].reset_index(drop=True)
    df_freq_mod = df_freq_mod[(df_freq_mod['now'] >= t_start - 1) & (
        df_freq_mod['now'] < t_end + 1)].reset_index(drop=True)
    df_perf_metric['tbt'] = df_perf_metric['now'].diff()

    fig, axs = plt.subplots(5, 1, figsize=(6, 5), sharex=True)

    def format_func(x_val, _):
        return f'{x_val - t_start:.2f}'
    for ax in axs:
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    seen_labels = set()

    # Plot 1: Inference batch timeline (Gantt)
    ax = axs[0]
    for idx, row in df_perf_metric.iterrows():
        start = row['pp_rank_0_start']
        end = row['pp_rank_0_idle']
        duration = end - start
        if row['num_prompt_tokens_iter'] == 0:
            color = 'tab:blue'  # decode-only
            label = 'Decode-only'
        else:
            color = 'tab:orange'  # hybrid
            label = 'Chunked-prefill'
        # Avoid duplicate labels in legend
        if label not in seen_labels:
            seen_labels.add(label)
        else:
            label = None
        ax.add_patch(patches.Rectangle((start, 0.05), duration, 0.9,
                     facecolor=color, edgecolor='black', label=label))

        # Add text to each triangle
        if start > t_start and end < t_end:
            text = str(row['num_tokens_iter'])
            ax.text(start + duration / 2, 0.3, text, ha='center', va='center', fontsize=10,
                    color='black', rotation='vertical')

    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', ncol=2)
    ax.set_xlim(t_start, t_end)
    ax.set_yticks([])
    ax.set_ylabel("Batches\n& tokens")

    # Plot 2: Frequency change intervals
    ax = axs[1]
    for idx, row in df_freq_mod.iterrows():
        mpc_start = row['mpc_start']
        mod_duration = row['freq_mod_end'] - mpc_start

        ax.add_patch(patches.Rectangle((mpc_start, 0.05), mod_duration, 0.9,
                                       facecolor='tab:green', edgecolor='black'))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("MPC &\nFreq Adj")

    # Plot 3: frequency
    ax = axs[2]
    ax.plot(df_power['Timestamp'], df_power['GPU_0_freq_mhz'])
    ax.set_ylabel('Freq\n(MHz)')
    ax.set_ylim(bottom=0)

    # Plot 4: power
    ax = axs[3]
    ax.plot(df_power['Timestamp'], df_power['GPU_0_power_w'])
    ax.set_ylabel('Power\n(W)')
    ax.set_ylim(bottom=0)

    # Plot 5: tbt
    ax = axs[4]
    ax.plot(df_perf_metric['now'], df_perf_metric['tbt'], marker='o')
    ax.set_ylabel('TBT (s)')
    ax.set_ylim((0, 0.27))
    ax.axhline(y=0.25, color='r', label='SLO', linestyle='--')
    ax.legend()

    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_path)
    pdfcrop(output_path)


def plot_latency_power_model_error(output_path=Path('figs/model_error.pdf')):
    fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
    _plot_latency_model_error(axs[0])
    _plot_power_model_error(axs[1])

    axs[0].set_ylabel('CDF')
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    pdfcrop(output_path)


def _plot_latency_model_error(ax):
    paths = [
        ['T4', Path(
            '/export2/obasit/EnergyEfficientServing/logs/Eurosys_all/T4/2025-05-07_Eurosys/T4_phi-2_qps1.0_reqs2000_mpc')],
        ['A40', Path('/export2/obasit/EnergyEfficientServing/logs/Eurosys_all/A40/A40Eurosys_logs/A40_Llama-3.1-8B-Instruct_qps5.66_reqs2000_mpc')],
        ['A100', Path('/export2/obasit/EnergyEfficientServing/logs/Eurosys_all/A100/2025-05-07_Eurosys.pdf/A100-SXM4-80GB_gemma-2-27b-it_qps1.8_reqs2000_mpc')],
        ['H100', Path('/export2/obasit/EnergyEfficientServing/logs/Eurosys_all/H100/2025-05-07_Eurosys/H100-80GB-HBM3_gemma-2-27b-it_qps2.5_reqs2000_mpc')],
        ['A100-TP4', Path('/export2/obasit/EnergyEfficientServing/logs/Eurosys_all/A100-TP4/2025-05-07_Eurosys/A100-SXM4-80GB_Llama-3.1-70B-Instruct_qps5.5_reqs2000_mpc')],
    ]
    for gpu, path in paths:
        color, linestyle = _gpu_to_color_and_style(gpu)
        df_perf_metric = load_logs_and_calc_difference(path)
        x, y = get_cdf_data(df_perf_metric['gpu_abs_rel'] * 100.0)
        ax.plot(x, y, label=gpu, c=color, linestyle=linestyle)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Latency Error (%)')


def _plot_power_model_error(ax):
    gpu_name_mapping = {
        'T4': 'T4_phi-2',
        'A40': 'A40_Llama-3.1-8B-Instruct',
        'A100': 'A100-SXM4-80GB_gemma-2-27b-it',
        'H100': 'H100-80GB-HBM3_gemma-2-27b-it',
        'A100-TP4': 'A100-SXM4-80GB_Llama-3.1-70B-Instruct',
    }
    for gpu, gpu_long in gpu_name_mapping.items():
        error_csv_path = Path('../../vidur/artifacts/power_model') / gpu_long / 'power_errors.csv'
        df_errors = pd.read_csv(error_csv_path)
        x, y = get_cdf_data(df_errors['Absolute Relative Error'] * 100.0)
        color, linestyle = _gpu_to_color_and_style(gpu)
        ax.plot(x, y, label=gpu, c=color, linestyle=linestyle)
    ax.set_xlim(0, 20)
    ax.set_xlabel('Power Error (%)')


def _gpu_to_color_and_style(gpu) -> tuple:
    return {
        'T4': ('C0', '--'),
        'A40': ('C1', '-'),
        'A100': ('C2', '-.'),
        'H100': ('C3', ':'),
        'A100-TP4': ('C4', 'dotted'),
    }[gpu]


def moving_average(a, win):
    return np.convolve(a, np.ones(win) / win, 'same')


def pdfcrop(filename):
    root, ext = os.path.splitext(filename)
    assert ext == '.pdf'
    os.system(f'pdfcrop {filename}')
    shutil.move(f'{root}-crop{ext}', filename)


if __name__ == '__main__':
    plot_qps_timelines()

    # plot_execution_timelines()

    # plot_qps_variance_time_curve(
    #     Path('/export2/kong102/energy_efficient_serving_results/datasets/processed/azure_2024_code_qps-default_req-cnt16803695.csv'))

    # plot_clock_switch_latency_cdf()

    # plot_microscopic_study()

    # plot_latency_power_model_error()
