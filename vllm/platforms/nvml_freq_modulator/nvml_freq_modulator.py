# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import gc
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from itertools import count, product
from multiprocessing import Process, SimpleQueue
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import msgspec
import numpy as np
import lightgbm as lgb
import pandas as pd
import pynvml

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.metrics.stats import IterationStats, NVMLFreqModulatorStats
from vllm.logger import init_logger
from vllm.platforms.nvml_utils import CSVWriter, get_gpu_name, get_preselected_freq
from vllm.utils import get_mp_context

logger = init_logger(__name__)

# Disable garbage collection for performance
gc.disable()

# Change this accordingly
PATH_TO_MODELS = Path(__file__).parent / 'tree_models'

# power model prefill
possible_freq = np.array([ 360,  570,  780, 1080, 1380, 1680, 1830])

class NvmlFreqModulatorInterface(ABC):

    @abstractmethod
    def step(self,
             scheduler_stats: Optional[NVMLFreqModulatorStats],) -> None:
        ...
    
    @abstractmethod
    def step_update_wait_q(self,
                             scheduler_stats: Optional[NVMLFreqModulatorStats],) -> None:
        ...

    def step_update_batch_ID_end(self,
                                 batch_ID: int,
                                 out_time: float) -> None:
        ...

    @abstractmethod
    def close(self):
        pass


class FreqModMsg(msgspec.Struct):
    """
    Msg from client to server.
    """
    now: float
    running_queue_tokens: list[int]     # tokens in batch just dispatched
    running_queue_pre_computed_tokens: list[int]  # precomputed tokens in batch just dispatched
    running_queue_wait_time: list[float]  # waiting times in running queue
    kv_cache_usage: float               # fraction of GPU memory used by KV cache
    waiting_queue_tokens: list[int]  # tokens in wait queue, for batch just dispatched
    waiting_queue_pre_computed_tokens: list[int]  # tokens in wait queue, for batch just dispatched (should be 0s)
    waiting_queue_wait_time: list[float]  # waiting times in waiting queue
    fromWho: str  # can be from scheduler or request_update or request_end
    batch_ID: int  # batch ID of the current batch

    def __post_init__(self):
        assert len(self.waiting_queue_tokens) == len(
            self.waiting_queue_wait_time)
        assert len(self.running_queue_tokens) == len(
            self.running_queue_wait_time)

@dataclass
class FutureState:
    """
    Represents the system state at a particular point in time.
    """
    num_prefills: int
    prefill_len_sum: int
    prefill_len_mean: float
    prefill_len_std: float
    num_decodes: int
    decode_len_sum: int
    decode_len_mean: float
    decode_len_std: float

@dataclass
class StateObservation:
    """
    Represents periodic state fingerprint for 
    handling latency models underprediction
    """
    timestamp: float
    running_len: int
    waiting_len: int
    running_top_req_tokens: int
    waiting_top_req_tokens: int
    running_top_computed_tokens: int


def nvml_freq_modulator(config: VllmConfig,
                                llm_engine) -> NvmlFreqModulatorInterface:
    '''
    Factory method to create an NvmlFreqModulator instance from a
    VllmConfig. Currently, always returns a RuleBasedNvmlFreqModulator.
    '''
    freq_choices = get_preselected_freq(get_gpu_name())

    return MPNvmlFreqModulatorClient(
        llm_engine,
        config,
        freq_choices=freq_choices,
        mod_interval=1,
        log_dir=Path(config.log_dir),
        tbt_sla=0.095,
        ttft_sla=0.54,
        optim_target='power',
    )


class MPNvmlFreqModulatorClient(NvmlFreqModulatorInterface):
    """
    Adjusts frequency in a separate process. Useful if the procedure of
    determining the frequency is computation heavy.
    """

    def __init__(
            self,
            llm_engine,
            vllm_config: VllmConfig,
            freq_choices: list[int],
            log_dir: Path,
            mod_interval: int = 1,
            future_window: int = 8,
            tbt_sla: float = 0.1,
            ttft_sla: float = 0.6,
            optim_target: str = 'power',  # 'energy' or 'power'factory
    ):
        self.llm_engine = llm_engine
        self.vllm_config = vllm_config

        self.token_budget = 100000 # big number for decode, we dont really care
        if vllm_config.scheduler_config.chunked_prefill_enabled:
            self.token_budget = vllm_config.scheduler_config.max_num_batched_tokens

        self.model = vllm_config.model_config.model

        self.q: SimpleQueue = get_mp_context().SimpleQueue()
        self.server = _MPNvmlFreqModulatorServer(vllm_config,
                                                 freq_choices,
                                                 self.q,
                                                 log_dir=log_dir,
                                                 mod_interval=mod_interval,
                                                 future_window=future_window,
                                                 tbt_sla=tbt_sla,
                                                 ttft_sla=ttft_sla,
                                                 optim_target=optim_target,
                                                 token_budget=self.token_budget,)
        self.server_process: Process = get_mp_context().Process(
            target=self.server.run)
        self.server_process.start()
        logger.info('_MPNvmlFreqModulatorServer process started.')

        self.stat_buffer = NVMLFreqModulatorStats()
        self.stat_buffer_lock = threading.Lock()

    def step(self,
             scheduler_stats: NVMLFreqModulatorStats) -> None:
        with self.stat_buffer_lock:
            self.stat_buffer = scheduler_stats
            msg = self.build_msg(scheduler_stats, fromWho="scheduler")
            msg_encoded = msgspec.msgpack.encode(msg)
            self.q.put(msg_encoded)

    def step_update_wait_q(self,
             scheduler_stats: NVMLFreqModulatorStats) -> None:
        if self.stat_buffer is None:
            return
        with self.stat_buffer_lock:
            time_elapsed = scheduler_stats.now - self.stat_buffer.now
            scheduler_stats.num_running_reqs = self.stat_buffer.num_running_reqs
            scheduler_stats.running_computed_tokens_list = self.stat_buffer.running_computed_tokens_list
            scheduler_stats.running_reqs_num_tokens = self.stat_buffer.running_reqs_num_tokens
            scheduler_stats.running_reqs_num_time = [x + time_elapsed for x in self.stat_buffer.running_reqs_num_time]
            scheduler_stats.batch_ID = self.stat_buffer.batch_ID
        msg = self.build_msg(scheduler_stats, fromWho="request_update")
        msg_encoded = msgspec.msgpack.encode(msg)
        self.q.put(msg_encoded)

    def step_update_batch_ID_end(self,
                                 batch_ID: int,
                                 out_time: float) -> None:
        msg = FreqModMsg(
            now = out_time,
            running_queue_tokens = [],
            running_queue_pre_computed_tokens = [],
            running_queue_wait_time = [],
            kv_cache_usage = 0.0,
            waiting_queue_tokens = [],
            waiting_queue_pre_computed_tokens = [],
            waiting_queue_wait_time = [],
            fromWho="request_end",
            batch_ID=batch_ID,
        )
        msg_encoded = msgspec.msgpack.encode(msg)
        self.q.put(msg_encoded)

    def close(self):
        self.q.put(None)
        self.server_process.join()
        logger.info('_MPNvmlFreqModulatorServer process terminated.')

    @staticmethod
    def build_msg(scheduler_stats: NVMLFreqModulatorStats, fromWho: str) -> FreqModMsg:
        return FreqModMsg(
            now = scheduler_stats.now,
            running_queue_tokens = scheduler_stats.running_reqs_num_tokens,
            running_queue_pre_computed_tokens = scheduler_stats.running_computed_tokens_list,
            running_queue_wait_time = scheduler_stats.running_reqs_num_time,
            kv_cache_usage = scheduler_stats.kv_cache_usage,
            waiting_queue_tokens = scheduler_stats.waiting_reqs_num_tokens,
            waiting_queue_pre_computed_tokens = scheduler_stats.waiting_computed_tokens_list,
            waiting_queue_wait_time = scheduler_stats.waiting_reqs_num_time,
            fromWho=fromWho,
            batch_ID=scheduler_stats.batch_ID,
        )

class _MPNvmlFreqModulatorServer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        freq_choices: list[int],
        q: SimpleQueue,
        log_dir: Path,
        optim_target: str,
        mod_interval: int,
        tbt_sla: float,
        ttft_sla: float,
        future_window: int = 4,
        mem_util_ceiling: float = 0.8,
        token_budget: int = 2048,
    ):
        self.vllm_config = vllm_config
        self.freq_choices = freq_choices
        self.q = q
        self.log_dir = log_dir

        self.future_windows = future_window
        self.mod_interval = mod_interval
        self.tbt_sla = tbt_sla
        self.ttft_sla = ttft_sla
        self.mem_util_ceiling = mem_util_ceiling
        self.optim_target = optim_target
        self.token_budget = token_budget

        self.model = vllm_config.model_config.model
        self.tp_degree = vllm_config.parallel_config.tensor_parallel_size

        model_name = vllm_config.model_config.model.split('/')[-1]
        combo_name = f'{get_gpu_name()}_{model_name}'

        self.power_model: lgb.Booster
        self.latency_model: lgb.Booster

        self.last_applied_freq: int = 1830

        self.underprediction_lock = None
        self.last_finished_ID: int = -1

        self._freq_daemon_queues = []
        self._freq_daemon_procs = []

        self.init_done = False
        self.csv_writer = None

        # ── Dynamic TBT SLA via EWMA feedback ──
        self.tbt_sla_max = tbt_sla          # original ceiling
        self.tbt_sla_min = 0.3 * tbt_sla    # floor = 30% of ceiling
        self.dynamic_tbt_sla = tbt_sla       # starts at max
        self._last_batch_end_time: Optional[float] = None
        # Asymmetric EWMA: tighten fast (~10 samples), relax slow (~200 samples)
        self._ewma_alpha_fast = 2.0 / (10.0 + 1.0)   # violation rate rising  → scale down SLA
        self._ewma_alpha_slow = 2.0 / (500.0 + 1.0)  # violation rate falling → scale up SLA
        self._ewma_violation_rate = 0.0       # EWMA of binary violations
        self._tbt_sample_count = 0            # warm-up counter


    def _load_models(self):
        if self.tp_degree == 8: 
            power_path = PATH_TO_MODELS / 'unified_power_model_tp8.txt'
            lat_path = PATH_TO_MODELS / 'unified_latency_model_tp8.txt'
        else:
            power_path = PATH_TO_MODELS / 'unified_power_model_tp2_tp4.txt'
            lat_path = PATH_TO_MODELS / 'unified_latency_model_tp2_tp4.txt'
        
        if power_path.exists():
            self.power_model = lgb.Booster(model_file=power_path)
        if lat_path.exists():
            self.latency_model = lgb.Booster(model_file=lat_path)
                    

    def run(self):
        if not self.init_done:
            # Explicitly disable GC in this process
            gc.disable()
            self.init_done = True
            self.underprediction_lock = threading.Lock()
            self._load_models()
            self.start_frequency_manager()
            self.csv_writer = CSVWriter(col_names=[
                'now', 'mpc_start', 'future_states_time', 'freq_mod_start', 'freq_mod_end',
                'target_freq', 'batch_lat', 'running_q_len', 'waiting_q_len',
                'max_running_q_wait', 'max_waiting_q_wait', 'fromWho',
                'dynamic_tbt_sla', 'ewma_violation_rate'
            ],
            filename=self.log_dir / 'freq_mod_log.csv')

        for step_id in count():
            msg_encoded = self.q.get()
            if msg_encoded is None:
                self.stop_frequency_manager()
                break
            if step_id % self.mod_interval > 0:
                continue
            
            mpc_start = time.time()
            msg: FreqModMsg = msgspec.msgpack.decode(msg_encoded,
                                                     type=FreqModMsg)
            # logger.debug('freq_mod_msg: %s', msg)
            if msg.fromWho == "request_end":
                with self.underprediction_lock:
                    if msg.batch_ID > self.last_finished_ID:
                        self.last_finished_ID = msg.batch_ID
                # ── Dynamic TBT SLA: track actual TBT from out_time diffs ──
                self._update_dynamic_tbt_sla(msg.now)
                continue

            future_states, prefill_cycles = self.get_future_states(
                msg, self.future_windows)
            
            future_states_time = time.time()
            selected_freq, pred_batch_lat = (
                self._get_next_freq(msg, future_states, prefill_cycles))
            
            if msg.kv_cache_usage >= self.mem_util_ceiling:
                selected_freq = max(self.freq_choices)


            freq_mod_start = time.time()
            with self.underprediction_lock:
                if self.last_applied_freq != selected_freq:
                    self.set_frequency_manager(selected_freq, msg.now)
                    self.last_applied_freq = selected_freq

            freq_mod_end = time.time()

            # decide later what to do with this
            timer_to_check_underpred = threading.Timer(float(pred_batch_lat+0.005),
                                                        self.check_underprediction, 
                                                        args=(msg.batch_ID,))
            timer_to_check_underpred.daemon = True
            timer_to_check_underpred.start()

            
            self.csv_writer.add_row([
                msg.now,
                mpc_start,
                future_states_time,
                freq_mod_start,
                freq_mod_end,
                self.last_applied_freq,
                pred_batch_lat,
                len(msg.running_queue_wait_time),
                len(msg.waiting_queue_wait_time),
                max(msg.running_queue_wait_time) if len(
                    msg.running_queue_wait_time) > 0 else 0.0,
                max(msg.waiting_queue_wait_time) if len(
                    msg.waiting_queue_wait_time) > 0 else 0.0,
                msg.fromWho,
                self.dynamic_tbt_sla,
                self._ewma_violation_rate,
            ])
        self.csv_writer.close()


    def _update_dynamic_tbt_sla(self, out_time: float):
        """
        Update the dynamic TBT SLA based on EWMA of observed TBT values.
        The diff of successive out_time gives the actual per-batch TBT.
        An EWMA violation rate drives the SLA between
        [tbt_sla_min, tbt_sla_max].
        """
        if self._last_batch_end_time is None:
            self._last_batch_end_time = out_time
            return

        actual_tbt = out_time - self._last_batch_end_time
        self._last_batch_end_time = out_time

        if actual_tbt <= 0:
            return  # ignore non-positive deltas (out-of-order, duplicates)

        self._tbt_sample_count += 1
        violation = 1.0 if actual_tbt > self.tbt_sla_max else 0.0

        if self._tbt_sample_count == 1:
            self._ewma_violation_rate = violation
        else:
            # Use fast alpha when violation rate is rising (tighten SLA faster),
            # slow alpha when it is falling (relax SLA slower).
            alpha = (self._ewma_alpha_fast
                     if violation > self._ewma_violation_rate
                     else self._ewma_alpha_slow)
            self._ewma_violation_rate = (
                alpha * violation + (1.0 - alpha) * self._ewma_violation_rate
            )

        # Map violation rate [0, 1] → SLA scaling [1.0, 0.5]
        # High violations → tighten SLA (lower); no violations → relax (higher)
        scale = 1.0 - 0.3 * self._ewma_violation_rate
        self.dynamic_tbt_sla = self.tbt_sla_max * scale
        # Clamp to [min, max] for safety
        self.dynamic_tbt_sla = max(self.tbt_sla_min,
                                   min(self.tbt_sla_max, self.dynamic_tbt_sla))

    def check_underprediction(self, ID: int,):
        """
        Check if the latency model underpredicted the actual latency.
        If yes, update the last known state for future corrections.
        """
        with self.underprediction_lock:
            if (self.last_finished_ID >= ID):
                # The running queue has changed since we set the timer.
                # Skip this check.
                return
            if self.last_applied_freq is not max(self.freq_choices):
                self.set_frequency_manager(max(self.freq_choices), time.time())
                self.last_applied_freq = max(self.freq_choices)
                logger.info('Underprediction detected, applied max freq')


    def _get_next_freq(self, freq_mod_msg: FreqModMsg,
                          future_states: list[FutureState], prefill_cycles):
        freq_choices_desc = sorted(copy.deepcopy(self.freq_choices),
                                   reverse=True)
        
        max_future_vision = self.future_windows
        future_states = future_states[:max_future_vision]
        # Pre-compute latency and power for each future window for each freq
        with ThreadPoolExecutor(max_workers=2) as executor:
            lat_future = executor.submit(self.predict_latencies_future_states, future_states, freq_choices_desc)
            power_future = executor.submit(self.predict_powers_future_states, future_states, freq_choices_desc)
            lat_mat = lat_future.result()
            power_mat = power_future.result()
        energy_mat = lat_mat * power_mat
        assert lat_mat.shape == (max_future_vision, len(freq_choices_desc))
        assert power_mat.shape == (max_future_vision, len(freq_choices_desc))

        # Build waiting time vector once (use numpy, ensure float32)
        # running_queue_wait_time contains decodes and then prefills. Don't care about decode wait times
        run_wait = np.asarray(freq_mod_msg.running_queue_wait_time[future_states[0].num_decodes:], dtype=np.float32)
        wait_wait = np.asarray(freq_mod_msg.waiting_queue_wait_time, dtype=np.float32)
        waiting_time_per_req = np.concatenate((run_wait, wait_wait), axis=0).reshape(-1, 1)
        
        # Start with the highest freq for each window
        selected_freq_ids = [0 for _ in range(max_future_vision)]
        for freq_idx in range(1, len(freq_choices_desc) - 1):
        # Collect the candidates from `selected_freqs`
            candidates_: list[list[int]] = [[]]
            for window_idx in range(max_future_vision):
                if selected_freq_ids[window_idx] == freq_idx - 1:
                    freq_ids_this_window = [freq_idx - 1, freq_idx, freq_idx + 1] 
                else:
                    freq_ids_this_window = [selected_freq_ids[window_idx]]
                candidates_ = [[
                    *c, f
                ] for c, f in product(candidates_, freq_ids_this_window)]
            candidates = np.array(candidates_)
            # [n_candidates, max_future_vision]
            assert candidates.shape[1] == max_future_vision

            # Keep candidates that meet SLA
            tbt_arr = lat_mat[np.arange(max_future_vision)[:, None],
                            candidates.T]
            # TBT mask
            sla_tbt_mask = np.all(tbt_arr <= self.dynamic_tbt_sla, axis=0)
            # Compute TTFT for all candidates in parallel
            time_till_finish_per_batch = np.cumsum(tbt_arr, axis=0)
            time_till_finish_per_req = time_till_finish_per_batch[
                np.array(prefill_cycles, dtype=int) - 1, :]
            ttft_arr = time_till_finish_per_req + waiting_time_per_req[: time_till_finish_per_req.shape[0], :]
            sla_ttft_mask = np.all(ttft_arr <= self.ttft_sla, axis=0)
            # Combine masks to filter valid candidates
            valid_mask = sla_tbt_mask & sla_ttft_mask
            candidates = candidates[valid_mask]
            # Select the min-energy candidate as `selected_freq_ids`
            if len(candidates) > 0:
                candidates = np.array(candidates)
                energy_per_batch = energy_mat[
                    np.arange(max_future_vision)[:, None], candidates.T]
                total_energy = np.sum(energy_per_batch, axis=0)
                if self.optim_target == 'energy':
                    selected_freq_ids = candidates[np.argmin(total_energy)]
                elif self.optim_target == 'power':
                    lat_per_batch = lat_mat[np.arange(max_future_vision)[:,
                                                                        None],
                                            candidates.T]
                    total_lat = np.sum(lat_per_batch, axis=0)
                    total_power = total_energy / total_lat
                    selected_freq_ids = candidates[np.argmin(total_power)]
            else:
                break
        # early freq increase if scheduler tells of new req arrivals
        if freq_mod_msg.fromWho == "request_update":
            selected_freq = max([
                freq_choices_desc[selected_freq_ids[i]]
                for i in range(2)
            ])
        else:
            selected_freq = max([
                freq_choices_desc[selected_freq_ids[i]]
                for i in range(self.mod_interval)
            ])
        predicted_batch_lat = lat_mat[0][selected_freq_ids[0]]

        return selected_freq, predicted_batch_lat


    def get_future_states(self, msg: FreqModMsg,
                          future_window: int) -> tuple[list, list]:
        """
        Get the future observation for the given index. The future observation
        is a list of observations for the next `future_windows` batches.
        Assumptions:
            - prefills are (poorly) chunked now
            - no decode requests reach EOS during future calculations
            - no new requests arrive
        """
        # A list that tells you for each request in the wait queue
        # how many iterations it will take to get the first token
        prefill_cycles = []

        

        # Construct a dummy wait queue to simulate future chunked prefills
        # list of (total tokens, processed tokens)
        dummy_wait_queue = []
        decodes = []
        for i in range(len(msg.running_queue_tokens)):
            total_tokens = msg.running_queue_tokens[i]
            processed_tokens = msg.running_queue_pre_computed_tokens[i]

            # separate prefills from decodes: if total tokens + 1 == processed_tokens, this is a decode
            if processed_tokens + 1 == total_tokens:
                # decode request, add directly to future states without chunking
                decodes.append(total_tokens)
            else:
                dummy_wait_queue.append((total_tokens, processed_tokens))


        # add the reqs in the wait queue, 
        dummy_wait_queue.extend([
            (m, n)
            for m, n in zip(msg.waiting_queue_tokens, msg.waiting_queue_pre_computed_tokens)
        ])

        future_states = []
        for i in range(future_window):
            budget_left = self.token_budget

            # decodes have priority over token budget
            num_decodes = len(decodes)
            if num_decodes > 0:
                decode_len_sum = np.sum(decodes).item()
                decode_len_mean = np.mean(decodes).item()
                decode_len_std = np.std(decodes).item()
            else:
                decode_len_sum = 0
                decode_len_mean = 0
                decode_len_std = 0.0

            budget_left -= num_decodes  # each decode progresses by 1 token per iteration
            decodes = [n + 1 for n in decodes]  # update decode lengths for future states

            prefills = []
            # fit prefills according to budget
            while budget_left > 0 and len(dummy_wait_queue) > 0:
                if i == 0 and len(dummy_wait_queue) <= len(msg.waiting_queue_tokens):
                    break  # in the first iteration, only consider the running queue, which is at the start of the dummy wait queue
                total_tokens, processed_tokens = dummy_wait_queue[0]
                num_tokens = min(budget_left, total_tokens - processed_tokens)

                budget_left -= num_tokens
                processed_tokens += num_tokens  # Update processed tokens

                # request completed without chunking
                if (processed_tokens == total_tokens):
                    # was never chunked
                    if dummy_wait_queue[0][1] == 0:
                        prefills.append(num_tokens)  # this req finishes in this future state, add total tokens to prefills
                    # was chunked
                    else:
                        prefills.append(num_tokens + int(0.05*dummy_wait_queue[0][1]))  # add small overhead for chunking, dont count towards chunking budget
                    
                    prefill_cycles.append(i + 1) # finished when the i-th iter finishes, so it takes i+1 cycles to finish
                    decodes.append(total_tokens + 1) # this req becomes a decode in the next future state, add to decodes
                    dummy_wait_queue.pop(0)

                # request chunked but not completed
                else: 
                    prefills.append(num_tokens + int(0.05*dummy_wait_queue[0][1])) # add overhead for chunking, dont count towards chunking budget
                    dummy_wait_queue[0] = (total_tokens, processed_tokens)  # Update tuple
                
                    
            # either chunking budget or reqs exhausted
            num_prefills = len(prefills)
            if num_prefills > 0:
                prefill_len_sum = np.sum(prefills).item()
                prefill_len_mean = np.mean(prefills).item()
                prefill_len_std = np.std(prefills).item()
            else:
                prefill_len_sum = 0
                prefill_len_mean = 0
                prefill_len_std = 0.0

            future_states.append(
                FutureState(
                    num_prefills,
                    prefill_len_sum,
                    prefill_len_mean,
                    prefill_len_std,
                    num_decodes,
                    decode_len_sum,
                    decode_len_mean,
                    decode_len_std,
                ))
        return future_states, prefill_cycles


    def predict_latencies_future_states(self, states: FutureState,
                          freq_choices) -> np.ndarray[np.ndarray[float]]:
        """
        Predict latency of the upcoming batch for each freq in `freq_choices`.
        Now expects `states` to be a list of FutureState and does a single
        batched inference for all (state, freq) pairs, returning a matrix
        shaped (n_states, n_freqs).
        """
        n_states = len(states)
        n_freqs = len(freq_choices)

        freq_arr = np.log1p(freq_choices, dtype=np.float32)
        prefill_bs_vec = np.log1p([st.num_prefills for st in states], dtype=np.float32)
        prefill_ils_vec = np.log1p([st.prefill_len_sum for st in states], dtype=np.float32)
        prefill_ilm_vec = np.log1p([st.prefill_len_mean for st in states], dtype=np.float32)
        prefill_ilsd_vec = np.log1p([st.prefill_len_std for st in states], dtype=np.float32)
        decode_bs_vec = np.log1p([st.num_decodes for st in states], dtype=np.float32)
        decode_ils_vec = np.log1p([st.decode_len_sum for st in states], dtype=np.float32)
        decode_ilm_vec = np.log1p([st.decode_len_mean for st in states], dtype=np.float32)
        decode_ilsd_vec = np.log1p([st.decode_len_std for st in states], dtype=np.float32)

        freqs = np.tile(freq_arr, n_states).astype(np.float32)
        batch_sizes = np.repeat(prefill_bs_vec, n_freqs).astype(np.float32)
        prefill_input_len_sums = np.repeat(prefill_ils_vec, n_freqs).astype(np.float32)
        prefill_input_len_means = np.repeat(prefill_ilm_vec, n_freqs).astype(np.float32)
        prefill_input_len_stds = np.repeat(prefill_ilsd_vec, n_freqs).astype(np.float32)
        decode_batch_sizes = np.repeat(decode_bs_vec, n_freqs).astype(np.float32)
        decode_input_len_sums = np.repeat(decode_ils_vec, n_freqs).astype(np.float32)
        decode_input_len_means = np.repeat(decode_ilm_vec, n_freqs).astype(np.float32)
        decode_input_len_stds = np.repeat(decode_ilsd_vec, n_freqs).astype(np.float32)
        tp_degrees = np.full(n_states * n_freqs, float(self.tp_degree), dtype=np.float32)

        # Build input feed for the latency model using numpy arrays directly
        input_feed = np.stack([
            batch_sizes,
            prefill_input_len_sums,
            prefill_input_len_means,
            prefill_input_len_stds,
            decode_batch_sizes,
            decode_input_len_sums,
            decode_input_len_means,
            decode_input_len_stds,
            tp_degrees,
            freqs,
        ], axis=1).astype(np.float32)

        out = self.latency_model.predict(input_feed)
        out = np.exp(out)
        out = np.clip(out, 0.005, None)
        latency_mat = out.reshape(n_states, n_freqs)
        return latency_mat


    def predict_powers_future_states(self, states: list[FutureState],
                                     freq_choices) -> np.ndarray[np.ndarray[float]]:
        """
        Predict power of the all batches for each freq in `freq_choices`.
        """
        n_states = len(states)
        n_freqs = len(freq_choices)

        freq_arr = np.log1p(freq_choices, dtype=np.float32)
        prefill_bs_vec = np.log1p([st.num_prefills for st in states], dtype=np.float32)
        prefill_ils_vec = np.log1p([st.prefill_len_sum for st in states], dtype=np.float32)
        prefill_ilm_vec = np.log1p([st.prefill_len_mean for st in states], dtype=np.float32)
        prefill_ilsd_vec = np.log1p([st.prefill_len_std for st in states], dtype=np.float32)
        decode_bs_vec = np.log1p([st.num_decodes for st in states], dtype=np.float32)
        decode_ils_vec = np.log1p([st.decode_len_sum for st in states], dtype=np.float32)
        decode_ilm_vec = np.log1p([st.decode_len_mean for st in states], dtype=np.float32)
        decode_ilsd_vec = np.log1p([st.decode_len_std for st in states], dtype=np.float32)

        freqs = np.tile(freq_arr, n_states).astype(np.float32)
        batch_sizes = np.repeat(prefill_bs_vec, n_freqs).astype(np.float32)
        prefill_input_len_sums = np.repeat(prefill_ils_vec, n_freqs).astype(np.float32)
        prefill_input_len_means = np.repeat(prefill_ilm_vec, n_freqs).astype(np.float32)
        prefill_input_len_stds = np.repeat(prefill_ilsd_vec, n_freqs).astype(np.float32)
        decode_batch_sizes = np.repeat(decode_bs_vec, n_freqs).astype(np.float32)
        decode_input_len_sums = np.repeat(decode_ils_vec, n_freqs).astype(np.float32)
        decode_input_len_means = np.repeat(decode_ilm_vec, n_freqs).astype(np.float32)
        decode_input_len_stds = np.repeat(decode_ilsd_vec, n_freqs).astype(np.float32)
        tp_degrees = np.full(n_states * n_freqs, float(self.tp_degree), dtype=np.float32)

        # Build input feed for the power model using numpy arrays directly
        input_feed = np.stack([
            batch_sizes,
            prefill_input_len_sums,
            prefill_input_len_means,
            prefill_input_len_stds,
            decode_batch_sizes,
            decode_input_len_sums,
            decode_input_len_means,
            decode_input_len_stds,
            tp_degrees,
            freqs,
        ], axis=1).astype(np.float32)

        out = self.power_model.predict(input_feed)
        # reshape back to (n_future_states, n_freq_choices)
        output_arr = np.asarray(out).clip(min=0.0).reshape(n_states, n_freqs)
        return output_arr


    def _persistent_gpu_worker(physical_gpu_index: int, queue: multiprocessing.Queue, log_dir: str):
        """
        Worker process that initializes NVML for a specific GPU and waits for
        frequency commands.

        Note: `self` is intentionally omitted so this function remains an
        unbound function object in the class dictionary; when referenced via
        `self.__class__._persistent_gpu_worker` it will be picklable by the
        spawn/forkserver start methods.
        """
        # Explicitly disable GC in this worker process
        gc.disable()
        
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_index)
            logger.info(f"Frequency daemon started for GPU index {physical_gpu_index}")
            csv_writer = CSVWriter(col_names=[
                'now',
                'freq_app_time',
                'target_freq',
                'skipped'
            ],
            filename=log_dir / f'freq_apply_log_{physical_gpu_index}.csv')
            prev_now = -float('inf')
            while True:
                # This blocks until a frequency is sent from the main process
                freq, now = queue.get()
                # Poison pill to stop the process
                if freq == -1 and now == -1:
                    break

                if not queue.empty():
                    qsize = queue.qsize()
                    for _skipped in range(qsize):
                        freq, now = queue.get()
                        csv_writer.add_row([
                            now, 
                            time.time(),
                            freq,
                            True  # skipped
                        ])
                        if freq == -1 and now == -1:
                            break

                if freq == -1 and now == -1:
                    break
                
                # Skip if now value is lower than previous now
                if now < prev_now:
                    continue
                
                prev_now = now
                    
                try:
                    # Apply the frequency
                    pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
                    csv_writer.add_row([
                        now, 
                        time.time(),
                        freq,
                        False  # not skipped
                    ])
                except pynvml.NVMLError as e:
                    logger.error(f"Daemon GPU {physical_gpu_index} failed to set freq {freq}: {e}")

        except Exception as e:
            logger.error(f"Crash in frequency daemon for GPU {physical_gpu_index}: {e}")
        finally:
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                pynvml.nvmlShutdown()
            except:
                pass
            csv_writer.close()


    def start_frequency_manager(self):
        """
        Spawns a separate process for each visible GPU. Each process waits
        indefinitely for frequency updates.
        """
        if self._freq_daemon_procs:
            logger.warning("Frequency manager already running.")
            return

        # Determine which physical GPUs we are using based on env var
        pynvml.nvmlInit()
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        if cuda_visible_devices:
            gpu_indices = [int(i) for i in cuda_visible_devices.split(',')]
        else:
            gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))

        ctx = get_mp_context()
        for physical_idx in gpu_indices:
            q = ctx.Queue()
            # Use the unbound function object from the class dict to avoid pickling the instance.
            p = ctx.Process(target=self.__class__._persistent_gpu_worker, args=(physical_idx, q, self.log_dir))
            p.start()
            
            self._freq_daemon_queues.append(q)
            self._freq_daemon_procs.append(p)
        logger.info(f"Started {len(self._freq_daemon_procs)} GPU frequency daemon processes.")


    def set_frequency_manager(self, freq: int, now: float):
        """
        Sends the new frequency to all waiting GPU processes.
        This function returns immediately after putting the freq in the queue.
        """
        if not self._freq_daemon_queues:
            logger.warning("Frequency manager not started. Call start_frequency_manager() first.")
            return

        for q in self._freq_daemon_queues:
            q.put((freq, now))


    def stop_frequency_manager(self):
        """
        Sends a stop signal (None) to all workers and joins the processes.
        """
        logger.info("Stopping frequency daemons...")
        
        for q in self._freq_daemon_queues:
            q.put((-1, -1))  # Poison pill

        for p in self._freq_daemon_procs:
            p.join()

        self._freq_daemon_queues = []
        self._freq_daemon_procs = []
        logger.info("Frequency daemons stopped.")

if __name__ == '__main__':
    q: SimpleQueue = SimpleQueue()
    from vllm.config import ParallelConfig, SchedulerConfig
    from vllm.config.kv_transfer import KVTransferConfig
    
    vllm_config = VllmConfig()
    vllm_config.kv_transfer_config = KVTransferConfig()
    vllm_config.model_config = ModelConfig(
        model='meta-llama/Llama-3.3-70B-Instruct',
        # Assign arbitrary values to remaining mandatory params
        task='draft',
        tokenizer='',
        tokenizer_mode='auto',
        trust_remote_code=False,
        dtype='float32',
        seed=0,
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.parallel_config.tensor_parallel_size = 4
    vllm_config.scheduler_config = SchedulerConfig()
    vllm_config.log_dir = './logs'
    
    freq_choices = get_preselected_freq(get_gpu_name())
    s = _MPNvmlFreqModulatorServer(
        vllm_config=vllm_config,
        freq_choices=freq_choices,
        q=q,
        log_dir=Path('./logs'),
        optim_target='power',
        mod_interval=1,
        tbt_sla=0.1,
        ttft_sla=0.6,
        future_window=8,
        token_budget=2048,
    )
    msg = [
        FreqModMsg(
            now=0.0,
            running_queue_tokens=[1024, 512],
            running_queue_pre_computed_tokens=[1023, 0],
            running_queue_wait_time=[0.02, 0.01,],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[500, ],
            waiting_queue_pre_computed_tokens=[0,],
            waiting_queue_wait_time=[0.10,],
            fromWho='scheduler',
            batch_ID=0,
        ),
        FreqModMsg(
            now=0.0,
            running_queue_tokens=[1024, 512],
            running_queue_pre_computed_tokens=[1023, 0],
            running_queue_wait_time=[0.02, 0.01,],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[500, 500 ],
            waiting_queue_pre_computed_tokens=[0, 0],
            waiting_queue_wait_time=[0.12, 0.02],
            fromWho='scheduler',
            batch_ID=1,
        ),
        FreqModMsg(
            now=0.0,
            running_queue_tokens=[1024, 512],
            running_queue_pre_computed_tokens=[1023, 0],
            running_queue_wait_time=[0.02, 0.01],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[500, 500],
            waiting_queue_pre_computed_tokens=[0, 0,],
            waiting_queue_wait_time=[0.12, 0.02,],
            fromWho='request_update',
            batch_ID=2,
        ),
        FreqModMsg(
            now=0.0,
            running_queue_tokens=[1024, 512],
            running_queue_pre_computed_tokens=[1023, 0],
            running_queue_wait_time=[0.02, 0.01,],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[1200, 100],
            waiting_queue_pre_computed_tokens=[0, 0],
            waiting_queue_wait_time=[0.10, 0.02],
            fromWho='scheduler',
            batch_ID=3,
        ),
        FreqModMsg(
            now=0.0,
            running_queue_tokens=[200, 512],
            running_queue_pre_computed_tokens=[0, 0],
            running_queue_wait_time=[0.001, 0.002],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[1200],
            waiting_queue_pre_computed_tokens=[0],
            waiting_queue_wait_time=[0.15],
            fromWho='scheduler',
            batch_ID=4,
        ),
        FreqModMsg(
            now=0.0,
            running_queue_tokens=[16],
            running_queue_pre_computed_tokens=[0],
            running_queue_wait_time=[0.001],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[],
            waiting_queue_pre_computed_tokens=[],
            waiting_queue_wait_time=[],
            fromWho='scheduler',
            batch_ID=5,
        ),
        FreqModMsg(
            now=0.0,
            running_queue_tokens=[1024, 512],
            running_queue_pre_computed_tokens=[520, 0],
            running_queue_wait_time=[0.01, 0.02],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[1200, 1200, 1200, 777, 666, 555, 888],
            waiting_queue_pre_computed_tokens=[0, 0, 0, 0, 0, 0, 0],
            waiting_queue_wait_time=[0.11, 0.12, 0.13, 0.13, 0.14, 0.157, 0.20],
            fromWho='scheduler',
            batch_ID=6,
        ),
        ]
    for i in range(len(msg)):
        q.put(msgspec.msgpack.encode(msg[i]))
    s.run()

