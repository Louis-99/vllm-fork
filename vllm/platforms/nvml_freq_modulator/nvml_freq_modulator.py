# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
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
import onnxruntime as ort
from scipy.interpolate import interpn
import pynvml

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.metrics.stats import IterationStats, NVMLFreqModulatorStats
from vllm.logger import init_logger
from vllm.platforms.nvml_utils import CSVWriter, get_gpu_name, get_preselected_freq
from vllm.utils import get_mp_context

logger = init_logger(__name__)

# Change this accordingly
PATH_TO_MODELS = Path(__file__).parent / 'tree_models'

# power model prefill
possible_freq = np.array([ 360,  570,  780, 1080, 1380, 1680, 1830])
possible_input_len = np.array(
    [32,   64,   96,  128,  160,  192,  256,  384,  512,  768, 1024, 1280, 1536, 1792, 2048])

busy_power_values_dict = {
    4: np.array([
        [ 766.79782832,  818.96870802,  865.25887505,  984.72127171, 1140.23307683, 1396.33445554, 1385.88617877],
        [ 814.60153684,  876.97261754,  924.91678831, 1082.53452469, 1286.32423938, 1549.65300951, 1563.94855435],
        [ 833.32759634,  902.68569682,  954.73581584, 1140.63229992, 1390.3780871 , 1680.76153648, 1734.70052576],
        [ 860.67616663,  960.86158035, 1018.43250191, 1221.78283476, 1543.07507994, 1880.22343364, 1942.6190668 ],
        [ 831.1066616 ,  962.64835685, 1021.0607414 , 1224.78029167, 1607.07943643, 1927.45478727, 2127.5052573 ],
        [ 847.49345326, 1002.65630459, 1084.28640911, 1291.95411139, 1742.84315608, 2198.67549647, 2404.429592  ],
        [ 868.87835047, 1036.20243426, 1143.74980904, 1367.73381494, 1882.58995999, 2417.12518785, 2634.21850238],
        [ 864.77862948, 1043.89451998, 1181.99973971, 1429.2342348 , 1956.98430597, 2582.52452   , 2703.46718597],
        [ 779.6524155 ,  956.80138795, 1106.30663166, 1350.266084  , 1895.1023916 , 2612.01642318, 2745.32649567],
        [ 814.26393778, 1005.21900045, 1180.96507466, 1481.40437675, 2063.47833033, 2676.77416513, 2702.09631725],
        [ 828.05580063, 1032.6387077 , 1217.45493742, 1520.95248079, 2177.22006411, 2759.28575851, 2759.6566107 ],
        [ 836.89135666, 1034.72175891, 1193.05040754, 1509.91219388, 2172.56049632, 2745.16476285, 2749.93814766],
        [ 809.83990546, 1005.36396346, 1180.79333461, 1488.51255534, 2203.16329111, 2766.10493464, 2769.15640533],
        [ 820.17031014, 1017.70414403, 1261.75169216, 1606.47416798, 2322.08879116, 2761.67362614, 2765.07185752],
        [ 827.67340895, 1029.70857166, 1235.87224067, 1559.00235443, 2283.79591686, 2776.05462126, 2775.78978486],
    ]),
    2: np.array([
        [ 467.95701759,  496.55788652,  504.8714655 ,  644.05910082,  788.11071287,  964.64911905,  996.81165047],
        [ 488.153836  ,  522.63585906,  538.43388779,  679.8367838 ,  848.52178567, 1058.02993614, 1115.76423859],
        [ 501.14422157,  541.09192822,  558.62907545,  706.88651643,  891.60417424, 1123.97111969, 1218.24403351],
        [ 528.47890244,  592.96161024,  610.07134923,  787.30547876, 1020.72949344, 1280.57615433, 1376.44100852],
        [ 485.13935138,  590.1152666 ,  612.87938317,  786.45673293, 1023.93927892, 1311.40308726, 1392.21389419],
        [ 517.1179014 ,  629.16407227,  657.19657127,  834.17186602, 1141.80856109, 1383.5541608 , 1385.80419169],
        [ 478.86860659,  598.14209491,  664.47926456,  828.20318831, 1160.58184678, 1384.09975325, 1386.41520802],
        [ 486.30697887,  615.57433168,  684.86457528,  862.61612102, 1236.03113618, 1384.74473118, 1385.64871904],
        [ 464.11729818,  589.60919617,  676.96543248,  856.05726866, 1240.74766892, 1379.47374536, 1383.89076739],
        [ 443.4393147 ,  562.81395845,  646.56773068,  847.77476953, 1195.56088575, 1386.44341108, 1388.97561606],
        [ 451.4055709 ,  575.64271474,  665.42824697,  880.82823165, 1249.25048523, 1389.56332899, 1386.52722456],
        [ 456.53185123,  580.18164108,  657.98246341,  869.99551703, 1257.0508306 , 1392.58820345, 1391.76599695],
        [ 455.30556431,  580.04494506,  631.10018873,  826.83978314, 1203.5316831 , 1393.6138576 , 1387.72965146],
        [ 446.82092724,  568.29904263,  696.76195988,  926.3252101 , 1336.88877153, 1393.38139544, 1391.86912304],
        [ 450.95068799,  574.6552041 ,  682.11812234,  906.90752387, 1317.28533085, 1392.48822538, 1391.85159044],
    ]),
}
idle_power_values_dict = {
    4: np.array([316.850818757042, 341.6161000279436, 351.3068106255578, 385.13777390054054, 436.57400559323213, 555.108677099649, 618.3753703198889]),
    2: np.array([161.39579638960902, 171.89891114741704, 166.3517919412156, 183.0507797413239, 210.4289325309798, 300.1692712489896, 330.3604991503286]),
}

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
                                 batch_ID: int) -> None:
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

        self.engine_role = 'decode'
        self.token_budget = 100000 # big number for decode, we dont really care
        if vllm_config.kv_transfer_config and vllm_config.kv_transfer_config.is_kv_producer:
            self.engine_role = 'prefill'
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
                                                 engine_role=self.engine_role,
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
                                 batch_ID: int) -> None:
        msg = FreqModMsg(
            now = time.time(),
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
        engine_role: str = 'prefill',
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
        self.engine_role = engine_role
        self.token_budget = token_budget

        self.model = vllm_config.model_config.model
        self.tp_degree = vllm_config.parallel_config.tensor_parallel_size

        model_name = vllm_config.model_config.model.split('/')[-1]
        combo_name = f'{get_gpu_name()}_{model_name}'
        self.latency_model_dir = (PATH_TO_MODELS / 'latency_model' /
                                  combo_name)
        self.power_model_dir = (PATH_TO_MODELS / 'power_model' / combo_name)

        self.power_model_prefill: ort.InferenceSession
        self.power_model_decode: ort.InferenceSession
        self.latency_model_prefill: ort.InferenceSession
        self.latency_model_decode: ort.InferenceSession

        self.last_applied_freq: int = 2000

        self.underprediction_lock = None
        self.last_finished_ID: int = -1

        self._freq_daemon_queues = []
        self._freq_daemon_procs = []

        self.init_done = False
        self.csv_writer = None


    def _load_models(self):
        dec = None
        pre = None
        dec_path = PATH_TO_MODELS / "decode_model_latency.onnx"
        pre_path = PATH_TO_MODELS / "prefill_model_latency.onnx"
        power_dec_path = PATH_TO_MODELS / "decode_model_power.onnx"
        if dec_path.exists():
            self.latency_model_decode = ort.InferenceSession(dec_path)
        if pre_path.exists():
            self.latency_model_prefill = ort.InferenceSession(pre_path)
        if power_dec_path.exists():
            self.power_model_decode = ort.InferenceSession(power_dec_path)             

    def run(self):
        if not self.init_done:
            self.init_done = True
            self.underprediction_lock = threading.Lock()
            self._load_models()
            self.start_frequency_manager()
            self.csv_writer = CSVWriter(col_names=[
                'now', 'mpc_start', 'freq_mod_start', 'freq_mod_end',
                'target_freq', 'batch_lat', 'running_q_len', 'waiting_q_len',
                'max_running_q_wait', 'max_waiting_q_wait',
            ],
            filename=self.log_dir / 'freq_mod_log.csv')

        for step_id in count():
            msg_encoded = self.q.get()
            if msg_encoded is None:
                self.stop_frequency_manager()
                break
            if step_id % self.mod_interval > 0:
                continue
            
            mpc_start = time.perf_counter()
            msg: FreqModMsg = msgspec.msgpack.decode(msg_encoded,
                                                     type=FreqModMsg)
            # logger.info('freq_mod_msg: %s', msg)
            logger.debug('freq_mod_msg: %s', msg)
            if msg.fromWho == "request_end":
                with self.underprediction_lock:
                    if msg.batch_ID > self.last_finished_ID:
                        self.last_finished_ID = msg.batch_ID
                continue

            future_states, prefill_cycles = self.get_future_states(
                msg, self.future_windows)

            selected_freq, pred_batch_lat = (
                self._get_next_freq(msg, future_states, prefill_cycles))

            if msg.kv_cache_usage >= self.mem_util_ceiling:
                selected_freq = max(self.freq_choices)

            freq_mod_start = time.perf_counter()
            with self.underprediction_lock:
                if self.last_applied_freq != selected_freq:
                    self.set_frequency_manager(selected_freq, msg.now)
                    self.last_applied_freq = selected_freq

            freq_mod_end = time.perf_counter()
            if self.engine_role == 'prefill':
                timer_to_check_underpred = threading.Timer(float(pred_batch_lat+0.005),
                                                           self.check_underprediction, 
                                                           args=(msg.batch_ID,))
                timer_to_check_underpred.daemon = True
                timer_to_check_underpred.start()

            
            self.csv_writer.add_row([
                msg.now,
                mpc_start,
                freq_mod_start,
                freq_mod_end,
                self.last_applied_freq,
                pred_batch_lat if self.last_applied_freq == selected_freq else pred_batch_lat+10.0,
                len(msg.running_queue_wait_time),
                len(msg.waiting_queue_wait_time),
                max(msg.running_queue_wait_time) if len(
                    msg.running_queue_wait_time) > 0 else 0.0,
                max(msg.waiting_queue_wait_time) if len(
                    msg.waiting_queue_wait_time) > 0 else 0.0,
            ])
            self.csv_writer.close()

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
        
        max_future_vision = self.future_windows if max(prefill_cycles) > self.future_windows else max(prefill_cycles)
        future_states = future_states[:max_future_vision]
        # Pre-compute latency and power for each future window for each freq
        lat_mat = self.predict_latencies_future_states(future_states,
                                    freq_choices_desc)
        power_mat = self.predict_powers_future_states(
            future_states, freq_choices_desc)
        energy_mat = lat_mat * power_mat
        assert lat_mat.shape == (max_future_vision, len(freq_choices_desc))
        assert power_mat.shape == (max_future_vision, len(freq_choices_desc))

        if self.engine_role == 'decode':
            # only consider the one future window for decode
            lat_mat = lat_mat[0:1, :]
            lat_mask = lat_mat <= self.tbt_sla
            if self.optim_target == 'power':
                power_mat = np.where(lat_mask, power_mat[0:1, :], np.inf)
                selected_freq_idx = [np.argmin(power_mat[0])]
            else:
                energy_mat = np.where(lat_mask, energy_mat[0:1, :], np.inf)
                selected_freq_idx = [np.argmin(energy_mat[0])]
            selected_freq = freq_choices_desc[selected_freq_idx[0]]
            predicted_batch_lat = lat_mat[0][selected_freq_idx[0]]
        else:
            # Build waiting time vector once (use numpy, ensure float32)
            run_wait = np.asarray(freq_mod_msg.running_queue_wait_time, dtype=np.float32)
            wait_wait = np.asarray(freq_mod_msg.waiting_queue_wait_time, dtype=np.float32)
            waiting_time_per_req = np.concatenate((run_wait, wait_wait), axis=0).reshape(-1, 1)
            # Start with the highest freq for each window
            selected_freq_ids = [0 for _ in range(self.future_windows)]
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
                # Compute TTFT for all candidates in parallel
                time_till_finish_per_batch = np.cumsum(tbt_arr, axis=0)
                time_till_finish_per_req = time_till_finish_per_batch[
                    np.array(prefill_cycles, dtype=int) - 1, :]
                ttft_arr = time_till_finish_per_req + waiting_time_per_req[: time_till_finish_per_req.shape[0], :]
                sla_ttft_mask = np.all(ttft_arr <= self.ttft_sla, axis=0)
                # Combine masks to filter valid candidates
                valid_mask = sla_ttft_mask
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
        # list of (total tokens, processed tokens, remaining tokens)
        dummy_wait_queue = []
        for i in range(len(msg.running_queue_tokens)):
            total_tokens = msg.running_queue_tokens[i]
            processed_tokens = msg.running_queue_pre_computed_tokens[i]
            remaining_tokens = (total_tokens+1) - processed_tokens
            if remaining_tokens > 0:
                dummy_wait_queue.append(
                    (total_tokens+1, processed_tokens, remaining_tokens))
                    # +1 because prefill generates the first token

        # add the reqs in the wait queue, 
        dummy_wait_queue.extend([
            (m+1, n, m - n)
            for m, n in zip(msg.waiting_queue_tokens, msg.waiting_queue_pre_computed_tokens)
        ])

        num_decodes = len(msg.running_queue_tokens)
        if num_decodes > 0:
            decode_len_sum = sum(msg.running_queue_tokens)
            decode_len_mean = np.mean(msg.running_queue_tokens).item()
            decode_len_std = np.std(
                msg.running_queue_tokens).item()
        else:
            decode_len_sum = 0
            decode_len_mean = 0
            decode_len_std = 0

        future_states = []
        extracted_reqs = 0
        for i in range(future_window):
            budget_left = self.token_budget
            prefills = []

            while budget_left > 0 and len(dummy_wait_queue) > 0:
                # only extract running queue reqs from the first cycle
                if i == 0 and extracted_reqs >= len(msg.running_queue_tokens):
                    break
                num_tokens = min(budget_left, dummy_wait_queue[0][2])
                # add small overhead for chunking, dont count towards chunking budget
                prefills.append(num_tokens + int(0.05*dummy_wait_queue[0][1]))

                total_tokens, \
                processed_tokens, \
                remaining_tokens = dummy_wait_queue[0]
                budget_left -= num_tokens
                processed_tokens += num_tokens  # Update processed tokens
                remaining_tokens -= num_tokens  # Update remaining tokens
                dummy_wait_queue[0] = (total_tokens, processed_tokens,
                                       remaining_tokens)  # Update tuple

                if dummy_wait_queue[0][2] == 0:
                    dummy_wait_queue.pop(0)
                    prefill_cycles.append(i + 1)
                extracted_reqs += 1
                    
            num_prefills = len(prefills)
            if num_prefills > 0:
                prefill_len_sum = np.sum(prefills).item()
                prefill_len_mean = np.mean(prefills).item()
                prefill_len_std = np.std(prefills).item()
            else:
                prefill_len_sum = 0
                prefill_len_mean = 0
                prefill_len_std = 0.0

            # then decode
            decode_len_sum += num_decodes   # all decode requests progress by 1
            decode_len_mean = decode_len_mean + 1 # mean increases by 1
            decode_len_std = decode_len_std # since everything increases by 1

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

        # Collect column-wise inputs for a single batched inference over
        # all state x freq combinations
        freqs = []
        batch_sizes = []
        input_len_sums = []
        input_len_means = []
        input_len_stds = []
        tp_degrees = []

        n_states = len(states)
        n_freqs = len(freq_choices)

        freq_arr = np.array(freq_choices, dtype=np.float32)
        if self.engine_role == 'prefill':
            bs_vec = np.array([st.num_prefills for st in states], dtype=np.float32)
            ils_vec = np.array([st.prefill_len_sum for st in states], dtype=np.float32)
            ilm_vec = np.array([st.prefill_len_mean for st in states], dtype=np.float32)
            ilsd_vec = np.array([st.prefill_len_std for st in states], dtype=np.float32)
        else:
            bs_vec = np.array([st.num_decodes for st in states], dtype=np.float32)
            ils_vec = np.array([st.decode_len_sum for st in states], dtype=np.float32)
            ilm_vec = np.array([st.decode_len_mean for st in states], dtype=np.float32)
            ilsd_vec = np.array([st.decode_len_std for st in states], dtype=np.float32)

        freqs = np.tile(freq_arr, n_states).astype(np.float32)
        batch_sizes = np.repeat(bs_vec, n_freqs).astype(np.float32)
        input_len_sums = np.repeat(ils_vec, n_freqs).astype(np.float32)
        input_len_means = np.repeat(ilm_vec, n_freqs).astype(np.float32)
        input_len_stds = np.repeat(ilsd_vec, n_freqs).astype(np.float32)
        tp_degrees = np.full(n_states * n_freqs, float(self.tp_degree), dtype=np.float32)

        # Keep numpy arrays; avoid converting to Python lists and back.
        n_rows = n_states * n_freqs

        # Select appropriate latency model for inference
        latency_model = (self.latency_model_prefill
                 if self.engine_role == 'prefill'
                 else self.latency_model_decode)

        # Build input feed for the latency model using numpy arrays directly
        # Build model column using np.full to avoid intermediate Python list
        model_col = np.full((n_rows, 1), self.model, dtype=str)
        input_feed = {
            "model": model_col,
            "batch_size": batch_sizes.reshape(n_rows, 1).astype(np.float32),
            "input_len_sum": input_len_sums.reshape(n_rows, 1).astype(np.float32),
            "input_len_mean": input_len_means.reshape(n_rows, 1).astype(np.float32),
            "input_len_std": input_len_stds.reshape(n_rows, 1).astype(np.float32),
            "tp_degree": tp_degrees.reshape(n_rows, 1).astype(np.float32),
            "freq_mhz": freqs.reshape(n_rows, 1).astype(np.float32),
        }

        out = latency_model.run(None, input_feed)[0]

        # Mark rows with batch_size == 0 so we can override model output after inference.
        batch_size_col = batch_sizes.reshape(n_rows)
        zero_mask = batch_size_col == 0
        out[zero_mask] = 0.005
        
        out = np.asarray(out)
        latency_mat = out.reshape(n_states, n_freqs)
        return latency_mat

    def predict_powers_future_states(self, future_states: list[FutureState],
                                     freq_choices) -> np.ndarray[np.ndarray[float]]:
        """
        Predict power of the all batches for each freq in `freq_choices`.
        """
        inputs = []
        for future_state in future_states:

            num_prefills = future_state.num_prefills
            prefill_len_sum = future_state.prefill_len_sum
            prefill_len_std = future_state.prefill_len_std
            prefill_len_mean = future_state.prefill_len_mean
            num_decodes = future_state.num_decodes
            decode_len_sum = future_state.decode_len_sum
            decode_len_std = future_state.decode_len_std
            decode_len_mean = future_state.decode_len_mean
            
            input = np.array([
                num_prefills,
                prefill_len_sum,
                prefill_len_mean,
                prefill_len_std,
                num_decodes,
                decode_len_sum,
                decode_len_mean,
                decode_len_std,
                self.tp_degree,
            ], dtype=np.float32)
            input = np.hstack([
                np.array(freq_choices).reshape(-1, 1),
                np.tile(input, (len(freq_choices), 1)),
            ])
            inputs.append(input)

        #prefill
        if self.engine_role == 'prefill':
            freqs = np.array(freq_choices, dtype=np.float32)
            n_future = len(inputs)
            n_freqs = len(freqs)

            # Determine which future states have batch_size == 0 (num_prefills == col 1)
            batch_sizes = np.array([float(inp[0, 1]) for inp in inputs], dtype=np.float32)
            idle_mask = batch_sizes == 0
            busy_mask = ~idle_mask

            output_arr = np.empty((n_future, n_freqs), dtype=np.float32)

            # Idle rows: directly lookup idle power table (map freq_choices -> nearest possible_freq)
            if np.any(idle_mask):
                # For each freq choice, pick nearest index in possible_freq
                freq_idx = np.argmin(np.abs(possible_freq.reshape(1, -1) - freqs.reshape(-1, 1)), axis=1)
                idle_vals_for_freqs = idle_power_values_dict[self.tp_degree][freq_idx]
                # Fill all idle future-state rows with the same idle vector
                idle_rows = np.where(idle_mask)[0]
                for r in idle_rows:
                    output_arr[r, :] = idle_vals_for_freqs

            # Busy rows: do interpolation like before, then place results back preserving original order
            if np.any(busy_mask):
                busy_indices = np.where(busy_mask)[0]
                busy_inputs = [inputs[i] for i in busy_indices]

                # Clamp and collect input_len (prefill_len_sum is column 2)
                input_lens = np.array([
                    np.clip(float(inp[0, 2]), possible_input_len.min(), possible_input_len.max())
                    for inp in busy_inputs
                ], dtype=np.float32)

                # Shape will be (n_busy, n_freqs)
                input_grid, freq_grid = np.meshgrid(input_lens, freqs, indexing='ij')
                xi_all = np.stack([input_grid.ravel(), freq_grid.ravel()], axis=-1)

                values = busy_power_values_dict[self.tp_degree]
                out_all = interpn(
                    points=(possible_input_len, possible_freq),
                    values=values,
                    xi=xi_all,
                    method='linear',
                    bounds_error=False,
                    fill_value=None,
                )
                busy_out = np.asarray(out_all, dtype=np.float32).reshape(len(busy_inputs), n_freqs)

                # Place back into output_arr according to original order
                for idx, row in enumerate(busy_indices):
                    output_arr[row, :] = busy_out[idx, :]

        else:
            # decode: batch-predict power for all future-state inputs at once
            power_model = self.power_model_decode
            
            # stack inputs: shape (n_future_states * n_freqs, n_cols)
            stacked = np.vstack(inputs).astype(np.float32)
            n_rows = stacked.shape[0]
            n_future = len(inputs)
            n_freqs = len(freq_choices)

            # Columns in stacked:
            # 0: freq, 1: num_prefills, 2: prefill_len_sum, 3: prefill_len_mean,
            # 4: prefill_len_std, 5: num_decodes, 6: decode_len_sum,
            # 7: decode_len_mean, 8: decode_len_std, 9: tp_degree
            model_col = np.full((n_rows, 1), self.model, dtype=str)
            input_feed = {
                "model": model_col,
                "batch_size": stacked[:, 5].reshape(n_rows, 1).astype(np.float32),
                "input_len_sum": stacked[:, 6].reshape(n_rows, 1).astype(np.float32),
                "input_len_mean": stacked[:, 7].reshape(n_rows, 1).astype(np.float32),
                "input_len_std": stacked[:, 8].reshape(n_rows, 1).astype(np.float32),
                "tp_degree": stacked[:, 9].reshape(n_rows, 1).astype(np.float32),
                "freq_mhz": stacked[:, 0].reshape(n_rows, 1).astype(np.float32),
            }

            out = power_model.run(None, input_feed)[0]
            out = np.asarray(out).reshape(-1)
            # reshape back to (n_future_states, n_freq_choices)
            output_arr = out.reshape(n_future, n_freqs)
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
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_index)
            logger.info(f"Frequency daemon started for GPU index {physical_gpu_index}")
            csv_writer = CSVWriter(col_names=[
                'now',
                'freq_app_time',
                'target_freq'
            ],
            filename=log_dir / f'freq_apply_log_{physical_gpu_index}.csv')
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
                        if freq == -1 and now == -1:
                            break

                if freq == -1 and now == -1:
                    break
                    
                try:
                    # Apply the frequency
                    pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
                    csv_writer.add_row([
                        now, 
                        time.time(),
                        freq,
                    ])
                    csv_writer.close()
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
    vllm_config = VllmConfig()
    from vllm.config.kv_transfer import KVTransferConfig
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
    vllm_config.parallel_config.tensor_parallel_size = 4
    freq_choices = get_preselected_freq(get_gpu_name())
    s = _MPNvmlFreqModulatorServer(freq_choices=freq_choices,
                                   vllm_config=vllm_config,
                                   q=q,
                                   log_dir=Path('./logs'),
                                   optim_target='power',
                                   mod_interval=1,
                                   future_window=8,
                                   engine_role='prefill',    
                                   tbt_sla=0.1,
                                   ttft_sla=0.6,
                                   token_budget=2048,
                                   )
    msg = [
        FreqModMsg(
            now=0.0,
            running_queue_tokens=[1024, 512, 256],
            running_queue_pre_computed_tokens=[520, 0, 0],
            running_queue_wait_time=[0.01, 0.02, 0.02],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[1200, 1200, 1200],
            waiting_queue_pre_computed_tokens=[0, 0, 0],
            waiting_queue_wait_time=[0.15, 0.15, 0.15],
        ),
        # FreqModMsg(
        #     now=0.0,
        #     running_queue_tokens=[200, 512],
        #     running_queue_pre_computed_tokens=[0, 0],
        #     running_queue_wait_time=[0.001, 0.002],
        #     kv_cache_usage=0.1,
        #     waiting_queue_tokens=[1200],
        #     waiting_queue_pre_computed_tokens=[0],
        #     waiting_queue_wait_time=[0.15],
        # ),
        # FreqModMsg(
        #     now=0.0,
        #     running_queue_tokens=[16],
        #     running_queue_pre_computed_tokens=[0],
        #     running_queue_wait_time=[0.001],
        #     kv_cache_usage=0.1,
        #     waiting_queue_tokens=[],
        #     waiting_queue_pre_computed_tokens=[],
        #     waiting_queue_wait_time=[],
        # ),
        # FreqModMsg(
        #     now=0.0,
        #     running_queue_tokens=[1024, 512],
        #     running_queue_pre_computed_tokens=[520, 0],
        #     running_queue_wait_time=[0.01, 0.02],
        #     kv_cache_usage=0.1,
        #     waiting_queue_tokens=[1200, 1200, 1200, 777, 666, 555, 888],
        #     waiting_queue_pre_computed_tokens=[0, 0, 0, 0, 0, 0, 0],
        #     waiting_queue_wait_time=[0.11, 0.12, 0.13, 0.13, 0.14, 0.157, 0.20],
        # ),
        ]
    for i in range(len(msg)):
        q.put(msgspec.msgpack.encode(msg[i]))
    s.run()

