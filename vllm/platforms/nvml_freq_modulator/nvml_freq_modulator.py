# SPDX-License-Identifier: Apache-2.0
import copy
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

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.logger import init_logger
from vllm.platforms.nvml_utils import CSVWriter, get_gpu_name, get_preselected_freq, nvml_set_freq
from vllm.utils import get_mp_context

logger = init_logger(__name__)

# Change this accordingly
PATH_TO_MODELS = Path(__file__).parent / 'tree_models'

# power model prefill
possible_freq = np.array([780, 1080, 1380, 1680, 1830])
possible_input_len = np.array(
    [32,   64,   96,  128,  160,  192,  256,  384,  512,  768, 1024, 1280, 1536, 1792, 2048])

busy_power_values_dict = {
    4: np.array([
        [ 865.25887505,  984.72127171, 1140.23307683, 1396.33445554, 1385.88617877],
        [ 924.91678831, 1082.53452469, 1286.32423938, 1549.65300951, 1563.94855435],
        [ 954.73581584, 1140.63229992, 1390.3780871 , 1680.76153648, 1734.70052576],
        [1018.43250191, 1221.78283476, 1543.07507994, 1880.22343364, 1942.6190668 ],
        [1021.0607414 , 1224.78029167, 1607.07943643, 1927.45478727, 2127.5052573 ],
        [1084.28640911, 1291.95411139, 1742.84315608, 2198.67549647, 2404.429592  ],
        [1143.74980904, 1367.73381494, 1882.58995999, 2417.12518785, 2634.21850238],
        [1181.99973971, 1429.2342348 , 1956.98430597, 2582.52452   , 2703.46718597],
        [1106.30663166, 1350.266084  , 1895.1023916 , 2612.01642318, 2745.32649567],
        [1180.96507466, 1481.40437675, 2063.47833033, 2676.77416513, 2702.09631725],
        [1217.45493742, 1520.95248079, 2177.22006411, 2759.28575851, 2759.6566107 ],
        [1193.05040754, 1509.91219388, 2172.56049632, 2745.16476285, 2749.93814766],
        [1180.79333461, 1488.51255534, 2203.16329111, 2766.10493464, 2769.15640533],
        [1261.75169216, 1606.47416798, 2322.08879116, 2761.67362614, 2765.07185752],
        [1235.87224067, 1559.00235443, 2283.79591686, 2776.05462126, 2775.78978486],
    ]),
    2: np.array([
        [ 504.8714655 ,  644.05910082,  788.11071287,  964.64911905,  996.81165047],
        [ 538.43388779,  679.8367838 ,  848.52178567, 1058.02993614, 1115.76423859],
        [ 558.62907545,  706.88651643,  891.60417424, 1123.97111969, 1218.24403351],
        [ 610.07134923,  787.30547876, 1020.72949344, 1280.57615433, 1376.44100852],
        [ 612.87938317,  786.45673293, 1023.93927892, 1311.40308726, 1392.21389419],
        [ 657.19657127,  834.17186602, 1141.80856109, 1383.5541608 , 1385.80419169],
        [ 664.47926456,  828.20318831, 1160.58184678, 1384.09975325, 1386.41520802],
        [ 684.86457528,  862.61612102, 1236.03113618, 1384.74473118, 1385.64871904],
        [ 676.96543248,  856.05726866, 1240.74766892, 1379.47374536, 1383.89076739],
        [ 646.56773068,  847.77476953, 1195.56088575, 1386.44341108, 1388.97561606],
        [ 665.42824697,  880.82823165, 1249.25048523, 1389.56332899, 1386.52722456],
        [ 657.98246341,  869.99551703, 1257.0508306 , 1392.58820345, 1391.76599695],
        [ 631.10018873,  826.83978314, 1203.5316831 , 1393.6138576 , 1387.72965146],
        [ 696.76195988,  926.3252101 , 1336.88877153, 1393.38139544, 1391.86912304],
        [ 682.11812234,  906.90752387, 1317.28533085, 1392.48822538, 1391.85159044],
    ]),
}


class NvmlFreqModulatorInterface(ABC):

    @abstractmethod
    def step(self,
             scheduler_stats: Optional[SchedulerStats],) -> None:
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
        log_dir=Path(config.log_dir),
        tbt_sla=0.1,
        ttft_sla=0.6,
        optim_target='energy',
        mod_interval=1,
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
            future_window: int = 4,
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

    def step(self,
             scheduler_stats: SchedulerStats) -> None:
        msg = self.build_msg(scheduler_stats)
        msg_encoded = msgspec.msgpack.encode(msg)
        self.q.put(msg_encoded)

    def close(self):
        self.q.put(None)
        self.server_process.join()
        logger.info('_MPNvmlFreqModulatorServer process terminated.')

    @staticmethod
    def build_msg(scheduler_stats: SchedulerStats) -> FreqModMsg:
        return FreqModMsg(
            now = scheduler_stats.now,
            running_queue_tokens = scheduler_stats.running_reqs_num_tokens,
            running_queue_pre_computed_tokens = scheduler_stats.running_computed_tokens_list,
            running_queue_wait_time = scheduler_stats.running_reqs_num_time,
            kv_cache_usage = scheduler_stats.kv_cache_usage,
            waiting_queue_tokens = scheduler_stats.waiting_reqs_num_tokens,
            waiting_queue_pre_computed_tokens = scheduler_stats.waiting_computed_tokens_list,
            waiting_queue_wait_time = scheduler_stats.waiting_reqs_num_time,
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
        mem_util_ceiling: float = 0.9,
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
        # Load models here rather than in __init__() so that we don't pass the
        # loaded models across processes
        self._load_models()

        # Column `now` used as key column to join with `perf_metrics.csv`
        csv_writer = CSVWriter(col_names=[
            'now', 'mpc_start', 'freq_mod_start', 'freq_mod_end',
            'target_freq', 'batch_lat', 
        ],
        filename=self.log_dir / 'freq_mod_log.csv')

        for step_id in count():
            msg_encoded = self.q.get()
            if msg_encoded is None:
                break
            if step_id % self.mod_interval > 0:
                continue
            mpc_start = time.perf_counter()
            msg: FreqModMsg = msgspec.msgpack.decode(msg_encoded,
                                                     type=FreqModMsg)
            logger.info('freq_mod_msg: %s', msg)
            # logger.debug('freq_mod_msg: %s', msg)

            future_states, prefill_cycles = self.get_future_states(
                msg, self.future_windows)

            selected_freq, pred_batch_lat = (
                self._get_next_freq(msg, future_states, prefill_cycles))

            if msg.kv_cache_usage >= self.mem_util_ceiling:
                selected_freq = max(self.freq_choices)

            freq_mod_start = time.perf_counter()
            if self.last_applied_freq != selected_freq:
                nvml_set_freq(selected_freq)
                self.last_applied_freq = selected_freq
            freq_mod_end = time.perf_counter()
            csv_writer.add_row([
                msg.now,
                mpc_start,
                freq_mod_start,
                freq_mod_end,
                selected_freq,
                pred_batch_lat,
            ])

        csv_writer.close()

    def _get_next_freq(self, freq_mod_msg: FreqModMsg,
                          future_states: list[FutureState], prefill_cycles):
        freq_choices_desc = sorted(copy.deepcopy(self.freq_choices),
                                   reverse=True)
        max_future_vision = self.future_windows
        # Pre-compute latency and power for each future window for each freq
        lat_mat_list = []
        power_mat_list = []
        for window_idx in range(max_future_vision):
            lat_mat_list.append(
                self.predict_latencies(future_states[window_idx],
                                       freq_choices_desc))
        power_mat_list = self.predict_powers_future_states(
            future_states, freq_choices_desc)
        lat_mat = np.array(lat_mat_list)
        power_mat = np.array(power_mat_list)
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
            waiting_time_per_req = np.concatenate(
                    (
                        np.array(freq_mod_msg.running_queue_wait_time[:len(prefill_cycles)])[:, None],
                        np.array(freq_mod_msg.waiting_queue_wait_time[:len(prefill_cycles)])[:, None],
                    ),
                    axis=0,
                )
            # Start with the highest freq for each window
            selected_freq_ids = [0 for _ in range(self.future_windows)]
            for freq_idx in range(1, len(freq_choices_desc)):
            # Collect the candidates from `selected_freqs`
                candidates_: list[list[int]] = [[]]
                for window_idx in range(max_future_vision):
                    if selected_freq_ids[window_idx] == freq_idx - 1:
                        freq_ids_this_window = [freq_idx - 1, freq_idx]
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
                ttft_arr = time_till_finish_per_req + waiting_time_per_req
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

            predicted_batch_lat = lat_mat_list[0][selected_freq_ids[0]]

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
            remaining_tokens = total_tokens - processed_tokens
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
                prefills.append(num_tokens)

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

    def predict_latencies(self, states: FutureState,
                          freq_choices) -> list[float]:
        """
        Predict latency of the upcoming batch for each freq in `freq_choices`.
        """
        num_prefills = states.num_prefills
        prefill_len_sum = states.prefill_len_sum
        prefill_len_std = states.prefill_len_std
        prefill_len_mean = states.prefill_len_mean
        num_decodes = states.num_decodes
        decode_len_sum = states.decode_len_sum
        decode_len_std = states.decode_len_std
        decode_len_mean = states.decode_len_mean

        # prefill model
        if self.engine_role == 'prefill':
            latency_model = self.latency_model_prefill
            # Build inputs with shape (len(freq_choices), 1) so the second dimension is 1
            input_feed = {
                "model": np.array([[self.model] for _ in range(len(freq_choices))], dtype=str),
                "batch_size": np.array([[num_prefills] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_sum": np.array([[prefill_len_sum] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_mean": np.array([[prefill_len_mean] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_std": np.array([[prefill_len_std] for _ in range(len(freq_choices))], dtype=np.float32),
                "tp_degree": np.array([[self.tp_degree] for _ in range(len(freq_choices))], dtype=np.float32),
                "freq_mhz": np.array([[freq] for freq in freq_choices], dtype=np.float32),
            }
        else:
            # decode model
            latency_model = self.latency_model_decode
            # Build inputs with shape (len(freq_choices), 1) so the second dimension is 1
            input_feed = {
                "model": np.array([[self.model] for _ in range(len(freq_choices))], dtype=str),
                "batch_size": np.array([[num_decodes] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_sum": np.array([[decode_len_sum] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_mean": np.array([[decode_len_mean] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_std": np.array([[decode_len_std] for _ in range(len(freq_choices))], dtype=np.float32),
                "tp_degree": np.array([[self.tp_degree] for _ in range(len(freq_choices))], dtype=np.float32),
                "freq_mhz": np.array([[freq] for freq in freq_choices], dtype=np.float32),
            }   
            

        latency_arr = latency_model.run(None, input_feed)[0]
        # ensure numpy array
        latency_arr = np.asarray(latency_arr)
        latency_arr = latency_arr[..., 0]
        return latency_arr

    def predict_powers_future_states(self, future_states: list[FutureState],
                                     freq_choices) -> list[list[float]]:
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
            # For each future-state input, interpolate power for every freq choice.
            outputs = []
            for inp in inputs:
                # inp shape: (len(freq_choices), Ncols) where
                # col0=freq, col1=num_prefills, col2=prefill_len_sum, ...
                # Extract prefill_len_sum from the first row (same for all rows).
                prefill_len_sum = float(inp[0, 2])
                # Clamp to the known grid bounds to avoid excessive extrapolation.
                input_len = float(np.clip(prefill_len_sum,
                            possible_input_len.min(),
                            possible_input_len.max()))
                freqs = np.array(freq_choices, dtype=np.float32)

                # xi should be shape (n_freqs, 2): [input_len, freq] pairs.
                xi = np.column_stack((
                    np.full(len(freqs), input_len, dtype=np.float32),
                    freqs,
                ))

                out = interpn(
                    points=(possible_input_len, possible_freq),
                    values=busy_power_values_dict[self.tp_degree],
                    xi=xi,
                    method='linear',
                    bounds_error=False,
                    fill_value=None,
                )
                outputs.append(np.asarray(out, dtype=np.float32).tolist())

            # output_arr shape: (n_future_states, n_freq_choices)
            output_arr = np.array(outputs, dtype=np.float32)
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
            input_feed = {
                "model": np.array([[self.model] for _ in range(n_rows)], dtype=str),
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
        return output_arr.tolist()

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
                                   optim_target='energy',
                                   mod_interval=1,
                                   future_window=4,
                                   engine_role='prefill',    
                                   tbt_sla=0.1,
                                   ttft_sla=0.6,
                                   token_budget=1024,
                                   )
    msg = [FreqModMsg(
            now=0.0,
            running_queue_tokens=[1024, 512],
            running_queue_pre_computed_tokens=[520, 0],
            running_queue_wait_time=[0.01, 0.05],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[1200],
            waiting_queue_pre_computed_tokens=[0],
            waiting_queue_wait_time=[0.15],
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
        ),
    FreqModMsg(
            now=0.0,
            running_queue_tokens=[200, 200, 200, 200],
            running_queue_pre_computed_tokens=[0, 0, 0, 0],
            running_queue_wait_time=[0.001, 0.002, 0.003, 0.004],
            kv_cache_usage=0.1,
            waiting_queue_tokens=[1200],
            waiting_queue_pre_computed_tokens=[0],
            waiting_queue_wait_time=[0.15],
        )]
    for i in range(3):
        q.put(msgspec.msgpack.encode(msg[i]))
    s.run()

