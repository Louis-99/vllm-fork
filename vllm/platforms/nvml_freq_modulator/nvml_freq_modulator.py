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
PATH_TO_MODELS = Path("/export2/obasit/ClusterLevelServing/DistServe/simdistserve/estimators/tree_models")

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
             scheduler_stats: Optional[SchedulerStats],
             iteration_stats: Optional[IterationStats],) -> None:
        ...

    @abstractmethod
    def close(self):
        pass


class FreqModMsg(msgspec.Struct):
    """
    Msg from client to server.
    """
    now: float
    num_prompt_tokens_reqs: list[int]   # for prefill, tokens in batch just executed
    num_generation_tokens_iter: list[int]  # for decode, tokens generated including the one right now
    kv_cache_usage: float               # fraction of GPU memory used by KV cache
    waiting_reqs_num_tokens: list[int]  # for prefill & decode, tokens in waiting queue
    waiting_reqs_num_time: list[float]  # for prefill, waiting time in waiting queue

    def __post_init__(self):
        assert len(self.waiting_reqs_num_tokens) == len(
            self.waiting_reqs_num_time)


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
        optim_target='power',
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
            tbt_sla: float = 0.1,
            ttft_sla: float = 0.6,
            optim_target: str = 'power',  # 'energy' or 'power'factory
    ):
        self.llm_engine = llm_engine
        self.vllm_config = vllm_config

        self.engine_role = 'decode'
        if vllm_config.kv_transfer_config.is_kv_producer:
            self.engine_role = 'prefill'

        self.model = vllm_config.model_config.model

        self.q: SimpleQueue = get_mp_context().SimpleQueue()
        self.server = _MPNvmlFreqModulatorServer(vllm_config,
                                                 freq_choices,
                                                 self.q,
                                                 log_dir=log_dir,
                                                 mod_interval=mod_interval,
                                                 tbt_sla=tbt_sla,
                                                 ttft_sla=ttft_sla,
                                                 optim_target=optim_target,
                                                 engine_role=self.engine_role)
        self.server_process: Process = get_mp_context().Process(
            target=self.server.run)
        self.server_process.start()
        logger.info('_MPNvmlFreqModulatorServer process started.')

    def step(self,
             scheduler_stats: Optional[SchedulerStats],
             iteration_stats: Optional[IterationStats],) -> None:
        if scheduler_stats and iteration_stats:
            msg = self.build_msg(scheduler_stats, iteration_stats)
            msg_encoded = msgspec.msgpack.encode(msg)
            self.q.put(msg_encoded)

    def close(self):
        self.q.put(None)
        self.server_process.join()
        logger.info('_MPNvmlFreqModulatorServer process terminated.')

    @staticmethod
    def build_msg(scheduler_stats: SchedulerStats, iteration_stats: IterationStats) -> FreqModMsg:
        return FreqModMsg(
            iteration_stats.iteration_timestamp,
            iteration_stats.num_prompt_tokens_reqs,
            iteration_stats.num_generation_tokens_iter,
            scheduler_stats.kv_cache_usage,
            scheduler_stats.waiting_reqs_num_tokens,
            scheduler_stats.waiting_reqs_num_time,
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

            num_waiting_reqs = len(msg.waiting_reqs_num_tokens)
            # Smaller if not all requests are prefilled in `future_windows`
            assert len(prefill_cycles) <= num_waiting_reqs

            selected_freq, pred_batch_lat = (
                self._get_next_freq(msg, future_states, prefill_cycles))

            if msg.kv_cache_usage >= self.mem_util_ceiling:
                selected_freq = max(self.freq_choices)

            freq_mod_start = time.perf_counter()
            nvml_set_freq(selected_freq)
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

        max_future_vision = 1

        print(self.engine_role)

        # Pre-compute latency and power for each future window for each freq
        lat_mat_list = []
        power_mat_list = []
        for window_idx in range(max_future_vision):
            lat_mat_list.append(
                self.predict_latencies(future_states[window_idx],
                                       freq_choices_desc))
            power_mat_list.append(
                self.predict_powers(future_states[window_idx],
                                    freq_choices_desc))
        lat_mat = np.array(lat_mat_list)
        power_mat = np.array(power_mat_list)
        energy_mat = lat_mat * power_mat
        print("lat_mat:", lat_mat)
        print("power_mat:", power_mat)
        print("energy_mat:", energy_mat)
        assert lat_mat.shape == (max_future_vision, len(freq_choices_desc))
        assert power_mat.shape == (max_future_vision, len(freq_choices_desc))

        # check SLO satisfaction and adjust frequencies
        batch_lats_all = lat_mat[0]
        oldest_waiting_time = max(freq_mod_msg.waiting_reqs_num_time)
        if self.engine_role == 'prefill':
            sla_mask = (batch_lats_all + oldest_waiting_time) <= self.ttft_sla
        else:
            sla_mask = batch_lats_all <= self.tbt_sla
        if self.optim_target == 'energy':
            feasible_freq_indices = np.where(sla_mask)[0]
            if len(feasible_freq_indices) > 0:
                selected_freq_idx = feasible_freq_indices[
                    np.argmin(energy_mat[0][feasible_freq_indices])]
            else:
                selected_freq_idx = 0
        elif self.optim_target == 'power':
            feasible_freq_indices = np.where(sla_mask)[0]
            if len(feasible_freq_indices) > 0:
                lat_feasible = lat_mat[0][feasible_freq_indices]
                energy_feasible = energy_mat[0][feasible_freq_indices]
                power_feasible = energy_feasible / lat_feasible
                selected_freq_idx = feasible_freq_indices[
                    np.argmin(power_feasible)]
            else:
                selected_freq_idx = 0
        selected_freq = freq_choices_desc[selected_freq_idx]
        predicted_batch_lat = lat_mat[0][selected_freq_idx]

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
            - no Chunking for now
        """
        # A list that tells you for each request in the wait queue
        # how many iterations it will take to get the first token
        prefill_cycles = []

        # Construct a dummy wait queue to simulate future chunked prefills
        # list of (total tokens, processed tokens, remaining tokens)
        dummy_wait_queue = []
        # TODO add the chunked reqs first if chunking enabled

        # add the reqs in the wait queue, 
        num_prefill_tokens = msg.waiting_reqs_num_tokens
        dummy_wait_queue.extend([
            (m, 0, m)
            for m in num_prefill_tokens
        ])

        prefill_cycles.extend([1 for _ in num_prefill_tokens])  # since no chunking


        num_decodes = len(msg.num_generation_tokens_iter)
        if num_decodes > 0:
            decode_len_sum = sum(msg.num_generation_tokens_iter)
            decode_len_mean = np.mean(msg.num_generation_tokens_iter).item()
            decode_len_std = np.std(
                msg.num_generation_tokens_iter).item()
        else:
            decode_len_sum = 0
            decode_len_mean = 0
            decode_len_std = 0

        future_states = []
        for i in range(future_window):
            # prefill first
            if i == 0:
                num_prefills = len(dummy_wait_queue) # see assumptions
            else: 
                num_prefills = 0

            if num_prefills > 0:
                prefills = np.array(
                    [req[2] for req in dummy_wait_queue])  # remaining tokens
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

    def predict_powers(self, future_state: FutureState,
                                freq_choices) -> list[list[float]]:

        num_prefills = future_state.num_prefills
        prefill_len_sum = future_state.prefill_len_sum
        prefill_len_std = future_state.prefill_len_std
        prefill_len_mean = future_state.prefill_len_mean
        num_decodes = future_state.num_decodes
        decode_len_sum = future_state.decode_len_sum
        decode_len_std = future_state.decode_len_std
        decode_len_mean = future_state.decode_len_mean

        #prefill
        if self.engine_role == 'prefill':
            input_len = int(min(2048, max(32, prefill_len_sum)))
            # build xi as a 2D array of (input_len, freq) pairs for all freq choices
            xi = np.array([[input_len, freq] for freq in freq_choices], dtype=np.float32)
            # interpolate power for each (input_len, freq) pair; allow extrapolation if needed
            output_arr = interpn(
                points=(possible_input_len, possible_freq),
                values=busy_power_values_dict[self.tp_degree],
                xi=xi,
                method='linear',
                bounds_error=False,
                fill_value=None,
            )
        else:
            #decode
            power_model = self.power_model_decode
            input_feed = {
                "model": np.array([[self.model] for _ in range(len(freq_choices))], dtype=str),
                "batch_size": np.array([[num_decodes] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_sum": np.array([[decode_len_sum] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_mean": np.array([[decode_len_mean] for _ in range(len(freq_choices))], dtype=np.float32),
                "input_len_std": np.array([[decode_len_std] for _ in range(len(freq_choices))], dtype=np.float32),
                "tp_degree": np.array([[self.tp_degree] for _ in range(len(freq_choices))], dtype=np.float32),
                "freq_mhz": np.array([[freq] for freq in freq_choices], dtype=np.float32),
            } 
            output_arr = power_model.run(None, input_feed)[0]
            output_arr = np.asarray(output_arr)
            output_arr = output_arr[..., 0]
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
    vllm_config.parallel_config.tensor_parallel_size = 2
    freq_choices = get_preselected_freq(get_gpu_name())
    s = _MPNvmlFreqModulatorServer(freq_choices=freq_choices,
                                   vllm_config=vllm_config,
                                   q=q,
                                   log_dir=Path('./logs'),
                                   optim_target='energy',
                                   mod_interval=1,
                                   tbt_sla=0.25,
                                   ttft_sla=1.0)
    msg = FreqModMsg(
            now=0.0,
            num_prompt_tokens_reqs=[1074],
            num_generation_tokens_iter=[2050, 789],
            kv_cache_usage=0.1,
            waiting_reqs_num_tokens=[0, 0],
            waiting_reqs_num_time=[0, 0.0],
        )
    for _ in range(10):
        q.put(msgspec.msgpack.encode(msg))
    s.run()

