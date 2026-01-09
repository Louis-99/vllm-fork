#!/usr/bin/env python3
"""
Offline profiling script for batch execution timing in vLLM.

This script allows you to profile execution time of batches by faking scheduler 
output. It supports separate specification of prefill and decode requests.

Usage:
    python profile_batch_execution.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --prefill-ctx-lens 128 256 512 \
        --decode-gen-lens 10 20 30 \
        --repeat 10

Based on vLLM V1 architecture documented in .github/copilot-instructions.md:
- Creates fake SchedulerOutput objects with NewRequestData/CachedRequestData
- Directly calls executor.execute_model() to bypass scheduler
- Measures pure execution time without scheduling overhead
"""

import argparse
import json
import multiprocessing
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.platforms.nvml_power_monitor import start_nvml_power_monitor
from vllm.v1.core.kv_cache_utils import KVCacheBlock, BlockHash
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)


def create_fake_block_ids(
    num_blocks: int,
    num_kv_groups: int = 1,
) -> tuple[list[int], ...]:
    """Create fake block IDs for KV cache allocation."""
    # Simulate block allocation - just sequential IDs
    block_ids = list(range(num_blocks))
    return tuple([block_ids.copy() for _ in range(num_kv_groups)])


def create_fake_new_request(
    req_id: str,
    context_length: int,
    block_size: int = 64,
    num_kv_groups: int = 1,
    num_computed_tokens: int = 0,
) -> NewRequestData:
    """
    Create a fake NewRequestData for a prefill request.
    
    Args:
        req_id: Request ID
        context_length: Number of prompt tokens
        block_size: KV cache block size (must match engine config)
        num_kv_groups: Number of KV cache groups
        num_computed_tokens: Number of tokens already computed (for partial prefill)
    
    Returns:
        NewRequestData object ready for SchedulerOutput
    """
    # Create fake prompt token IDs
    prompt_token_ids = list(range(context_length))
    
    # Calculate number of blocks needed
    num_blocks = (context_length + block_size - 1) // block_size
    block_ids = create_fake_block_ids(num_blocks, num_kv_groups)
    
    # Create sampling params (for generation tasks)
    from vllm.sampling_params import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
    )
    
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],  # No multimodal features
        sampling_params=sampling_params,
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
    )


def create_fake_cached_request(
    req_id: str,
    num_generated_tokens: int,
    total_tokens: int,
    block_size: int = 64,
    num_kv_groups: int = 1,
) -> tuple[str, int, Optional[tuple[list[int], ...]]]:
    """
    Create data for a fake cached request (decode phase).
    
    Args:
        req_id: Request ID
        num_generated_tokens: Number of tokens already generated
        total_tokens: Total tokens (prompt + generated)
        block_size: KV cache block size
        num_kv_groups: Number of KV cache groups
    
    Returns:
        Tuple of (req_id, num_computed_tokens, new_block_ids)
    """
    num_computed_tokens = total_tokens  # Everything is computed
    
    # Calculate blocks needed for all tokens
    num_blocks = (total_tokens + block_size - 1) // block_size
    new_block_ids = create_fake_block_ids(num_blocks, num_kv_groups)
    
    return req_id, num_computed_tokens, new_block_ids


def create_fake_scheduler_output(
    prefill_ctx_lens: list[int],
    decode_gen_lens: list[int],
    block_size: int = 64,
    num_kv_groups: int = 1,
    iteration: int = 0,
    prefill_precomputed_tokens: list[int] = None,
    decode_req_ids: list[str] = None,
    decode_token_counts: list[int] = None,
) -> SchedulerOutput:
    """
    Create a fake SchedulerOutput combining prefill and decode requests.
    
    Args:
        prefill_ctx_lens: List of context lengths for prefill requests
        decode_gen_lens: List of generated token counts for decode requests
        block_size: KV cache block size
        num_kv_groups: Number of KV cache groups
        iteration: Iteration number to ensure unique request IDs
        prefill_precomputed_tokens: List of pre-computed tokens per prefill request
        decode_req_ids: Persistent IDs for decode requests (must be same across iterations)
        decode_token_counts: Current token count for each decode request
    
    Returns:
        SchedulerOutput object ready for execution
    """
    scheduled_new_reqs = []
    req_id_counter = 0
    
    # Default to 0 pre-computed tokens if not specified
    if prefill_precomputed_tokens is None:
        prefill_precomputed_tokens = [0] * len(prefill_ctx_lens)
    
    # Create new requests (prefill phase)
    for ctx_len, num_precomputed in zip(prefill_ctx_lens, prefill_precomputed_tokens):
        req_id = f"prefill_req_iter{iteration}_{req_id_counter}"
        new_req = create_fake_new_request(
            req_id, ctx_len, block_size, num_kv_groups, num_precomputed
        )
        scheduled_new_reqs.append(new_req)
        req_id_counter += 1
    
    # Create cached requests (decode phase)
    req_ids = []
    resumed_from_preemption = []
    new_token_ids = []
    new_block_ids = []
    num_computed_tokens_list = []
    
    for i in range(len(decode_gen_lens)):
        # Use persistent decode request IDs and token counts provided by caller
        req_id = decode_req_ids[i] if decode_req_ids else f"decode_req_{i}"
        current_tokens = decode_token_counts[i] if decode_token_counts else 100
        
        # Calculate blocks needed for current token count (after adding 1 token this iteration)
        total_tokens = current_tokens + 1  # We're about to generate 1 more token
        num_blocks = (total_tokens + block_size - 1) // block_size
        
        # Check if we need a new block this iteration
        prev_num_blocks = (current_tokens + block_size - 1) // block_size
        if num_blocks > prev_num_blocks:
            # Need to allocate a new block
            new_block_ids_for_req = create_fake_block_ids(num_blocks, num_kv_groups)
        else:
            # No new blocks needed this iteration
            new_block_ids_for_req = None
        
        req_ids.append(req_id)
        resumed_from_preemption.append(False)
        # For PP, we'd need actual token IDs, but for profiling we can use dummy
        new_token_ids.append([1])  # Single new token for decode step
        new_block_ids.append(new_block_ids_for_req)
        num_computed_tokens_list.append(total_tokens)
        
        req_id_counter += 1
    
    cached_reqs_data = CachedRequestData(
        req_ids=req_ids,
        resumed_from_preemption=resumed_from_preemption,
        new_token_ids=new_token_ids,
        new_block_ids=new_block_ids,
        num_computed_tokens=num_computed_tokens_list,
    )
    
    # Calculate total scheduled tokens
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    
    for req in scheduled_new_reqs:
        # For prefill, schedule all prompt tokens
        num_tokens = len(req.prompt_token_ids)
        num_scheduled_tokens[req.req_id] = num_tokens
        total_num_scheduled_tokens += num_tokens
    
    for req_id in req_ids:
        # For decode, schedule 1 token per request
        num_scheduled_tokens[req_id] = 1
        total_num_scheduled_tokens += 1
    
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=cached_reqs_data,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},  # No speculative decoding
        scheduled_encoder_inputs={},  # No encoder inputs
        num_common_prefix_blocks=[0] * num_kv_groups,  # No prefix sharing
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
        kv_connector_metadata=None,
        batch_ID=0,
    )


def profile_batch_execution(
    executor: Executor,
    prefill_ctx_lens: list[int],
    decode_gen_lens: list[int],
    block_size: int,
    num_kv_groups: int,
    repeat: int = 1,
    warmup: int = 1,
    prefill_precomputed_tokens: list[int] = None,
    test_id: str = "default",
    enable_power_logging: bool = False,
    power_log_file: Optional[Path] = None,
) -> dict:
    """
    Profile execution time of a single batch configuration.
    
    Creates a fresh executor for each profiling run to ensure clean state.
    Only profiles prefill (new requests) to avoid complexity of maintaining
    request state across iterations.
    
    Args:
        executor: vLLM executor instance
        prefill_ctx_lens: List of context lengths for prefill requests
        decode_gen_lens: List of generated token counts for decode requests  
        block_size: KV cache block size
        num_kv_groups: Number of KV cache groups
        repeat: Number of times to repeat for averaging
        warmup: Number of warmup iterations
        prefill_precomputed_tokens: List of pre-computed tokens per prefill request
    
    Returns:
        Dictionary with timing statistics
    """
    # For profiling mixed prefill/decode batches:
    # - Iteration 0 (warmup): Initialize ALL requests as prefill to populate model runner state
    # - Iteration 1+: Use actual batch composition (prefill + decode)
    
    logger.info(
        f"Profiling batch: {len(prefill_ctx_lens)} prefill reqs, "
        f"{len(decode_gen_lens)} decode reqs, "
        f"ctx_lens: {prefill_ctx_lens}, decode_lens: {decode_gen_lens}"
    )
    
    # Generate persistent decode request IDs (unique per test)
    decode_req_ids = [f"decode_req_{test_id}_{i}" for i in range(len(decode_gen_lens))]
    
    # Track token counts for decode requests (starts at initial context length from dummy prefill)
    decode_token_counts = decode_gen_lens.copy() if decode_gen_lens else []
    
    # Warmup + timed iterations
    all_latencies = []
    previous_prefill_req_ids = set()  # Track prefill request IDs to clean up
    decode_initialized = False  # Track if decode requests have been initialized
    
    # Start power monitoring if enabled
    power_monitor_process = None
    if enable_power_logging and power_log_file:
        logger.info(f"Starting power monitoring: {power_log_file}")
        power_monitor_process = multiprocessing.Process(
            target=start_nvml_power_monitor,
            kwargs={
                'interval': 0.01,  # 10ms sampling interval
                'csv_filename': str(power_log_file),
                'log_interval': 1.0,  # Write to CSV every 1 second
                'power_queue': None,
            },
            daemon=True
        )
        power_monitor_process.start()
        time.sleep(0.5)  # Give power monitor time to initialize
    
    for iteration in range(warmup + repeat):
        # First iteration: Initialize decode requests as dummy prefill
        # This populates the model runner's state so they can be used as cached requests later
        if iteration == 0 and decode_gen_lens:
            # Create dummy prefill for decode requests to initialize them
            # Use decode_gen_lens as the context lengths for dummy prefill
            dummy_prefill_lens = decode_gen_lens.copy()
            
            # Create new requests for actual prefill
            scheduler_output = create_fake_scheduler_output(
                prefill_ctx_lens=prefill_ctx_lens,
                decode_gen_lens=[],  # No decode yet
                block_size=block_size,
                num_kv_groups=num_kv_groups,
                iteration=iteration,
                prefill_precomputed_tokens=prefill_precomputed_tokens,
                decode_req_ids=[],
            )
            
            # Add dummy prefill requests for decode initialization using persistent IDs
            for i, (req_id, ctx_len) in enumerate(zip(decode_req_ids, dummy_prefill_lens)):
                new_req = create_fake_new_request(
                    req_id, ctx_len, block_size, num_kv_groups, num_computed_tokens=0
                )
                scheduler_output.scheduled_new_reqs.append(new_req)
                # Update token counts
                scheduler_output.num_scheduled_tokens[req_id] = ctx_len
                scheduler_output.total_num_scheduled_tokens += ctx_len
            
            decode_initialized = True
        else:
            # Normal profiling with actual batch composition
            scheduler_output = create_fake_scheduler_output(
                prefill_ctx_lens=prefill_ctx_lens,
                decode_gen_lens=decode_gen_lens if decode_initialized else [],
                block_size=block_size,
                num_kv_groups=num_kv_groups,
                iteration=iteration,
                prefill_precomputed_tokens=prefill_precomputed_tokens,
                decode_req_ids=decode_req_ids if decode_initialized else [],
                decode_token_counts=decode_token_counts if decode_initialized else [],
            )
            
            # Don't update decode token counts - keep them constant for profiling
            # This ensures we're profiling the same batch configuration repeatedly
            # rather than simulating decode progression
        
        # Mark previous iteration's PREFILL requests as finished (decode requests persist)
        scheduler_output.finished_req_ids = previous_prefill_req_ids
        
        # Track current prefill request IDs for next iteration cleanup
        current_prefill_req_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        # Don't include decode request IDs - they stay alive across iterations
        if iteration == 0 and decode_gen_lens:
            # Remove decode request IDs from the set to finish
            current_prefill_req_ids -= set(decode_req_ids)
        previous_prefill_req_ids = current_prefill_req_ids
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # print(scheduler_output)
        ret = executor.execute_model(scheduler_output, non_block=False)
        # print(ret)
        # print("\n\n")
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        if iteration < warmup:
            logger.info(f"Warmup iteration {iteration+1}/{warmup}: {latency_ms:.2f}ms")
        else:
            all_latencies.append(latency_ms)
            if (len(all_latencies)) % max(1, repeat // 10) == 0:
                logger.info(f"Timed iteration {len(all_latencies)}/{repeat}: {latency_ms:.2f}ms")
    
    # Stop power monitoring
    if power_monitor_process is not None:
        logger.info("Stopping power monitoring...")
        power_monitor_process.terminate()
        power_monitor_process.join(timeout=5)
        if power_monitor_process.is_alive():
            power_monitor_process.kill()
    
    return {
        "mean_ms": np.mean(all_latencies),
        "std_ms": np.std(all_latencies),
        "min_ms": np.min(all_latencies),
        "max_ms": np.max(all_latencies),
        "median_ms": np.median(all_latencies),
        "p95_ms": np.percentile(all_latencies, 95),
        "p99_ms": np.percentile(all_latencies, 99),
        "num_prefill_reqs": len(prefill_ctx_lens),
        "num_decode_reqs": len(decode_gen_lens),
        "total_tokens": sum(prefill_ctx_lens) + len(decode_gen_lens),  # prefill tokens + 1 token per decode
        "prefill_precomputed_tokens": prefill_precomputed_tokens,
    }


def run_single_test(
    executor,
    test_config: dict,
    block_size: int,
    num_kv_groups: int,
    model_name: str,
    cleanup_previous_test: bool = False,
    save_dir: Optional[Path] = None,
    enable_power_logging: bool = False,
) -> dict:
    """Run a single profiling test from config."""
    test_name = test_config.get("name", "unnamed_test")
    prefill_ctx_lens = test_config.get("prefill_ctx_lens", [])
    prefill_precomputed_tokens = test_config.get("prefill_precomputed_tokens", None)
    decode_gen_lens = test_config.get("decode_gen_lens", [])
    repeat = test_config.get("repeat", 10)
    warmup = test_config.get("warmup", 2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running test: {test_name}")
    logger.info(f"{'='*80}")
    
    # Setup power log file if enabled
    power_log_file = None
    if enable_power_logging and save_dir:
        test_save_dir = save_dir / test_name
        test_save_dir.mkdir(parents=True, exist_ok=True)
        power_log_file = test_save_dir / "power_log.csv"
        logger.info(f"Power logging enabled: {power_log_file}")
    
    # Clean up state from previous test if needed
    if cleanup_previous_test:
        logger.info("Cleaning up state from previous test...")
        # Create empty scheduler output to flush all requests
        cleanup_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[],
                resumed_from_preemption=[],
                new_token_ids=[],
                new_block_ids=[],
                num_computed_tokens=[],
            ),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * num_kv_groups,
            finished_req_ids=set(),  # Will be populated below
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
            kv_connector_metadata=None,
            batch_ID=0,
        )
        # Mark all existing requests in model runner as finished
        # Access the model runner's request state to get all active request IDs
        try:
            # Get all request IDs from the model runner
            model_runner_requests = executor.driver_worker.model_runner.requests
            if model_runner_requests:
                cleanup_output.finished_req_ids = set(model_runner_requests.keys())
                logger.info(f"Finishing {len(cleanup_output.finished_req_ids)} requests from previous test")
                executor.execute_model(cleanup_output, non_block=False)
        except Exception as e:
            logger.warning(f"Could not clean up previous test state: {e}")
    
    results = profile_batch_execution(
        executor=executor,
        prefill_ctx_lens=prefill_ctx_lens,
        decode_gen_lens=decode_gen_lens,
        block_size=block_size,
        num_kv_groups=num_kv_groups,
        repeat=repeat,
        warmup=warmup,
        prefill_precomputed_tokens=prefill_precomputed_tokens,
        test_id=test_name,
        enable_power_logging=enable_power_logging,
        power_log_file=power_log_file,
    )
    
    # Add test metadata to results
    results["test_name"] = test_name
    results["model"] = model_name
    
    # Print results
    # print("\n" + "="*80)
    print(f"TEST: {test_name} done")
    # print("="*80)
    # print(f"Configuration:")
    # print(f"  Model: {model_name}")
    # print(f"  Prefill requests: {results['num_prefill_reqs']} "
    #       f"(ctx_lens: {prefill_ctx_lens})")
    # if results['prefill_precomputed_tokens'] is not None:
    #     print(f"  Prefill precomputed tokens: {results['prefill_precomputed_tokens']}")
    # print(f"  Decode requests: {results['num_decode_reqs']} "
    #       f"(gen_lens: {decode_gen_lens})")
    # print(f"  Total tokens: {results['total_tokens']}")
    # print(f"\nTiming Statistics ({repeat} iterations):")
    # print(f"  Mean:   {results['mean_ms']:.2f} ms")
    # print(f"  Median: {results['median_ms']:.2f} ms")
    # print(f"  Std:    {results['std_ms']:.2f} ms")
    # print(f"  Min:    {results['min_ms']:.2f} ms")
    # print(f"  Max:    {results['max_ms']:.2f} ms")
    # print(f"  P95:    {results['p95_ms']:.2f} ms")
    # print(f"  P99:    {results['p99_ms']:.2f} ms")
    # print(f"\nThroughput:")
    # print(f"  {results['total_tokens'] / (results['mean_ms'] / 1000):.2f} tokens/sec")
    # print("="*80)
    
    # Save results to file if save_dir is specified
    if save_dir:
        test_save_dir = save_dir / test_name
        test_save_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = test_save_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile batch execution time in vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to JSON config file with test configurations",
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism degree",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="KV cache block size (must match across prefill/decode in disagg mode)",
    )
    
    # Batch configuration
    parser.add_argument(
        "--prefill-ctx-lens",
        type=int,
        nargs="+",
        default=[],
        help="List of context lengths for prefill requests (e.g., 128 256 512)",
    )
    parser.add_argument(
        "--prefill-precomputed-tokens",
        type=int,
        nargs="+",
        default=None,
        help="List of pre-computed tokens for each prefill request (default: 0 for all). "
             "Must have same length as --prefill-ctx-lens.",
    )
    parser.add_argument(
        "--decode-gen-lens",
        type=int,
        nargs="+",
        default=[],
        help="List of generated token counts for decode requests (e.g., 10 20 30)",
    )
    
    # Profiling parameters
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of times to repeat each batch for averaging",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations before measurement",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory to use for KV cache (default: 0.8)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=2048,
        help="Maximum number of sequences that can be processed (default: 2048)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per batch. Default: 4096",
    )
    parser.add_argument(
        "--enable-power-logging",
        action="store_true",
        help="Enable GPU power logging during profiling (saves to power_log.csv in each test dir)",
    )
    
    args = parser.parse_args()
    
    # Load config file if provided
    save_dir = None
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            parser.error(f"Config file not found: {args.config_file}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        test_configs = config.get("tests", [])
        if not test_configs:
            parser.error("Config file must contain 'tests' array")
        
        # Get save directory from config
        save_dir_str = config.get("save_dir", None)
        if save_dir_str:
            save_dir = Path(save_dir_str)
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Results will be saved to: {save_dir}")
            
            # Copy config file to save directory
            config_copy_path = save_dir / "config.json"
            shutil.copy2(config_path, config_copy_path)
            logger.info(f"Config file copied to: {config_copy_path}")
    else:
        # Create single test config from command-line arguments
        if not args.prefill_ctx_lens and not args.decode_gen_lens:
            parser.error("At least one of --prefill-ctx-lens or --decode-gen-lens must be provided")
        
        test_configs = [{
            "name": "command_line_test",
            "prefill_ctx_lens": args.prefill_ctx_lens,
            "prefill_precomputed_tokens": args.prefill_precomputed_tokens,
            "decode_gen_lens": args.decode_gen_lens,
            "repeat": args.repeat,
            "warmup": args.warmup,
        }]
    
    # Validate test configurations
    for test_idx, test_config in enumerate(test_configs):
        prefill_ctx_lens = test_config.get("prefill_ctx_lens", [])
        prefill_precomputed_tokens = test_config.get("prefill_precomputed_tokens", None)
        
        if prefill_precomputed_tokens is not None:
            if len(prefill_precomputed_tokens) != len(prefill_ctx_lens):
                parser.error(
                    f"Test {test_idx}: prefill_precomputed_tokens must have {len(prefill_ctx_lens)} values "
                    f"(matching prefill_ctx_lens length), got {len(prefill_precomputed_tokens)}"
                )
            # Validate that precomputed tokens don't exceed context length
            for i, (precomputed, ctx_len) in enumerate(zip(prefill_precomputed_tokens, prefill_ctx_lens)):
                if precomputed > ctx_len:
                    parser.error(
                        f"Test {test_idx}, Prefill request {i}: precomputed tokens ({precomputed}) "
                        f"cannot exceed context length ({ctx_len})"
                    )
    
    # Create engine arguments
    engine_args = EngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        enforce_eager=False,  # Disable CUDA graphs for profiling
        enable_prefix_caching=False,  # Simplify for profiling
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    
    vllm_config = engine_args.create_engine_config()
    
    # Initialize executor
    logger.info("Initializing executor...")
    executor_class = Executor.get_class(vllm_config)
    executor = executor_class(vllm_config)
    
    # Get KV cache specs and initialize
    logger.info("Initializing KV caches...")
    kv_cache_specs = executor.get_kv_cache_specs()
    
    # Import here to avoid circular dependency
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config, unify_kv_cache_configs
    
    # Calculate available memory based on gpu_memory_utilization
    # Get actual GPU memory and apply utilization factor
    import torch
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = [int(total_gpu_memory * args.gpu_memory_utilization)] * len(kv_cache_specs)
    
    kv_cache_configs = [
        get_kv_cache_config(vllm_config, spec, mem)
        for spec, mem in zip(kv_cache_specs, available_memory)
    ]
    unify_kv_cache_configs(kv_cache_configs)
    
    executor.initialize_from_config(kv_cache_configs)
    
    num_kv_groups = len(kv_cache_configs[0].kv_cache_groups)
    
    # Profile execution - run all tests
    logger.info(f"Starting profiling with {len(test_configs)} test(s)...")
    all_results = []
    
    for test_idx, test_config in enumerate(test_configs):
        result = run_single_test(
            executor=executor,
            test_config=test_config,
            block_size=args.block_size,
            num_kv_groups=num_kv_groups,
            model_name=args.model,
            cleanup_previous_test=(test_idx > 0),  # Clean up before 2nd+ tests
            save_dir=save_dir,
            enable_power_logging=args.enable_power_logging,
        )
        all_results.append(result)
    
    # Print summary
    # print("\n" + "="*80)
    # print("PROFILING SUMMARY")
    # print("="*80)
    # print(f"Total tests run: {len(all_results)}")
    # for result in all_results:
    #     print(f"\n  Test: {result['test_name']}")
    #     print(f"    Mean latency: {result['mean_ms']:.2f} ms")
    #     print(f"    Throughput: {result['total_tokens'] / (result['mean_ms'] / 1000):.2f} tokens/sec")
    # print("="*80)
    print("done")


if __name__ == "__main__":
    main()
