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
from tqdm import tqdm

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


def calculate_warmup_iterations(
    prefill_ctx_lens: list[int],
    decode_gen_lens: list[int],
    max_num_batched_tokens: int,
) -> int:
    """
    Calculate number of warmup iterations needed to initialize decode requests
    without violating max_num_batched_tokens limit.
    
    Args:
        prefill_ctx_lens: Context lengths for prefill requests
        decode_gen_lens: Context lengths for decode initialization
        max_num_batched_tokens: Maximum tokens per batch
    
    Returns:
        Number of warmup iterations needed (minimum 2)
    """
    if not decode_gen_lens:
        # No decode requests, use default warmup
        return 2
    
    # Calculate tokens needed for prefill requests
    prefill_tokens = sum(prefill_ctx_lens)
    
    # Calculate total tokens needed for decode initialization (all as prefill)
    decode_init_tokens = sum(decode_gen_lens)
    
    # If both prefill and decode init fit in one batch, use default warmup
    if prefill_tokens + decode_init_tokens <= max_num_batched_tokens:
        return 2
    
    # Need to spread decode initialization across multiple iterations
    # Each warmup iteration can initialize decode requests up to the limit
    available_tokens_per_iter = max_num_batched_tokens - prefill_tokens
    
    if available_tokens_per_iter <= 0:
        # Prefill alone exceeds limit - still need at least 2 warmup iterations
        # Decode will be initialized in subsequent iterations without prefill
        available_tokens_per_iter = max_num_batched_tokens
        
        # Calculate iterations needed for decode init (after first iteration with prefill)
        warmup_for_decode = (decode_init_tokens + available_tokens_per_iter - 1) // available_tokens_per_iter
        return 1 + warmup_for_decode  # 1 for prefill + N for decode init
    
    # Calculate how many warmup iterations needed to initialize all decode requests
    warmup_for_decode = (decode_init_tokens + available_tokens_per_iter - 1) // available_tokens_per_iter
    
    # Return at least 2, but more if needed for decode initialization
    return max(2, warmup_for_decode)


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
        max_tokens=1000,  # Set very high to avoid stopping during profiling
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
    repeat: int = None,
    runtime: float = None,
    prefill_precomputed_tokens: list[int] = None,
    test_id: str = "default",
    enable_power_logging: bool = False,
    power_log_file: Optional[Path] = None,
    max_num_batched_tokens: int = 4096,
) -> dict:
    """
    Profile execution time of a single batch configuration.
    
    Args:
        executor: vLLM executor instance
        prefill_ctx_lens: List of context lengths for prefill requests
        decode_gen_lens: List of generated token counts for decode requests  
        block_size: KV cache block size
        num_kv_groups: Number of KV cache groups
        repeat: Number of times to repeat for averaging (mutually exclusive with runtime)
        runtime: Runtime in seconds for profiling (mutually exclusive with repeat)
        prefill_precomputed_tokens: List of pre-computed tokens per prefill request
        max_num_batched_tokens: Maximum tokens per batch (for warmup calculation)
    
    Returns:
        Dictionary with timing statistics
    """
    # Validate that exactly one of repeat or runtime is provided
    if repeat is None and runtime is None:
        raise ValueError("Either 'repeat' or 'runtime' must be specified")
    if repeat is not None and runtime is not None:
        raise ValueError("Cannot specify both 'repeat' and 'runtime'")
    
    use_runtime_mode = runtime is not None
    # Calculate warmup iterations based on decode initialization needs
    warmup = calculate_warmup_iterations(
        prefill_ctx_lens, decode_gen_lens, max_num_batched_tokens
    )
    
    logger.info(
        f"Profiling batch: {len(prefill_ctx_lens)} prefill reqs, "
        f"{len(decode_gen_lens)} decode reqs, "
        f"ctx_lens: {prefill_ctx_lens}, decode_lens: {decode_gen_lens}, "
        f"prefill_precomputed_tokens: {prefill_precomputed_tokens}, "
        f"warmup iterations: {warmup}, "
        f"mode: {'runtime=' + str(runtime) + 's' if use_runtime_mode else 'repeat=' + str(repeat)}"
    )
    
    # Generate persistent decode request IDs (unique per test)
    decode_req_ids = [f"decode_req_{test_id}_{i}" for i in range(len(decode_gen_lens))]
    
    # Track token counts for decode requests (starts at initial context length from dummy prefill)
    decode_token_counts = decode_gen_lens.copy() if decode_gen_lens else []
    
    # Generate persistent prefill request IDs for precomputation
    prefill_req_ids = [f"prefill_req_{test_id}_{i}" for i in range(len(prefill_ctx_lens))]
    prefill_precomputed_initialized = False  # Track if prefill precomputation is done
    
    # Warmup + timed iterations
    all_latencies = []
    previous_prefill_req_ids = set()  # Track prefill request IDs to clean up
    decode_initialized_count = 0  # Track how many decode requests initialized
    
    # Start power monitoring if enabled
    power_monitor_process = None
    if enable_power_logging and power_log_file:
        logger.info(f"Starting power monitoring: {power_log_file}")
        power_monitor_process = multiprocessing.Process(
            target=start_nvml_power_monitor,
            kwargs={
                'interval': 0.01,  # 10ms sampling interval
                'csv_filename': str(power_log_file),
                'log_interval': 0.3,  # Write to CSV every 0.3 seconds
                'power_queue': None,
            },
            daemon=True
        )
    
    iteration = 0
    profiling_start_time = None
    
    while True:
        # Check termination condition
        if iteration >= warmup:
            # We're in profiling phase
            if use_runtime_mode:
                if profiling_start_time is None:
                    profiling_start_time = time.perf_counter()
                elapsed = time.perf_counter() - profiling_start_time
                if elapsed >= runtime:
                    break
            else:
                # repeat mode
                if len(all_latencies) >= repeat:
                    break
        
        # During warmup: Initialize prefill precomputation and decode requests
        # Prefill precomputation: Initialize prefill requests with precomputed tokens
        if iteration < warmup and prefill_precomputed_tokens and not prefill_precomputed_initialized:
            # Create prefill requests with only the precomputed portion to initialize KV cache
            # We only need to compute up to the precomputed tokens during initialization
            scheduler_output = create_fake_scheduler_output(
                prefill_ctx_lens=prefill_precomputed_tokens,  # Only process precomputed portion
                decode_gen_lens=[],  # No decode during prefill precomputation
                block_size=block_size,
                num_kv_groups=num_kv_groups,
                iteration=iteration,
                prefill_precomputed_tokens=[0] * len(prefill_precomputed_tokens),  # Process all tokens in precomputed portion
                decode_req_ids=[],
            )
            
            # Override request IDs to use persistent IDs
            for i, req in enumerate(scheduler_output.scheduled_new_reqs):
                old_req_id = req.req_id
                req.req_id = prefill_req_ids[i]
                scheduler_output.num_scheduled_tokens[prefill_req_ids[i]] = scheduler_output.num_scheduled_tokens.pop(old_req_id)
            
            prefill_precomputed_initialized = True
            logger.info(f"Warmup {iteration+1}/{warmup}: Initialized prefill precomputation for {len(prefill_ctx_lens)} requests")
        # During warmup: Initialize decode requests as dummy prefill
        # Spread initialization across iterations to respect max_num_batched_tokens
        elif iteration < warmup and decode_gen_lens and decode_initialized_count < len(decode_gen_lens):
            # Create base scheduler output with prefill requests
            scheduler_output = create_fake_scheduler_output(
                prefill_ctx_lens=prefill_ctx_lens,
                decode_gen_lens=[],  # No decode yet
                block_size=block_size,
                num_kv_groups=num_kv_groups,
                iteration=iteration,
                prefill_precomputed_tokens=prefill_precomputed_tokens,
                decode_req_ids=[],
            )
            
            # Calculate how many tokens we can use for decode initialization
            prefill_tokens = sum(prefill_ctx_lens) if prefill_ctx_lens else 0
            available_tokens = max_num_batched_tokens - prefill_tokens
            
            # If prefill alone exceeds limit, skip prefill in decode init iterations
            if available_tokens <= 0:
                # Clear prefill from this iteration
                scheduler_output.scheduled_new_reqs = []
                scheduler_output.num_scheduled_tokens = {}
                scheduler_output.total_num_scheduled_tokens = 0
                available_tokens = max_num_batched_tokens
            
            # Add as many decode requests as fit within the token limit
            tokens_used = 0
            while decode_initialized_count < len(decode_gen_lens):
                ctx_len = decode_gen_lens[decode_initialized_count]
                if tokens_used + ctx_len > available_tokens:
                    break
                
                req_id = decode_req_ids[decode_initialized_count]
                new_req = create_fake_new_request(
                    req_id, ctx_len, block_size, num_kv_groups, num_computed_tokens=0
                )
                scheduler_output.scheduled_new_reqs.append(new_req)
                scheduler_output.num_scheduled_tokens[req_id] = ctx_len
                scheduler_output.total_num_scheduled_tokens += ctx_len
                
                tokens_used += ctx_len
                decode_initialized_count += 1
            
            logger.info(f"Warmup {iteration+1}/{warmup}: Initialized {decode_initialized_count}/{len(decode_gen_lens)} decode requests")
        elif iteration < warmup:
            # Remaining warmup iterations without special initialization
            # Just create empty scheduler output to advance warmup
            scheduler_output = create_fake_scheduler_output(
                prefill_ctx_lens=[],
                decode_gen_lens=[],
                block_size=block_size,
                num_kv_groups=num_kv_groups,
                iteration=iteration,
            )
            logger.info(f"Warmup {iteration+1}/{warmup}: No action")
        else:
            if iteration == warmup and power_monitor_process is not None:
                power_monitor_process.start()
            # Normal profiling with actual batch composition
            decode_fully_initialized = (decode_initialized_count == len(decode_gen_lens))
            
            # For prefill requests: use persistent IDs and precomputed tokens if initialized
            if prefill_precomputed_initialized:
                # Create scheduler output with decode requests
                scheduler_output = create_fake_scheduler_output(
                    prefill_ctx_lens=[],
                    decode_gen_lens=decode_gen_lens if decode_fully_initialized else [],
                    block_size=block_size,
                    num_kv_groups=num_kv_groups,
                    iteration=iteration,
                    prefill_precomputed_tokens=None,
                    decode_req_ids=decode_req_ids if decode_fully_initialized else [],
                    decode_token_counts=decode_token_counts if decode_fully_initialized else [],
                )
                
                # Add prefill requests as NEW requests with full context and num_computed_tokens set
                # This allows the model runner to use cached KV from warmup while processing remaining tokens
                if prefill_ctx_lens:
                    for i, (ctx_len, num_precomputed) in enumerate(zip(prefill_ctx_lens, prefill_precomputed_tokens)):
                        req_id = prefill_req_ids[i]
                        
                        # Create as NewRequestData with full context but indicate precomputed tokens
                        new_req = create_fake_new_request(
                            req_id=req_id,
                            context_length=ctx_len,
                            block_size=block_size,
                            num_kv_groups=num_kv_groups,
                            num_computed_tokens=num_precomputed,
                        )
                        
                        scheduler_output.scheduled_new_reqs.append(new_req)
                        scheduler_output.num_scheduled_tokens[req_id] = ctx_len - num_precomputed
                        scheduler_output.total_num_scheduled_tokens += ctx_len - num_precomputed
            else:
                # No precomputation: create normal prefill requests
                scheduler_output = create_fake_scheduler_output(
                    prefill_ctx_lens=prefill_ctx_lens,
                    decode_gen_lens=decode_gen_lens if decode_fully_initialized else [],
                    block_size=block_size,
                    num_kv_groups=num_kv_groups,
                    iteration=iteration,
                    prefill_precomputed_tokens=prefill_precomputed_tokens,
                    decode_req_ids=decode_req_ids if decode_fully_initialized else [],
                    decode_token_counts=decode_token_counts if decode_fully_initialized else [],
                )
            
            # Update decode token counts to match model runner's internal state
            # The model runner tracks output tokens internally, so num_computed_tokens
            # must increment each iteration to stay in sync
            if decode_fully_initialized:
                for i in range(len(decode_token_counts)):
                    decode_token_counts[i] += 1
        
        # Mark previous iteration's PREFILL requests as finished (decode requests persist)
        scheduler_output.finished_req_ids = previous_prefill_req_ids
        
        # Track current prefill request IDs for next iteration cleanup
        current_prefill_req_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        # Don't include decode request IDs - they stay alive across iterations
        if iteration < warmup and decode_gen_lens:
            # Remove decode request IDs from the set to finish (they persist)
            current_prefill_req_ids -= set(decode_req_ids[:decode_initialized_count])
        # Don't finish prefill requests with precomputation - they persist
        if prefill_precomputed_initialized and iteration < warmup:
            current_prefill_req_ids -= set(prefill_req_ids)
        previous_prefill_req_ids = current_prefill_req_ids
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        ret = executor.execute_model(scheduler_output, non_block=False)
        # print(scheduler_output)
        # print(ret)
        # print("\n\n")
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        if iteration < warmup:
            logger.info(f"Warmup iteration {iteration+1}/{warmup}: {latency_ms:.2f}ms")
        else:
            all_latencies.append(latency_ms)
            if use_runtime_mode:
                elapsed = time.perf_counter() - profiling_start_time
                if len(all_latencies) % max(1, 10) == 0 or elapsed >= runtime:
                    logger.info(f"Profiling iteration {len(all_latencies)}: {latency_ms:.2f}ms (elapsed: {elapsed:.1f}s/{runtime}s)")
            else:
                if (len(all_latencies)) % max(1, repeat // 10) == 0:
                    logger.info(f"Timed iteration {len(all_latencies)}/{repeat}: {latency_ms:.2f}ms")
        
        iteration += 1
    
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
        "mean_ctx_len": np.mean(prefill_ctx_lens) if prefill_ctx_lens else 0,
        "std_ctx_len": np.std(prefill_ctx_lens) if prefill_ctx_lens else 0,
        "num_decode_reqs": len(decode_gen_lens),
        "mean_decode_len": np.mean(decode_gen_lens) + (iteration/2) if decode_gen_lens else 0,
        "std_decode_len": np.std(decode_gen_lens) if decode_gen_lens else 0,
        "prefill_precomputed_tokens": prefill_precomputed_tokens,
        "mean_ctx_len_w_precomputed": (
            np.mean([cl + pt for cl, pt in zip(prefill_ctx_lens, prefill_precomputed_tokens)])
            if prefill_precomputed_tokens else np.mean(prefill_ctx_lens) if prefill_ctx_lens else 0
        ),
        "std_ctx_len_w_precomputed": (
            np.std([cl + pt for cl, pt in zip(prefill_ctx_lens, prefill_precomputed_tokens)])
            if prefill_precomputed_tokens else np.std(prefill_ctx_lens) if prefill_ctx_lens else 0
        ),
        "num_iterations": len(all_latencies),
        "total_profiling_time_s": sum(all_latencies) / 1000,
        "mode": "runtime" if use_runtime_mode else "repeat",
        "target_runtime_s": runtime if use_runtime_mode else None,
        "target_repeat": repeat if not use_runtime_mode else None,
    }


def run_single_test(
    executor,
    test_config: dict,
    block_size: int,
    num_kv_groups: int,
    model_name: str,
    max_num_batched_tokens: int,
    repeat: int = None,
    runtime: float = None,
    cleanup_previous_test: bool = False,
    save_dir: Optional[Path] = None,
    enable_power_logging: bool = False,
) -> dict:
    """Run a single profiling test from config."""
    test_name = test_config.get("name", "unnamed_test")
    prefill_ctx_lens = test_config.get("prefill_ctx_lens", [])
    prefill_precomputed_tokens = test_config.get("prefill_precomputed_tokens", None)
    decode_gen_lens = test_config.get("decode_gen_lens", [])
    
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
        runtime=runtime,
        prefill_precomputed_tokens=prefill_precomputed_tokens,
        test_id=test_name,
        enable_power_logging=enable_power_logging,
        power_log_file=power_log_file,
        max_num_batched_tokens=max_num_batched_tokens,
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
        default=None,
        help="Number of times to repeat each batch for averaging (mutually exclusive with --runtime)",
    )
    parser.add_argument(
        "--runtime",
        type=float,
        default=None,
        help="Runtime in seconds for profiling each test (mutually exclusive with --repeat)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="Fraction of GPU memory to use for KV cache (default: 0.5)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1024,
        help="Maximum number of sequences that can be processed (default: 1024)",
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
    repeat = args.repeat  # Default from command line
    runtime = args.runtime
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            parser.error(f"Config file not found: {args.config_file}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        test_configs = config.get("tests", [])
        if not test_configs:
            parser.error("Config file must contain 'tests' array")
        
        # Get repeat or runtime from config, fall back to command line arg
        config_repeat = config.get("repeat", None)
        config_runtime = config.get("runtime", None)
        
        # Validate config has exactly one
        if config_repeat is None and config_runtime is None:
            # Use command line values (already validated)
            pass
        elif config_repeat is not None and config_runtime is not None:
            parser.error("Config file cannot specify both 'repeat' and 'runtime'")
        else:
            # Config overrides command line
            repeat = config_repeat if config_repeat is not None else None
            runtime = config_runtime if config_runtime is not None else None
        
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
        }]
        # repeat and runtime already set from args
    
    # Validate that exactly one of repeat or runtime is provided
    if repeat is None and runtime is None:
        parser.error("Either --repeat or --runtime must be specified (via command line or config file)")
    if repeat is not None and runtime is not None:
        parser.error("Cannot specify both --repeat and --runtime")
    
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
    # Get actual GPU memory and subtract model memory for each GPU (TP > 1 support)
    import torch
    torch.cuda.synchronize()
    
    # Get memory info for each GPU in the tensor parallel group
    num_gpus = args.tensor_parallel_size
    available_memory = []
    
    for gpu_id in range(num_gpus):
        model_memory = torch.cuda.memory_allocated(gpu_id)
        total_gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        remaining_memory = total_gpu_memory - model_memory
        available_mem = int(remaining_memory * args.gpu_memory_utilization)
        available_memory.append(available_mem)
        
        logger.info(f"GPU {gpu_id} Memory - Total: {total_gpu_memory / 1e9:.2f} GB, "
                    f"Model: {model_memory / 1e9:.2f} GB, "
                    f"Remaining: {remaining_memory / 1e9:.2f} GB, "
                    f"Available for KV cache: {available_mem / 1e9:.2f} GB")
    
    # Ensure available_memory list matches kv_cache_specs length
    if len(available_memory) < len(kv_cache_specs):
        # Extend with last value if needed (shouldn't normally happen)
        available_memory.extend([available_memory[-1]] * (len(kv_cache_specs) - len(available_memory)))
    elif len(available_memory) > len(kv_cache_specs):
        # Trim if needed
        available_memory = available_memory[:len(kv_cache_specs)]
    
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
    
    for test_idx, test_config in enumerate(tqdm(test_configs, desc="Tests", unit="test")):
        result = run_single_test(
            executor=executor,
            test_config=test_config,
            block_size=args.block_size,
            num_kv_groups=num_kv_groups,
            model_name=args.model,
            max_num_batched_tokens=args.max_num_batched_tokens,
            repeat=repeat,
            runtime=runtime,
            cleanup_previous_test=(test_idx > 0),  # Clean up before 2nd+ tests
            save_dir=save_dir,
            enable_power_logging=args.enable_power_logging,
        )
        all_results.append(result)

    # Shutdown executor across all ranks
    logger.info("Shutting down executor...")
    executor.shutdown()

    
    
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
