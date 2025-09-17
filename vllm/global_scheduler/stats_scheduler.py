# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from torch.distributed import TCPStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

VALID_STATS_KEYS = {"kv_cache_usage", "num_running_reqs", "num_waiting_reqs", "num_uncomputed_tokens"}

class Scheduler:
    def __init__(self, instances: list[dict], stats_stores: list[TCPStore], stats_keys: list[str]):
        self.instances = instances
        self.stats_stores = stats_stores
        assert len(self.instances) == len(self.stats_stores)
        assert len(self.instances) > 0
        self.num_instances = len(instances)
        self.current_idx = 0
        assert set(stats_keys) <= VALID_STATS_KEYS
        self.stats_keys = stats_keys
        

    def schedule(self, request: Request) -> dict:
        stats_values = [tuple(map(float, store.multi_get(self.stats_keys))) for store in self.stats_stores]
        
        self.current_idx = (self.current_idx + 1) % self.num_instances
        min_value = stats_values[self.current_idx]
        for i in range(1, self.num_instances):
            idx = (self.current_idx + i) % self.num_instances
            if stats_values[idx] < min_value:
                min_value = stats_values[idx]
                self.current_idx = idx
        return self.instances[self.current_idx]
        

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    prefill_clients = []
    prefill_stats_stores = []
    decode_clients = []
    decode_stats_stores = []

    # Create prefill clients
    for i, (host, port, stats_port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f'http://{host}:{port}/v1'
        prefill_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=prefiller_base_url),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })
        prefill_stats_stores.append(
            TCPStore(host, stats_port, is_master=False))

    # Create decode clients
    for i, (host, port, stats_port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f'http://{host}:{port}/v1'
        decode_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=decoder_base_url),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })
        decode_stats_stores.append(
            TCPStore(host, stats_port, is_master=False))
        
    app.state.prefill_scheduler = Scheduler(prefill_clients, prefill_stats_stores, global_args.prefill_stats_keys)
    app.state.decode_scheduler = Scheduler(decode_clients, decode_stats_stores, global_args.decode_stats_keys) 

    print(f"Initialized {len(prefill_clients)} prefill clients "
          f"and {len(decode_clients)} decode clients.")

    yield

    # Shutdown: Close all clients
    for client_info in prefill_clients:
        await client_info['client'].aclose()

    for client_info in decode_clients:
        await client_info['client'].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    parser.add_argument("--prefiller-hosts",
                        "--prefiller-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        "--prefiller-port",
                        type=int,
                        nargs="+",
                        default=[8100])

    parser.add_argument("--prefiller-stats-ports",
                        "--prefiller-stats-port",
                        type=int,
                        nargs="+",
                        default=[8101])

    # For decoder instances
    parser.add_argument("--decoder-hosts",
                        "--decoder-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports",
                        "--decoder-port",
                        type=int,
                        nargs="+",
                        default=[8200])
    
    parser.add_argument("--decoder-stats-ports",
                        "--decoder-stats-port",
                        type=int,
                        nargs="+",
                        default=[8201])

    parser.add_argument("--prefill-stats-keys",
                        type=str,
                        nargs="*",
                        default=[])

    parser.add_argument("--decode-stats-keys",
                        type=str,
                        nargs="*",
                        default=[])    

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port, stats_ports) for each service type
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports, args.prefiller_stats_ports))
    args.decoder_instances = list(
        zip(args.decoder_hosts, args.decoder_ports, args.decoder_stats_ports))
    
    assert set(args.decode_stats_keys) <= VALID_STATS_KEYS
    assert set(args.prefill_stats_keys) <= VALID_STATS_KEYS

    return args


async def send_request_to_service(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    response = await client_info['client'].post(endpoint,
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response


async def stream_service_response(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    async with client_info['client'].stream("POST",
                                            endpoint,
                                            json=req_data,
                                            headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        app: FastAPI = request.app

        # Get the next prefill client in round-robin fashion
        prefill_client_info = app.state.prefill_scheduler.schedule(request)

        # Send request to prefill service
        response = await send_request_to_service(prefill_client_info, api,
                                                 req_data, request_id)

        # Extract the needed fields
        response_json = response.json()
        kv_transfer_params = response_json.get('kv_transfer_params', {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = app.state.decode_scheduler.schedule(request)

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            async for chunk in stream_service_response(decode_client_info,
                                                       api,
                                                       req_data,
                                                       request_id=request_id):
                yield chunk

        return StreamingResponse(generate_stream(),
                                 media_type="application/json")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
