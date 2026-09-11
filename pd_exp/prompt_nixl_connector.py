"""vLLM 0.11 NIXL compatibility for prompt-embedding requests.

Upstream vLLM 0.11's NixlConnectorScheduler assumes prompt_token_ids is a
list.  Prompt-embedding requests intentionally set it to None and expose the
correct common length as Request.num_prompt_tokens.  Keep this one-line
semantic adaptation in the research connector instead of patching the
installed package.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector,
    NixlConnectorScheduler,
    NixlConnectorWorker,
)


def _append_trace(event: str, **fields) -> None:
    """Append one small process-safe trace record when profiling is enabled."""
    trace_path = os.environ.get("COSY_PD_TRACE_FILE")
    if not trace_path:
        return
    record = {
        "event": event,
        "monotonic_ns": time.perf_counter_ns(),
        "wall_time_ns": time.time_ns(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        **fields,
    }
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


class CosyPromptNixlConnectorScheduler(NixlConnectorScheduler):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            count = int(request.num_prompt_tokens) - num_computed_tokens
            if count > 0:
                return count, True
        return 0, False


class CosyPromptNixlConnectorWorker(NixlConnectorWorker):
    """NIXL worker with opt-in, request-level GPU READ measurements.

    vLLM 0.11 only reports a completed-transfer count.  This experimental
    worker records the actual asynchronous READ submission and the first
    completion observation without changing NIXL's data path.
    """

    def __init__(self, vllm_config, engine_id: str):
        if (
            os.environ.get("COSY_PD_STEP_TRACE_FILE")
            or os.environ.get("COSY_PD_GRAPH_SAFE_REMOTE_PREFILL") == "1"
        ):
            from pd_exp.graph_runtime import install_vllm_runtime_patch

            install_vllm_runtime_patch()
        super().__init__(vllm_config, engine_id)
        self._cosy_trace_context = threading.local()
        self._cosy_transfer_started_ns: dict[str, int] = {}
        self._cosy_transfer_bytes: dict[str, int] = {}
        original_prepare = self.nixl_wrapper.make_prepped_xfer
        original_transfer = self.nixl_wrapper.transfer

        def traced_prepare(*args, **kwargs):
            request_id = getattr(
                self._cosy_trace_context, "request_id", "unknown"
            )
            started_ns = time.perf_counter_ns()
            handle = original_prepare(*args, **kwargs)
            _append_trace(
                "kv_prepare_complete",
                request_id=request_id,
                duration_ns=time.perf_counter_ns() - started_ns,
            )
            return handle

        def traced_transfer(handle):
            request_id = getattr(
                self._cosy_trace_context, "request_id", "unknown"
            )
            started_ns = time.perf_counter_ns()
            self._cosy_transfer_started_ns[request_id] = started_ns
            _append_trace(
                "kv_transfer_submit",
                request_id=request_id,
                estimated_bytes=self._cosy_transfer_bytes.get(request_id),
            )
            if torch.cuda.is_available():
                # An NVTX point complements the JSONL interval.  A normal
                # range cannot safely span this asynchronous call because
                # several requests may complete out of submission order.
                torch.cuda.nvtx.mark("COSY_PD_KV_RECV")
            return original_transfer(handle)

        self.nixl_wrapper.make_prepped_xfer = traced_prepare
        self.nixl_wrapper.transfer = traced_transfer

    def _read_blocks(
        self,
        local_block_ids,
        remote_block_ids,
        dst_engine_id,
        request_id,
    ):
        self._cosy_trace_context.request_id = request_id
        # Each selected block transfers one block from every KV region.
        # block_len_per_layer already contains the registered byte size for
        # each layer/region in the upstream connector.
        estimated_bytes = len(local_block_ids) * sum(self.block_len_per_layer)
        self._cosy_transfer_bytes[request_id] = int(estimated_bytes)
        _append_trace(
            "kv_read_prepare_start",
            request_id=request_id,
            local_block_count=len(local_block_ids),
            remote_block_count=len(remote_block_ids),
            estimated_bytes=int(estimated_bytes),
        )
        try:
            return super()._read_blocks(
                local_block_ids,
                remote_block_ids,
                dst_engine_id,
                request_id,
            )
        finally:
            self._cosy_trace_context.request_id = None

    def _pop_done_transfers(self, transfers):
        receiving = transfers is self._recving_transfers
        done = super()._pop_done_transfers(transfers)
        if receiving:
            completed_ns = time.perf_counter_ns()
            for request_id in done:
                started_ns = self._cosy_transfer_started_ns.pop(
                    request_id, None
                )
                estimated_bytes = self._cosy_transfer_bytes.pop(
                    request_id, None
                )
                _append_trace(
                    "kv_transfer_complete",
                    request_id=request_id,
                    duration_ns=(
                        None
                        if started_ns is None
                        else completed_ns - started_ns
                    ),
                    estimated_bytes=estimated_bytes,
                )
                if torch.cuda.is_available():
                    torch.cuda.nvtx.mark("COSY_PD_KV_RECV_DONE")
        return done


class CosyPromptNixlConnector(NixlConnector):
    """NixlConnector with prompt-embedding-aware scheduler length."""

    def __init__(self, vllm_config, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = CosyPromptNixlConnectorScheduler(
                vllm_config, self.engine_id,
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            worker_class = (
                CosyPromptNixlConnectorWorker
                if os.environ.get("COSY_PD_INSTRUMENT_NIXL") == "1"
                else NixlConnectorWorker
            )
            self.connector_worker = worker_class(vllm_config, self.engine_id)
        else:
            raise ValueError(f"Unsupported KV connector role: {role}")
