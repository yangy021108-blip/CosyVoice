"""CosyVoice prompt-embedding adapter for vLLM 0.25.1 NIXL connector."""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlPullConnector,
    NixlPullConnectorScheduler,
    NixlPullConnectorWorker,
)
if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

def _append_trace(event: str, **fields) -> None:
    path_value = os.environ.get("COSY_PD_TRACE_FILE")
    if not path_value:
        return
    record = {"event": event, "monotonic_ns": time.perf_counter_ns(),
              "wall_time_ns": time.time_ns(), "pid": os.getpid(),
              "thread_id": threading.get_ident(), **fields}
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)

class CosyPromptNixlPullConnectorScheduler(NixlPullConnectorScheduler):
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int):
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            prompt_length = int(request.num_prompt_tokens)
            actual = self._get_remote_prefill_token_count(prompt_length)
            count = actual - int(num_computed_tokens)
            if count > 0:
                return count, True
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

class CosyPromptNixlPullConnectorWorker(NixlPullConnectorWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kv_hash_pending = {}
        self._kv_hash_enabled = os.environ.get("COSY_PD_KV_HASH") == "1"
        self._kv_hash_file = os.environ.get("COSY_PD_KV_HASH_FILE")
        self._cosy_trace_context = threading.local()
        self._cosy_transfer_started_ns = {}
        self._cosy_transfer_bytes = {}
        if os.environ.get("COSY_PD_INSTRUMENT_NIXL") == "1":
            original_prepare = self.nixl_wrapper.make_prepped_xfer
            original_transfer = self.nixl_wrapper.transfer
            def traced_prepare(*args, **kwargs):
                request_id = getattr(self._cosy_trace_context, "request_id", "unknown")
                started_ns = time.perf_counter_ns()
                handle = original_prepare(*args, **kwargs)
                _append_trace("kv_prepare_complete", request_id=request_id,
                              duration_ns=time.perf_counter_ns() - started_ns,
                              estimated_bytes=self._cosy_transfer_bytes.get(request_id))
                return handle
            def traced_transfer(handle):
                request_id = getattr(self._cosy_trace_context, "request_id", "unknown")
                started_ns = time.perf_counter_ns()
                self._cosy_transfer_started_ns[request_id] = started_ns
                _append_trace("kv_transfer_submit", request_id=request_id,
                              estimated_bytes=self._cosy_transfer_bytes.get(request_id))
                return original_transfer(handle)
            self.nixl_wrapper.make_prepped_xfer = traced_prepare
            self.nixl_wrapper.transfer = traced_transfer

    @staticmethod
    def _normalise_block_ids(block_ids):
        return [list(map(int, group)) for group in (block_ids or [])]

    def _hash_blocks(self, block_ids):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        digest = hashlib.sha256()
        per_cache = {}
        for cache_name, cache_value in sorted(self.device_kv_caches.items()):
            values = cache_value if isinstance(cache_value, (tuple, list)) else (cache_value,)
            cache_hash = hashlib.sha256()
            for part_index, tensor in enumerate(values):
                if not torch.is_tensor(tensor):
                    continue
                for group in block_ids:
                    if not group:
                        continue
                    max_id = max(group)
                    block_dim = 1 if tensor.ndim >= 2 and tensor.shape[1] > max_id else 0
                    index = torch.as_tensor(group, dtype=torch.long, device=tensor.device)
                    selected = tensor.index_select(block_dim, index).detach().contiguous()
                    raw = selected.view(torch.uint8).cpu().numpy().tobytes()
                    cache_hash.update(str(part_index).encode())
                    cache_hash.update(raw)
                    digest.update(cache_name.encode())
                    digest.update(str(part_index).encode())
                    digest.update(raw)
            per_cache[cache_name] = cache_hash.hexdigest()
        return digest.hexdigest(), per_cache

    def _write_kv_hash(self, request_id, snapshot, stage):
        if not self._kv_hash_enabled or not self._kv_hash_file:
            return
        block_ids = snapshot.get("physical_block_ids", [])
        overall, per_cache = self._hash_blocks(block_ids)
        record = {"stage": stage, "engine_id": self.engine_id,
                  "kv_role": getattr(self.kv_transfer_config, "kv_role", None),
                  "request_id": request_id,
                  "remote_request_id": snapshot.get("remote_request_id"),
                  "logical_block_ids": snapshot.get("logical_block_ids", []),
                  "physical_block_ids": block_ids, "sha256": overall,
                  "per_cache_sha256": per_cache}
        path = Path(self._kv_hash_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_source_kv_hashes(self, metadata):
        if not self._kv_hash_enabled:
            return
        for request_id, meta in getattr(metadata, "reqs_to_save", {}).items():
            physical = self._normalise_block_ids(self._logical_to_kernel_block_ids(meta.local_block_ids))
            snapshot = {"stage": "source",
                        "logical_block_ids": self._normalise_block_ids(meta.local_block_ids),
                        "physical_block_ids": physical, "remote_request_id": None}
            self._write_kv_hash(request_id, snapshot, "source_after_prefill")
            self._kv_hash_pending.pop(request_id, None)

    def start_load_kv(self, metadata):
        if self._kv_hash_enabled:
            for request_id, meta in getattr(metadata, "reqs_to_save", {}).items():
                physical = self._normalise_block_ids(self._logical_to_kernel_block_ids(meta.local_block_ids))
                self._kv_hash_pending[request_id] = {"stage": "source",
                    "logical_block_ids": self._normalise_block_ids(meta.local_block_ids),
                    "physical_block_ids": physical, "remote_request_id": None}
            for request_id, meta in getattr(metadata, "reqs_to_recv", {}).items():
                remote = meta.remote
                self._kv_hash_pending[request_id] = {"stage": "receiver",
                    "logical_block_ids": self._normalise_block_ids(meta.local_block_ids),
                    "physical_block_ids": self._normalise_block_ids(meta.local_block_ids),
                    "remote_request_id": getattr(remote, "request_id", None)}
        return super().start_load_kv(metadata)

    def _read_blocks(self, read_spec, dst_engine_id, request_id, remote_request_id,
                     local_xfer_side_handle, remote_xfer_side_handle):
        self._cosy_trace_context.request_id = request_id
        local_groups = getattr(read_spec, "local_block_ids", []) or []
        remote_groups = getattr(read_spec, "remote_block_ids", []) or []
        estimated_bytes = sum(len(group) for group in local_groups) * sum(self.block_len_per_layer)
        self._cosy_transfer_bytes[request_id] = int(estimated_bytes)
        _append_trace("kv_read_prepare_start", request_id=request_id,
                      remote_request_id=remote_request_id,
                      local_block_count=sum(len(group) for group in local_groups),
                      remote_block_count=sum(len(group) for group in remote_groups),
                      estimated_bytes=int(estimated_bytes))
        try:
            return super()._read_blocks(read_spec, dst_engine_id, request_id,
                                        remote_request_id, local_xfer_side_handle,
                                        remote_xfer_side_handle)
        finally:
            self._cosy_trace_context.request_id = None

    def _pop_done_transfers(self, transfers):
        receiving = transfers is self._recving_transfers
        done = super()._pop_done_transfers(transfers)
        if receiving and os.environ.get("COSY_PD_INSTRUMENT_NIXL") == "1":
            completed_ns = time.perf_counter_ns()
            for request_id in done:
                started_ns = self._cosy_transfer_started_ns.pop(request_id, None)
                estimated_bytes = self._cosy_transfer_bytes.pop(request_id, None)
                _append_trace("kv_transfer_complete", request_id=request_id,
                              duration_ns=(None if started_ns is None else completed_ns - started_ns),
                              estimated_bytes=estimated_bytes)
        return done

    def shutdown(self):
        if (self._kv_hash_enabled and
            getattr(self.kv_transfer_config, "kv_role", None) == "kv_producer" and
            self._kv_hash_file):
            ids = [int(item) for item in os.environ.get("COSY_PD_KV_HASH_BLOCK_IDS", "").split(",") if item.strip()]
            if ids:
                self._write_kv_hash("source-blocks", {"logical_block_ids": [ids],
                                  "physical_block_ids": [ids],
                                  "remote_request_id": None}, "source_shutdown")
        return super().shutdown()

    def get_finished(self):
        snapshots = dict(self._kv_hash_pending) if self._kv_hash_enabled else {}
        done_sending, done_recving = super().get_finished()
        if self._kv_hash_enabled:
            for request_id in done_sending:
                snapshot = snapshots.get(request_id)
                if snapshot and snapshot.get("stage") == "source":
                    self._write_kv_hash(request_id, snapshot, "source_after_transfer")
                    self._kv_hash_pending.pop(request_id, None)
            for request_id in done_recving:
                snapshot = snapshots.get(request_id)
                if snapshot and snapshot.get("stage") == "receiver":
                    self._write_kv_hash(request_id, snapshot, "receiver_after_transfer")
                    self._kv_hash_pending.pop(request_id, None)
        if done_recving and os.environ.get("COSY_PD_NIXL_SYNC") == "1":
            import torch
            torch.cuda.synchronize()
        return done_sending, done_recving

class CosyPromptNixlConnector(NixlPullConnector):
    def wait_for_save(self):
        if (os.environ.get("COSY_PD_KV_HASH") == "1" and self.connector_worker is not None
            and self._connector_metadata is not None):
            self.connector_worker.record_source_kv_hashes(self._connector_metadata)
        return super().wait_for_save()

    def shutdown(self):
        if (os.environ.get("COSY_PD_KV_HASH") == "1" and self.connector_worker is not None
            and self._connector_metadata is not None):
            self.connector_worker.record_source_kv_hashes(self._connector_metadata)
        return super().shutdown()

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = CosyPromptNixlPullConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = CosyPromptNixlPullConnectorWorker(vllm_config, self.engine_id, kv_cache_config)
        else:
            raise ValueError(f"Unsupported KV connector role: {role}")
