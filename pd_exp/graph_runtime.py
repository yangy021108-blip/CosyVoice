"""Opt-in diagnostics and graph safety for the CosyVoice PD experiment.

This module patches vLLM only inside the experiment process.  It neither
modifies site-packages nor changes the production API backend.
"""

from __future__ import annotations

import json
import os
import threading
import torch
import time
from pathlib import Path
from typing import Any


_TRACE_LOCK = threading.Lock()
_TLS = threading.local()


def _trace(event: str, **fields: Any) -> None:
    path_values = [
        value
        for value in (
            os.environ.get("COSY_PD_STEP_TRACE_FILE"),
            os.environ.get("COSY_PD_LOGITS_DUMP_FILE"),
        )
        if value
    ]
    if not path_values:
        return
    record = {
        "event": event,
        "monotonic_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        **fields,
    }
    payload = json.dumps(record, sort_keys=True) + "\n"
    with _TRACE_LOCK:
        for path_value in dict.fromkeys(path_values):
            path = Path(path_value)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(payload)


def _prompt_length(new_request: Any) -> int | None:
    embeds = getattr(new_request, "prompt_embeds", None)
    if embeds is not None:
        return int(embeds.shape[0])
    token_ids = getattr(new_request, "prompt_token_ids", None)
    return None if token_ids is None else len(token_ids)


def is_single_token_prompt_continuation(scheduler_output: Any, model_runner: Any | None = None) -> bool:
    """Return true when a one-token batch is still computing the prompt.

    vLLM 0.11 labels every uniform query-length-one batch as decode.  A PD
    consumer can, however, receive all but the last prompt KV tokens and then
    locally compute one final prompt embedding.  That step is prefill, not
    decode, and must not replay a decode-only CUDA graph.
    """
    # vLLM 0.25.1 provides the authoritative computed count for cached
    # requests in SchedulerOutput.scheduled_cached_reqs. Use it before the
    # worker InputBatch is updated for this iteration.
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is not None and model_runner is not None:
        batch = getattr(model_runner, "input_batch", None)
        req_ids = list(getattr(cached, "req_ids", ()))
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})
        batch_req_ids = list(getattr(batch, "req_ids", ())) if batch is not None else []
        prompt_lengths = getattr(batch, "num_prompt_tokens", ()) if batch is not None else ()
        for index, req_id in enumerate(req_ids):
            if int(scheduled_tokens.get(req_id, 0)) != 1:
                continue
            try:
                batch_index = batch_req_ids.index(req_id)
                prompt_length = int(prompt_lengths[batch_index])
            except (ValueError, IndexError, TypeError):
                continue
            computed = int(cached.num_computed_tokens[index])
            # Once the first speech token has been emitted, the request is in
            # generation even if the scheduler's cached count lags by one.
            output_tokens = int(cached.num_output_tokens[index])
            if output_tokens == 0 and computed + int(scheduled_tokens.get(req_id, 0)) == prompt_length:
                return True

    # vLLM 0.25.1 places an already-running request in
    # scheduled_cached_reqs, so it is not present in scheduled_new_reqs.
    # InputBatch retains the prompt length and current computed count for both
    # request kinds and is available before _prepare_inputs mutates the batch.
    if model_runner is not None and cached is None:
        batch = getattr(model_runner, "input_batch", None)
        req_ids = list(getattr(batch, "req_ids", ())) if batch is not None else []
        scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})
        prompt_lengths = getattr(batch, "num_prompt_tokens", ()) if batch is not None else ()
        computed_tokens = getattr(batch, "num_computed_tokens_cpu", ()) if batch is not None else ()
        for index, req_id in enumerate(req_ids):
            scheduled = int(scheduled_tokens.get(req_id, 0))
            if scheduled != 1 or index >= len(prompt_lengths) or index >= len(computed_tokens):
                continue
            if int(computed_tokens[index]) + scheduled == int(prompt_lengths[index]):
                return True
    for request in getattr(scheduler_output, "scheduled_new_reqs", ()):
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}).get(
            request.req_id, 0
        )
        prompt_length = _prompt_length(request)
        if (
            scheduled == 1
            and prompt_length is not None
            and int(request.num_computed_tokens) + int(scheduled) == prompt_length
        ):
            return True
    return False


def _sampling_metadata_debug(metadata: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalties",
        "presence_penalties",
        "frequency_penalties",
    ):
        value = getattr(metadata, name, None)
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        result[name] = value
    output_token_ids = getattr(metadata, "output_token_ids", None)
    if output_token_ids is not None:
        result["output_token_ids"] = output_token_ids
    return result


def _topk(logits: Any, k: int = 10) -> list[dict[str, list[float] | list[int]]]:
    if logits is None:
        return []
    values, indices = logits.detach().float().topk(min(k, logits.shape[-1]), dim=-1)
    return [
        {"token_ids": row_ids.tolist(), "values": row_values.tolist()}
        for row_ids, row_values in zip(indices.cpu(), values.cpu())
    ]


def install_vllm_runtime_patch() -> None:
    """Install opt-in step tracing and the targeted PD graph correctness fix."""
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_cosy_pd_runtime_patch", False):
        return

    original_execute = GPUModelRunner.execute_model
    original_prepare = GPUModelRunner._prepare_inputs
    original_sample = GPUModelRunner._sample
    original_dispatch = CudagraphDispatcher.dispatch
    from vllm.v1.sample.sampler import Sampler
    original_sampler_sample = Sampler.sample

    def sampler_sample(self: Any, logits: Any, sampling_metadata: Any, *args: Any, **kwargs: Any):
        if (
            getattr(_TLS, "context", None) is not None
            and os.environ.get("COSY_PD_LOGITS_DUMP_FILE")
        ):
            _TLS.processed_logits = logits.detach().float().cpu().tolist()
        return original_sampler_sample(self, logits, sampling_metadata, *args, **kwargs)

    def execute_model(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
        force_eager = (
            os.environ.get("COSY_PD_GRAPH_SAFE_REMOTE_PREFILL") == "1"
            and is_single_token_prompt_continuation(scheduler_output, self)
        )
        previous = self.uniform_decode_query_len
        if force_eager:
            # Make the existing uniform-decode predicate false for this one
            # model invocation. FULL_DECODE_ONLY then dispatches to NONE.
            self.uniform_decode_query_len = -1
            _trace("graph_safety_bypass", reason="single_token_prompt_continuation")
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            self.uniform_decode_query_len = previous

    def prepare_inputs(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
        result = original_prepare(self, scheduler_output, *args, **kwargs)
        if not os.environ.get("COSY_PD_STEP_TRACE_FILE"):
            return result
        req_ids = list(self.input_batch.req_ids)
        query_lens = [
            int(scheduler_output.num_scheduled_tokens[req_id]) for req_id in req_ids
        ]
        total = sum(query_lens)
        rows = []
        position_offset = 0
        block_tables = self.input_batch.block_table.block_tables
        for req_index, (req_id, query_len) in enumerate(zip(req_ids, query_lens)):
            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
            prompt_len = int(self.input_batch.num_prompt_tokens[req_index])
            block_table_rows = []
            for table in block_tables:
                count = int(table.num_blocks_per_row[req_index])
                block_table_rows.append(
                    table.block_table.np[req_index, :count].astype(int).tolist()
                )
            positions = self.positions.detach().cpu().numpy()[
                position_offset : position_offset + query_len
            ].astype(int).tolist()
            slot_mapping = [
                table.slot_mapping.gpu.detach().cpu().numpy()[
                    position_offset : position_offset + query_len
                ].astype(int).tolist()
                for table in block_tables
            ]
            rows.append({
                "request_id": req_id,
                "position": positions,
                "seq_len": num_computed + query_len,
                "query_len": query_len,
                "num_computed_tokens": num_computed,
                "num_prompt_tokens": prompt_len,
                "prompt_continuation": num_computed < prompt_len,
                "block_table": block_table_rows,
                "slot_mapping": slot_mapping,
            })
            position_offset += query_len
        context = {"rows": rows, "total_num_scheduled_tokens": total}
        _TLS.context = context
        _TLS.processed_logits = None
        _trace("decode_step_input", **context)
        return result

    def dispatch(self: Any, *args: Any, **kwargs: Any):
        # vLLM 0.25.1 dispatches with scalar keyword arguments; older vLLM
        # passed a BatchDescriptor. Preserve both call contracts so tracing
        # remains opt-in and cannot affect normal execution.
        mode, resolved = original_dispatch(self, *args, **kwargs)
        context = getattr(_TLS, "context", None)
        if context is not None:
            context["cudagraph_mode"] = getattr(mode, "name", str(mode))
            descriptor = args[0] if args else None
            uniform_decode = kwargs.get(
                "uniform_decode", getattr(descriptor, "uniform_decode", False)
            )
            graph_num_tokens = kwargs.get(
                "num_tokens", getattr(descriptor, "num_tokens", 0)
            )
            context["uniform_decode"] = bool(uniform_decode)
            context["graph_num_tokens"] = int(graph_num_tokens)
            _trace(
                "cudagraph_dispatch",
                cudagraph_mode=context["cudagraph_mode"],
                uniform_decode=context["uniform_decode"],
                graph_num_tokens=context["graph_num_tokens"],
            )
        return mode, resolved

    def sample(self: Any, logits: Any, spec_decode_metadata: Any):
        argmax_mode = os.environ.get("COSY_PD_ARGMAX") == "1"
        dump_logits = bool(os.environ.get("COSY_PD_LOGITS_DUMP_FILE"))
        if argmax_mode and logits is not None:
            from vllm.v1.outputs import SamplerOutput

            argmax_ids = logits.argmax(dim=-1).to(torch.int32).view(-1, 1)
            output = SamplerOutput(
                sampled_token_ids=argmax_ids,
                logprobs_tensors=None,
            )
        else:
            output = original_sample(self, logits, spec_decode_metadata)
        processed_topk = _topk(logits)
        sampled = output.sampled_token_ids.detach().cpu().tolist()
        context = getattr(_TLS, "context", {})
        extra = {"argmax_mode": argmax_mode}
        if dump_logits and logits is not None:
            extra["logits_full"] = logits.detach().float().cpu().tolist()
        processed_logits = getattr(_TLS, "processed_logits", None)
        if processed_logits is not None:
            extra["logits_processed_before_top_p"] = processed_logits
        _TLS.processed_logits = None
        if os.environ.get("COSY_PD_DUMP_SAMPLING_METADATA") == "1":
            metadata = self.input_batch.sampling_metadata
            extra["sampling_metadata"] = _sampling_metadata_debug(metadata)
        _trace(
            "decode_step_output",
            rows=context.get("rows", []),
            cudagraph_mode=context.get("cudagraph_mode"),
            uniform_decode=context.get("uniform_decode"),
            logits_topk=processed_topk,
            logits_stage="model_logits_before_sampler",
            sampled_token_ids=sampled,
            **extra,
        )
        return output

    GPUModelRunner.execute_model = execute_model
    GPUModelRunner._prepare_inputs = prepare_inputs
    GPUModelRunner._sample = sample
    CudagraphDispatcher.dispatch = dispatch
    Sampler.sample = sampler_sample
    GPUModelRunner._cosy_pd_runtime_patch = True
