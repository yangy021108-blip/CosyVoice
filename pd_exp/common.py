"""Shared primitives for the isolated PD experiment.

This module deliberately does not change the production CosyVoice inference
path.  It uses the same prompt construction and sampling values while driving
vLLM directly so KV-transfer metadata can be observed.
"""

from __future__ import annotations

import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch


# Keep the source ASCII-only so Windows/SSH locale conversions cannot silently
# change the fixed correctness input.
DEFAULT_TEXT = (
    "\u4f60\u597d\uff0c\u8fd9\u662f CosyVoice3 vLLM Prefill Decode "
    "\u5206\u79bb\u6d4b\u8bd5\u3002"
)
DEFAULT_SEED = 20260726
DEFAULT_TOP_K = 25
DEFAULT_TOP_P = 0.8
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_file(path: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.05)


def register_cosyvoice_model() -> None:
    """Register only when the exported checkpoint names the custom class."""
    from vllm import ModelRegistry

    if (
        os.environ.get("COSY_PD_STEP_TRACE_FILE")
        or os.environ.get("COSY_PD_GRAPH_SAFE_REMOTE_PREFILL") == "1"
        or os.environ.get("COSY_PD_ARGMAX") == "1"
        or os.environ.get("COSY_PD_LOGITS_DUMP_FILE")
    ):
        from pd_exp.cosyvoice_debug_model import CosyVoice2ForCausalLM
    else:
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)


def wrap_nvtx_method(obj: Any, name: str, label: str) -> None:
    """Add an NVTX range without changing the wrapped return contract."""
    original = getattr(obj, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(label)
        try:
            result = original(*args, **kwargs)
        except BaseException:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
            raise
        if inspect.isgenerator(result):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()

            def marked_generator():
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(label)
                try:
                    yield from result
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()

            return marked_generator()
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
        return result

    setattr(obj, name, wrapped)


def build_engine(
    model_dir: Path,
    *,
    kv_role: str | None = None,
    eager: bool = True,
    gpu_memory_utilization: float = 0.2,
    max_num_seqs: int = 1,
):
    register_cosyvoice_model()
    from vllm import EngineArgs, LLMEngine
    from vllm.config import KVTransferConfig

    kv_config = None
    if kv_role is not None:
        kv_config = KVTransferConfig(
            kv_connector="CosyPromptNixlConnector",
            kv_role=kv_role,
            kv_buffer_device="cuda",
            kv_connector_module_path=os.environ.get("COSY_PD_CONNECTOR_MODULE", "pd_exp.prompt_nixl_connector"),
        )
    engine_kwargs = {}
    if not eager:
        from vllm.config import CUDAGraphMode, CompilationConfig

        engine_kwargs["compilation_config"] = CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_copy_inputs=os.environ.get("COSY_PD_CUDAGRAPH_COPY_INPUTS") == "1",
        )
    args = EngineArgs(
        model=str(model_dir),
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        enforce_eager=eager,
        max_num_seqs=max_num_seqs,
        kv_transfer_config=kv_config,
        **engine_kwargs,
    )
    return LLMEngine.from_engine_args(args)


def make_sampling_params(
    metadata: dict[str, Any],
    *,
    kv_transfer_params: dict[str, Any] | None = None,
    prefill_only: bool = False,
):
    from vllm import SamplingParams

    extra_args = None
    if kv_transfer_params is not None:
        extra_args = {"kv_transfer_params": kv_transfer_params}
    return SamplingParams(
        top_k=int(metadata["top_k"]),
        top_p=float(metadata["top_p"]),
        repetition_penalty=float(metadata["repetition_penalty"]),
        presence_penalty=float(metadata["presence_penalty"]),
        frequency_penalty=float(metadata["frequency_penalty"]),
        stop_token_ids=[int(item) for item in metadata["stop_token_ids"]],
        min_tokens=0 if prefill_only else int(metadata["min_tokens"]),
        max_tokens=1 if prefill_only else int(metadata["max_tokens"]),
        seed=int(metadata["seed"]),
        extra_args=extra_args,
    )


def indexed_path(path: Path, index: int, iterations: int) -> Path:
    if iterations == 1:
        return path
    return path.with_name(f"{path.stem}_{index:04d}{path.suffix}")


@torch.inference_mode()
def run_engine_request(
    engine: Any,
    prompt_embeds: torch.Tensor,
    sampling_params: Any,
    *,
    request_id: str | None = None,
    timeout_seconds: float = 300.0,
    nvtx_label: str | None = None,
) -> tuple[list[int], dict[str, Any], dict[str, Any] | None]:
    request_id = request_id or f"cosy-pd-{uuid.uuid4()}"
    prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
    nvtx_pushed = False
    if nvtx_label and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(nvtx_label)
        nvtx_pushed = True
    started = time.perf_counter()
    engine.add_request(
        request_id,
        {"prompt_embeds": prompt_embeds},
        sampling_params,
    )
    first_token_at: float | None = None
    token_times: list[float] = []
    output_batch_sizes: list[int] = []
    previous_count = 0
    final_output = None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        request_outputs = engine.step()
        now = time.perf_counter()
        for output in request_outputs:
            if output.request_id != request_id:
                raise RuntimeError(
                    f"Unexpected request {output.request_id}; expected {request_id}"
                )
            final_output = output
            tokens = list(output.outputs[0].token_ids)
            added = len(tokens) - previous_count
            if added > 0:
                output_batch_sizes.append(added)
                if first_token_at is None:
                    first_token_at = now
                token_times.extend([now] * added)
                previous_count = len(tokens)
        if final_output is not None and final_output.finished:
            break
        if not request_outputs:
            # NIXL handshake and async GPU READ can legitimately span several
            # empty scheduler iterations. Mirror the production driver's
            # cooperative polling instead of consuming a fixed step budget.
            time.sleep(0.001)
    if final_output is None or not final_output.finished:
        engine.abort_request([request_id])
        raise RuntimeError("vLLM request did not finish within the step bound")

    finished = time.perf_counter()
    returned_tokens = [int(item) for item in final_output.outputs[0].token_ids]
    # CosyVoice's production inference_wrapper stops before yielding any of
    # the three reserved stop tokens.  vLLM may include the matched stop token
    # in CompletionOutput.token_ids, so mirror that behavior here before Flow.
    stop_ids = set(int(item) for item in sampling_params.stop_token_ids)
    stop_index = next(
        (index for index, token in enumerate(returned_tokens) if token in stop_ids),
        None,
    )
    tokens = (
        returned_tokens if stop_index is None else returned_tokens[:stop_index]
    )
    token_times = token_times[: len(tokens)]
    intervals = [
        token_times[index] - token_times[index - 1]
        for index in range(1, len(token_times))
    ]
    metrics_obj = getattr(final_output, "metrics", None)
    output_delivery_span = (
        None
        if len(token_times) < 2
        else token_times[-1] - token_times[0]
    )
    tpot_observation_valid = (
        max(output_batch_sizes, default=0) <= 1
        and (
            output_delivery_span is None
            or output_delivery_span >= 0.5 * (finished - started)
        )
    )
    metrics = {
        "request_id": request_id,
        "total_seconds": finished - started,
        "first_token_seconds": (
            None if first_token_at is None else first_token_at - started
        ),
        "token_timestamps_seconds": [value - started for value in token_times],
        "token_intervals_seconds": intervals,
        "mean_tpot_seconds": (
            sum(intervals) / len(intervals) if intervals else None
        ),
        "token_count": len(tokens),
        "output_batch_sizes": output_batch_sizes,
        "max_tokens_per_output_event": max(output_batch_sizes, default=0),
        "output_delivery_span_seconds": output_delivery_span,
        "tpot_observation_valid": tpot_observation_valid,
        "returned_token_count_including_stop": len(returned_tokens),
        "reserved_stop_token": (
            None if stop_index is None else returned_tokens[stop_index]
        ),
        "num_cached_tokens": getattr(final_output, "num_cached_tokens", None),
        "finish_reason": getattr(final_output.outputs[0], "finish_reason", None),
        "stop_reason": getattr(final_output.outputs[0], "stop_reason", None),
        "engine_metrics": {
            name: getattr(metrics_obj, name, None)
            for name in (
                "arrival_time",
                "first_scheduled_time",
                "first_token_time",
                "finished_time",
                "scheduler_time",
                "model_forward_time",
                "model_execute_time",
            )
        },
    }
    if nvtx_pushed:
        torch.cuda.nvtx.range_pop()
    return tokens, metrics, getattr(final_output, "kv_transfer_params", None)


@torch.inference_mode()
def run_engine_requests(
    engine: Any,
    requests: list[dict[str, Any]],
    *,
    timeout_seconds: float = 300.0,
    nvtx_label: str | None = None,
) -> dict[str, tuple[list[int], dict[str, Any], dict[str, Any] | None]]:
    """Drive several requests concurrently through one offline V1 engine."""
    if not requests:
        return {}
    nvtx_pushed = False
    if nvtx_label and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(nvtx_label)
        nvtx_pushed = True
    states: dict[str, dict[str, Any]] = {}
    for item in requests:
        request_id = str(item["request_id"])
        prompt_embeds = item["prompt_embeds"].to(
            device="cuda", dtype=torch.bfloat16,
        )
        started = time.perf_counter()
        engine.add_request(
            request_id,
            {"prompt_embeds": prompt_embeds},
            item["sampling_params"],
        )
        states[request_id] = {
            "started": started,
            "first_token_at": None,
            "token_times": [],
            "previous_count": 0,
            "output_batch_sizes": [],
            "final_output": None,
            "finished": None,
            "sampling_params": item["sampling_params"],
        }

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        request_outputs = engine.step()
        now = time.perf_counter()
        for output in request_outputs:
            if output.request_id not in states:
                raise RuntimeError(f"Unexpected request {output.request_id}")
            state = states[output.request_id]
            state["final_output"] = output
            tokens = list(output.outputs[0].token_ids)
            added = len(tokens) - state["previous_count"]
            if added > 0:
                state["output_batch_sizes"].append(added)
                if state["first_token_at"] is None:
                    state["first_token_at"] = now
                state["token_times"].extend([now] * added)
                state["previous_count"] = len(tokens)
                progress_path = os.environ.get("COSY_PD_PROGRESS_FILE")
                progress_request = os.environ.get(
                    "COSY_PD_PROGRESS_REQUEST_ID"
                )
                progress_tokens = int(
                    os.environ.get("COSY_PD_PROGRESS_TOKENS", "40")
                )
                if (
                    progress_path
                    and output.request_id == progress_request
                    and len(tokens) >= progress_tokens
                    and not state.get("progress_emitted")
                ):
                    write_json(Path(progress_path), {
                        "request_id": output.request_id,
                        "observed_tokens": len(tokens),
                        "monotonic_ns": time.perf_counter_ns(),
                    })
                    state["progress_emitted"] = True
            if output.finished and state["finished"] is None:
                state["finished"] = now
        if all(state["finished"] is not None for state in states.values()):
            break
        if not request_outputs:
            time.sleep(0.001)

    unfinished = [
        request_id
        for request_id, state in states.items()
        if state["finished"] is None
    ]
    if unfinished:
        engine.abort_request(unfinished)
        raise RuntimeError(f"vLLM requests timed out: {unfinished}")

    results = {}
    for request_id, state in states.items():
        final_output = state["final_output"]
        sampling_params = state["sampling_params"]
        returned_tokens = [
            int(item) for item in final_output.outputs[0].token_ids
        ]
        stop_ids = set(int(item) for item in sampling_params.stop_token_ids)
        stop_index = next(
            (
                index
                for index, token in enumerate(returned_tokens)
                if token in stop_ids
            ),
            None,
        )
        tokens = (
            returned_tokens
            if stop_index is None
            else returned_tokens[:stop_index]
        )
        token_times = state["token_times"][: len(tokens)]
        intervals = [
            token_times[index] - token_times[index - 1]
            for index in range(1, len(token_times))
        ]
        total_seconds = state["finished"] - state["started"]
        output_delivery_span = (
            None
            if len(token_times) < 2
            else token_times[-1] - token_times[0]
        )
        batch_sizes = state["output_batch_sizes"]
        metrics_obj = getattr(final_output, "metrics", None)
        metrics = {
            "request_id": request_id,
            "total_seconds": total_seconds,
            "first_token_seconds": (
                None
                if state["first_token_at"] is None
                else state["first_token_at"] - state["started"]
            ),
            "token_timestamps_seconds": [
                value - state["started"] for value in token_times
            ],
            "token_intervals_seconds": intervals,
            "mean_tpot_seconds": (
                sum(intervals) / len(intervals) if intervals else None
            ),
            "token_count": len(tokens),
            "returned_token_count_including_stop": len(returned_tokens),
            "reserved_stop_token": (
                None if stop_index is None else returned_tokens[stop_index]
            ),
            "output_batch_sizes": batch_sizes,
            "max_tokens_per_output_event": max(batch_sizes, default=0),
            "output_delivery_span_seconds": output_delivery_span,
            "tpot_observation_valid": (
                max(batch_sizes, default=0) <= 1
                and (
                    output_delivery_span is None
                    or output_delivery_span >= 0.5 * total_seconds
                )
            ),
            "num_cached_tokens": getattr(
                final_output, "num_cached_tokens", None
            ),
            "finish_reason": getattr(
                final_output.outputs[0], "finish_reason", None
            ),
            "stop_reason": getattr(
                final_output.outputs[0], "stop_reason", None
            ),
            "engine_metrics": {
                name: getattr(metrics_obj, name, None)
                for name in (
                    "arrival_time",
                    "first_scheduled_time",
                    "first_token_time",
                    "finished_time",
                    "scheduler_time",
                    "model_forward_time",
                    "model_execute_time",
                )
            },
        }
        results[request_id] = (
            tokens,
            metrics,
            getattr(final_output, "kv_transfer_params", None),
        )
    if nvtx_pushed:
        torch.cuda.nvtx.range_pop()
    return results


def prompt_payload(
    prompt_embeds: torch.Tensor,
    *,
    spans: dict[str, Any],
    minimum_tokens: int,
    maximum_tokens: int,
    seed: int,
    stop_token_ids: list[int],
) -> dict[str, Any]:
    return {
        "prompt_embeds": prompt_embeds.detach().to("cpu", torch.bfloat16),
        "metadata": {
            "prompt_embeds_shape": list(prompt_embeds.shape),
            "prompt_embeds_dtype": "bfloat16",
            "spans": spans,
            "min_tokens": minimum_tokens,
            "max_tokens": maximum_tokens,
            "seed": seed,
            "top_k": DEFAULT_TOP_K,
            "top_p": DEFAULT_TOP_P,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "presence_penalty": DEFAULT_PRESENCE_PENALTY,
            "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
            "stop_token_ids": stop_token_ids,
        },
    }


def load_prompt_payload(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value["prompt_embeds"], value["metadata"]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    intervals = [float(item) for item in metrics["token_intervals_seconds"]]
    return {
        "mean": percentile(intervals, 0.5) if len(intervals) == 1 else (
            sum(intervals) / len(intervals) if intervals else None
        ),
        "p50": percentile(intervals, 0.50),
        "p95": percentile(intervals, 0.95),
        "p99": percentile(intervals, 0.99),
    }
