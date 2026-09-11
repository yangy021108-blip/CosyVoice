#!/usr/bin/env python3
"""NIXL consumer process for the isolated CosyVoice PD experiment."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prefill-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--side-channel-port", type=int, default=5601)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--instrument-nixl", action="store_true")
    parser.add_argument("--standard-optimized", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--progress-request-id", default=None)
    parser.add_argument("--progress-tokens", type=int, default=40)
    parser.add_argument("--step-trace", type=Path, default=None)
    parser.add_argument("--graph-safe-remote-prefill", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(args.side_channel_port)
    os.environ.setdefault("UCX_TLS", "tcp,cuda_ipc,cuda_copy,sm,self")
    os.environ.setdefault("UCX_NET_DEVICES", "lo")
    if args.trace is not None:
        os.environ["COSY_PD_TRACE_FILE"] = str(args.trace)
    if args.instrument_nixl:
        os.environ["COSY_PD_INSTRUMENT_NIXL"] = "1"
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(args.profile_dir)
        os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "0"
    if args.progress_file is not None:
        if not args.progress_request_id:
            raise ValueError("--progress-request-id is required")
        os.environ["COSY_PD_PROGRESS_FILE"] = str(args.progress_file)
        os.environ["COSY_PD_PROGRESS_REQUEST_ID"] = args.progress_request_id
        os.environ["COSY_PD_PROGRESS_TOKENS"] = str(args.progress_tokens)
    if args.step_trace is not None:
        os.environ["COSY_PD_STEP_TRACE_FILE"] = str(args.step_trace)
    if args.graph_safe_remote_prefill:
        os.environ["COSY_PD_GRAPH_SAFE_REMOTE_PREFILL"] = "1"
    from pd_exp.common import (
        build_engine,
        indexed_path,
        load_prompt_payload,
        make_sampling_params,
        read_json,
        run_engine_requests,
        wait_for_file,
        write_json,
    )

    engine = None
    try:
        prompt_embeds, metadata = load_prompt_payload(args.input)
        engine = build_engine(
            args.model_dir,
            kv_role="kv_consumer",
            eager=not args.standard_optimized,
            max_num_seqs=args.batch_size,
        )
        write_json(args.ready, {"status": "ready", "gpu": args.gpu})
        if args.profile_dir is not None:
            engine.start_profile()
        total_requests = args.iterations * args.batch_size
        for index in range(args.iterations):
            request_specs = []
            request_ids = []
            for batch_index in range(args.batch_size):
                linear_index = index * args.batch_size + batch_index
                prefill_output = indexed_path(
                    args.prefill_output, linear_index, total_requests
                )
                wait_for_file(prefill_output, args.timeout)
                prefill = read_json(prefill_output)
                if prefill.get("status") != "ok":
                    raise RuntimeError(f"Prefill worker failed: {prefill}")
                sampling = make_sampling_params(
                    metadata,
                    kv_transfer_params=prefill["kv_transfer_params"],
                    prefill_only=False,
                )
                request_id = (
                    "cosy-pd-decode"
                    if total_requests == 1
                    else f"cosy-pd-decode-{linear_index:04d}"
                )
                request_ids.append(request_id)
                request_specs.append({
                    "request_id": request_id,
                    "prompt_embeds": prompt_embeds,
                    "sampling_params": sampling,
                })
            results = run_engine_requests(
                engine,
                request_specs,
                timeout_seconds=args.timeout,
                nvtx_label="COSY_PD_DECODE",
            )
            for batch_index, request_id in enumerate(request_ids):
                linear_index = index * args.batch_size + batch_index
                tokens, metrics, _ = results[request_id]
                output = indexed_path(args.output, linear_index, total_requests)
                release = indexed_path(
                    args.release, linear_index, total_requests
                )
                write_json(output, {
                    "status": "ok",
                    "gpu": args.gpu,
                    "iteration": index,
                    "request_in_batch": batch_index,
                    "raw_speech_tokens": tokens,
                    "metrics": metrics,
                })
                write_json(release, {
                    "status": "decode_complete",
                    "iteration": index,
                    "request_in_batch": batch_index,
                })
        return 0
    except Exception as exc:
        write_json(args.output, {
            "status": "error",
            "gpu": args.gpu,
            "stage": "decode",
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        write_json(args.release, {"status": "decode_failed"})
        return 1
    finally:
        if (
            engine is not None
            and args.profile_dir is not None
            and callable(getattr(engine, "stop_profile", None))
        ):
            engine.stop_profile()
        if engine is not None and callable(getattr(engine, "shutdown", None)):
            engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
