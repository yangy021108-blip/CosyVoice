#!/usr/bin/env python3
"""NIXL producer process for the isolated CosyVoice PD experiment."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--input-list", type=Path, default=None,
                        help="JSON list of payload paths, one per batch slot.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--side-channel-port", type=int, default=5600)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--instrument-nixl", action="store_true")
    parser.add_argument("--standard-optimized", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--truncate-last-prompt-token", action="store_true",
                        help="Experimental: export KV for prompt[:-1] only.")
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
    from pd_exp.common import (
        build_engine,
        load_prompt_payload,
        make_sampling_params,
        indexed_path,
        run_engine_requests,
        wait_for_file,
        write_json,
    )
    import torch

    engine = None
    try:
        if (args.input is None) == (args.input_list is None):
            raise ValueError("provide exactly one of --input or --input-list")
        payload_paths = None
        if args.input_list is not None:
            import json
            payload_paths = [Path(item) for item in json.loads(
                args.input_list.read_text(encoding="utf-8")
            )]
            if len(payload_paths) < args.batch_size:
                raise ValueError("input-list must contain at least batch-size payloads")
        first_payload = args.input if payload_paths is None else payload_paths[0]
        prompt_embeds, metadata = load_prompt_payload(first_payload)
        if args.truncate_last_prompt_token:
            if prompt_embeds.shape[0] <= 1:
                raise ValueError("Cannot truncate a one-token prompt")
            prompt_embeds = prompt_embeds[:-1]
            metadata = dict(metadata)
            metadata["prompt_embeds_shape"] = list(prompt_embeds.shape)
        engine = build_engine(
            args.model_dir,
            kv_role="kv_producer",
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
                transfer_request = {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                }
                request_payload = (
                    args.input if payload_paths is None
                    else payload_paths[batch_index]
                )
                request_embeds, request_metadata = load_prompt_payload(
                    request_payload
                )
                sampling = make_sampling_params(
                    request_metadata,
                    kv_transfer_params=transfer_request,
                    prefill_only=True,
                )
                request_id = (
                    "cosy-pd-prefill"
                    if total_requests == 1
                    else f"cosy-pd-prefill-{linear_index:04d}"
                )
                request_ids.append(request_id)
                request_specs.append({
                    "request_id": request_id,
                    "prompt_embeds": request_embeds,
                    "sampling_params": sampling,
                })
            results = run_engine_requests(
                engine,
                request_specs,
                timeout_seconds=args.timeout,
                nvtx_label="COSY_PD_PREFILL",
            )
            for batch_index, request_id in enumerate(request_ids):
                linear_index = index * args.batch_size + batch_index
                tokens, metrics, transfer_params = results[request_id]
                output = indexed_path(
                    args.output, linear_index, total_requests
                )
                if not transfer_params:
                    raise RuntimeError(
                        "NixlConnector returned no KV transfer metadata"
                    )
                if torch.cuda.is_available():
                    # NIXL is pull-based on this host: this mark means the
                    # producer has exposed metadata/KV, not a CPU-side send.
                    torch.cuda.nvtx.mark("COSY_PD_KV_SEND")
                write_json(output, {
                    "status": "ok",
                    "gpu": args.gpu,
                    "iteration": index,
                    "request_in_batch": batch_index,
                    "auxiliary_tokens_discarded": tokens,
                    "metrics": metrics,
                    "kv_transfer_params": transfer_params,
                })
            for batch_index in range(args.batch_size):
                linear_index = index * args.batch_size + batch_index
                release = indexed_path(
                    args.release, linear_index, total_requests
                )
                wait_for_file(release, args.timeout)
        return 0
    except Exception as exc:
        write_json(args.output, {
            "status": "error",
            "gpu": args.gpu,
            "stage": "prefill",
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
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
