#!/usr/bin/env python3
"""Unified-vLLM control benchmark for the isolated PD experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prompt-payload", type=Path, default=None)
    parser.add_argument("--prompt-payload-list", type=Path, default=None,
                        help="JSON list of payloads, one per batch slot.")
    parser.add_argument("--baseline-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--engine-mode", choices=("eager", "standard_optimized"),
        default="eager",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "pretrained_models/Fun-CosyVoice3-0.5B/vllm",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.iterations < 1:
        raise ValueError("concurrency and iterations must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from pd_exp.common import (
        build_engine,
        load_prompt_payload,
        make_sampling_params,
        read_json,
        run_engine_requests,
        timing_summary,
        write_json,
    )
    from pd_exp.compare_tokens import compare

    if (args.prompt_payload is None) == (args.prompt_payload_list is None):
        raise ValueError("provide exactly one of --prompt-payload/--prompt-payload-list")
    payload_paths = [args.prompt_payload]
    if args.prompt_payload_list is not None:
        payload_paths = [Path(item) for item in json.loads(
            args.prompt_payload_list.read_text(encoding="utf-8")
        )]
        if len(payload_paths) < args.concurrency:
            raise ValueError("prompt-payload-list must contain at least concurrency payloads")
    payloads = [load_prompt_payload(path) for path in payload_paths]
    baseline = read_json(args.baseline_tokens)
    engine = build_engine(
        args.model_dir,
        eager=args.engine_mode == "eager",
        max_num_seqs=args.concurrency,
    )
    started = time.perf_counter()
    records = []
    try:
        for iteration in range(args.iterations):
            specs = []
            for request_in_batch in range(args.concurrency):
                request_id = (
                    f"cosy-unified-{iteration:04d}-{request_in_batch:04d}"
                )
                request_embeds, request_metadata = payloads[request_in_batch]
                specs.append({
                    "request_id": request_id,
                    "prompt_embeds": request_embeds,
                    "sampling_params": make_sampling_params(request_metadata),
                })
            results = run_engine_requests(
                engine,
                specs,
                nvtx_label="COSY_UNIFIED_LLM",
            )
            iteration_records = []
            for request_in_batch, spec in enumerate(specs):
                tokens, metrics, _ = results[spec["request_id"]]
                iteration_records.append({
                    "iteration": iteration,
                    "request_in_batch": request_in_batch,
                    "raw_speech_tokens": tokens,
                    "metrics": metrics,
                    "tpot": timing_summary(metrics),
                    "token_correctness": compare(
                        baseline["raw_speech_tokens"], tokens,
                    ),
                })
            batch_wall_seconds = max(
                item["metrics"]["total_seconds"]
                for item in iteration_records
            )
            for item in iteration_records:
                item["batch_wall_seconds_upper_bound"] = batch_wall_seconds
            records.extend(iteration_records)
    finally:
        if callable(getattr(engine, "shutdown", None)):
            engine.shutdown()

    warm_iteration = args.iterations - 1
    warm = [item for item in records if item["iteration"] == warm_iteration]
    warm_wall = max(item["batch_wall_seconds_upper_bound"] for item in warm)
    audio_duration = 6.16
    summary = {
        "mode": "unified_vllm",
        "engine_mode": args.engine_mode,
        "gpu": str(args.gpu),
        "concurrency": args.concurrency,
        "iterations": args.iterations,
        "canonical_prompt_payload": (
            None if args.prompt_payload is None
            else str(args.prompt_payload.resolve())
        ),
        "prompt_payload_list": (
            None if args.prompt_payload_list is None
            else str(args.prompt_payload_list.resolve())
        ),
        "all_tokens_equal": all(
            item["token_correctness"]["token_by_token_equal"]
            for item in records
        ),
        "warm_iteration": warm_iteration,
        "warm_batch_wall_seconds_upper_bound": warm_wall,
        "warm_requests_per_second_lower_bound": args.concurrency / warm_wall,
        "warm_audio_seconds_per_second_lower_bound": (
            args.concurrency * audio_duration / warm_wall
        ),
        "total_script_seconds": time.perf_counter() - started,
        "torch_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "records": records,
    }
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_tokens_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
