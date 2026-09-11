#!/usr/bin/env python3
"""Measure decode-A TPOT while injecting a long prefill-B on Unified vLLM."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prompt-payload", type=Path, required=True)
    parser.add_argument("--baseline-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument("--inject-after", type=int, default=40)
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "pretrained_models/Fun-CosyVoice3-0.5B/vllm",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from pd_exp.common import (
        build_engine,
        load_prompt_payload,
        make_sampling_params,
        read_json,
        run_engine_request,
        write_json,
    )
    from pd_exp.compare_tokens import compare

    prompt, metadata = load_prompt_payload(args.prompt_payload)
    baseline = read_json(args.baseline_tokens)
    engine = build_engine(
        args.model_dir, eager=False, max_num_seqs=2,
    )
    try:
        control_tokens, control_metrics, _ = run_engine_request(
            engine,
            prompt,
            make_sampling_params(metadata),
            request_id="cosy-interference-control-a",
        )

        a_id = "cosy-interference-a"
        b_id = "cosy-interference-b"
        engine.add_request(
            a_id,
            {"prompt_embeds": prompt.to("cuda", torch.bfloat16)},
            make_sampling_params(metadata),
        )
        previous = 0
        a_tokens: list[int] = []
        a_times: list[float] = []
        a_finished = False
        b_finished = False
        injected = False
        injection_time = None
        started = time.perf_counter()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and not (a_finished and b_finished):
            outputs = engine.step()
            now = time.perf_counter()
            for output in outputs:
                if output.request_id == a_id:
                    a_tokens = list(output.outputs[0].token_ids)
                    added = len(a_tokens) - previous
                    if added > 0:
                        a_times.extend([now] * added)
                        previous = len(a_tokens)
                    a_finished = output.finished
                elif output.request_id == b_id:
                    b_finished = output.finished
            if not injected and len(a_tokens) >= args.inject_after:
                long_prompt = prompt.repeat((args.repeat, 1)).to(
                    "cuda", torch.bfloat16,
                )
                torch.cuda.nvtx.range_push("COSY_UNIFIED_PREFILL_B_INJECT")
                engine.add_request(
                    b_id,
                    {"prompt_embeds": long_prompt},
                    make_sampling_params(metadata, prefill_only=True),
                )
                torch.cuda.nvtx.range_pop()
                injected = True
                injection_time = time.perf_counter()
            if a_finished and not injected:
                raise RuntimeError("A finished before B injection")
        if not (a_finished and b_finished):
            raise RuntimeError("Unified interference requests timed out")

        stop_ids = set(metadata["stop_token_ids"])
        stop_index = next(
            (i for i, token in enumerate(a_tokens) if token in stop_ids), None
        )
        if stop_index is not None:
            a_tokens = a_tokens[:stop_index]
            a_times = a_times[:stop_index]
        intervals = [
            a_times[i] - a_times[i - 1] for i in range(1, len(a_times))
        ]
        pre = [
            value for i, value in enumerate(intervals, start=1)
            if a_times[i] < injection_time
        ]
        post = [
            value for i, value in enumerate(intervals, start=1)
            if a_times[i] >= injection_time
        ]
        result = {
            "mode": "unified_interference",
            "gpu": args.gpu,
            "long_prompt_tokens": len(prompt) * args.repeat,
            "inject_after_tokens": args.inject_after,
            "control": {
                "correctness": compare(
                    baseline["raw_speech_tokens"], control_tokens,
                ),
                "metrics": control_metrics,
            },
            "interference": {
                "correctness": compare(
                    baseline["raw_speech_tokens"], a_tokens,
                ),
                "total_seconds": time.perf_counter() - started,
                "pre_tpot_p95_seconds": percentile(pre, 0.95),
                "pre_tpot_p99_seconds": percentile(pre, 0.99),
                "post_tpot_p95_seconds": percentile(post, 0.95),
                "post_tpot_p99_seconds": percentile(post, 0.99),
                "max_post_tpot_seconds": max(post, default=None),
                "pre_interval_count": len(pre),
                "post_interval_count": len(post),
            },
        }
        write_json(args.output, result)
        return 0
    finally:
        if callable(getattr(engine, "shutdown", None)):
            engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
