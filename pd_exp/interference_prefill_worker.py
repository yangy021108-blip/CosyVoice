#!/usr/bin/env python3
"""Inject one synthetic long prefill on the P GPU at a progress marker."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-payload", type=Path, required=True)
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from pd_exp.common import (
        build_engine,
        load_prompt_payload,
        make_sampling_params,
        run_engine_request,
        wait_for_file,
        write_json,
    )

    engine = None
    try:
        prompt, metadata = load_prompt_payload(args.prompt_payload)
        long_prompt = prompt.repeat((args.repeat, 1))
        engine = build_engine(
            args.model_dir, eager=True, gpu_memory_utilization=0.2,
        )
        write_json(args.ready, {
            "status": "ready", "prompt_tokens": len(long_prompt),
        })
        wait_for_file(args.signal, args.timeout)
        sampling = make_sampling_params(metadata, prefill_only=True)
        tokens, metrics, _ = run_engine_request(
            engine,
            long_prompt,
            sampling,
            request_id="cosy-interference-prefill-b",
            nvtx_label="COSY_PD_INTERFERENCE_PREFILL_B",
        )
        write_json(args.output, {
            "status": "ok",
            "prompt_tokens": len(long_prompt),
            "auxiliary_tokens": tokens,
            "metrics": metrics,
        })
        return 0
    except Exception as exc:
        write_json(args.output, {
            "status": "error",
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        return 1
    finally:
        if engine is not None and callable(getattr(engine, "shutdown", None)):
            engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
