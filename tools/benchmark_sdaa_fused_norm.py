#!/usr/bin/env python3
"""Validate and benchmark the CosyVoice SDAA fused Flow norm POC."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = (
    ROOT
    / "cosyvoice"
    / "sdaa_ops"
    / "sdaa_mm_encoder_fa_poc_ext.cpython-310-x86_64-linux-gnu.so"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def synchronize() -> None:
    torch.sdaa.synchronize()


def measure(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    args = parse_args()
    torch.ops.load_library(str(EXTENSION))
    torch.manual_seed(20260729)
    device = torch.device("sdaa")
    dtype = torch.float16
    shape = (args.batch, args.sequence, args.dim)
    parameter_shape = (args.batch, args.dim)
    input_tensor = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype)
    gate = torch.randn(parameter_shape, device=device, dtype=dtype)
    gamma = torch.randn(parameter_shape, device=device, dtype=dtype)
    beta = torch.randn(parameter_shape, device=device, dtype=dtype)
    eps = 1e-6

    def reference():
        residual_out = residual + gate[:, None, :] * input_tensor
        output = F.layer_norm(
            residual_out,
            (args.dim,),
            eps=eps,
        )
        output = output * (1 + gamma[:, None, :]) + beta[:, None, :]
        return output, residual_out

    results = {}
    reference_output, reference_residual = reference()

    def custom():
        gated_input = gate[:, None, :] * input_tensor
        output, residual_out = torch.ops._C_sdaa_poc.flow_fused_norm(
            gated_input,
            residual,
            gamma,
            beta,
            eps,
        )
        return output, residual_out

    output, residual_out = custom()
    synchronize()
    results["output_max_abs_diff"] = float(
        (output - reference_output).abs().max().cpu()
    )
    results["residual_max_abs_diff"] = float(
        (residual_out - reference_residual).abs().max().cpu()
    )
    results["custom_mean_ms"] = measure(
        custom,
        args.warmup,
        args.iterations,
    )

    results["reference_mean_ms"] = measure(
        reference,
        args.warmup,
        args.iterations,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
