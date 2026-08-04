#!/usr/bin/env python3
"""Validate and benchmark TECO LMK FFN for the CosyVoice Flow GELU MLP."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def measure(function, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    torch.sdaa.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        function()
        torch.sdaa.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.mean(samples), statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-lengths", default="128,256,512")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    import vllm_sdaa._C  # noqa: F401

    torch.manual_seed(20260803)
    device = torch.device("sdaa")
    hidden_size = 1024
    intermediate_size = 2048
    weight_in = torch.randn(
        intermediate_size, hidden_size, device=device, dtype=torch.float16
    ) / hidden_size**0.5
    bias_in = torch.randn(
        intermediate_size, device=device, dtype=torch.float16
    ) / hidden_size**0.5
    weight_out = torch.randn(
        hidden_size, intermediate_size, device=device, dtype=torch.float16
    ) / intermediate_size**0.5
    bias_out = torch.randn(
        hidden_size, device=device, dtype=torch.float16
    ) / intermediate_size**0.5

    permuted_in = torch.empty_like(weight_in)
    permuted_out = torch.empty_like(weight_out)
    torch.ops._C_sdaa.weight_permute(
        permuted_in, None, weight_in.permute(1, 0).contiguous(), 2, 0, -1
    )
    torch.ops._C_sdaa.weight_permute(
        permuted_out, None, weight_out.permute(1, 0).contiguous(), 3, 0, -1
    )
    torch.sdaa.synchronize()

    results = []
    for sequence_length in [int(value) for value in args.sequence_lengths.split(",")]:
        inputs = torch.randn(
            2 * sequence_length,
            hidden_size,
            device=device,
            dtype=torch.float16,
        )
        fused_output = torch.empty_like(inputs)

        def reference():
            hidden = F.linear(inputs, weight_in, bias_in)
            hidden = F.gelu(hidden, approximate="tanh")
            return F.linear(hidden, weight_out, bias_out)

        def fused():
            torch.ops._C_sdaa.ffnv2(
                fused_output,
                inputs,
                permuted_in,
                permuted_out,
                bias_in,
                "default",
                "gelu",
            )
            return fused_output + bias_out

        reference_output = reference()
        candidate_output = fused()
        torch.sdaa.synchronize()
        difference = (candidate_output - reference_output).float().abs()
        reference_mean, reference_median = measure(
            reference, args.warmup, args.iterations
        )
        fused_mean, fused_median = measure(
            fused, args.warmup, args.iterations
        )
        results.append(
            {
                "sequence_length": sequence_length,
                "rows": 2 * sequence_length,
                "reference_mean_ms": reference_mean,
                "reference_median_ms": reference_median,
                "fused_mean_ms": fused_mean,
                "fused_median_ms": fused_median,
                "mean_change_percent": (fused_mean / reference_mean - 1) * 100,
                "max_absolute_difference": difference.max().item(),
                "mean_absolute_difference": difference.mean().item(),
            }
        )
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
