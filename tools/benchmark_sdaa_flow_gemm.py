#!/usr/bin/env python3
"""Benchmark SDAA Flow linear layers against the TECO fused GEMM backend."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F
import torch_sdaa  # noqa: F401
import vllm_sdaa._C  # noqa: F401


def measure(function, iterations: int) -> float:
    values = []
    for _ in range(iterations):
        torch.sdaa.synchronize()
        started = time.perf_counter()
        function()
        torch.sdaa.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return statistics.fmean(values)


def run_shape(rows: int, input_dim: int, output_dim: int,
              iterations: int) -> dict[str, object]:
    x = torch.randn(rows, input_dim, device="sdaa", dtype=torch.float16)
    weight = torch.randn(
        output_dim, input_dim, device="sdaa", dtype=torch.float16
    ) / input_dim**0.5
    bias = torch.randn(output_dim, device="sdaa", dtype=torch.float16)
    packed_weight = torch.ops._C_sdaa.blas_gemm_fusion_weight(
        weight.T.contiguous().cpu()
    ).to("sdaa")

    def reference() -> torch.Tensor:
        return F.linear(x, weight, bias)

    def candidate() -> torch.Tensor:
        return torch.ops._C_sdaa.blas_gemm_fusion(
            x, packed_weight, False, False
        ).add_(bias)

    for _ in range(10):
        reference()
        candidate()
    torch.sdaa.synchronize()
    expected = reference()
    actual = candidate()
    torch.sdaa.synchronize()
    max_abs = float((expected - actual).abs().max().cpu())
    reference_ms = measure(reference, iterations)
    candidate_ms = measure(candidate, iterations)
    return {
        "shape": [rows, input_dim, output_dim],
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "change_percent": (candidate_ms / reference_ms - 1.0) * 100.0,
        "max_abs_diff": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    shapes = [
        (600, 1024, 1024),
        (800, 1024, 1024),
        (1000, 1024, 1024),
        (600, 1024, 6144),
        (800, 1024, 6144),
        (1000, 1024, 6144),
    ]
    print(json.dumps({
        "results": [
            run_shape(rows, input_dim, output_dim, args.iterations)
            for rows, input_dim, output_dim in shapes
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
