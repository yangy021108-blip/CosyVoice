#!/usr/bin/env python3
"""Benchmark separate and fused CosyVoice Flow QKV projections on SDAA."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding, rotate_half


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def measure(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.sdaa.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.sdaa.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260802)
    device = torch.device("sdaa")
    dtype = torch.float16
    inputs = torch.randn(
        args.batch,
        args.sequence,
        args.dim,
        device=device,
        dtype=dtype,
    )
    weights = [
        torch.randn(args.dim, args.dim, device=device, dtype=dtype)
        for _ in range(3)
    ]
    biases = [
        torch.randn(args.dim, device=device, dtype=dtype)
        for _ in range(3)
    ]
    fused_weight = torch.cat(weights, dim=0).contiguous()
    fused_bias = torch.cat(biases, dim=0).contiguous()
    fused_qk_weight = torch.cat(weights[:2], dim=0).contiguous()
    fused_qk_bias = torch.cat(biases[:2], dim=0).contiguous()
    frequencies, _ = RotaryEmbedding(64).forward_from_seq_len(
        args.sequence
    )
    frequencies = frequencies.to(device)
    cosine = frequencies.cos()
    sine = frequencies.sin()

    def apply_rope(tensor):
        rotated = tensor[..., :64]
        rotated = (
            rotated * cosine + rotate_half(rotated) * sine
        )
        return torch.cat((rotated, tensor[..., 64:]), dim=-1)

    def separate():
        return tuple(
            F.linear(inputs, weight, bias)
            for weight, bias in zip(weights, biases)
        )

    def fused():
        return F.linear(inputs, fused_weight, fused_bias).chunk(3, dim=-1)

    def separate_qk_rope():
        return tuple(
            apply_rope(F.linear(inputs, weight, bias))
            for weight, bias in zip(weights[:2], biases[:2])
        )

    def fused_qk_rope():
        query, key = F.linear(
            inputs, fused_qk_weight, fused_qk_bias
        ).chunk(2, dim=-1)
        return apply_rope(query), apply_rope(key)

    separate_outputs = separate()
    fused_outputs = fused()
    separate_qk_outputs = separate_qk_rope()
    fused_qk_outputs = fused_qk_rope()
    torch.sdaa.synchronize()
    results = {
        "max_abs_diff": max(
            float((left - right).abs().max().cpu())
            for left, right in zip(separate_outputs, fused_outputs)
        ),
        "qk_rope_max_abs_diff": max(
            float((left - right).abs().max().cpu())
            for left, right in zip(
                separate_qk_outputs, fused_qk_outputs
            )
        ),
        "separate_mean_ms": measure(
            separate, args.warmup, args.iterations
        ),
        "fused_mean_ms": measure(fused, args.warmup, args.iterations),
        "separate_qk_rope_mean_ms": measure(
            separate_qk_rope, args.warmup, args.iterations
        ),
        "fused_qk_rope_mean_ms": measure(
            fused_qk_rope, args.warmup, args.iterations
        ),
    }
    results["change_percent"] = (
        results["fused_mean_ms"] / results["separate_mean_ms"] - 1.0
    ) * 100.0
    results["qk_rope_change_percent"] = (
        results["fused_qk_rope_mean_ms"]
        / results["separate_qk_rope_mean_ms"]
        - 1.0
    ) * 100.0
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
