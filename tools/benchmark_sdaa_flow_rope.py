#!/usr/bin/env python3
"""Validate and benchmark CosyVoice Flow RoPE paths on SDAA."""

from __future__ import annotations

import argparse
import json
import time
import torch
from x_transformers.x_transformers import (
    apply_rotary_pos_emb,
    RotaryEmbedding,
    rotate_half,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
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
    torch.manual_seed(20260729)
    device = torch.device("sdaa")
    query = torch.randn(
        (args.batch, args.sequence, args.dim),
        device=device,
        dtype=torch.float16,
    )
    key = torch.randn_like(query)
    frequencies, _ = RotaryEmbedding(
        args.rotary_dim
    ).forward_from_seq_len(args.sequence)
    frequencies = frequencies.to(device)
    cosine = frequencies.cos()
    sine = frequencies.sin()

    def original():
        return (
            apply_rotary_pos_emb(query, frequencies, 1.0),
            apply_rotary_pos_emb(key, frequencies, 1.0),
        )

    def precomputed_python():
        query_rotated = (
            query[..., : args.rotary_dim] * cosine
            + rotate_half(query[..., : args.rotary_dim]) * sine
        )
        key_rotated = (
            key[..., : args.rotary_dim] * cosine
            + rotate_half(key[..., : args.rotary_dim]) * sine
        )
        return (
            torch.cat(
                (query_rotated, query[..., args.rotary_dim :]),
                dim=-1,
            ).type(query.dtype),
            torch.cat(
                (key_rotated, key[..., args.rotary_dim :]),
                dim=-1,
            ).type(key.dtype),
        )

    original_query, original_key = original()
    python_query, python_key = precomputed_python()
    synchronize()
    results = {
        "python_query_max_abs_diff": float(
            (python_query - original_query).abs().max().cpu()
        ),
        "python_key_max_abs_diff": float(
            (python_key - original_key).abs().max().cpu()
        ),
        "precomputed_python_mean_ms": measure(
            precomputed_python,
            args.warmup,
            args.iterations,
        ),
        "original_mean_ms": measure(
            original,
            args.warmup,
            args.iterations,
        ),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
