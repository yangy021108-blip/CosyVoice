#!/usr/bin/env python3
"""Compare Torch SDPA with the SDAA non-causal fixed flash-attention op."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _timed_ms(fn, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        torch.sdaa.synchronize()
        started = time.perf_counter()
        fn()
        torch.sdaa.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 384, 512, 640])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.ops.load_library(str(args.extension.resolve()))
    torch.manual_seed(args.seed)
    torch.sdaa.manual_seed_all(args.seed)
    device = torch.device("sdaa")
    scale = 1.0 / math.sqrt(args.head_dim)
    results = []

    for seq_len in args.seq_lens:
        q = torch.randn(
            (args.batch, args.heads, seq_len, args.head_dim),
            dtype=torch.float16,
            device=device,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def torch_sdpa():
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )

        def sdaa_flash():
            return torch.ops._C_sdaa_poc.mm_encoder_flash_attention_fixed(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale,
            ).transpose(1, 2)

        for _ in range(args.warmup):
            torch_sdpa()
            sdaa_flash()
        torch.sdaa.synchronize()
        reference = torch_sdpa()
        candidate = sdaa_flash()
        torch.sdaa.synchronize()
        difference = (candidate.float() - reference.float()).abs()
        sdpa_ms = _timed_ms(torch_sdpa, args.repeats)
        flash_ms = _timed_ms(sdaa_flash, args.repeats)
        results.append(
            {
                "batch": args.batch,
                "heads": args.heads,
                "seq_len": seq_len,
                "head_dim": args.head_dim,
                "torch_sdpa_mean_ms": statistics.fmean(sdpa_ms),
                "torch_sdpa_median_ms": statistics.median(sdpa_ms),
                "sdaa_flash_mean_ms": statistics.fmean(flash_ms),
                "sdaa_flash_median_ms": statistics.median(flash_ms),
                "speedup_mean_x": (
                    statistics.fmean(sdpa_ms) / statistics.fmean(flash_ms)
                ),
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
            }
        )

    output = {
        "extension": str(args.extension.resolve()),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
