#!/usr/bin/env python3
"""Compare exact nearest-neighbor 1D upsample implementations on SDAA."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


CASES = (
    ("f0", (1, 1, 500), 256),
    ("harmonics", (1, 9, 500), 256),
    ("upsample_1", (1, 512, 500), 8),
    ("upsample_2", (1, 256, 4000), 8),
)


def time_method(method, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        method()
    torch.sdaa.synchronize()
    elapsed_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            method()
        torch.sdaa.synchronize()
        elapsed_ms.append(
            (time.perf_counter() - start) * 1000.0 / iterations
        )
    return elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    results = []
    with torch.inference_mode():
        for dtype in (torch.float16, torch.float32):
            for case_name, shape, factor in CASES:
                x = torch.randn(shape, dtype=dtype, device="sdaa")
                indices = torch.arange(
                    shape[-1],
                    device=x.device,
                ).repeat_interleave(factor)
                methods = {
                    "interpolate": lambda: F.interpolate(
                        x,
                        scale_factor=factor,
                        mode="nearest",
                    ),
                    "repeat_interleave": lambda: x.repeat_interleave(
                        factor,
                        dim=-1,
                    ),
                    "expand_reshape": lambda: (
                        x.unsqueeze(-1)
                        .expand(*x.shape, factor)
                        .reshape(*x.shape[:-1], -1)
                    ),
                    "index_select": lambda: x.index_select(-1, indices),
                }
                reference = methods["interpolate"]()
                for method_name, method in methods.items():
                    candidate = method()
                    exact = torch.equal(reference, candidate)
                    timings = time_method(
                        method,
                        args.warmup,
                        args.iterations,
                        args.repeats,
                    )
                    results.append(
                        {
                            "case": case_name,
                            "shape": shape,
                            "factor": factor,
                            "dtype": str(dtype),
                            "method": method_name,
                            "exact": exact,
                            "mean_ms": statistics.fmean(timings),
                            "median_ms": statistics.median(timings),
                            "timings_ms": timings,
                        }
                    )

    payload = {"results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
