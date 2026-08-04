#!/usr/bin/env python3
"""Print compact hotspot tables from analyze_perfetto_trace.py output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_PATTERN = (
    r"scaled_dot|flash|gemm|linear|addmm|matmul|bmm|sin|cos|"
    r"masked_fill|bitwise_not|cat|copy|convolution|layer_norm|"
    r"item|local_scalar|pad|repeat|softmax|transpose|where"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--top", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data: dict[str, Any] = json.loads(
        args.analysis.read_text(encoding="utf-8")
    )
    pattern = re.compile(args.pattern, re.IGNORECASE)
    rows: list[tuple[float, str, int, float, str]] = []
    for category, events in data["top_events_by_category"].items():
        for event in events:
            if pattern.search(event["name"]):
                rows.append(
                    (
                        float(event["total_ms"]),
                        category,
                        int(event["calls"]),
                        float(event["mean_us"]),
                        event["name"],
                    )
                )
    rows.sort(reverse=True)
    print("total_ms\tcategory\tcalls\tmean_us\tname")
    for total_ms, category, calls, mean_us, name in rows[: args.top]:
        print(
            f"{total_ms:.3f}\t{category}\t{calls}\t"
            f"{mean_us:.3f}\t{name}"
        )


if __name__ == "__main__":
    main()
