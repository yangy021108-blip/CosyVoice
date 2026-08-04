#!/usr/bin/env python3
"""Aggregate Perfetto operators by CosyVoice request phase."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO


PROFILE_CATEGORIES = {
    "cpu_op",
    "gpu_memcpy",
    "kernel",
    "privateuse1_runtime",
}
STAGE_PREFIX = "cosyvoice."


def _open_trace(path: Path) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _interval(event: dict[str, Any]) -> tuple[float, float]:
    start = float(event["ts"])
    return start, start + float(event.get("dur", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    with _open_trace(args.trace) as stream:
        trace = json.load(stream)
    events = trace["traceEvents"]
    full_requests = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event.get("name") == "cosyvoice.full_request"
        ),
        key=lambda event: float(event["ts"]),
    )
    request_ranges = [_interval(event) for event in full_requests]
    stage_ranges = []
    for event in events:
        if (
            event.get("cat") == "user_annotation"
            and str(event.get("name", "")).startswith(STAGE_PREFIX)
            and event.get("name") != "cosyvoice.full_request"
        ):
            start, end = _interval(event)
            stage_ranges.append(
                (start, end, str(event["name"]).removeprefix(STAGE_PREFIX))
            )

    totals: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    )
    for event in events:
        category = event.get("cat")
        if category not in PROFILE_CATEGORIES or event.get("ph") != "X":
            continue
        start = float(event["ts"])
        if not any(left <= start <= right for left, right in request_ranges):
            continue
        phase = "unlabeled_or_llm"
        for left, right, stage_name in stage_ranges:
            if left <= start <= right:
                phase = stage_name
                break
        row = totals[phase][str(category)][str(event.get("name", ""))]
        row[0] += 1
        row[1] += float(event.get("dur", 0.0))

    output: dict[str, Any] = {"requests": len(full_requests), "phases": {}}
    for phase, categories in totals.items():
        output["phases"][phase] = {}
        for category, names in categories.items():
            rows = [
                {
                    "name": name,
                    "calls": int(values[0]),
                    "total_ms": values[1] / 1000.0,
                    "mean_us": values[1] / values[0],
                }
                for name, values in names.items()
            ]
            rows.sort(key=lambda row: row["total_ms"], reverse=True)
            output["phases"][phase][category] = rows[: args.top]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
