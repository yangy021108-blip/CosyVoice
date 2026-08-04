#!/usr/bin/env python3
"""Attribute a Round 8 Perfetto trace to Flow and HiFT hotspot tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASES = {
    "cosyvoice.flow": "flow",
    "cosyvoice.hift": "hift",
    "cosyvoice.full_request": "full_request",
}
TARGET_PATTERNS = {
    "gemm": ("gemm", "linear", "addmm", "matmul", "::mm", "bmm"),
    "ffn": ("ffn", "mlp"),
    "attention": ("attention", "flash", "scaled_dot", "softmax"),
    "layer_norm": ("norm",),
    "gate_multiply": ("mul",),
    "rope": ("rope", "rotary", "sin", "cos"),
    "cat": ("cat", "concat"),
    "copy_stride": ("copy", "stride"),
    "transpose_contiguous": ("transpose", "permute", "contiguous"),
    "conv1d": ("conv1d", "convolution"),
    "conv_transpose1d": ("conv_transpose", "deconvolution"),
    "stft_istft": ("stft", "istft", "fft"),
    "host_device_copy": ("memcpy", "h2d", "d2h"),
    "synchronization": ("synchron", "wait", "item", "local_scalar"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--kernel-summary-csv", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def load_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    return list(data.get("traceEvents", []))


def overlaps(event: dict[str, Any], interval: tuple[float, float]) -> bool:
    start = float(event.get("ts", -1))
    duration = float(event.get("dur", 0))
    end = start + duration
    return duration > 0 and start < interval[1] and end > interval[0]


def target_bucket(name: str) -> str | None:
    lower = name.lower()
    for bucket, patterns in TARGET_PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            return bucket
    return None


def aggregate(
    events: list[dict[str, Any]], intervals: list[tuple[float, float]], top: int,
    category_filter: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    totals: dict[tuple[str, str], float] = defaultdict(float)
    calls: dict[tuple[str, str], int] = defaultdict(int)
    shapes: dict[tuple[str, str], str] = {}
    buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "total_us": 0.0})
    for event in events:
        if event.get("ph") != "X" or not any(overlaps(event, interval) for interval in intervals):
            continue
        category = str(event.get("cat", "unknown"))
        if category_filter is not None and category != category_filter:
            continue
        name = str(event.get("name", "unknown"))
        duration = float(event.get("dur", 0.0))
        key = (category, name)
        totals[key] += duration
        calls[key] += 1
        args = event.get("args")
        if isinstance(args, dict) and key not in shapes:
            value = args.get("Input Dims") or args.get("input_dims") or args.get("Concrete Inputs")
            if value is not None:
                shapes[key] = str(value)
        bucket = target_bucket(name)
        if bucket:
            buckets[bucket]["calls"] += 1
            buckets[bucket]["total_us"] += duration
    rows = [
        {
            "category": category,
            "operator": name,
            "calls": calls[(category, name)],
            "total_us": duration,
            "total_ms": duration / 1000.0,
            "mean_us": duration / calls[(category, name)],
            "input_shape_or_args": shapes.get((category, name), ""),
        }
        for (category, name), duration in totals.items()
    ]
    rows.sort(key=lambda row: float(row["total_us"]), reverse=True)
    return rows[:top], dict(buckets)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge potentially nested profiler intervals without double counting."""
    if not intervals:
        return []
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def cpu_gap_summary(
    events: list[dict[str, Any]], full_intervals: list[tuple[float, float]]
) -> dict[str, Any]:
    """Quantify trace-visible gaps between ATen CPU-op intervals.

    The CPU trace is nested, so this is deliberately an observation metric,
    not a claim that the host thread was idle.  It nevertheless identifies
    intervals where the trace contains neither an ATen CPU op nor an enclosing
    ATen leaf activity while a full request is in flight.
    """
    per_request: list[dict[str, float | int]] = []
    all_gaps: list[float] = []
    kernel_intervals = merge_intervals([
        (float(event.get("ts", -1)), float(event.get("ts", -1)) + float(event.get("dur", 0.0)))
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "kernel"
        and float(event.get("dur", 0.0)) > 0
    ])
    kernel_overlap_total = 0.0
    gaps_with_kernel_overlap = 0
    for request_start, request_end in full_intervals:
        cpu_intervals: list[tuple[float, float]] = []
        for event in events:
            if event.get("ph") != "X":
                continue
            name = str(event.get("name", ""))
            if not name.startswith("aten::"):
                continue
            event_start = float(event.get("ts", -1))
            event_end = event_start + float(event.get("dur", 0.0))
            if event_end <= request_start or event_start >= request_end:
                continue
            cpu_intervals.append((
                max(request_start, event_start), min(request_end, event_end)
            ))
        covered = merge_intervals(cpu_intervals)
        gaps: list[float] = []
        cursor = request_start
        for start, end in covered:
            if start > cursor:
                gaps.append(start - cursor)
            cursor = max(cursor, end)
        if cursor < request_end:
            gaps.append(request_end - cursor)
        visible_gaps = [gap for gap in gaps if gap >= 100.0]
        gap_kernel_overlap = 0.0
        cursor = request_start
        visible_gap_intervals: list[tuple[float, float]] = []
        for start, end in covered:
            if start - cursor >= 100.0:
                visible_gap_intervals.append((cursor, start))
            cursor = max(cursor, end)
        if request_end - cursor >= 100.0:
            visible_gap_intervals.append((cursor, request_end))
        for gap_start, gap_end in visible_gap_intervals:
            for kernel_start, kernel_end in kernel_intervals:
                if kernel_start >= gap_end:
                    break
                gap_kernel_overlap += max(
                    0.0, min(gap_end, kernel_end) - max(gap_start, kernel_start)
                )
        all_gaps.extend(visible_gaps)
        kernel_overlap_total += gap_kernel_overlap
        gaps_with_kernel_overlap += sum(
            any(
                min(gap_end, kernel_end) > max(gap_start, kernel_start)
                for kernel_start, kernel_end in kernel_intervals
            )
            for gap_start, gap_end in visible_gap_intervals
        )
        per_request.append({
            "request_duration_ms": (request_end - request_start) / 1000.0,
            "aten_covered_ms": sum(end - start for start, end in covered) / 1000.0,
            "visible_gap_count_ge_100us": len(visible_gaps),
            "visible_gap_ms_ge_100us": sum(visible_gaps) / 1000.0,
            "largest_visible_gap_ms": max(visible_gaps, default=0.0) / 1000.0,
            "kernel_overlap_during_visible_gaps_ms": gap_kernel_overlap / 1000.0,
        })
    return {
        "definition": (
            "Gaps >=100us between merged aten:: CPU-op intervals during a "
            "cosyvoice.full_request annotation. Nested profiler events and "
            "untraced Python/device work mean this is not proof of CPU idleness."
        ),
        "request_count": len(per_request),
        "total_visible_gap_ms_ge_100us": sum(all_gaps) / 1000.0,
        "largest_visible_gap_ms": max(all_gaps, default=0.0) / 1000.0,
        "visible_gap_count_with_kernel_overlap": gaps_with_kernel_overlap,
        "kernel_overlap_during_visible_gaps_ms": kernel_overlap_total / 1000.0,
        "mean_visible_gap_ms_per_request": (
            sum(item["visible_gap_ms_ge_100us"] for item in per_request)
            / len(per_request) if per_request else 0.0
        ),
        "per_request": per_request,
    }


def main() -> int:
    args = parse_args()
    events = load_events(args.trace)
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        name = str(event.get("name", ""))
        if event.get("ph") == "X" and name in PHASES:
            intervals[PHASES[name]].append(
                (float(event.get("ts", 0.0)), float(event.get("ts", 0.0)) + float(event.get("dur", 0.0)))
            )
    phases: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for phase in ("flow", "hift", "full_request"):
        rows, buckets = aggregate(events, intervals[phase], args.top)
        kernel_rows, _ = aggregate(
            events, intervals[phase], args.top, category_filter="kernel"
        )
        phase_duration_ms = sum(end - start for start, end in intervals[phase]) / 1000.0
        phases[phase] = {
            "event_count": len(intervals[phase]),
            "total_stage_duration_ms": phase_duration_ms,
            "top_operators": rows,
            "top_sdaa_kernels": kernel_rows,
            "target_operation_totals": {
                name: {
                    "calls": int(value["calls"]),
                    "total_ms": value["total_us"] / 1000.0,
                    "mean_us": value["total_us"] / value["calls"] if value["calls"] else 0.0,
                }
                for name, value in buckets.items()
            },
        }
        for row in rows:
            csv_rows.append({"phase": phase, "row_type": "top_operator", **row})
        for row in kernel_rows:
            csv_rows.append({"phase": phase, "row_type": "top_sdaa_kernel", **row})

    result = {
        "trace": str(args.trace),
        "event_count": len(events),
        "phases": phases,
        "cpu_gap_analysis": cpu_gap_summary(events, intervals["full_request"]),
        "method_note": (
            "Operator durations are summed profiler durations for events whose "
            "interval overlaps the named phase. Nested CPU events are therefore "
            "not additive wall time; use stage duration for wall-clock attribution."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.kernel_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase", "row_type", "category", "operator", "calls", "total_us", "total_ms",
        "mean_us", "input_shape_or_args",
    ]
    with args.kernel_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
