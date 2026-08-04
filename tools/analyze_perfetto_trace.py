#!/usr/bin/env python3
"""Stream a PyTorch Perfetto JSON trace and summarize events without loading it."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path
from typing import Any


EVENT_RE = re.compile(
    r'"ph":\s*"X".*"cat":\s*"([^"]+)".*"name":\s*"([^"]+)"'
)
DURATION_RE = re.compile(r'"dur":\s*([0-9.eE+-]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    totals: dict[tuple[str, str], float] = collections.defaultdict(float)
    counts: dict[tuple[str, str], int] = collections.defaultdict(int)
    pending: tuple[str, str] | None = None
    stage_events: list[dict[str, Any]] = []

    open_trace = gzip.open if args.trace.suffix == ".gz" else Path.open
    with open_trace(
        args.trace, "rt", encoding="utf-8", errors="replace"
    ) as trace:
        for line in trace:
            event_match = EVENT_RE.search(line)
            if event_match:
                pending = (event_match.group(1), event_match.group(2))
                continue
            if pending is None:
                continue
            duration_match = DURATION_RE.search(line)
            if not duration_match:
                continue
            duration_us = float(duration_match.group(1))
            totals[pending] += duration_us
            counts[pending] += 1
            category, name = pending
            if category == "user_annotation" and name.startswith("cosyvoice."):
                stage_events.append(
                    {
                        "name": name,
                        "duration_us": duration_us,
                        "duration_ms": duration_us / 1000.0,
                    }
                )
            pending = None

    categories: dict[str, list[dict[str, Any]]] = {}
    category_names = sorted({category for category, _ in totals})
    for category in category_names:
        rows = [
            {
                "name": name,
                "calls": counts[(category, name)],
                "total_us": total,
                "total_ms": total / 1000.0,
                "mean_us": total / counts[(category, name)],
            }
            for (event_category, name), total in totals.items()
            if event_category == category
        ]
        rows.sort(key=lambda row: row["total_us"], reverse=True)
        categories[category] = rows[: args.top]

    full_requests = [
        event["duration_ms"]
        for event in stage_events
        if event["name"] == "cosyvoice.full_request"
    ]
    phase_totals: dict[str, list[float]] = collections.defaultdict(list)
    for event in stage_events:
        if event["name"] != "cosyvoice.full_request":
            phase_totals[event["name"]].append(event["duration_ms"])
    request_breakdown = []
    for index, full_ms in enumerate(full_requests):
        phases = {
            name: durations[index]
            for name, durations in phase_totals.items()
            if index < len(durations)
        }
        measured_ms = sum(phases.values())
        request_breakdown.append(
            {
                "request_index": index,
                "full_request_ms": full_ms,
                "phases_ms": phases,
                "unlabeled_or_llm_ms": full_ms - measured_ms,
            }
        )

    output = {
        "trace": str(args.trace),
        "stage_events": stage_events,
        "request_breakdown": request_breakdown,
        "top_events_by_category": categories,
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
