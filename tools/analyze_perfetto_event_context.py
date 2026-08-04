#!/usr/bin/env python3
"""Report enclosing CPU operators for selected Perfetto events."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--event", default="sdaaEventSynchronize")
    parser.add_argument("--skip-context", action="store_true")
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    by_thread: dict[tuple[object, object], list[dict[str, object]]] = {}
    by_external_id: dict[object, list[dict[str, object]]] = {}
    targets: list[dict[str, object]] = []
    for event in events:
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        key = (event.get("pid"), event.get("tid"))
        by_thread.setdefault(key, []).append(event)
        external_id = event.get("args", {}).get("External id")
        if external_id is not None:
            by_external_id.setdefault(external_id, []).append(event)
        if event.get("name") == args.event:
            targets.append(event)
    parent_counts: Counter[str] = Counter()
    external_context_counts: Counter[str] = Counter()
    input_shape_counts: Counter[str] = Counter()
    chains: Counter[tuple[str, ...]] = Counter()
    for target in targets:
        target_args = target.get("args", {})
        input_shapes = target_args.get("Input Dims")
        if input_shapes is not None:
            input_shape_counts[json.dumps(input_shapes, sort_keys=True)] += 1
        names: tuple[str, ...] = ()
        if not args.skip_context:
            start = float(target["ts"])
            end = start + float(target["dur"])
            key = (target.get("pid"), target.get("tid"))
            enclosing = [
                event
                for event in by_thread[key]
                if event is not target
                and float(event["ts"]) <= start
                and float(event["ts"]) + float(event["dur"]) >= end
            ]
            enclosing.sort(key=lambda event: float(event["dur"]))
            names = tuple(str(event.get("name", "")) for event in enclosing[:8])
        parent_counts[names[0] if names else "<none>"] += 1
        chains[names] += 1
        external_id = target.get("args", {}).get("External id")
        external_names = {
            str(event.get("name", ""))
            for event in by_external_id.get(external_id, [])
            if event is not target
        }
        external_context_counts.update(external_names or {"<none>"})
    result = {
        "target": args.event,
        "count": len(targets),
        "samples": targets[:3],
        "immediate_parent_counts": parent_counts.most_common(),
        "external_id_context_counts": external_context_counts.most_common(),
        "input_shape_counts": input_shape_counts.most_common(),
        "top_chains_inner_to_outer": [
            {"count": count, "chain": chain}
            for chain, count in chains.most_common(20)
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
