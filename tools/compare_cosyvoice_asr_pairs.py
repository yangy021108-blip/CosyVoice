#!/usr/bin/env python3
"""Compare paired ASR transcripts from two benchmark variants."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RUN_PATTERN = re.compile(r"run_\d+_(?P<variant>.+)_cycle_(?P<cycle>\d+)")


def normalize_text(value: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", value.lower()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--baseline-variant", required=True)
    parser.add_argument("--candidate-variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def record_key(record: dict[str, str]) -> tuple[str, int, str]:
    path = Path(record["audio"])
    for part in path.parts:
        match = RUN_PATTERN.fullmatch(part)
        if match:
            return (
                match.group("variant"),
                int(match.group("cycle")),
                path.name,
            )
    raise ValueError(f"cannot parse benchmark run from {path}")


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = {record_key(record): record for record in records}
    pairs = []
    for (variant, cycle, name), baseline in sorted(indexed.items()):
        if variant != args.baseline_variant:
            continue
        candidate_key = (args.candidate_variant, cycle, name)
        if candidate_key not in indexed:
            raise KeyError(f"missing candidate transcript for {candidate_key}")
        candidate = indexed[candidate_key]
        baseline_text = baseline["text"]
        candidate_text = candidate["text"]
        pairs.append(
            {
                "cycle": cycle,
                "name": name,
                "baseline_text": baseline_text,
                "candidate_text": candidate_text,
                "raw_equal": baseline_text == candidate_text,
                "normalized_equal": (
                    normalize_text(baseline_text)
                    == normalize_text(candidate_text)
                ),
            }
        )
    result = {
        "pair_count": len(pairs),
        "raw_equal_count": sum(pair["raw_equal"] for pair in pairs),
        "normalized_equal_count": sum(
            pair["normalized_equal"] for pair in pairs
        ),
        "nonempty_baseline_count": sum(
            bool(normalize_text(pair["baseline_text"])) for pair in pairs
        ),
        "nonempty_candidate_count": sum(
            bool(normalize_text(pair["candidate_text"])) for pair in pairs
        ),
        "different_pairs": [
            pair for pair in pairs if not pair["normalized_equal"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
