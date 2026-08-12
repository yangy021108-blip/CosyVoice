#!/usr/bin/env python3
"""Freeze five suspected and five clear trajectories for the onset pilot."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.phase2_5_core import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--suspected", type=int, default=5)
    parser.add_argument("--clear", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def suspicion_score(record: dict[str, Any]) -> tuple[float, float, str]:
    sources = set(record["review_sources"])
    source_score = 0.0
    if "MANUAL_NOMINATION" in sources:
        source_score += 1000.0
    if "CONTENT_BORDERLINE" in sources:
        source_score += 100.0
    if "ACOUSTIC_TOO_SILENT" in sources:
        source_score += 25.0
    prefix = record["prefix_transcript_metrics"]
    prefix_score = 10.0 * float(prefix["3"]["cer"])
    prefix_score += 5.0 * float(prefix["5"]["cer"])
    prefix_score += float(prefix["10"]["cer"])
    return source_score + prefix_score, prefix_score, str(record["decode_id"])


def diverse_first(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_samples: set[str] = set()
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in used_samples:
            continue
        selected.append(record)
        used_samples.add(sample_id)
        if len(selected) == count:
            return selected
    for record in records:
        if record not in selected:
            selected.append(record)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} records available; requested {count}")


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.features.resolve())
    suspicious_pool = [
        record
        for record in records
        if set(record["review_sources"])
        & {"MANUAL_NOMINATION", "CONTENT_BORDERLINE", "ACOUSTIC_TOO_SILENT"}
    ]
    suspicious_pool.sort(key=suspicion_score, reverse=True)
    suspicious = diverse_first(suspicious_pool, args.suspected)

    clear_pool = [
        record
        for record in records
        if "FIXED_RANDOM_CLEAR_CONTROL" in record["review_sources"]
        and str(record["content_label"]) == "GOOD"
        and str(record["quality_status"]) == "PASS"
        and all(
            float(record["prefix_transcript_metrics"][str(length)]["cer"]) == 0.0
            for length in (3, 5, 10)
        )
    ]
    rng = random.Random(args.random_seed)
    rng.shuffle(clear_pool)
    clear = diverse_first(clear_pool, args.clear)

    output: list[dict[str, Any]] = []
    for group, selected in (
        ("SUSPECTED_DEGRADATION_CANDIDATE", suspicious),
        ("CLEAR_CONTROL_CANDIDATE", clear),
    ):
        for index, record in enumerate(selected):
            item = {
                **record,
                "pilot_group": group,
                "pilot_group_index": index,
                "selection_is_human_confirmation": False,
            }
            if group.startswith("SUSPECTED"):
                item["selection_score"] = suspicion_score(record)[0]
                item["selection_reason"] = (
                    "frozen candidate ranking from review source and prefix ASR; "
                    "not an articulation label"
                )
            else:
                item["selection_reason"] = (
                    "fixed-seed random draw from GOOD/PASS controls with zero "
                    "prefix CER at 3, 5 and 10 characters"
                )
            output.append(item)
    write_jsonl(args.output.resolve(), output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "record_count": len(output),
                "suspected_count": len(suspicious),
                "clear_count": len(clear),
                "random_seed": args.random_seed,
                "sample_ids": [record["sample_id"] for record in output],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
