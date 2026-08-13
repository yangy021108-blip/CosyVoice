#!/usr/bin/env python3
"""Append one exact sample/seed nomination for Phase-2.75 review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--llm-seed", type=int, required=True)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    existing: list[dict] = []
    if output.exists():
        existing = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    key = (args.sample_id, args.llm_seed)
    if any((str(row["sample_id"]), int(row["llm_seed"])) == key for row in existing):
        raise ValueError(f"Nomination already exists: {args.sample_id}/{args.llm_seed}")
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "sample_id": args.sample_id,
        "llm_seed": args.llm_seed,
        "notes": args.notes or None,
    }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
