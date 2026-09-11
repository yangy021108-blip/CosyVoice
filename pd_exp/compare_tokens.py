#!/usr/bin/env python3
"""Compare unified and PD speech-token trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(left: list[int], right: list[int]) -> dict[str, object]:
    first_mismatch = next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        None,
    )
    if first_mismatch is None and len(left) != len(right):
        first_mismatch = min(len(left), len(right))
    return {
        "left_count": len(left),
        "right_count": len(right),
        "token_by_token_equal": left == right,
        "first_mismatch_index": first_mismatch,
        "common_prefix_tokens": (
            min(len(left), len(right)) if first_mismatch is None else first_mismatch
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--field", default="filtered_speech_tokens")
    args = parser.parse_args()
    left = json.loads(args.left.read_text(encoding="utf-8"))[args.field]
    right = json.loads(args.right.read_text(encoding="utf-8"))[args.field]
    result = compare(left, right)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["token_by_token_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
