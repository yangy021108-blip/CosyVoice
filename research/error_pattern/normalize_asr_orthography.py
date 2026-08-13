#!/usr/bin/env python3
"""Add Simplified-Chinese scoring fields to fixed-ASR JSONL records.

Run this helper in the Qwen3-ASR environment, where ``zhconv`` is installed.
It preserves the raw transcript and adds separate fields used only for CER
scoring, so orthographic conversion can never overwrite audit evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def main() -> None:
    args = parse_args()
    try:
        from zhconv import convert
    except ImportError as exc:
        raise RuntimeError(
            "zhconv is required. Run this helper with the fixed Qwen3-ASR "
            "environment, or install zhconv there."
        ) from exc

    records = read_jsonl(args.input.resolve())
    changed_transcripts = 0
    changed_references = 0
    output_records: list[dict[str, Any]] = []
    for record in records:
        transcript = str(record.get("text", ""))
        expected = str(record.get("expected_text", ""))
        normalized_transcript = convert(transcript, "zh-cn")
        normalized_expected = convert(expected, "zh-cn")
        changed_transcripts += normalized_transcript != transcript
        changed_references += normalized_expected != expected
        output_records.append(
            {
                **record,
                "orthography_normalizer": "zhconv:zh-cn",
                "orthography_normalized_text": normalized_transcript,
                "orthography_normalized_expected": normalized_expected,
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    summary = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "output": str(output),
        "record_count": len(output_records),
        "changed_transcript_count": changed_transcripts,
        "changed_reference_count": changed_references,
        "normalizer": "zhconv:zh-cn",
    }
    if args.summary:
        summary_path = args.summary.resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
