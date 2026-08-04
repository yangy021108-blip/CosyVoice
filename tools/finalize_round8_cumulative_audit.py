#!/usr/bin/env python3
"""Regenerate Round 8 aggregate files from immutable completed raw samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_round8_cumulative_audit import (
    VARIANT_ORDER,
    aggregate,
    pairwise,
    write_pairwise_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    metadata = json.loads((results_dir / "metadata.json").read_text(encoding="utf-8"))
    records = load_jsonl(results_dir / "raw_samples.jsonl")
    by_variant = {
        variant: [record for record in records if record["variant"] == variant]
        for variant in VARIANT_ORDER
    }
    expected_count = int(metadata["requests_per_run"]) * 4
    for variant, variant_records in by_variant.items():
        if len(variant_records) != expected_count:
            raise RuntimeError(
                f"{variant}: expected {expected_count} raw samples, got {len(variant_records)}"
            )

    comparisons: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    for first, second in (
        ("baseline_sdaa", "round5_final"),
        ("round5_final", "round8_final"),
        ("baseline_sdaa", "round8_final"),
    ):
        comparison, rows = pairwise(
            first, second, by_variant, args.bootstrap_resamples
        )
        comparisons[f"{first}_to_{second}"] = comparison
        pair_rows.extend(rows)

    write_pairwise_csv(results_dir / "pairwise_results.csv", pair_rows)
    summary = {
        "metadata": metadata,
        "variants": {
            variant: aggregate(variant_records)
            for variant, variant_records in by_variant.items()
        },
        "comparisons": comparisons,
        "raw_samples_path": "raw_samples.jsonl",
        "pairwise_results_path": "pairwise_results.csv",
        "diagnostics_dir": str(
            results_dir.parents[1] / "cosyvoice_api_outputs" / "perf"
            / "round8_cumulative_audit_20260805" / "runs"
        ),
        "aggregation": {
            "source": "completed raw_samples.jsonl",
            "raw_sample_count": len(records),
            "bootstrap_resamples": args.bootstrap_resamples,
        },
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
