#!/usr/bin/env python3
"""Build a machine-readable CosyVoice SDAA baseline/optimized comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(baseline: float, optimized: float) -> dict[str, float]:
    return {
        "baseline": baseline,
        "optimized": optimized,
        "change_percent": (optimized / baseline - 1.0) * 100.0,
        "speedup": baseline / optimized,
    }


def increasing_metric(
    baseline: float, optimized: float
) -> dict[str, float]:
    return {
        "baseline": baseline,
        "optimized": optimized,
        "change_percent": (optimized / baseline - 1.0) * 100.0,
        "multiplier": optimized / baseline,
    }


def mean_request_value(
    analysis: dict[str, Any],
    key: str,
) -> float:
    return statistics.mean(
        float(request[key]) for request in analysis["request_breakdown"]
    )


def mean_phase_value(
    analysis: dict[str, Any],
    phase: str,
) -> float:
    return statistics.mean(
        float(request["phases_ms"][phase])
        for request in analysis["request_breakdown"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "cosyvoice_api_outputs"
        / "perf"
        / "cosyvoice_sdaa_optimization_20260726",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "comparison_summary.json"
    )

    baseline_eval = load_json(
        root / "baseline" / "evalscope_c1" / "tts_summary.json"
    )
    optimized_eval = load_json(
        root / "optimized" / "evalscope_c1" / "tts_summary.json"
    )
    baseline_perfetto = load_json(
        root / "baseline" / "perfetto" / "baseline.analysis.json"
    )
    optimized_perfetto = load_json(
        root / "optimized" / "perfetto" / "optimized.analysis.json"
    )
    baseline_asr = load_json(
        root / "baseline" / "evalscope_c1" / "asr_summary.json"
    )
    optimized_asr = load_json(
        root / "optimized" / "evalscope_c1" / "asr_summary.json"
    )

    comparison = {
        "evalscope": {
            "success_count": {
                "baseline": baseline_eval["success_count"],
                "optimized": optimized_eval["success_count"],
            },
            "mean_latency_seconds": metric(
                baseline_eval["client_latency_seconds"]["mean"],
                optimized_eval["client_latency_seconds"]["mean"],
            ),
            "p50_latency_seconds": metric(
                baseline_eval["client_latency_seconds"]["p50"],
                optimized_eval["client_latency_seconds"]["p50"],
            ),
            "p95_latency_seconds": metric(
                baseline_eval["client_latency_seconds"]["p95"],
                optimized_eval["client_latency_seconds"]["p95"],
            ),
            "mean_real_time_factor": metric(
                baseline_eval["real_time_factor"]["mean"],
                optimized_eval["real_time_factor"]["mean"],
            ),
            "request_throughput_per_second": increasing_metric(
                baseline_eval["request_throughput_per_second"],
                optimized_eval["request_throughput_per_second"],
            ),
            "audio_throughput_realtime_x": increasing_metric(
                baseline_eval["audio_throughput_realtime_x"],
                optimized_eval["audio_throughput_realtime_x"],
            ),
        },
        "perfetto": {
            "mean_full_request_ms": metric(
                mean_request_value(
                    baseline_perfetto, "full_request_ms"
                ),
                mean_request_value(
                    optimized_perfetto, "full_request_ms"
                ),
            ),
            "mean_flow_ms": metric(
                mean_phase_value(baseline_perfetto, "cosyvoice.flow"),
                mean_phase_value(optimized_perfetto, "cosyvoice.flow"),
            ),
            "mean_hift_ms": metric(
                mean_phase_value(baseline_perfetto, "cosyvoice.hift"),
                mean_phase_value(optimized_perfetto, "cosyvoice.hift"),
            ),
            "mean_unlabeled_or_llm_ms": metric(
                mean_request_value(
                    baseline_perfetto, "unlabeled_or_llm_ms"
                ),
                mean_request_value(
                    optimized_perfetto, "unlabeled_or_llm_ms"
                ),
            ),
        },
        "semantic_regression": {
            "baseline_audio_count": baseline_asr["audio_count"],
            "optimized_audio_count": optimized_asr["audio_count"],
            "baseline_nonempty_transcript_count": baseline_asr[
                "nonempty_transcript_count"
            ],
            "optimized_nonempty_transcript_count": optimized_asr[
                "nonempty_transcript_count"
            ],
            "baseline_character_error_rate": baseline_asr[
                "character_error_rate"
            ],
            "optimized_character_error_rate": optimized_asr[
                "character_error_rate"
            ],
            "character_error_rate_delta": optimized_asr[
                "character_error_rate"
            ]
            - baseline_asr["character_error_rate"],
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
