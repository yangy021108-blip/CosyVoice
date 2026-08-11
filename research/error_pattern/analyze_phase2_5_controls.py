#!/usr/bin/env python3
"""Merge Phase-2.5 decode/ASR records and evaluate the three controls."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.evaluate_outputs import (
    edit_operations,
    normalize_characters,
)
from research.error_pattern.phase2_5_core import read_json, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decode-results", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def index_asr(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        audio = Path(str(record["audio"]))
        for key in dict.fromkeys((audio.name, audio.stem, str(audio))):
            if key in result:
                raise ValueError(f"Duplicate ASR key: {key}")
            result[key] = record
    return result


def find_asr(record: dict[str, Any], indexed: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    audio = Path(str(record["audio_path"]))
    return indexed.get(audio.name) or indexed.get(audio.stem) or indexed.get(str(audio))


def label_for_cer(cer: float) -> str:
    if cer <= 0.05:
        return "GOOD"
    if cer <= 0.20:
        return "BORDERLINE"
    return "BAD"


def rounded_range(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    controls = config["controls"]
    decoded = read_jsonl(args.decode_results.resolve())
    indexed = index_asr(read_jsonl(args.asr.resolve()))
    records: list[dict[str, Any]] = []
    for record in decoded:
        asr = find_asr(record, indexed)
        reference = normalize_characters(record["text"])
        asr_text = None if asr is None else str(asr.get("text", ""))
        hypothesis = normalize_characters(asr_text or "")
        edits = edit_operations(reference, hypothesis)
        cer = sum(edits.values()) / max(len(reference), 1)
        generation_status = str(record.get("generation_status", "SUCCESS"))
        if generation_status != "SUCCESS" or asr is None:
            content_label = "UNKNOWN"
            reasons = ["missing_generated_audio_or_asr"]
        else:
            content_label = label_for_cer(cer)
            reasons = [f"cer_{content_label.lower()}_range"]
        records.append(
            {
                **record,
                "asr_text": asr_text,
                "cer": cer,
                **edits,
                "content_label": content_label,
                "content_label_reasons": reasons,
                "quality_status": (
                    "PASS" if record.get("quality_acceptable") else "REJECTED"
                ),
                "error_source": "UNKNOWN",
            }
        )

    experiment_a = [record for record in records if record["experiment"] == "A"]
    experiment_b = [record for record in records if record["experiment"] == "B"]
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in experiment_a:
        by_sample[record["sample_id"]].append(record)
    a_summary = {}
    for sample_id, sample_records in sorted(by_sample.items()):
        cers = [float(record["cer"]) for record in sample_records]
        a_summary[sample_id] = {
            "observation_count": len(sample_records),
            "unique_trajectory_count": len(
                {record["trajectory_sha256"] for record in sample_records}
            ),
            "content_label_counts": dict(
                Counter(record["content_label"] for record in sample_records)
            ),
            "cer": rounded_range(cers),
        }

    b_cers = [float(record["cer"]) for record in experiment_b]
    b_labels = {record["content_label"] for record in experiment_b}
    b_summary = {
        "observation_count": len(experiment_b),
        "unique_trajectory_count": len(
            {record["trajectory_sha256"] for record in experiment_b}
        ),
        "unique_wav_count": len({record["wav_sha256"] for record in experiment_b}),
        "unique_flow_noise_count": len(
            {record.get("flow_noise_sha256") for record in experiment_b}
        ),
        "production_noise_match_at_seed_zero": all(
            bool(record.get("flow_noise_matches_production"))
            for record in experiment_b
            if int(record["flow_seed"]) == 0
        ),
        "counterfactual_noise_differs_from_production": all(
            not bool(record.get("flow_noise_matches_production"))
            for record in experiment_b
            if int(record["flow_seed"]) != 0
        ),
        "content_label_counts": dict(Counter(record["content_label"] for record in experiment_b)),
        "cer": rounded_range(b_cers),
        "content_label_stable": len(b_labels) == 1,
    }

    c_records = sorted(
        [
            record
            for record in experiment_a
            if record["sample_id"] == controls["repeat_sample_id"]
            and int(record["llm_seed"]) == int(controls["repeat_llm_seed"])
        ],
        key=lambda record: int(record["llm_repeat_index"]),
    )
    if len(c_records) != 2:
        raise ValueError("Experiment C requires exactly two repeated records")
    c_summary = {
        "observation_count": len(c_records),
        "trajectory_identical": c_records[0]["trajectory_sha256"] == c_records[1]["trajectory_sha256"],
        "stop_signature_identical": (
            c_records[0]["trajectory_sha256"] == c_records[1]["trajectory_sha256"]
        ),
        "wav_byte_identical": c_records[0]["wav_sha256"] == c_records[1]["wav_sha256"],
        "pcm_identical": c_records[0]["pcm_sha256"] == c_records[1]["pcm_sha256"],
        "cer_values": [record["cer"] for record in c_records],
    }

    b_unstable = not b_summary["content_label_stable"] or b_summary["cer"]["range"] > 0.05
    for record in records:
        if record["content_label"] != "BAD":
            continue
        if record["experiment"] == "B" and b_unstable:
            record["error_source"] = "LIKELY_ACOUSTIC"
        elif record["experiment"] == "A" and not b_unstable:
            record["error_source"] = "LLM_OR_TOKEN_PATH_REQUIRES_ASR_REVIEW"

    summary = {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id"),
        "observation_count": len(records),
        "independent_text_group_count": len(by_sample),
        "content_label_counts": dict(Counter(record["content_label"] for record in records)),
        "quality_status_counts": dict(Counter(record["quality_status"] for record in records)),
        "experiment_A_fixed_flow_varied_llm": a_summary,
        "experiment_B_fixed_trajectory_varied_flow": b_summary,
        "experiment_C_repeatability": c_summary,
        "controls_passed": bool(
            b_summary["unique_trajectory_count"] == 1
            and b_summary["unique_flow_noise_count"] == len(experiment_b)
            and b_summary["unique_wav_count"] == len(experiment_b)
            and b_summary["production_noise_match_at_seed_zero"]
            and b_summary["counterfactual_noise_differs_from_production"]
            and c_summary["trajectory_identical"]
            and c_summary["wav_byte_identical"]
            and all(record["generation_status"] == "SUCCESS" for record in records)
            and all(record["content_label"] != "UNKNOWN" for record in records)
        ),
        "interpretation_limits": [
            "ASR CER remains an end-to-end proxy and is not direct proof of an LLM defect.",
            "Top-k entropy fields are approximations over returned logprobs, not exact vocabulary entropy.",
            "The three control texts are a mechanism check, not an effect-size dataset.",
        ],
    }
    write_jsonl(args.output.resolve(), records)
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
