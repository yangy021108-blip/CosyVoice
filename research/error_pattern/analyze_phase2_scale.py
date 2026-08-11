#!/usr/bin/env python3
"""Build the Phase-2 scale dataset from token, decode, and fixed-ASR records."""

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
    asr_index,
    classify,
    content_only_characters,
    edit_operations,
    find_asr,
    longest_repeated_ngram,
    normalize_characters,
    tokenize_words,
)
from research.error_pattern.phase2_5_core import read_json, read_jsonl, write_jsonl


REVIEW_TAGS = {
    "numbers",
    "date",
    "abbreviation",
    "mixed_language",
    "pronunciation_sensitive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--decode-results", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    return parser.parse_args()


def trajectory_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        generation_id = str(record["generation_id"])
        if generation_id in result:
            raise ValueError(f"Duplicate trajectory generation_id: {generation_id}")
        chunks = record["chunks"]
        result[generation_id] = {
            "trajectory_sha256": record["trajectory_sha256"],
            "frontend_chunks": record["frontend_chunks"],
            "frontend_chunk_count": len(chunks),
            "raw_speech_token_count": sum(
                int(chunk["raw_speech_token_count"]) for chunk in chunks
            ),
            "filtered_speech_token_count": sum(
                int(chunk["filtered_speech_token_count"]) for chunk in chunks
            ),
            "dropped_silent_token_count": sum(
                len(chunk["dropped_silent_token_indexes"]) for chunk in chunks
            ),
            "stop_reasons": [str(chunk["stop_reason"]) for chunk in chunks],
            "max_length_reached": any(
                bool(chunk["max_length_reached"]) for chunk in chunks
            ),
            "text_frontend_backend": record.get("text_frontend_backend"),
        }
    return result


def build_metrics(
    decoded: dict[str, Any],
    trajectory: dict[str, Any],
    asr: dict[str, Any] | None,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    reference_chars = normalize_characters(decoded["text"])
    asr_text = None if asr is None else str(asr.get("text", ""))
    hypothesis_chars = normalize_characters(asr_text or "")
    reference_words = tokenize_words(decoded["text"])
    hypothesis_words = tokenize_words(asr_text or "")
    reference_content = content_only_characters(decoded["text"])
    hypothesis_content = content_only_characters(asr_text or "")
    char_edits = edit_operations(reference_chars, hypothesis_chars)
    word_edits = edit_operations(reference_words, hypothesis_words)
    content_edits = edit_operations(reference_content, hypothesis_content)
    duration = float(decoded.get("audio_duration_seconds") or 0.0)
    record: dict[str, Any] = {
        **decoded,
        **trajectory,
        "asr_text": asr_text,
        "asr_raw_text": None if asr is None else asr.get("raw_text"),
        "normalized_reference_text": "".join(reference_chars),
        "normalized_asr_text": "".join(hypothesis_chars),
        "reference_character_count": len(reference_chars),
        "asr_character_count": len(hypothesis_chars),
        "cer": sum(char_edits.values()) / max(len(reference_chars), 1),
        "wer": sum(word_edits.values()) / max(len(reference_words), 1),
        "content_only_cer": sum(content_edits.values())
        / max(len(reference_content), 1),
        **char_edits,
        "word_insertions": word_edits["insertions"],
        "word_deletions": word_edits["deletions"],
        "word_substitutions": word_edits["substitutions"],
        "transcript_length_ratio": len(hypothesis_chars)
        / max(len(reference_chars), 1),
        "duration": duration,
        "duration_per_reference_char": duration / max(len(reference_chars), 1),
        "silent_frame_ratio": float(decoded.get("silent_frame_ratio") or 0.0),
        "clipping_ratio": float(decoded.get("clipping_ratio") or 0.0),
        "longest_repeated_asr_ngram": longest_repeated_ngram(hypothesis_chars),
    }
    label, reasons = classify(record, thresholds)
    record["content_label"] = label
    record["content_label_reasons"] = reasons
    record["quality_status"] = (
        "PASS" if decoded.get("quality_acceptable") else "REJECTED"
    )
    if record["quality_status"] == "REJECTED":
        record["error_source"] = "ACOUSTIC_QUALITY_GATE"
    elif label == "BAD":
        record["error_source"] = "LLM_OR_TOKEN_PATH_REQUIRES_REVIEW"
    else:
        record["error_source"] = "UNKNOWN"
    return record


def deterministic_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[str(record["sample_id"])].append(record)
    pairs: list[dict[str, Any]] = []
    for sample_id, sample_records in sorted(by_sample.items()):
        good = sorted(
            [record for record in sample_records if record["content_label"] == "GOOD"],
            key=lambda record: (int(record["llm_seed"]), str(record["decode_id"])),
        )
        bad = sorted(
            [
                record
                for record in sample_records
                if record["content_label"] == "BAD"
                and not (REVIEW_TAGS & set(record.get("tags", [])))
            ],
            key=lambda record: (int(record["llm_seed"]), str(record["decode_id"])),
        )
        for pair_index, (good_record, bad_record) in enumerate(zip(good, bad)):
            pairs.append(
                {
                    "pair_id": f"{sample_id}__pair_{pair_index:02d}",
                    "sample_id": sample_id,
                    "text": good_record["text"],
                    "good_llm_seed": good_record["llm_seed"],
                    "good_trajectory_sha256": good_record["trajectory_sha256"],
                    "good_audio_path": good_record["audio_path"],
                    "good_cer": good_record["cer"],
                    "bad_llm_seed": bad_record["llm_seed"],
                    "bad_trajectory_sha256": bad_record["trajectory_sha256"],
                    "bad_audio_path": bad_record["audio_path"],
                    "bad_cer": bad_record["cer"],
                    "bad_reasons": bad_record["content_label_reasons"],
                }
            )
    return pairs


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    sweep = config["sweep"]
    thresholds = config["label_thresholds"]
    trajectory_index = trajectory_summaries(read_jsonl(args.trajectories.resolve()))
    decoded = read_jsonl(args.decode_results.resolve())
    indexed_asr = asr_index(read_jsonl(args.asr.resolve()))

    records: list[dict[str, Any]] = []
    for decoded_record in decoded:
        generation_id = str(decoded_record["generation_id"])
        trajectory = trajectory_index.get(generation_id)
        if trajectory is None:
            raise ValueError(f"Missing trajectory for {generation_id}")
        records.append(
            build_metrics(
                decoded_record,
                trajectory,
                find_asr(decoded_record, indexed_asr),
                thresholds,
            )
        )

    pairs = deterministic_pairs(records)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[str(record["sample_id"])].append(record)
    per_sample: dict[str, Any] = {}
    for sample_id, sample_records in sorted(by_sample.items()):
        cers = [float(record["cer"]) for record in sample_records]
        per_sample[sample_id] = {
            "observation_count": len(sample_records),
            "unique_trajectory_count": len(
                {record["trajectory_sha256"] for record in sample_records}
            ),
            "content_label_counts": dict(
                Counter(record["content_label"] for record in sample_records)
            ),
            "quality_status_counts": dict(
                Counter(record["quality_status"] for record in sample_records)
            ),
            "cer_min": min(cers),
            "cer_max": max(cers),
            "cer_mean": sum(cers) / len(cers),
        }

    expected_total = int(sweep["expected_sample_count"]) * int(
        sweep["expected_seed_count"]
    )
    label_counts = Counter(record["content_label"] for record in records)
    quality_counts = Counter(record["quality_status"] for record in records)
    stop_counts = Counter(
        reason for record in records for reason in record["stop_reasons"]
    )
    paired_groups = {pair["sample_id"] for pair in pairs}
    gate = config["phase3_gate"]
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "observation_count": len(records),
        "expected_observation_count": expected_total,
        "independent_text_group_count": len(by_sample),
        "expected_text_group_count": int(sweep["expected_sample_count"]),
        "collection_complete": len(records) == expected_total,
        "asr_result_count": sum(record["asr_text"] is not None for record in records),
        "content_label_counts": dict(label_counts),
        "quality_status_counts": dict(quality_counts),
        "unique_trajectory_count": len(
            {record["trajectory_sha256"] for record in records}
        ),
        "raw_speech_token_count": sum(
            int(record["raw_speech_token_count"]) for record in records
        ),
        "filtered_speech_token_count": sum(
            int(record["filtered_speech_token_count"]) for record in records
        ),
        "dropped_silent_token_count": sum(
            int(record["dropped_silent_token_count"]) for record in records
        ),
        "max_length_reached_count": sum(
            bool(record["max_length_reached"]) for record in records
        ),
        "stop_reason_counts": dict(stop_counts),
        "audio_duration_seconds": sum(float(record["duration"]) for record in records),
        "mean_cer": sum(float(record["cer"]) for record in records)
        / max(len(records), 1),
        "matched_good_bad_pair_count": len(pairs),
        "matched_good_bad_text_group_count": len(paired_groups),
        "matched_good_bad_text_groups": sorted(paired_groups),
        "pairing_policy": "deterministic_one_to_one_within_sample_id",
        "required_split_group": "sample_id",
        "phase3_gate": gate,
        "phase3_ready": bool(
            len(pairs) >= int(gate["minimum_matched_pairs"])
            and len(paired_groups) >= int(gate["minimum_bad_text_groups"])
        ),
        "per_sample": per_sample,
        "limitations": [
            "ASR CER is an end-to-end proxy and all BAD rows require audio or pronunciation review.",
            "Number, date, abbreviation, mixed-language, and pronunciation-sensitive rows cannot be auto-GOOD.",
            "Flow seed zero is fixed to the production CausalConditionalCFM noise buffer.",
            "Top-k logprob entropy is approximate, not exact vocabulary entropy.",
        ],
    }
    write_jsonl(args.output.resolve(), records)
    write_jsonl(args.pairs.resolve(), pairs)
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
