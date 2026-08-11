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
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--decode-results", type=Path, nargs="+", required=True)
    parser.add_argument("--asr", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--adjudication",
        type=Path,
        help="Optional human-adjudication JSONL keyed by decode_id.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    return parser.parse_args()


def read_many_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path.resolve()))
    return records


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
    asr_text = None if asr is None else str(asr.get("text", ""))
    scoring_reference = str(decoded["text"])
    scoring_hypothesis = asr_text or ""
    orthography_normalizer = None
    if asr is not None and asr.get("orthography_normalizer"):
        scoring_reference = str(
            asr.get("orthography_normalized_expected", scoring_reference)
        )
        scoring_hypothesis = str(
            asr.get("orthography_normalized_text", scoring_hypothesis)
        )
        orthography_normalizer = str(asr["orthography_normalizer"])
    reference_chars = normalize_characters(scoring_reference)
    hypothesis_chars = normalize_characters(scoring_hypothesis)
    reference_words = tokenize_words(scoring_reference)
    hypothesis_words = tokenize_words(scoring_hypothesis)
    reference_content = content_only_characters(scoring_reference)
    hypothesis_content = content_only_characters(scoring_hypothesis)
    char_edits = edit_operations(reference_chars, hypothesis_chars)
    word_edits = edit_operations(reference_words, hypothesis_words)
    content_edits = edit_operations(reference_content, hypothesis_content)
    duration = float(decoded.get("audio_duration_seconds") or 0.0)
    record: dict[str, Any] = {
        **decoded,
        **trajectory,
        "asr_text": asr_text,
        "asr_raw_text": None if asr is None else asr.get("raw_text"),
        "orthography_normalizer": orthography_normalizer,
        "orthography_normalized_reference_text": scoring_reference,
        "orthography_normalized_asr_text": scoring_hypothesis,
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


def deterministic_pairs(
    records: list[dict[str, Any]], *, confirmed_only: bool = False
) -> list[dict[str, Any]]:
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
                and (
                    record.get("adjudication_label")
                    == "CONFIRMED_CONTENT_ERROR"
                    if confirmed_only
                    else not (REVIEW_TAGS & set(record.get("tags", [])))
                )
            ],
            key=lambda record: (int(record["llm_seed"]), str(record["decode_id"])),
        )
        for pair_index, (good_record, bad_record) in enumerate(zip(good, bad)):
            pairs.append(
                {
                    "pair_id": f"{sample_id}__pair_{pair_index:02d}",
                    "sample_id": sample_id,
                    "challenge_category": good_record.get("challenge_category"),
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
                    "bad_adjudication_label": bad_record.get(
                        "adjudication_label", "NOT_REVIEWED"
                    ),
                }
            )
    return pairs


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    sweep = config["sweep"]
    thresholds = config["label_thresholds"]
    trajectory_index = trajectory_summaries(read_many_jsonl(args.trajectories))
    decoded = read_many_jsonl(args.decode_results)
    indexed_asr = asr_index(read_many_jsonl(args.asr))
    adjudication_index: dict[str, dict[str, Any]] = {}
    if args.adjudication:
        for adjudication in read_jsonl(args.adjudication.resolve()):
            decode_id = str(adjudication["decode_id"])
            if decode_id in adjudication_index:
                raise ValueError(f"Duplicate adjudication decode_id: {decode_id}")
            adjudication_index[decode_id] = adjudication

    records: list[dict[str, Any]] = []
    for decoded_record in decoded:
        generation_id = str(decoded_record["generation_id"])
        trajectory = trajectory_index.get(generation_id)
        if trajectory is None:
            raise ValueError(f"Missing trajectory for {generation_id}")
        record = build_metrics(
            decoded_record,
            trajectory,
            find_asr(decoded_record, indexed_asr),
            thresholds,
        )
        adjudication = adjudication_index.get(str(record["decode_id"]))
        record["adjudication_label"] = (
            "NOT_REVIEWED"
            if adjudication is None
            else str(adjudication.get("adjudication_label", "UNRESOLVED"))
        )
        record["adjudication_notes"] = (
            None if adjudication is None else adjudication.get("notes")
        )
        records.append(record)

    pairs = deterministic_pairs(records)
    confirmed_pairs = deterministic_pairs(records, confirmed_only=True)
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

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = record.get("challenge_category")
        if category:
            by_category[str(category)].append(record)
    per_category = {
        category: {
            "observation_count": len(category_records),
            "independent_text_group_count": len(
                {record["sample_id"] for record in category_records}
            ),
            "content_label_counts": dict(
                Counter(record["content_label"] for record in category_records)
            ),
            "confirmed_content_error_count": sum(
                record["adjudication_label"] == "CONFIRMED_CONTENT_ERROR"
                for record in category_records
            ),
        }
        for category, category_records in sorted(by_category.items())
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
    paired_categories = {
        str(pair["challenge_category"])
        for pair in pairs
        if pair.get("challenge_category")
    }
    confirmed_paired_groups = {pair["sample_id"] for pair in confirmed_pairs}
    confirmed_paired_categories = {
        str(pair["challenge_category"])
        for pair in confirmed_pairs
        if pair.get("challenge_category")
    }
    gate = config["phase3_gate"]
    minimum_bad_categories = int(gate.get("minimum_bad_categories", 0))
    candidate_ready = bool(
        len(pairs) >= int(gate["minimum_matched_pairs"])
        and len(paired_groups) >= int(gate["minimum_bad_text_groups"])
        and len(paired_categories) >= minimum_bad_categories
    )
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "observation_count": len(records),
        "expected_observation_count": expected_total,
        "independent_text_group_count": len(by_sample),
        "expected_text_group_count": int(sweep["expected_sample_count"]),
        "independent_category_count": len(by_category),
        "expected_category_count": sweep.get("expected_category_count"),
        "collection_complete": len(records) == expected_total,
        "asr_result_count": sum(record["asr_text"] is not None for record in records),
        "orthography_normalized_count": sum(
            record["orthography_normalizer"] is not None for record in records
        ),
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
        "matched_good_bad_category_count": len(paired_categories),
        "matched_good_bad_categories": sorted(paired_categories),
        "confirmed_matched_pair_count": len(confirmed_pairs),
        "confirmed_matched_text_group_count": len(confirmed_paired_groups),
        "confirmed_matched_text_groups": sorted(confirmed_paired_groups),
        "confirmed_matched_category_count": len(confirmed_paired_categories),
        "confirmed_matched_categories": sorted(confirmed_paired_categories),
        "pairing_policy": "deterministic_one_to_one_within_sample_id",
        "required_split_group": "sample_id",
        "phase3_gate": gate,
        "phase3_candidate_ready": candidate_ready,
        "phase3_ready": bool(
            len(confirmed_pairs) >= int(gate["minimum_matched_pairs"])
            and len(confirmed_paired_groups)
            >= int(gate["minimum_bad_text_groups"])
            and len(confirmed_paired_categories) >= minimum_bad_categories
        ),
        "per_sample": per_sample,
        "per_category": per_category,
        "limitations": [
            "ASR CER is an end-to-end proxy and all BAD rows require audio or pronunciation review.",
            "Simplified/Traditional Chinese is folded only when the ASR JSONL contains explicit zhconv scoring fields; raw transcripts are preserved.",
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
