#!/usr/bin/env python3
"""Merge Phase-2 generation and ASR records and assign reproducible labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        records.append(value)
    return records


def normalize_characters(text: str) -> list[str]:
    return list("".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())))


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())


def content_only_characters(text: str) -> list[str]:
    """Remove number/Latin forms whose spoken spelling is ASR-dependent.

    This is not a replacement for CER. It is a guard against labeling a sample
    BAD solely because, for example, the reference uses ``2026`` and ASR emits
    ``二零二六``. Such samples remain BORDERLINE and require targeted review.
    """
    normalized = "".join(normalize_characters(text))
    normalized = re.sub(r"[a-z0-9]+", "", normalized)
    normalized = re.sub(r"[零〇一二三四五六七八九十百千万亿两幺点]+", "", normalized)
    return list(normalized)


def edit_operations(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        operations[row][0] = "deletion"
    for column in range(1, columns):
        costs[0][column] = column
        operations[0][column] = "insertion"
    priority = {"equal": 0, "substitution": 1, "deletion": 2, "insertion": 3}
    for row in range(1, rows):
        for column in range(1, columns):
            diagonal_name = (
                "equal" if reference[row - 1] == hypothesis[column - 1]
                else "substitution"
            )
            choices = [
                (costs[row - 1][column - 1] + (diagonal_name != "equal"), diagonal_name),
                (costs[row - 1][column] + 1, "deletion"),
                (costs[row][column - 1] + 1, "insertion"),
            ]
            cost, name = min(choices, key=lambda item: (item[0], priority[item[1]]))
            costs[row][column] = int(cost)
            operations[row][column] = name
    counts = Counter({"insertions": 0, "deletions": 0, "substitutions": 0})
    row, column = len(reference), len(hypothesis)
    while row or column:
        operation = operations[row][column]
        if operation == "equal":
            row -= 1
            column -= 1
        elif operation == "substitution":
            counts["substitutions"] += 1
            row -= 1
            column -= 1
        elif operation == "deletion":
            counts["deletions"] += 1
            row -= 1
        elif operation == "insertion":
            counts["insertions"] += 1
            column -= 1
        else:
            raise RuntimeError(f"Invalid edit backtrace at {row}, {column}")
    return dict(counts)


def longest_repeated_ngram(tokens: Sequence[str], maximum_n: int = 8) -> int:
    longest = 0
    for ngram_size in range(1, min(maximum_n, len(tokens) // 2) + 1):
        for start in range(len(tokens) - 2 * ngram_size + 1):
            first = tokens[start : start + ngram_size]
            second = tokens[start + ngram_size : start + 2 * ngram_size]
            if first == second:
                longest = max(longest, ngram_size)
    return longest


def asr_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        audio = Path(str(record["audio"]))
        for key in dict.fromkeys((audio.name, audio.stem, str(audio))):
            if key in result:
                raise ValueError(f"Duplicate ASR audio key: {key}")
            result[key] = record
    return result


def find_asr(record: dict[str, Any], indexed: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not record.get("audio_path"):
        return None
    audio = Path(str(record["audio_path"]))
    return indexed.get(audio.name) or indexed.get(audio.stem) or indexed.get(str(audio))


def classify(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if metrics["generation_status"] != "SUCCESS":
        return "UNKNOWN", ["service_failure_has_no_content_label"]
    if metrics["asr_text"] is None:
        return "UNKNOWN", ["missing_asr_result"]
    if not metrics["normalized_asr_text"]:
        return "BAD", ["empty_asr_transcript"]
    if metrics["transcript_length_ratio"] < thresholds["early_transcript_ratio"]:
        reasons.append("abnormal_early_termination")
    if metrics["transcript_length_ratio"] > thresholds["long_transcript_ratio"]:
        reasons.append("abnormal_long_transcript")
    if metrics["duration_per_reference_char"] < thresholds["min_duration_per_reference_char"]:
        reasons.append("abnormally_short_audio")
    if metrics["duration_per_reference_char"] > thresholds["max_duration_per_reference_char"]:
        reasons.append("abnormally_long_audio")
    if metrics["silent_frame_ratio"] > thresholds["max_silent_frame_ratio"]:
        reasons.append("excessive_silence")
    if metrics["clipping_ratio"] > thresholds["max_clipping_ratio"]:
        reasons.append("excessive_clipping")
    if metrics["longest_repeated_asr_ngram"] >= 4:
        reasons.append("phrase_repetition")
    orthography_sensitive = bool(
        {
            "numbers",
            "date",
            "abbreviation",
            "mixed_language",
            "pronunciation_sensitive",
        }
        & set(metrics.get("tags", []))
    )
    # Plain ASR CER cannot validate the pronunciation of numbers, dates or
    # abbreviations. Such rows require a pronunciation-aware/manual review and
    # must never be auto-promoted to GOOD, even when raw CER happens to be low.
    if orthography_sensitive and not reasons:
        return "BORDERLINE", ["orthography_sensitive_terms_require_review"]
    if metrics["cer"] > thresholds["borderline_cer_max"]:
        reasons.append("high_cer")
    if reasons:
        return "BAD", reasons
    if metrics["cer"] <= thresholds["good_cer_max"]:
        return "GOOD", ["cer_within_good_threshold"]
    return "BORDERLINE", ["cer_within_borderline_range"]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    thresholds = config["label_thresholds"]
    generation = read_jsonl(args.generation)
    indexed_asr = asr_index(read_jsonl(args.asr))
    output_records: list[dict[str, Any]] = []

    for generated in generation:
        asr = find_asr(generated, indexed_asr)
        reference_chars = normalize_characters(generated["text"])
        asr_text = None if asr is None else str(asr.get("text", ""))
        hypothesis_chars = normalize_characters(asr_text or "")
        reference_words = tokenize_words(generated["text"])
        hypothesis_words = tokenize_words(asr_text or "")
        reference_content = content_only_characters(generated["text"])
        hypothesis_content = content_only_characters(asr_text or "")
        char_edits = edit_operations(reference_chars, hypothesis_chars)
        word_edits = edit_operations(reference_words, hypothesis_words)
        content_edits = edit_operations(reference_content, hypothesis_content)
        duration = float(generated.get("audio_duration_seconds") or 0.0)
        metrics: dict[str, Any] = {
            **generated,
            "generation_status": (
                "SUCCESS" if generated.get("status") == "success" else "SERVICE_FAIL"
            ),
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
            "transcript_length_ratio": len(hypothesis_chars) / max(len(reference_chars), 1),
            "duration": duration,
            "duration_per_reference_char": duration / max(len(reference_chars), 1),
            "silent_frame_ratio": float(
                generated.get("silent_frame_ratio_local") or 0.0
            ),
            "clipping_ratio": float(generated.get("clipping_ratio") or 0.0),
            "longest_repeated_asr_ngram": longest_repeated_ngram(hypothesis_chars),
            "speech_token_repetition_ratio": None,
            "longest_repeated_speech_token_ngram": None,
            "eos_position": None,
            "output_input_token_ratio": None,
        }
        label, reasons = classify(metrics, thresholds)
        metrics["content_label"] = label
        metrics["content_label_reasons"] = reasons
        metrics["error_source"] = "UNKNOWN"
        output_records.append(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in output_records:
        by_sample[str(record["sample_id"])].append(record)
    pairs: list[dict[str, Any]] = []
    for sample_id, records in sorted(by_sample.items()):
        good = [
            record
            for record in records
            if record["content_label"] == "GOOD" and record.get("audio_path")
        ]
        bad = [
            record
            for record in records
            if record["content_label"] == "BAD" and record.get("audio_path")
        ]
        # Deterministic one-to-one matching prevents GOOD x BAD Cartesian
        # pseudo-replication. A sample can appear in at most min(GOOD, BAD)
        # independent pairs.
        good.sort(key=lambda record: (int(record["seed"]), str(record["audio_path"])))
        bad.sort(key=lambda record: (int(record["seed"]), str(record["audio_path"])))
        for pair_index, (good_record, bad_record) in enumerate(zip(good, bad)):
            pairs.append(
                {
                    "pair_id": f"{sample_id}__pair_{pair_index:02d}",
                    "sample_id": sample_id,
                    "text": good_record["text"],
                    "good_seed": good_record["seed"],
                    "good_audio_path": good_record.get("audio_path"),
                    "good_cer": good_record["cer"],
                    "bad_seed": bad_record["seed"],
                    "bad_audio_path": bad_record.get("audio_path"),
                    "bad_cer": bad_record["cer"],
                    "bad_reasons": bad_record["content_label_reasons"],
                }
            )
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")

    label_counts = Counter(record["content_label"] for record in output_records)
    valid_cer = [
        record["cer"]
        for record in output_records
        if record["generation_status"] == "SUCCESS" and record["asr_text"] is not None
    ]
    summary = {
        "schema_version": 2,
        "experiment_id": config.get("experiment_id"),
        "sample_count": len(output_records),
        "independent_text_group_count": len(by_sample),
        "generation_success_count": sum(
            record["generation_status"] == "SUCCESS" for record in output_records
        ),
        "service_failure_count": sum(
            record["generation_status"] == "SERVICE_FAIL" for record in output_records
        ),
        "successful_audio_bad_count": sum(
            record["generation_status"] == "SUCCESS"
            and record["content_label"] == "BAD"
            for record in output_records
        ),
        "asr_result_count": sum(record["asr_text"] is not None for record in output_records),
        "content_label_counts": dict(label_counts),
        "mean_cer_on_generated_audio": (
            sum(valid_cer) / len(valid_cer) if valid_cer else None
        ),
        "matched_good_bad_audio_pair_count": len(pairs),
        "pairing_policy": "deterministic_one_to_one_within_sample_id",
        "required_split_group": "sample_id",
        "thresholds": thresholds,
        "limitations": [
            "Phase-2 API rows do not contain the intermediate speech-token trajectory.",
            "CosyVoice3 production CFM noise is fixed; Phase 2.5 verifies it separately.",
            "ASR mismatch is an end-to-end signal, not proof of an LLM error.",
            "Speech-token and EOS metrics are unavailable until Phase 3 instrumentation.",
            "Chinese WER tokenizes each Han character and groups Latin alphanumerics.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
