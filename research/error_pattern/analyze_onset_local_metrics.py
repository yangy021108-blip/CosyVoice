#!/usr/bin/env python3
"""Compute transcript-prefix, acoustic-window and onset-position features."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.evaluate_outputs import normalize_characters
from research.error_pattern.phase2_5_core import read_jsonl, write_jsonl

WINDOWS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0))
PREFIX_LENGTHS = (3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asr", type=Path, nargs="+")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def edit_counts(reference: Sequence[Any], hypothesis: Sequence[Any]) -> dict[str, int]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)
    ]
    for i in range(1, rows):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        table[0][j] = (j, j, 0, 0)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            insertion = table[i][j - 1]
            deletion = table[i - 1][j]
            substitution = table[i - 1][j - 1]
            candidates = [
                (insertion[0] + 1, insertion[1] + 1, insertion[2], insertion[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (
                    substitution[0] + 1,
                    substitution[1],
                    substitution[2],
                    substitution[3] + 1,
                ),
            ]
            table[i][j] = min(candidates)
    distance, insertions, deletions, substitutions = table[-1][-1]
    return {
        "distance": distance,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
    }


def index_asr(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        value = str(record.get("audio") or record.get("audio_path") or "")
        if not value:
            continue
        path = Path(value)
        for key in (str(path), path.name, path.stem):
            if key in result and result[key] is not record:
                raise ValueError(f"Duplicate ASR lookup key: {key}")
            result[key] = record
    return result


def find_asr(record: dict[str, Any], indexed: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    path = Path(str(record["audio_path"]))
    return indexed.get(str(path)) or indexed.get(path.name) or indexed.get(path.stem)


def frame_rms(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(waveform) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(waveform) < frame_length:
        padded = np.pad(waveform, (0, frame_length - len(waveform)))
        return np.array([float(np.sqrt(np.mean(np.square(padded))))])
    count = 1 + (len(waveform) - frame_length) // hop_length
    return np.array(
        [
            float(
                np.sqrt(
                    np.mean(
                        np.square(waveform[i * hop_length : i * hop_length + frame_length])
                    )
                )
            )
            for i in range(count)
        ],
        dtype=np.float64,
    )


def acoustic_window(
    waveform: np.ndarray, sample_rate: int, start: float, end: float
) -> dict[str, Any]:
    segment = waveform[int(start * sample_rate) : min(int(end * sample_rate), len(waveform))]
    actual_duration = len(segment) / sample_rate
    rms = frame_rms(segment, max(int(0.025 * sample_rate), 1), max(int(0.010 * sample_rate), 1))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    result: dict[str, Any] = {
        "start_seconds": start,
        "end_seconds": end,
        "available_duration_seconds": actual_duration,
        "rms": float(np.sqrt(np.mean(np.square(segment)))) if len(segment) else 0.0,
        "silence_ratio": float(np.mean(rms < 0.001)) if len(rms) else 1.0,
        "silence_definition": "25ms frame RMS < 0.001 full-scale amplitude",
        "energy_continuity_mean_abs_db_step": (
            float(np.mean(np.abs(np.diff(rms_db)))) if len(rms_db) > 1 else None
        ),
        "energy_continuity_max_abs_db_step": (
            float(np.max(np.abs(np.diff(rms_db)))) if len(rms_db) > 1 else None
        ),
        "voiced_ratio": None,
        "f0_median_hz": None,
        "f0_continuity_median_abs_cents_step": None,
        "f0_status": "unavailable",
    }
    if len(segment) < int(0.08 * sample_rate):
        result["f0_status"] = "segment_too_short"
        return result
    try:
        import librosa

        f0, voiced_flag, _ = librosa.pyin(
            segment,
            fmin=60.0,
            fmax=min(600.0, sample_rate / 2.0 - 1.0),
            sr=sample_rate,
            frame_length=1024,
            hop_length=max(int(0.010 * sample_rate), 1),
        )
        finite = np.isfinite(f0)
        result["voiced_ratio"] = float(np.mean(voiced_flag)) if len(voiced_flag) else 0.0
        if np.any(finite):
            voiced_f0 = f0[finite]
            result["f0_median_hz"] = float(np.median(voiced_f0))
            if len(voiced_f0) > 1:
                cents = 1200.0 * np.diff(np.log2(voiced_f0))
                result["f0_continuity_median_abs_cents_step"] = float(
                    np.median(np.abs(cents))
                )
            result["f0_status"] = "pyin_proxy"
        else:
            result["f0_status"] = "no_voiced_frames"
    except Exception as error:  # feature availability must not abort review
        result["f0_status"] = f"unavailable:{type(error).__name__}"
    return result


def token_prefix_metrics(
    tokens: list[int],
    length: int,
    token_counts: Counter[int],
    transitions: Counter[tuple[int, int]],
) -> dict[str, Any]:
    prefix = tokens[:length]
    bigrams = list(zip(prefix, prefix[1:]))
    trigrams = list(zip(prefix, prefix[1:], prefix[2:]))
    rare_count = sum(token_counts[token] <= 2 for token in prefix)
    return {
        "available_token_count": len(prefix),
        "unique_token_count": len(set(prefix)),
        "token_diversity": len(set(prefix)) / max(len(prefix), 1),
        "adjacent_repetition_count": sum(a == b for a, b in bigrams),
        "repeated_bigram_count": len(bigrams) - len(set(bigrams)),
        "repeated_trigram_count": len(trigrams) - len(set(trigrams)),
        "rare_token_count": rare_count,
        "rare_token_frequency": rare_count / max(len(prefix), 1),
        "mean_corpus_transition_frequency": (
            float(np.mean([transitions[pair] for pair in bigrams])) if bigrams else None
        ),
        "mapping_to_audio": "approximate_at_25_tokens_per_second_not_one_to_one",
    }


def approximate_alignment_spans(asr: dict[str, Any] | None) -> list[dict[str, Any]]:
    if asr is None:
        return []
    segments = asr.get("segments") or asr.get("chunks") or []
    result: list[dict[str, Any]] = []
    for segment in segments:
        timestamp = segment.get("timestamp")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            continue
        result.append(
            {
                "reference_span": None,
                "asr_span": str(segment.get("text", "")),
                "start_time": timestamp[0],
                "end_time": timestamp[1],
                "confidence": None,
                "status": "approximate_asr_segment_not_forced_alignment",
            }
        )
    return result


def phi_effect(a: int, b: int, c: int, d: int) -> float | None:
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denominator if denominator else None


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest.resolve())
    asr_index = (
        index_asr(
            [
                record
                for path in args.asr
                for record in read_jsonl(path.resolve())
            ]
        )
        if args.asr
        else {}
    )
    reviews = read_jsonl(args.reviews.resolve()) if args.reviews and args.reviews.exists() else []
    review_index = {str(record["blind_id"]): record for record in reviews}
    if len(review_index) != len(reviews):
        raise ValueError("Duplicate blind_id in review records")

    token_counts: Counter[int] = Counter()
    transitions: Counter[tuple[int, int]] = Counter()
    for record in manifest:
        tokens = [int(token) for token in record["filtered_speech_tokens"]]
        token_counts.update(tokens)
        transitions.update(zip(tokens, tokens[1:]))

    features: list[dict[str, Any]] = []
    for record in manifest:
        waveform, sample_rate = sf.read(
            Path(str(record["audio_path"])), dtype="float32", always_2d=False
        )
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        asr = find_asr(record, asr_index)
        hypothesis = "" if asr is None else str(asr.get("text", ""))
        reference_chars = normalize_characters(str(record["text"]))
        hypothesis_chars = normalize_characters(hypothesis)
        prefix_metrics: dict[str, Any] = {}
        for length in PREFIX_LENGTHS:
            edits = edit_counts(reference_chars[:length], hypothesis_chars[:length])
            prefix_metrics[str(length)] = {
                "reference": "".join(reference_chars[:length]),
                "hypothesis": "".join(hypothesis_chars[:length]),
                "cer": edits["distance"] / max(min(length, len(reference_chars)), 1),
                **edits,
            }
        tokens = [int(token) for token in record["filtered_speech_tokens"]]
        review = review_index.get(str(record["blind_id"]))
        features.append(
            {
                **record,
                "asr_text": hypothesis if asr is not None else None,
                "prefix_transcript_metrics": prefix_metrics,
                "first_3_chars_error": prefix_metrics["3"]["distance"] > 0,
                "first_5_chars_error": prefix_metrics["5"]["distance"] > 0,
                "first_10_chars_error": prefix_metrics["10"]["distance"] > 0,
                "audio_window_metrics": {
                    f"{start:.1f}-{end:.1f}s": acoustic_window(
                        waveform, int(sample_rate), start, end
                    )
                    for start, end in WINDOWS
                },
                "first_0.5s_error": None,
                "first_1.0s_error": None,
                "first_2.0s_error": None,
                "time_local_error_status": "unavailable_without_reliable_character_alignment",
                "alignment_method": (
                    None if asr is None else asr.get("alignment_method")
                ),
                "alignment_confidence": (
                    None if asr is None else asr.get("alignment_confidence")
                ),
                "alignment_spans": approximate_alignment_spans(asr),
                "speech_token_prefix_metrics": {
                    str(length): token_prefix_metrics(
                        tokens, length, token_counts, transitions
                    )
                    for length in (10, 25, 50)
                },
                "human_review": review,
            }
        )

    write_jsonl(args.output.resolve(), features)
    label_counts = Counter(
        record["human_review"]["articulation_label"]
        for record in features
        if record["human_review"] is not None
    )
    degraded = [
        record
        for record in features
        if record["human_review"] is not None
        and record["human_review"]["articulation_label"]
        in {"MILD_DEGRADATION", "SEVERE_DEGRADATION"}
    ]
    onset_buckets = Counter()
    approximate_token_buckets = Counter()
    interval_overlap_counts = Counter()
    known_timing_count = 0
    for record in degraded:
        onset = record["human_review"].get("error_start_seconds")
        if onset is None:
            onset_buckets["UNKNOWN"] += 1
        else:
            known_timing_count += 1
            if onset < 0.5:
                onset_buckets["0-0.5s"] += 1
            elif onset < 1.0:
                onset_buckets["0.5-1.0s"] += 1
            elif onset < 2.0:
                onset_buckets["1.0-2.0s"] += 1
            else:
                onset_buckets[">=2.0s"] += 1
            approximate_token = float(onset) * 25.0
            if approximate_token < 10:
                approximate_token_buckets["0-10"] += 1
            elif approximate_token < 25:
                approximate_token_buckets["10-25"] += 1
            elif approximate_token < 50:
                approximate_token_buckets["25-50"] += 1
            else:
                approximate_token_buckets[">50"] += 1
            error_end = record["human_review"].get("error_end_seconds")
            end = float(onset) if error_end is None else float(error_end)
            for window_start, window_end in WINDOWS:
                if float(onset) < window_end and end >= window_start:
                    interval_overlap_counts[
                        f"{window_start:.1f}-{window_end:.1f}s"
                    ] += 1
    known_reviews = [
        record
        for record in features
        if record["human_review"] is not None
        and record["human_review"]["articulation_label"] != "UNKNOWN"
    ]
    pass_clear = sum(
        record["quality_status"] == "PASS"
        and record["human_review"]["articulation_label"] == "CLEAR"
        for record in known_reviews
    )
    pass_degraded = sum(
        record["quality_status"] == "PASS"
        and record["human_review"]["articulation_label"] != "CLEAR"
        for record in known_reviews
    )
    silent_clear = sum(
        record["quality_reason"] == "too_silent"
        and record["human_review"]["articulation_label"] == "CLEAR"
        for record in known_reviews
    )
    silent_degraded = sum(
        record["quality_reason"] == "too_silent"
        and record["human_review"]["articulation_label"] != "CLEAR"
        for record in known_reviews
    )
    summary = {
        "schema_version": 1,
        "observation_count": len(features),
        "asr_matched_count": sum(record["asr_text"] is not None for record in features),
        "human_review_count": len(reviews),
        "articulation_label_counts": dict(label_counts),
        "degraded_error_onset_buckets": dict(onset_buckets),
        "degraded_error_onset_bucket_probability": {
            key: value / known_timing_count
            for key, value in onset_buckets.items()
            if key != "UNKNOWN" and known_timing_count
        },
        "degraded_error_window_overlap_probability": {
            key: value / known_timing_count
            for key, value in interval_overlap_counts.items()
            if known_timing_count
        },
        "degraded_error_known_timing_count": known_timing_count,
        "token_position_approximation": {
            "assumed_rate_tokens_per_second": 25,
            "buckets": ["0-10", "10-25", "25-50", ">50"],
            "observed_counts": dict(approximate_token_buckets),
            "observed_probability": {
                key: value / known_timing_count
                for key, value in approximate_token_buckets.items()
                if known_timing_count
            },
            "warning": "speech-token position is not a one-to-one audio alignment",
        },
        "too_silent_contingency": {
            "PASS": {"CLEAR": pass_clear, "DEGRADED": pass_degraded},
            "TOO_SILENT": {"CLEAR": silent_clear, "DEGRADED": silent_degraded},
            "phi_effect_size": phi_effect(
                pass_clear, pass_degraded, silent_clear, silent_degraded
            ),
            "causal_interpretation_allowed": False,
        },
        "alignment": {
            "character_level_available": False,
            "time_window_error_fields_are_null": True,
            "reason": (
                "Whisper returned phrase-level segments without confidence; no "
                "forced aligner is installed in the fixed ASR environment."
            ),
        },
    }
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
