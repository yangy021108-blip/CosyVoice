#!/usr/bin/env python3
"""Calculate acoustic and ASR quality checks for the cumulative audit."""

from __future__ import annotations

import argparse
import json
import math
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


COMPARISONS = (
    ("baseline_sdaa", "round5_final"),
    ("round5_final", "round8_final"),
    ("baseline_sdaa", "round8_final"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-jsonl", type=Path)
    return parser.parse_args()


def normalized_text(value: str) -> str:
    return "".join(
        character.lower()
        for character in value
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.getnframes()
        payload = source.readframes(frames)
    if channels != 1 or sample_width != 2:
        raise ValueError(f"expected mono PCM16 WAV: {path}")
    values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    return values, sample_rate


def audio_health(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    waveform, sample_rate = load_wav(path)
    finite = bool(np.isfinite(waveform).all())
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.999)) if waveform.size else 0.0
    return {
        "audio": str(path),
        "sample_rate": sample_rate,
        "pcm_frames": int(waveform.size),
        "duration_seconds": waveform.size / sample_rate,
        "finite": finite,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "silent_frame_ratio": record.get("silent_frame_ratio"),
        "quality_retry_count": record.get("quality_retry_count"),
        "abnormal_silence": bool(record.get("silent_frame_ratio", 0.0) > 0.9),
        "possible_truncation": bool(waveform.size == 0),
    }


def acoustic_pair(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_waveform, first_rate = load_wav(Path(first["wav_path"]))
    second_waveform, second_rate = load_wav(Path(second["wav_path"]))
    common = min(first_waveform.size, second_waveform.size)
    if common == 0:
        correlation = float("nan")
        snr_db = float("nan")
        spectral_cosine = float("nan")
    else:
        reference = first_waveform[:common]
        candidate = second_waveform[:common]
        reference_norm = float(np.linalg.norm(reference))
        candidate_norm = float(np.linalg.norm(candidate))
        correlation = (
            float(np.dot(reference, candidate) / (reference_norm * candidate_norm))
            if reference_norm and candidate_norm else float("nan")
        )
        error = candidate - reference
        signal_power = float(np.sum(np.square(reference)))
        error_power = float(np.sum(np.square(error)))
        snr_db = 10.0 * math.log10(signal_power / max(error_power, 1e-20))
        first_spectrum = np.abs(np.fft.rfft(reference))
        second_spectrum = np.abs(np.fft.rfft(candidate))
        first_spectrum_norm = float(np.linalg.norm(first_spectrum))
        second_spectrum_norm = float(np.linalg.norm(second_spectrum))
        spectral_cosine = (
            float(
                np.dot(first_spectrum, second_spectrum)
                / (first_spectrum_norm * second_spectrum_norm)
            )
            if first_spectrum_norm and second_spectrum_norm else float("nan")
        )
    return {
        "pair_key": first["pair_key"],
        "sample_rate_equal": first_rate == second_rate,
        "pcm_frames_equal": first_waveform.size == second_waveform.size,
        "duration_equal": first["audio_duration_seconds"] == second["audio_duration_seconds"],
        "wav_sha256_equal": first["wav_sha256"] == second["wav_sha256"],
        "speech_tokens_equal": first["speech_tokens"] == second["speech_tokens"],
        "token_count_equal": first["token_count"] == second["token_count"],
        "waveform_correlation": correlation,
        "snr_db": snr_db,
        "spectral_cosine_similarity": spectral_cosine,
        "common_pcm_frames": common,
    }


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def load_asr(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(Path(record["audio"]).resolve()): record for record in records}


def main() -> int:
    args = parse_args()
    samples = [
        json.loads(line) for line in args.raw_samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    asr_by_audio = load_asr(args.asr_jsonl)
    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    health: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_variant[sample["variant"]][sample["pair_key"]] = sample
        health[sample["variant"]].append(audio_health(Path(sample["wav_path"]), sample))

    quality: dict[str, Any] = {
        "sample_count": len(samples),
        "per_variant": {},
        "comparisons": {},
        "asr_available": bool(asr_by_audio),
    }
    for variant, records in health.items():
        quality["per_variant"][variant] = {
            "sample_count": len(records),
            "finite_count": sum(record["finite"] for record in records),
            "abnormal_silence_count": sum(record["abnormal_silence"] for record in records),
            "possible_truncation_count": sum(record["possible_truncation"] for record in records),
            "quality_retry_count": sum(int(record["quality_retry_count"]) for record in records),
            "mean_rms": mean([float(record["rms"]) for record in records]),
            "mean_peak": mean([float(record["peak"]) for record in records]),
            "mean_clipping_ratio": mean([float(record["clipping_ratio"]) for record in records]),
        }

    for first_variant, second_variant in COMPARISONS:
        first_records = by_variant[first_variant]
        second_records = by_variant[second_variant]
        if set(first_records) != set(second_records):
            raise RuntimeError(f"pair keys differ: {first_variant}, {second_variant}")
        pairs = [
            acoustic_pair(first_records[key], second_records[key])
            for key in sorted(first_records)
        ]
        asr_pairs = []
        for key in sorted(first_records):
            first = first_records[key]
            second = second_records[key]
            first_asr = asr_by_audio.get(str(Path(first["wav_path"]).resolve()))
            second_asr = asr_by_audio.get(str(Path(second["wav_path"]).resolve()))
            if first_asr is None or second_asr is None:
                continue
            first_text = str(first_asr["text"])
            second_text = str(second_asr["text"])
            expected = str(first["input"])
            asr_pairs.append(
                {
                    "pair_key": key,
                    "first_text": first_text,
                    "second_text": second_text,
                    "raw_equal": first_text == second_text,
                    "normalized_equal": normalized_text(first_text) == normalized_text(second_text),
                    "first_cer": edit_distance(normalized_text(expected), normalized_text(first_text))
                    / max(len(normalized_text(expected)), 1),
                    "second_cer": edit_distance(normalized_text(expected), normalized_text(second_text))
                    / max(len(normalized_text(expected)), 1),
                }
            )
        label = f"{first_variant}_to_{second_variant}"
        quality["comparisons"][label] = {
            "pair_count": len(pairs),
            "speech_tokens_equal_count": sum(pair["speech_tokens_equal"] for pair in pairs),
            "token_count_equal_count": sum(pair["token_count_equal"] for pair in pairs),
            "wav_sha256_equal_count": sum(pair["wav_sha256_equal"] for pair in pairs),
            "sample_rate_equal_count": sum(pair["sample_rate_equal"] for pair in pairs),
            "pcm_frames_equal_count": sum(pair["pcm_frames_equal"] for pair in pairs),
            "duration_equal_count": sum(pair["duration_equal"] for pair in pairs),
            "mean_waveform_correlation": mean([pair["waveform_correlation"] for pair in pairs]),
            "min_waveform_correlation": min(pair["waveform_correlation"] for pair in pairs),
            "mean_snr_db": mean([pair["snr_db"] for pair in pairs]),
            "mean_spectral_cosine_similarity": mean([pair["spectral_cosine_similarity"] for pair in pairs]),
            "min_spectral_cosine_similarity": min(pair["spectral_cosine_similarity"] for pair in pairs),
            "pairs": pairs,
            "asr": {
                "pair_count": len(asr_pairs),
                "raw_equal_count": sum(pair["raw_equal"] for pair in asr_pairs),
                "normalized_equal_count": sum(pair["normalized_equal"] for pair in asr_pairs),
                "first_mean_cer": mean([pair["first_cer"] for pair in asr_pairs]),
                "second_mean_cer": mean([pair["second_cer"] for pair in asr_pairs]),
                "pairs": asr_pairs,
            } if asr_by_audio else {"status": "not_run"},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "per_variant": quality["per_variant"],
        "comparisons": {
            key: {
                metric: value for metric, value in result.items()
                if metric not in ("pairs", "asr")
            }
            for key, result in quality["comparisons"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
