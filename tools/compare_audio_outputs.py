#!/usr/bin/env python3
"""Compare paired mono PCM16 WAV files from baseline and optimized runs."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError(f"{path} is not mono PCM16")
        sample_rate = wav_file.getframerate()
        samples = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()),
            dtype="<i2",
        ).astype(np.float64)
    return sample_rate, samples / 32768.0


def _spectral_cosine(left: np.ndarray, right: np.ndarray) -> float:
    frame_size = 1024
    hop = 256
    usable = min(len(left), len(right))
    if usable < frame_size:
        return float("nan")
    frame_count = 1 + (usable - frame_size) // hop
    indices = np.arange(frame_size)[None, :] + hop * np.arange(frame_count)[:, None]
    window = np.hanning(frame_size)[None, :]
    left_mag = np.abs(np.fft.rfft(left[indices] * window, axis=1))
    right_mag = np.abs(np.fft.rfft(right[indices] * window, axis=1))
    numerator = np.sum(left_mag * right_mag, axis=1)
    denominator = (
        np.linalg.norm(left_mag, axis=1) * np.linalg.norm(right_mag, axis=1)
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_files = sorted(args.baseline_dir.glob("*.wav"))
    optimized_files = sorted(args.optimized_dir.glob("*.wav"))
    if [path.name for path in baseline_files] != [
        path.name for path in optimized_files
    ]:
        raise RuntimeError("baseline and optimized WAV file sets differ")

    records = []
    for baseline_path, optimized_path in zip(
        baseline_files, optimized_files
    ):
        baseline_rate, baseline = _read_wav(baseline_path)
        optimized_rate, optimized = _read_wav(optimized_path)
        if baseline_rate != optimized_rate:
            raise RuntimeError("sample rates differ")
        usable = min(len(baseline), len(optimized))
        left = baseline[:usable]
        right = optimized[:usable]
        error = right - left
        signal_power = float(np.mean(left**2))
        noise_power = float(np.mean(error**2))
        correlation = float(np.corrcoef(left, right)[0, 1])
        records.append(
            {
                "name": baseline_path.name,
                "sample_rate": baseline_rate,
                "baseline_samples": len(baseline),
                "optimized_samples": len(optimized),
                "duration_delta_seconds": (
                    len(optimized) - len(baseline)
                )
                / baseline_rate,
                "waveform_correlation": correlation,
                "waveform_snr_db": (
                    10.0 * math.log10(signal_power / noise_power)
                    if noise_power > 0
                    else float("inf")
                ),
                "mean_absolute_error": float(np.mean(np.abs(error))),
                "max_absolute_error": float(np.max(np.abs(error))),
                "spectral_cosine_similarity": _spectral_cosine(left, right),
                "baseline_rms": math.sqrt(signal_power),
                "optimized_rms": float(math.sqrt(np.mean(right**2))),
            }
        )

    output = {"pairs": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
