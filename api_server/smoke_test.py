"""Small dependency-free smoke client for a running CosyVoice API."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import wave
from array import array
from collections.abc import Mapping
from pathlib import Path


def validate_wav_response(
    content: bytes,
    headers: Mapping[str, str],
    *,
    expected_sample_rate: int,
    duration_tolerance_seconds: float,
    max_clipping_ratio: float,
) -> dict[str, float | int]:
    content_type = (headers.get("Content-Type") or "").split(";", 1)[0]
    if content_type.lower() != "audio/wav":
        raise ValueError(f"expected audio/wav, got {content_type!r}")

    try:
        header_duration = float(headers["X-Audio-Duration"])
        real_time_factor = float(headers["X-Real-Time-Factor"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing or invalid audio metric response headers") from exc
    if header_duration <= 0:
        raise ValueError("X-Audio-Duration must be greater than zero")
    if real_time_factor < 0:
        raise ValueError("X-Real-Time-Factor must not be negative")

    with wave.open(io.BytesIO(content), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if channels != 1:
        raise ValueError(f"expected mono WAV, got {channels} channels")
    if sample_width != 2:
        raise ValueError(f"expected 16-bit WAV, got {sample_width * 8}-bit")
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"expected {expected_sample_rate} Hz WAV, got {sample_rate} Hz"
        )
    if frame_count <= 0:
        raise ValueError("WAV contains no audio frames")

    actual_duration = frame_count / sample_rate
    if abs(actual_duration - header_duration) > duration_tolerance_seconds:
        raise ValueError(
            "WAV duration differs from X-Audio-Duration by more than "
            f"{duration_tolerance_seconds} seconds"
        )

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        raise ValueError("WAV is entirely silent")
    clipping_ratio = sum(abs(sample) >= 32767 for sample in samples) / len(samples)
    if clipping_ratio > max_clipping_ratio:
        raise ValueError(
            f"WAV clipping ratio {clipping_ratio:.6f} exceeds "
            f"{max_clipping_ratio:.6f}"
        )

    return {
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "actual_duration": actual_duration,
        "peak_pcm16": peak,
        "clipping_ratio": clipping_ratio,
        "real_time_factor": real_time_factor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--model", default="cosyvoice3-0.5b")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--text", default="你好，这是一次接口冒烟测试。")
    parser.add_argument("--instructions")
    parser.add_argument("--output", type=Path, default=Path("smoke.wav"))
    parser.add_argument("--expected-sample-rate", type=int, default=24000)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.02)
    parser.add_argument("--max-clipping-ratio", type=float, default=0.05)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": "wav",
        "speed": 1.0,
    }
    if args.instructions:
        payload["instructions"] = args.instructions
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/audio/speech",
        data=body,
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        content = response.read()
        metrics = validate_wav_response(
            content,
            response.headers,
            expected_sample_rate=args.expected_sample_rate,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            max_clipping_ratio=args.max_clipping_ratio,
        )
        request_id = response.headers.get("X-Request-ID")
        audio_duration = response.headers.get("X-Audio-Duration")
        rtf = response.headers.get("X-Real-Time-Factor")
    elapsed = time.perf_counter() - started
    args.output.write_bytes(content)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": len(content),
                "client_elapsed_seconds": round(elapsed, 3),
                "request_id": request_id,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "wav_validation": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
