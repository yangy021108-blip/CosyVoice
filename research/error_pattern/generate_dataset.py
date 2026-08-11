#!/usr/bin/env python3
"""Generate a reproducible Phase-2 CosyVoice error-screening dataset."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any


METRIC_HEADERS = (
    "x-request-id",
    "x-audio-sample-rate",
    "x-audio-channels",
    "x-audio-duration",
    "x-queue-wait-ms",
    "x-inference-latency-ms",
    "x-real-time-factor",
    "x-generation-seed",
    "x-quality-retry-count",
    "x-silent-frame-ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


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


def read_api_key(path: Path | None) -> str:
    value = os.environ.get("COSYVOICE_API_KEY", "").strip()
    if path is not None:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, candidate = line.split("=", 1)
                if name.strip() != "COSYVOICE_API_KEY":
                    continue
                value = candidate.strip().strip("\"'")
                break
            value = line.strip("\"'")
            break
    if not value:
        raise ValueError(
            "No API key found; set COSYVOICE_API_KEY or pass --api-key-file"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError(f"Invalid sample_id: {value!r}")
    return cleaned


def wav_statistics(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)
    if channels != 1 or sample_width != 2:
        raise ValueError(
            f"Expected mono 16-bit WAV, got channels={channels}, "
            f"sample_width={sample_width}"
        )
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    count = max(len(samples), 1)
    squares = sum(int(value) * int(value) for value in samples)
    rms = math.sqrt(squares / count)
    clipping_ratio = sum(abs(value) >= 32767 for value in samples) / count
    frame_size = max(int(sample_rate * 0.02), 1)
    silent_frames = 0
    total_frames = 0
    silent_threshold = 32767.0 * (10.0 ** (-50.0 / 20.0))
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        frame_rms = math.sqrt(
            sum(int(value) * int(value) for value in frame) / len(frame)
        )
        silent_frames += frame_rms < silent_threshold
        total_frames += 1
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "audio_duration_seconds": frame_count / sample_rate,
        "rms_dbfs": 20.0 * math.log10(max(rms, 1e-12) / 32767.0),
        "peak_absolute": max((abs(value) for value in samples), default=0),
        "clipping_ratio": clipping_ratio,
        "silent_frame_ratio_local": silent_frames / max(total_frames, 1),
        "audio_sha256": sha256_file(path),
    }


def parse_error_body(body: bytes) -> Any:
    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


def compact_headers(headers: Any) -> dict[str, str]:
    lowered = {key.lower(): value for key, value in headers.items()}
    result = {
        name: lowered[name] for name in METRIC_HEADERS if name in lowered
    }
    if "content-type" in lowered:
        result["content-type"] = lowered["content-type"]
    return result


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_asr_inputs(output_dir: Path, records: list[dict[str, Any]]) -> None:
    successful = sorted(
        (record for record in records if record.get("status") == "success"),
        key=lambda record: str(record["audio_path"]),
    )
    with (output_dir / "asr_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for record in successful:
            item = {
                "sample_id": record["sample_id"],
                "seed": record["seed"],
                "audio_path": record["audio_path"],
                "expected_text": record["text"],
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    expected = "\n".join(record["text"] for record in successful)
    if expected:
        expected += "\n"
    (output_dir / "asr_expected_texts.txt").write_text(
        expected, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    samples = read_jsonl(args.samples)
    api_key = read_api_key(args.api_key_file)
    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    errors_dir = output_dir / "errors"
    results_path = output_dir / "generation_results.jsonl"

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --resume or a new run"
        )
    audio_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if args.resume and results_path.exists():
        existing = read_jsonl(results_path)
    completed = {(item["sample_id"], int(item["seed"])) for item in existing}

    manifest = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "experiment_id": config.get("experiment_id"),
        "base_url": args.base_url.rstrip("/"),
        "endpoint": "/v1/audio/speech",
        "config": config,
        "config_sha256": sha256_file(args.config),
        "samples_sha256": sha256_file(args.samples),
        "api_key_source": str(args.api_key_file) if args.api_key_file else "environment",
        "api_key_recorded": False,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": command_output(["git", "status", "--porcelain"]),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "instrumentation": {
            "raw_speech_tokens": None,
            "filtered_speech_tokens": None,
            "frontend_chunks": None,
            "stop_reason": None,
            "text_token_num": None,
            "reason": "not exposed by the Phase-2 external API",
        },
    }
    voice_config = Path("api_server/voices.json")
    if voice_config.is_file():
        manifest["voice_config_path"] = str(voice_config.resolve())
        manifest["voice_config_sha256"] = sha256_file(voice_config)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    expanded: list[tuple[int, dict[str, Any], int]] = []
    for sample in samples:
        for seed in sample.get("seeds", []):
            expanded.append((len(expanded), sample, int(seed)))
    if not expanded:
        raise ValueError("No sample seeds were defined")

    endpoint = args.base_url.rstrip("/") + "/v1/audio/speech"
    timeout = float(config.get("request_timeout_seconds", 900))
    for request_index, sample, seed in expanded:
        sample_id = str(sample["sample_id"])
        if (sample_id, seed) in completed:
            continue
        safe_id = safe_name(sample_id)
        stem = f"{request_index:04d}_{safe_id}__seed_{seed}"
        audio_path = audio_dir / f"{stem}.wav"
        request_id = f"phase2-{safe_id}-{seed}-{uuid.uuid4().hex[:8]}"
        payload = {
            "model": config["model"],
            "input": sample["text"],
            "voice": config["voice"],
            "response_format": config.get("response_format", "wav"),
            "speed": config.get("speed", 1.0),
            "seed": seed,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
        )
        started = time.perf_counter()
        base_record: dict[str, Any] = {
            "schema_version": 1,
            "request_index": request_index,
            "sample_id": sample_id,
            "length_group": sample.get("length_group"),
            "tags": sample.get("tags", []),
            "text": sample["text"],
            "seed": seed,
            "request": payload,
            "normalized_frontend_chunks": None,
            "chunk_boundaries": None,
            "text_token_num": None,
            "raw_speech_tokens": None,
            "filtered_speech_tokens": None,
            "speech_token_num": None,
            "stop_reason": None,
        }
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                headers = compact_headers(response.headers)
                content_type = headers.get("content-type", "")
                if not content_type.startswith("audio/"):
                    raise ValueError(
                        f"Expected audio response, received {content_type!r}"
                    )
                audio_path.write_bytes(body)
                stats = wav_statistics(audio_path)
                expected_rate = int(config.get("expected_sample_rate", 24000))
                if stats["sample_rate"] != expected_rate:
                    raise ValueError(
                        f"Expected sample rate {expected_rate}, got {stats['sample_rate']}"
                    )
                record = {
                    **base_record,
                    "status": "success",
                    "http_status": response.status,
                    "wall_latency_seconds": time.perf_counter() - started,
                    "response_headers": headers,
                    "effective_seed": int(headers.get("x-generation-seed", seed)),
                    "quality_retry_count": int(
                        headers.get("x-quality-retry-count", "0")
                    ),
                    "audio_path": str(audio_path),
                    **stats,
                }
        except urllib.error.HTTPError as exc:
            error_body = exc.read()
            error_path = errors_dir / f"{stem}.json"
            parsed = parse_error_body(error_body)
            error_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            record = {
                **base_record,
                "status": "http_error",
                "http_status": exc.code,
                "wall_latency_seconds": time.perf_counter() - started,
                "response_headers": compact_headers(exc.headers),
                "effective_seed": None,
                "quality_retry_count": None,
                "audio_path": None,
                "error_path": str(error_path),
                "error": parsed,
            }
        except (urllib.error.URLError, TimeoutError, ValueError, wave.Error) as exc:
            record = {
                **base_record,
                "status": "client_error",
                "http_status": None,
                "wall_latency_seconds": time.perf_counter() - started,
                "response_headers": {},
                "effective_seed": None,
                "quality_retry_count": None,
                "audio_path": str(audio_path) if audio_path.exists() else None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        append_jsonl(results_path, record)
        existing.append(record)
        print(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "seed": seed,
                    "status": record["status"],
                    "latency": round(record["wall_latency_seconds"], 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_asr_inputs(output_dir, existing)
    success_count = sum(item.get("status") == "success" for item in existing)
    print(
        json.dumps(
            {
                "requests": len(existing),
                "successes": success_count,
                "failures": len(existing) - success_count,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
