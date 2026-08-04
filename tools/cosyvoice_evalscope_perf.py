#!/usr/bin/env python3
"""Run EvalScope perf against the CosyVoice binary speech endpoint.

EvalScope's built-in OpenAI plugin expects JSON/SSE text completions.  This
adapter keeps EvalScope's request scheduler, SQLite results, latency
percentiles, and request-throughput metrics, while validating and saving every
WAV response and adding TTS-specific RTF/audio-throughput statistics.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, AsyncGenerator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALSCOPE_PATH = (
    ROOT / "cosyvoice_api_outputs" / "perf_tools" / "evalscope_py"
)
if DEFAULT_EVALSCOPE_PATH.is_dir():
    sys.path.insert(0, str(DEFAULT_EVALSCOPE_PATH))

# EvalScope imports torch for optional local-device accounting.  The load
# generator itself is HTTP-only and does not need to auto-load torch_sdaa.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import aiohttp  # noqa: E402
from evalscope.perf.arguments import Arguments  # noqa: E402
from evalscope.perf.main import run_perf_benchmark  # noqa: E402
from evalscope.perf.plugin.api.base import ApiPluginBase  # noqa: E402
from evalscope.perf.plugin.registry import register_api  # noqa: E402


RECORDS: list[dict[str, Any]] = []
RUN_CONFIG: dict[str, Any] = {}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_api_key(path: Path | None) -> str:
    existing = os.getenv("COSYVOICE_API_KEY", "").strip()
    if existing:
        return existing
    if path is None:
        raise RuntimeError(
            "Set COSYVOICE_API_KEY or pass --api-key-env-file"
        )
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "COSYVOICE_API_KEY":
            value = value.strip().strip("\"'")
            if value:
                return value
    raise RuntimeError(f"COSYVOICE_API_KEY is missing from {path}")


def _wav_metadata(content: bytes) -> tuple[int, int, float]:
    with wave.open(io.BytesIO(content), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
    if channels != 1 or sample_width != 2 or sample_rate <= 0 or frames <= 0:
        raise ValueError(
            "unexpected WAV format: "
            f"channels={channels}, sample_width={sample_width}, "
            f"sample_rate={sample_rate}, frames={frames}"
        )
    return sample_rate, frames, frames / sample_rate


def _prompt_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list) and messages:
        item = messages[-1]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    return str(messages)


@register_api("cosyvoice_tts")
class CosyVoiceTtsPlugin(ApiPluginBase):
    """EvalScope API plugin for non-streaming WAV responses."""

    def __init__(self, param: Arguments) -> None:
        super().__init__(param)
        self._request_index = 0

    def build_request(
        self, messages: Any, param: Arguments | None = None
    ) -> dict[str, Any]:
        text = _prompt_text(messages)
        index = self._request_index
        self._request_index += 1
        return {
            "model": RUN_CONFIG["model"],
            "input": text,
            "voice": RUN_CONFIG["voice"],
            "response_format": "wav",
            "speed": RUN_CONFIG["speed"],
            "seed": (RUN_CONFIG["seed"] + index) % (2**32),
        }

    def parse_responses(
        self,
        responses: list[Any],
        request: Any = None,
        **kwargs: Any,
    ) -> tuple[int, int]:
        # TTS has no output-token stream.  One completion unit per successful
        # WAV keeps EvalScope's request records valid; TTS metrics below are
        # computed from audio duration and response headers.
        prompt_units = max(1, len(str((request or {}).get("input", ""))))
        return prompt_units, 1

    async def process_request(
        self,
        client_session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, Any],
        body: dict[str, Any],
    ) -> AsyncGenerator[tuple[bool, int, str], None]:
        request_index = len(RECORDS)
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUN_CONFIG['api_key']}",
            "X-Request-ID": (
                f"evalscope-{RUN_CONFIG['run_name']}-{request_index:04d}"
            ),
            **headers,
        }
        started = time.perf_counter()
        try:
            async with client_session.post(
                url,
                json=body,
                headers=request_headers,
            ) as response:
                content = await response.read()
                client_latency = time.perf_counter() - started
                if response.status != 200:
                    error_text = content.decode("utf-8", errors="replace")
                    yield True, response.status, error_text
                    return

                sample_rate, frames, duration = _wav_metadata(content)
                audio_dir = Path(RUN_CONFIG["audio_dir"])
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / f"request_{request_index:04d}.wav"
                audio_path.write_bytes(content)
                record = {
                    "request_index": request_index,
                    "input": body["input"],
                    "seed": body["seed"],
                    "http_status": response.status,
                    "client_latency_seconds": client_latency,
                    "inference_latency_ms": float(
                        response.headers.get(
                            "X-Inference-Latency-Ms", "nan"
                        )
                    ),
                    "queue_wait_ms": float(
                        response.headers.get("X-Queue-Wait-Ms", "nan")
                    ),
                    "real_time_factor": float(
                        response.headers.get("X-Real-Time-Factor", "nan")
                    ),
                    "audio_duration_seconds": duration,
                    "sample_rate": sample_rate,
                    "frames": frames,
                    "silent_frame_ratio": float(
                        response.headers.get("X-Silent-Frame-Ratio", "nan")
                    ),
                    "quality_retry_count": int(
                        response.headers.get("X-Quality-Retry-Count", "0")
                    ),
                    "wav_bytes": len(content),
                    "wav_sha256": hashlib.sha256(content).hexdigest(),
                    "wav_path": str(audio_path),
                }
                RECORDS.append(record)
                yield False, response.status, json.dumps(
                    {
                        "audio_duration_seconds": duration,
                        "wav_bytes": len(content),
                        "wav_sha256": record["wav_sha256"],
                    },
                    ensure_ascii=False,
                )
        except Exception as exc:
            yield True, 500, f"{type(exc).__name__}: {exc}"


def _warmup(
    url: str,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    speed: float,
    seed: int,
    count: int,
) -> None:
    for index in range(count):
        payload = json.dumps(
            {
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "speed": speed,
                "seed": (seed + index) % (2**32),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "X-Request-ID": f"evalscope-warmup-{index:04d}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                content = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"warmup request failed: HTTP {exc.code}: {detail}"
            ) from exc
        _wav_metadata(content)


def _summary(
    records: list[dict[str, Any]],
    wall_seconds: float,
    evalscope_result: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    latencies = [record["client_latency_seconds"] for record in records]
    inference_ms = [record["inference_latency_ms"] for record in records]
    queue_ms = [record["queue_wait_ms"] for record in records]
    rtfs = [record["real_time_factor"] for record in records]
    durations = [record["audio_duration_seconds"] for record in records]
    return {
        "tool": "evalscope perf",
        "evalscope_version": "0.17.1",
        "endpoint": args.url,
        "model": args.model,
        "voice": args.voice,
        "number": args.number,
        "parallel": args.parallel,
        "warmup": args.warmup,
        "seed": args.seed,
        "success_count": len(records),
        "wall_seconds": wall_seconds,
        "request_throughput_per_second": (
            len(records) / wall_seconds if wall_seconds else 0.0
        ),
        "audio_seconds_generated": sum(durations),
        "audio_throughput_realtime_x": (
            sum(durations) / wall_seconds if wall_seconds else 0.0
        ),
        "client_latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "server_inference_ms": {
            "mean": statistics.fmean(inference_ms) if inference_ms else None,
            "p50": _percentile(inference_ms, 50),
            "p95": _percentile(inference_ms, 95),
        },
        "queue_wait_ms": {
            "mean": statistics.fmean(queue_ms) if queue_ms else None,
            "p95": _percentile(queue_ms, 95),
        },
        "real_time_factor": {
            "mean": statistics.fmean(rtfs) if rtfs else None,
            "p50": _percentile(rtfs, 50),
            "p95": _percentile(rtfs, 95),
        },
        "evalscope_result": evalscope_result,
        "metric_note": (
            "EvalScope output-token, TTFT, TPOT and ITL fields are not "
            "applicable to a non-streaming binary TTS response. Each WAV is "
            "counted as one completion unit; use latency, request throughput, "
            "audio throughput and RTF for A/B decisions."
        ),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8021/v1/audio/speech",
    )
    parser.add_argument("--model", default="cosyvoice3-0.5b")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--api-key-env-file", type=Path)
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--run-name", default="cosyvoice-sdaa")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outputs_dir = args.outputs_dir.resolve()
    args.prompt_file = args.prompt_file.resolve()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    api_key = _load_api_key(
        args.api_key_env_file.resolve() if args.api_key_env_file else None
    )
    prompts = [
        line.strip()
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise RuntimeError("prompt file is empty")

    RUN_CONFIG.update(
        {
            "api_key": api_key,
            "model": args.model,
            "voice": args.voice,
            "speed": args.speed,
            "seed": args.seed,
            "run_name": args.run_name,
            "audio_dir": str(args.outputs_dir / "audio"),
        }
    )

    _warmup(
        args.url,
        api_key,
        args.model,
        args.voice,
        prompts[0],
        args.speed,
        args.seed,
        args.warmup,
    )
    RECORDS.clear()
    evalscope_args = Arguments(
        model=args.model,
        url=args.url,
        api="cosyvoice_tts",
        api_key=None,
        headers={},
        number=args.number,
        parallel=args.parallel,
        dataset="line_by_line",
        dataset_path=str(args.prompt_file),
        outputs_dir=str(args.outputs_dir / "evalscope"),
        name=args.run_name,
        stream=False,
        seed=args.seed,
        connect_timeout=30,
        read_timeout=600,
        no_test_connection=True,
        log_every_n_query=max(1, min(args.number, 10)),
    )
    started = time.perf_counter()
    evalscope_result = run_perf_benchmark(evalscope_args)
    wall_seconds = time.perf_counter() - started
    summary = _summary(
        RECORDS,
        wall_seconds,
        evalscope_result,
        args,
    )
    summary_path = args.outputs_dir / "tts_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(RECORDS) != args.number:
        raise RuntimeError(
            f"only {len(RECORDS)} of {args.number} requests succeeded"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
