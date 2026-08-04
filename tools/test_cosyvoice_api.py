#!/usr/bin/env python3
"""Run real CosyVoice API checks without shell variables or curl functions."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from array import array
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "cosyvoice_api_outputs" / "manual"
API_KEY_FILE = REPO_ROOT / "cosyvoice_sdaa_vllm_api.env"
BACKEND_URLS = {
    "pytorch": "http://127.0.0.1:8020",
    "vllm": "http://127.0.0.1:8021",
}


def read_api_key(path: Path) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator and key.strip() == "COSYVOICE_API_KEY":
            api_key = value.strip()
            if api_key:
                return api_key
    raise ValueError(f"COSYVOICE_API_KEY is missing from {path}")


def request(
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    headers: dict[str, str] = {}
    body = None
    method = "GET"
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
    api_request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(api_request, timeout=timeout) as response:
            return (
                response.status,
                {key.lower(): value for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            {
                key.lower(): value
                for key, value in (error.headers.items() if error.headers else [])
            },
            error.read(),
        )


def parse_json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw_body": body.decode("utf-8", errors="replace")}


def wait_until_ready(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_result: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            status, _, body = request(
                f"{base_url}/ready",
                timeout=min(5.0, timeout),
            )
        except urllib.error.URLError as error:
            last_result = {"network_error": str(error)}
        else:
            last_result = {
                "status": status,
                "body": parse_json(body),
            }
            if status == 200:
                return last_result
        time.sleep(2)
    raise TimeoutError(
        f"{base_url}/ready did not return HTTP 200 within {timeout} seconds; "
        f"last_result={last_result}"
    )


def write_headers(
    path: Path,
    status: int,
    headers: dict[str, str],
) -> None:
    lines = [f"HTTP-Status: {status}"]
    lines.extend(f"{key}: {value}" for key, value in sorted(headers.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_wav(content: bytes) -> dict[str, float | int]:
    with wave.open(io.BytesIO(content), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)
    if channels != 1:
        raise ValueError(f"expected mono WAV, got {channels} channels")
    if sample_width != 2:
        raise ValueError(f"expected PCM16 WAV, got {sample_width * 8}-bit")
    if sample_rate != 24000:
        raise ValueError(f"expected 24000 Hz WAV, got {sample_rate} Hz")
    if frame_count <= 0:
        raise ValueError("WAV contains no frames")
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        raise ValueError("WAV is entirely silent")
    clipping_ratio = sum(abs(sample) >= 32767 for sample in samples) / len(
        samples
    )
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / sample_rate, 3),
        "peak_pcm16": peak,
        "clipping_ratio": round(clipping_ratio, 8),
    }


def validate_pcm(content: bytes) -> dict[str, float | int]:
    if not content or len(content) % 2:
        raise ValueError("PCM response is empty or has an odd byte count")
    samples = array("h")
    samples.frombytes(content)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        raise ValueError("PCM response is entirely silent")
    return {
        "sample_rate": 24000,
        "channels": 1,
        "sample_width": 2,
        "sample_count": len(samples),
        "duration_seconds": round(len(samples) / 24000, 3),
        "peak_pcm16": peak,
    }


def save_audio_response(
    output_dir: Path,
    name: str,
    status: int,
    headers: dict[str, str],
    body: bytes,
    *,
    expected_format: str,
) -> dict[str, Any]:
    for suffix in (".wav", ".pcm", ".error.json"):
        stale_path = output_dir / f"{name}{suffix}"
        if stale_path.exists():
            stale_path.unlink()
    write_headers(output_dir / f"{name}.headers.txt", status, headers)
    content_type = headers.get("content-type", "").split(";", 1)[0].lower()
    expected_content_type = (
        "audio/wav" if expected_format == "wav" else "audio/pcm"
    )
    result: dict[str, Any] = {
        "name": name,
        "status": status,
        "content_type": content_type,
        "generation_seed": headers.get("x-generation-seed"),
        "quality_retry_count": headers.get("x-quality-retry-count"),
        "inference_latency_ms": headers.get("x-inference-latency-ms"),
        "real_time_factor": headers.get("x-real-time-factor"),
    }
    if status == 200 and content_type == expected_content_type:
        metrics = (
            validate_wav(body)
            if expected_format == "wav"
            else validate_pcm(body)
        )
        output_path = output_dir / f"{name}.{expected_format}"
        output_path.write_bytes(body)
        result["output"] = str(output_path)
        result["audio"] = metrics
        result["ok"] = True
        return result

    error_path = output_dir / f"{name}.error.json"
    error_body = parse_json(body)
    error_path.write_text(
        json.dumps(error_body, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result["output"] = str(error_path)
    result["error"] = error_body
    result["ok"] = False
    return result


def speech_request(
    base_url: str,
    api_key: str,
    output_dir: Path,
    name: str,
    payload: dict[str, Any],
    *,
    request_timeout: float,
) -> dict[str, Any]:
    request_bytes = len(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    started = time.monotonic()
    status, headers, body = request(
        f"{base_url}/v1/audio/speech",
        api_key=api_key,
        payload=payload,
        timeout=request_timeout,
    )
    elapsed_seconds = time.monotonic() - started
    result = save_audio_response(
        output_dir,
        name,
        status,
        headers,
        body,
        expected_format=payload.get("response_format", "wav"),
    )
    result["request_bytes"] = request_bytes
    result["response_bytes"] = len(body)
    result["elapsed_seconds"] = elapsed_seconds
    return result


def ensure_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed_root = (REPO_ROOT / "cosyvoice_api_outputs").resolve()
    if allowed_root not in resolved.parents:
        raise ValueError(
            f"output directory must stay under {allowed_root}: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def normalize_voice_mode(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"zero_shot", "sft"}:
        raise argparse.ArgumentTypeError(
            "voice mode must be 'zero_shot' (or 'zero-shot') or 'sft'"
        )
    return normalized


def select_voice(
    voices_body: Any,
    voice_mode: str,
    requested_voice: str | None,
) -> dict[str, Any]:
    if not isinstance(voices_body, dict):
        raise ValueError("voices endpoint did not return a JSON object")
    records = voices_body.get("data")
    if not isinstance(records, list):
        raise ValueError("voices endpoint response is missing a data list")
    voices = [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("id"), str)
        and isinstance(record.get("mode"), str)
    ]
    available = ", ".join(
        f"{record['id']}({record['mode']})" for record in voices
    ) or "none"
    if requested_voice is not None:
        selected = next(
            (
                record
                for record in voices
                if record["id"] == requested_voice
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f"voice {requested_voice!r} is not registered; "
                f"available voices: {available}"
            )
        if selected["mode"] != voice_mode:
            raise ValueError(
                f"voice {requested_voice!r} has mode "
                f"{selected['mode']!r}, not {voice_mode!r}"
            )
        return selected
    selected = next(
        (record for record in voices if record["mode"] == voice_mode),
        None,
    )
    if selected is None:
        raise ValueError(
            f"no {voice_mode} voice is registered; "
            f"available voices: {available}"
        )
    return selected


def build_speech_payload(
    *,
    text: str,
    voice: str,
    voice_mode: str,
    response_format: str,
    seed: int,
    instructions: str | None,
) -> dict[str, Any]:
    if instructions is not None and voice_mode != "zero_shot":
        raise ValueError(
            "instructions are supported only with a zero_shot voice"
        )
    payload: dict[str, Any] = {
        "model": "cosyvoice3-0.5b",
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": 1.0,
        "seed": seed,
    }
    if instructions is not None:
        normalized = " ".join(instructions.split())
        if not normalized:
            raise ValueError("instructions must not be empty")
        payload["instructions"] = normalized
    return payload


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def format_transfer_size(byte_count: int) -> str:
    if byte_count >= 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.1f}M"
    if byte_count >= 1024:
        return f"{byte_count // 1024}k"
    return str(byte_count)


def format_transfer_time(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"


def terminal_summary(summary: dict[str, Any], verbose: bool) -> str:
    if verbose:
        return json.dumps(summary, ensure_ascii=False, indent=2)
    requests = summary.get("speech_requests", [])
    if not requests:
        message = summary.get(
            "validation_error",
            "no speech request was sent",
        )
        return f"test_cosyvoice_api: {message}"
    result = requests[-1]
    if not result.get("ok"):
        error = result.get("error", {}).get("error", {})
        status = result.get("status", "unknown")
        code = error.get("code", "request_failed")
        message = error.get("message", "speech request failed")
        return (
            f"curl: (22) The requested URL returned error: {status}\n"
            f"{code}: {message}"
        )
    request_bytes = int(result.get("request_bytes", 0))
    response_bytes = int(result.get("response_bytes", 0))
    elapsed_seconds = max(
        float(result.get("elapsed_seconds", 0.0)),
        1e-9,
    )
    total_bytes = request_bytes + response_bytes
    download_speed = int(response_bytes / elapsed_seconds)
    upload_speed = int(request_bytes / elapsed_seconds)
    total_speed = int(total_bytes / elapsed_seconds)
    elapsed = format_transfer_time(elapsed_seconds)
    return (
        "  % Total    % Received % Xferd  Average Speed   Time    "
        "Time     Time  Current\n"
        "                                 Dload  Upload   Total   "
        "Spent    Left  Speed\n"
        f"100 {format_transfer_size(total_bytes):>5}  "
        f"100 {format_transfer_size(response_bytes):>5}  "
        f"100 {format_transfer_size(request_bytes):>5}  "
        f"{download_speed:>6} {upload_speed:>6}  "
        f"{elapsed} {elapsed} --:--:-- {total_speed:>6}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKEND_URLS), required=True)
    parser.add_argument(
        "--voice-mode",
        type=normalize_voice_mode,
        required=True,
        help="zero_shot/zero-shot or sft; checked against /v1/audio/voices",
    )
    parser.add_argument(
        "--voice",
        help="registered voice ID; defaults to the first voice of --voice-mode",
    )
    parser.add_argument(
        "--instructions",
        help="style instruction; valid only for a zero_shot voice",
    )
    parser.add_argument(
        "--response-format",
        choices=("wav", "pcm"),
        default="wav",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--text", default="你好，这是 CosyVoice API 推理测试。")
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the complete results.json instead of a curl-style summary",
    )
    args = parser.parse_args()

    base_url = BACKEND_URLS[args.backend]
    api_key = read_api_key(API_KEY_FILE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instruction_label = (
        "instruction" if args.instructions is not None else "no_instruction"
    )
    case_name = (
        f"{args.voice_mode}_{instruction_label}_{args.response_format}"
    )
    output_dir = ensure_output_dir(
        args.output_dir
        if args.output_dir is not None
        else OUTPUT_ROOT / f"{args.backend}_{case_name}_{timestamp}"
    )

    summary: dict[str, Any] = {
        "backend": args.backend,
        "base_url": base_url,
        "case": case_name,
        "requested_voice_mode": args.voice_mode,
        "requested_voice": args.voice,
        "response_format": args.response_format,
        "has_instructions": args.instructions is not None,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "ready": wait_until_ready(base_url, args.ready_timeout),
        "endpoints": {},
        "speech_requests": [],
    }
    failed = False

    for name, path, protected in (
        ("health", "/health", False),
        ("models", "/v1/models", True),
        ("voices", "/v1/audio/voices", True),
    ):
        status, _, body = request(
            f"{base_url}{path}",
            api_key=api_key if protected else None,
            timeout=10,
        )
        endpoint_result = {
            "status": status,
            "body": parse_json(body),
        }
        summary["endpoints"][name] = endpoint_result
        (output_dir / f"{name}.json").write_text(
            json.dumps(endpoint_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        failed = failed or status != 200

    if failed:
        summary["ok"] = False
        summary["validation_error"] = (
            "one or more discovery endpoints did not return HTTP 200"
        )
        write_summary(output_dir, summary)
        print(terminal_summary(summary, args.verbose), file=sys.stderr)
        raise SystemExit(1)

    try:
        selected_voice = select_voice(
            summary["endpoints"]["voices"]["body"],
            args.voice_mode,
            args.voice,
        )
        payload = build_speech_payload(
            text=args.text,
            voice=selected_voice["id"],
            voice_mode=args.voice_mode,
            response_format=args.response_format,
            seed=args.seed,
            instructions=args.instructions,
        )
    except ValueError as error:
        summary["ok"] = False
        summary["validation_error"] = str(error)
        write_summary(output_dir, summary)
        print(terminal_summary(summary, args.verbose), file=sys.stderr)
        raise SystemExit(1) from error

    summary["selected_voice"] = selected_voice
    (output_dir / "request.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = speech_request(
        base_url,
        api_key,
        output_dir,
        f"{args.backend}_{args.voice_mode}_{instruction_label}",
        payload,
        request_timeout=args.request_timeout,
    )
    summary["speech_requests"].append(result)
    failed = not result["ok"]

    summary["ok"] = not failed
    write_summary(output_dir, summary)
    print(
        terminal_summary(summary, args.verbose),
        file=sys.stderr if failed else sys.stdout,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
