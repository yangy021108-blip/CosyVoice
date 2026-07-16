"""Small dependency-free smoke client for a running CosyVoice API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--model", default="cosyvoice3-0.5b")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--text", default="你好，这是一次接口冒烟测试。")
    parser.add_argument("--output", type=Path, default=Path("smoke.wav"))
    args = parser.parse_args()

    body = json.dumps(
        {
            "model": args.model,
            "input": args.text,
            "voice": args.voice,
            "response_format": "wav",
            "speed": 1.0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
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
        response_headers = dict(response.headers.items())
    elapsed = time.perf_counter() - started
    args.output.write_bytes(content)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": len(content),
                "client_elapsed_seconds": round(elapsed, 3),
                "request_id": response_headers.get("X-Request-ID"),
                "audio_duration": response_headers.get("X-Audio-Duration"),
                "rtf": response_headers.get("X-Real-Time-Factor"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
