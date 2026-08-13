#!/usr/bin/env python3
"""Exercise the CosyVoice WebSocket stream and save both audio forms."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import io
import json
import subprocess
import wave
from pathlib import Path

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="ws://127.0.0.1:8021/v1/audio/speech/stream"
    )
    authentication = parser.add_mutually_exclusive_group(required=True)
    authentication.add_argument("--api-key")
    authentication.add_argument("--api-key-file", type=Path)
    parser.add_argument("--model", default="cosyvoice3-0.5b")
    parser.add_argument("--input", required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--response-format", choices=("wav", "pcm"), default="wav")
    parser.add_argument("--instructions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--play",
        action="store_true",
        help="play live PCM through ffplay while receiving it",
    )
    return parser.parse_args()


def read_api_key(args: argparse.Namespace) -> str:
    if args.api_key is not None:
        return args.api_key
    for line in args.api_key_file.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "COSYVOICE_API_KEY":
            return value.strip().strip("'\"")
    raise RuntimeError(
        f"COSYVOICE_API_KEY is missing from {args.api_key_file}"
    )


def connect_headers(api_key: str) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {api_key}"}
    parameters = inspect.signature(websockets.connect).parameters
    name = "additional_headers" if "additional_headers" in parameters else "extra_headers"
    return {name: headers}


def start_player(sample_rate: int) -> subprocess.Popen | None:
    try:
        return subprocess.Popen(
            [
                "ffplay",
                "-loglevel",
                "error",
                "-nodisp",
                "-autoexit",
                "-f",
                "s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("--play requires ffplay on PATH") from exc


def wav_pcm(content: bytes) -> bytes:
    with wave.open(io.BytesIO(content), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise RuntimeError("complete WAV is not mono PCM16")
        return wav_file.readframes(wav_file.getnframes())


async def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "model": args.model,
        "input": args.input,
        "voice": args.voice,
        "response_format": args.response_format,
        "speed": 1.0,
        "seed": args.seed,
    }
    if args.instructions:
        request["instructions"] = args.instructions

    live_pcm = bytearray()
    complete_audio: bytes | None = None
    complete_format: str | None = None
    transcript: list[dict] = []
    player = None

    try:
        async with websockets.connect(
            args.url,
            max_size=None,
            **connect_headers(read_api_key(args)),
        ) as websocket:
            ready = json.loads(await websocket.recv())
            if ready.get("event") != "ready":
                raise RuntimeError(f"expected ready event, received {ready}")
            transcript.append(ready)
            sample_rate = int(ready["sample_rate"])
            if args.play:
                player = start_player(sample_rate)

            await websocket.send(json.dumps(request, ensure_ascii=False))
            while True:
                event = json.loads(await websocket.recv())
                transcript.append(event)
                event_name = event.get("event")
                if event_name == "audio_chunk":
                    content = await websocket.recv()
                    if not isinstance(content, bytes):
                        raise RuntimeError("audio_chunk was not followed by binary PCM")
                    if len(content) != event["bytes"]:
                        raise RuntimeError("audio_chunk byte count does not match metadata")
                    live_pcm.extend(content)
                    if player is not None and player.stdin is not None:
                        player.stdin.write(content)
                        player.stdin.flush()
                    print(
                        f"chunk={event['index']} bytes={len(content)} "
                        f"audio_seconds={len(live_pcm) / 2 / sample_rate:.3f}",
                        flush=True,
                    )
                elif event_name == "complete_audio":
                    content = await websocket.recv()
                    if not isinstance(content, bytes):
                        raise RuntimeError(
                            "complete_audio was not followed by binary audio"
                        )
                    if len(content) != event["bytes"]:
                        raise RuntimeError(
                            "complete_audio byte count does not match metadata"
                        )
                    complete_audio = content
                    complete_format = event["format"]
                elif event_name == "error":
                    raise RuntimeError(json.dumps(event, ensure_ascii=False))
                elif event_name == "end":
                    print(json.dumps(event, ensure_ascii=False, indent=2))
                    break
    finally:
        if player is not None:
            if player.stdin is not None:
                player.stdin.close()
            player.wait(timeout=10)

    if complete_audio is None or complete_format is None:
        raise RuntimeError("server ended without complete_audio")
    final_pcm = wav_pcm(complete_audio) if complete_format == "wav" else complete_audio
    if bytes(live_pcm) != final_pcm:
        raise RuntimeError("concatenated live PCM differs from complete audio")

    live_path = args.output_dir / "stream_live.pcm"
    complete_path = args.output_dir / f"stream_complete.{complete_format}"
    metadata_path = args.output_dir / "stream_events.json"
    live_path.write_bytes(live_pcm)
    complete_path.write_bytes(complete_audio)
    metadata_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"live_pcm={live_path}")
    print(f"complete_audio={complete_path}")
    print("validation=live_pcm_matches_complete_audio")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
