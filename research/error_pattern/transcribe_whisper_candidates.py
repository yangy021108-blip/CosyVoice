#!/usr/bin/env python3
"""Transcribe adjudication candidates with an independent Whisper model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="zh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    model_path = args.model.resolve()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(model_path)
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=args.device,
    )

    records: list[dict[str, Any]] = []
    for audio_path_arg in args.audio:
        audio_path = audio_path_arg.resolve()
        waveform, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False,
        )
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        started = time.perf_counter()
        result = transcriber(
            {"raw": waveform, "sampling_rate": sample_rate},
            generate_kwargs={"language": args.language, "task": "transcribe"},
        )
        records.append(
            {
                "schema_version": 1,
                "audio": str(audio_path),
                "model": str(model_path),
                "language": args.language,
                "sample_rate": int(sample_rate),
                "text": str(result["text"]).strip(),
                "latency_seconds": time.perf_counter() - started,
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(
        json.dumps(
            {"output": str(output), "record_count": len(records)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
