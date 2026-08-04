#!/usr/bin/env python3
"""Profile full CosyVoice vLLM inference and export a Perfetto JSON trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATCHA_PATH = ROOT / "third_party" / "Matcha-TTS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MATCHA_PATH) not in sys.path:
    sys.path.append(str(MATCHA_PATH))

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from api_server.config import Settings
from api_server.engine import CosyVoiceEngine
from api_server.schemas import SpeechRequest
from api_server.voice_store import VoiceStore


def _activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    for name in ("SDAA", "PrivateUse1"):
        activity = getattr(ProfilerActivity, name, None)
        if activity is not None and activity not in activities:
            activities.append(activity)
    if len(activities) == 1:
        raise RuntimeError(
            "this torch build does not expose ProfilerActivity.SDAA or "
            "ProfilerActivity.PrivateUse1"
        )
    return activities


def _wrap_method(obj: Any, method_name: str, label: str) -> None:
    original = getattr(obj, method_name)

    def wrapped(*args: Any, **kwargs: Any):
        with record_function(label):
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)


def _install_phase_labels(engine: CosyVoiceEngine) -> None:
    model = engine.backend.model
    _wrap_method(model.llm, "inference", "cosyvoice.llm_speech_tokens")
    _wrap_method(model.flow, "inference", "cosyvoice.flow")
    _wrap_method(model.hift, "inference", "cosyvoice.hift")

    import cosyvoice.cli.model as model_module

    original_clear = model_module._clear_accelerator_cache

    def profiled_clear(device: torch.device) -> None:
        with record_function("cosyvoice.clear_accelerator_cache"):
            original_clear(device)

    model_module._clear_accelerator_cache = profiled_clear


def _settings(model_dir: Path) -> Settings:
    return Settings(
        root_dir=ROOT,
        model_alias="cosyvoice3-0.5b",
        model_dir=model_dir,
        voices_file=ROOT / "api_server" / "voices.json",
        api_key=None,
        host="127.0.0.1",
        port=0,
        max_text_characters=2000,
        max_concurrency=1,
        max_queue_size=0,
        request_timeout_seconds=600,
        fp16=True,
        load_vllm=True,
        allow_unauthenticated=True,
        default_seed=20260726,
        quality_check_enabled=True,
        quality_max_retries=0,
    )


def _request(text: str, seed: int) -> SpeechRequest:
    return SpeechRequest(
        model="cosyvoice3-0.5b",
        input=text,
        voice="default",
        response_format="wav",
        speed=1.0,
        seed=seed,
    )


def _run_one(
    engine: CosyVoiceEngine,
    voice: Any,
    request: SpeechRequest,
    output_path: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    with record_function("cosyvoice.full_request"):
        result = engine.synthesize(request, voice)
    wall_seconds = time.perf_counter() - started
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(result.content)
    return {
        "input": request.input,
        "seed": request.seed,
        "wall_seconds": wall_seconds,
        "inference_seconds": result.inference_seconds,
        "audio_duration_seconds": result.duration_seconds,
        "real_time_factor": result.real_time_factor,
        "wav_sha256": hashlib.sha256(result.content).hexdigest(),
        "wav_bytes": len(result.content),
        "wav_path": str(output_path) if output_path is not None else None,
        "silent_frame_ratio": result.silent_frame_ratio,
        "quality_retry_count": result.quality_retry_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/model"))
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outputs_dir = args.outputs_dir.resolve()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        line.strip()
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise RuntimeError("prompt file is empty")

    settings = _settings(args.model_dir)
    voice_store = VoiceStore.from_json(
        settings.voices_file,
        settings.root_dir,
    )
    engine = CosyVoiceEngine(settings, voice_store)
    engine.load()
    voice = voice_store.get("default")
    if voice is None:
        raise RuntimeError("default voice is not configured")
    _install_phase_labels(engine)

    warmup_records = []
    for index in range(args.warmup):
        warmup_records.append(
            _run_one(
                engine,
                voice,
                _request(prompts[index % len(prompts)], args.seed + index),
                None,
            )
        )

    trace_path = args.outputs_dir / f"{args.tag}.perfetto.json.gz"
    table_path = args.outputs_dir / f"{args.tag}.operator_table.txt"
    records = []
    activities = _activities()
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        with_stack=args.with_stack,
        profile_memory=False,
    ) as profiler:
        for index in range(args.active):
            records.append(
                _run_one(
                    engine,
                    voice,
                    _request(
                        prompts[index % len(prompts)],
                        args.seed + index,
                    ),
                    args.outputs_dir / "audio" / f"request_{index:04d}.wav",
                )
            )

    profiler.export_chrome_trace(str(trace_path))
    table = profiler.key_averages().table(
        sort_by="self_device_time_total",
        row_limit=100,
    )
    table_path.write_text(table, encoding="utf-8")
    summary = {
        "tool": "PyTorch Profiler / Perfetto",
        "tag": args.tag,
        "activities": [str(activity) for activity in activities],
        "warmup": args.warmup,
        "active": args.active,
        "record_shapes": args.record_shapes,
        "with_stack": args.with_stack,
        "trace_path": str(trace_path),
        "operator_table_path": str(table_path),
        "trace_bytes": trace_path.stat().st_size,
        "mean_wall_seconds": statistics.fmean(
            record["wall_seconds"] for record in records
        ),
        "mean_inference_seconds": statistics.fmean(
            record["inference_seconds"] for record in records
        ),
        "mean_real_time_factor": statistics.fmean(
            record["real_time_factor"] for record in records
        ),
        "warmup_records": warmup_records,
        "records": records,
    }
    summary_path = args.outputs_dir / f"{args.tag}.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
