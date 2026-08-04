#!/usr/bin/env python3
"""Benchmark SDAA inference candidates without generating profiler traces."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATCHA_PATH = ROOT / "third_party" / "Matcha-TTS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MATCHA_PATH) not in sys.path:
    sys.path.append(str(MATCHA_PATH))

from api_server.engine import CosyVoiceEngine
from api_server.voice_store import VoiceStore
from tools.cosyvoice_perfetto_profile import _request, _run_one, _settings


FLAG_NAMES = (
    "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH",
    "COSYVOICE_SDAA_HIFT_REFLECTION_PAD",
    "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE",
    "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE",
    "COSYVOICE_SDAA_FLOW_STEPS",
    "COSYVOICE_SDAA_FLOW_FUSED_NORM",
    "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE",
    "COSYVOICE_SDAA_HIFT_F0_FP32",
    "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK",
    "COSYVOICE_SDAA_HIFT_CONV_STFT",
    "COSYVOICE_SDAA_FLOW_FUSED_FFN",
    "COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS",
    "COSYVOICE_SDAA_FLOW_FUSED_GEMM",
    "COSYVOICE_SDAA_FLOW_FUSED_GEMM_MIN_ROWS",
)
VARIANTS = {
    "current": {},
    "unmasked": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
    },
    "unmasked_reflect": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
    },
    "unmasked_hift": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
    },
    "unmasked_hift_steps8": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
    },
    "unmasked_hift_steps8_contiguous": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
    },
    "unmasked_hift_steps8_contiguous_fused_norm": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
    },
    "unmasked_hift_steps8_contiguous_fused_norm_precomputed_rope": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
    },
    "unmasked_hift_steps8_contiguous_fused_norm_precomputed_rope_f0_fp32": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
    },
    "optimized_trusted_mask": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
        "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "1",
    },
    "optimized_trusted_mask_conv_stft": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
        "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "1",
        "COSYVOICE_SDAA_HIFT_CONV_STFT": "1",
    },
    "optimized_trusted_mask_conv_stft_fused_ffn": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
        "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "1",
        "COSYVOICE_SDAA_HIFT_CONV_STFT": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_FFN": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS": "800",
    },
    "optimized_trusted_mask_conv_stft_fused_ffn_gemm": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
        "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "1",
        "COSYVOICE_SDAA_HIFT_CONV_STFT": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_FFN": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS": "800",
        "COSYVOICE_SDAA_FLOW_FUSED_GEMM": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_GEMM_MIN_ROWS": "512",
    },
    "unmasked_hift_steps6": {
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "6",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/model"))
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=["current", "unmasked"],
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def set_variant(name: str) -> None:
    os.environ["COSYVOICE_SDAA_FLOW_FLASH_ATTN"] = "1"
    for flag in FLAG_NAMES:
        os.environ[flag] = "0"
    os.environ.update(VARIANTS[name])


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "request_count": len(records),
        "mean_wall_seconds": statistics.fmean(
            record["wall_seconds"] for record in records
        ),
        "median_wall_seconds": statistics.median(
            record["wall_seconds"] for record in records
        ),
        "mean_inference_seconds": statistics.fmean(
            record["inference_seconds"] for record in records
        ),
        "mean_real_time_factor": statistics.fmean(
            record["real_time_factor"] for record in records
        ),
        "quality_retry_count": sum(
            record["quality_retry_count"] for record in records
        ),
        "records": records,
    }


def main() -> None:
    args = parse_args()
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

    by_variant: dict[str, list[dict[str, Any]]] = {
        name: [] for name in args.variants
    }
    for round_index in range(args.rounds):
        order = (
            args.variants
            if round_index % 2 == 0
            else list(reversed(args.variants))
        )
        for variant in order:
            set_variant(variant)
            for warmup_index in range(args.warmup):
                index = warmup_index % len(prompts)
                _run_one(
                    engine,
                    voice,
                    _request(prompts[index], args.seed + index),
                    None,
                )
            for request_index in range(args.requests):
                prompt_index = request_index % len(prompts)
                record = _run_one(
                    engine,
                    voice,
                    _request(
                        prompts[prompt_index],
                        args.seed + prompt_index,
                    ),
                    (
                        args.audio_dir
                        / variant
                        / (
                            f"round_{round_index:02d}_"
                            f"request_{request_index:02d}_"
                            f"seed_{args.seed + prompt_index}.wav"
                        )
                        if args.audio_dir is not None
                        else None
                    ),
                )
                record["round"] = round_index
                by_variant[variant].append(record)

    variants = {
        name: aggregate(records) for name, records in by_variant.items()
    }
    baseline = variants[args.variants[0]]["mean_inference_seconds"]
    for name, result in variants.items():
        candidate = result["mean_inference_seconds"]
        result["inference_change_percent_vs_first"] = (
            candidate / baseline - 1.0
        ) * 100.0
        result["speedup_vs_first"] = baseline / candidate

    output = {
        "variants_order": args.variants,
        "rounds": args.rounds,
        "requests_per_round": args.requests,
        "warmup_per_variant_round": args.warmup,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
