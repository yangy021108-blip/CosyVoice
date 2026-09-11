#!/usr/bin/env python3
"""Reproducible one-H100 unified vLLM baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B",
    )
    parser.add_argument("--voices-json", type=Path, default=ROOT / "pd_exp/voices.json")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--text", default=None)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "pd_exp/results")
    parser.add_argument("--prompt-payload", type=Path, default=None)
    parser.add_argument("--skip-acoustic", action="store_true")
    parser.add_argument(
        "--matcha-path", type=Path,
        default=ROOT / "third_party" / "Matcha-TTS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if str(args.matcha_path) not in sys.path:
        sys.path.append(str(args.matcha_path))

    import torch
    from pd_exp.common import (
        DEFAULT_TEXT,
        build_engine,
        make_sampling_params,
        prompt_payload,
        run_engine_request,
        timing_summary,
        wrap_nvtx_method,
        write_json,
    )
    from research.error_pattern.phase2_5_core import (
        _prepare_lm_input,
        build_model_inputs,
        decode_fixed_trajectory,
        filter_silent_tokens,
        load_research_backend,
        write_waveform,
    )
    from tools.cosyvoice_phase_timing import PhaseCollector, wrap_method

    args.results_dir.mkdir(parents=True, exist_ok=True)
    text = args.text or DEFAULT_TEXT
    backend, _ = load_research_backend(
        model_dir=args.model_dir,
        voices_json=args.voices_json,
        voice_id=args.voice,
        repository_root=ROOT,
        load_vllm=False,
        fp16=False,
    )
    # Drive the installed vLLM 0.11 engine directly.  The merged production
    # loader currently contains optional profiler arguments introduced after
    # 0.11; keeping this compatibility shim inside pd_exp avoids changing the
    # production backend before the PD correctness gate.
    llm = backend.model.llm
    llm.vllm = build_engine(args.model_dir / "vllm", eager=False)
    llm.lock = threading.Lock()
    del llm.llm.model.model.layers
    overall_started = time.perf_counter()
    frontend_started = time.perf_counter()
    chunks, model_inputs = build_model_inputs(
        backend, text=text, voice_id=args.voice, text_frontend=True,
    )
    frontend_seconds = time.perf_counter() - frontend_started
    if len(model_inputs) != 1:
        raise RuntimeError("The first PD baseline requires exactly one frontend chunk")

    lm_input, minimum_tokens, maximum_tokens, spans = _prepare_lm_input(
        llm,
        model_inputs[0],
        min_token_text_ratio=2.0,
        max_token_text_ratio=20.0,
    )
    payload = prompt_payload(
        lm_input.squeeze(0),
        spans=spans,
        minimum_tokens=minimum_tokens,
        maximum_tokens=maximum_tokens,
        seed=args.seed,
        stop_token_ids=list(llm.stop_token_ids),
    )
    if args.prompt_payload is not None:
        payload = torch.load(
            args.prompt_payload, map_location="cpu", weights_only=False,
        )
        spans = payload["metadata"]["spans"]
    # This file is the canonical cross-process correctness input.  Reusing it
    # avoids allowing frontend normalization drift to masquerade as a PD/KV
    # mismatch.
    torch.save(payload, args.results_dir / "prompt_payload.pt")
    sampling = make_sampling_params(payload["metadata"])
    raw_tokens, llm_metrics, _ = run_engine_request(
        llm.vllm, payload["prompt_embeds"], sampling,
        request_id="cosy-unified-baseline",
    )
    filtered_tokens, dropped_indexes = filter_silent_tokens(
        raw_tokens, backend.model.silent_tokens,
    )

    if args.skip_acoustic:
        phases = {}
        acoustic_metrics = {
            "decode_latency_seconds": 0.0,
            "audio_duration_seconds": None,
            "skipped": True,
        }
        audio_metadata = {"skipped": True}
    else:
        trajectory = {
            "text": text,
            "frontend_chunks": chunks,
            "chunks": [{
                "spans": spans,
                "filtered_speech_tokens": filtered_tokens,
            }],
        }
        collector = PhaseCollector()
        wrap_nvtx_method(backend.model.flow, "inference", "COSY_FLOW")
        wrap_nvtx_method(backend.model.hift, "inference", "COSY_HIFT")
        wrap_method(backend.model.flow, "inference", "flow_seconds", collector)
        wrap_method(backend.model.hift, "inference", "hift_seconds", collector)
        mark = collector.mark()
        waveform, acoustic_metrics = decode_fixed_trajectory(
            backend,
            trajectory=trajectory,
            flow_seed=0,
            voice_id=args.voice,
            text_frontend=True,
            speed=1.0,
        )
        phases = collector.since(mark)
        audio_metadata = write_waveform(
            args.results_dir / "baseline.wav",
            waveform,
            int(backend.sample_rate),
        )
    e2e_seconds = time.perf_counter() - overall_started
    tokens = {
        "text": text,
        "seed": args.seed,
        "frontend_chunks": chunks,
        "spans": spans,
        "raw_speech_tokens": raw_tokens,
        "filtered_speech_tokens": filtered_tokens,
        "dropped_silent_token_indexes": dropped_indexes,
    }
    metrics = {
        "mode": "unified_vllm",
        "gpu": str(args.gpu),
        "model_dir": str(args.model_dir.resolve()),
        "text_token_count": len(spans["target_text_token_ids"]),
        "prompt_text_token_count": len(spans["prompt_text_token_ids"]),
        "prompt_speech_token_count": len(spans["prompt_speech_token_ids"]),
        "prompt_embeds_shape": payload["metadata"]["prompt_embeds_shape"],
        "generated_raw_speech_token_count": len(raw_tokens),
        "generated_filtered_speech_token_count": len(filtered_tokens),
        "frontend_seconds": frontend_seconds,
        "llm_total_seconds": llm_metrics["total_seconds"],
        "ttft_speech_token_seconds": llm_metrics["first_token_seconds"],
        "speech_token_decode_seconds": (
            llm_metrics["total_seconds"] - (llm_metrics["first_token_seconds"] or 0.0)
        ),
        "tpot": timing_summary(llm_metrics),
        "flow_seconds": phases.get("flow_seconds", 0.0),
        "hift_seconds": phases.get("hift_seconds", 0.0),
        "acoustic_total_seconds": acoustic_metrics["decode_latency_seconds"],
        "e2e_seconds": e2e_seconds,
        "audio_duration_seconds": acoustic_metrics["audio_duration_seconds"],
        "rtf": (
            None
            if acoustic_metrics["audio_duration_seconds"] is None
            else e2e_seconds / acoustic_metrics["audio_duration_seconds"]
        ),
        "llm_detail": llm_metrics,
        "audio": {**acoustic_metrics, **audio_metadata},
        "torch_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    write_json(args.results_dir / "baseline_tokens.json", tokens)
    write_json(args.results_dir / "baseline_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
