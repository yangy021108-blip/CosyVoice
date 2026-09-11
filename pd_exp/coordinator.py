#!/usr/bin/env python3
"""Coordinate a real prompt-embedding NIXL 1P1D CosyVoice experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--acoustic-gpu", default="2")
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B",
    )
    parser.add_argument("--voices-json", type=Path, default=ROOT / "pd_exp/voices.json")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--text", default=None)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "pd_exp/results/pd_run")
    parser.add_argument("--baseline-tokens", type=Path, default=ROOT / "pd_exp/results/baseline_tokens.json")
    parser.add_argument(
        "--prompt-payload", type=Path, default=None,
        help="Canonical prompt_payload.pt saved by baseline.py.",
    )
    parser.add_argument(
        "--skip-acoustic", action="store_true",
        help="Profile only P/D using the already validated fixed token path.",
    )
    parser.add_argument("--instrument-nixl", action="store_true")
    parser.add_argument(
        "--standard-optimized", action="store_true",
        help="Enable the existing vLLM CUDA graph path after eager Gate 1.",
    )
    parser.add_argument(
        "--torch-profile", action="store_true",
        help="Write separate vLLM EngineCore PyTorch traces for P and D.",
    )
    parser.add_argument("--decode-progress-file", type=Path, default=None)
    parser.add_argument("--decode-progress-iteration", type=int, default=1)
    parser.add_argument("--decode-progress-tokens", type=int, default=40)
    parser.add_argument(
        "--step-trace", action="store_true",
        help="Record per-step decode inputs, graph mode, logits top-k and token.",
    )
    parser.add_argument(
        "--graph-safe-remote-prefill", action="store_true",
        help="Run the consumer's final local prompt token eagerly, then use graphs.",
    )
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--prefill-side-channel-port", type=int, default=5650)
    parser.add_argument("--decode-side-channel-port", type=int, default=5651)
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Keep both vLLM engines alive and run N sequential requests.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of requests admitted together in each iteration.",
    )
    parser.add_argument(
        "--matcha-path", type=Path,
        default=ROOT / "third_party" / "Matcha-TTS",
    )
    return parser.parse_args()


def _wait_process_files(
    paths: list[Path], processes: list[subprocess.Popen[str]], timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while not all(path.exists() for path in paths):
        failed = [proc for proc in processes if proc.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"Worker exited early with code {failed[0].returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {paths}")
        time.sleep(0.1)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _kv_trace_summary(records: list[dict], request_id: str) -> dict:
    selected = [item for item in records if item.get("request_id") == request_id]
    prepared = next(
        (item for item in selected if item["event"] == "kv_prepare_complete"),
        None,
    )
    submitted = next(
        (item for item in selected if item["event"] == "kv_transfer_submit"),
        None,
    )
    completed = next(
        (item for item in selected if item["event"] == "kv_transfer_complete"),
        None,
    )
    duration_ns = None if completed is None else completed.get("duration_ns")
    estimated_bytes = None
    if completed is not None:
        estimated_bytes = completed.get("estimated_bytes")
    if estimated_bytes is None and submitted is not None:
        estimated_bytes = submitted.get("estimated_bytes")
    seconds = None if duration_ns is None else duration_ns / 1e9
    bandwidth = None
    if seconds and estimated_bytes is not None:
        bandwidth = estimated_bytes / seconds
    return {
        "request_id": request_id,
        "prepare_seconds": (
            None if prepared is None else prepared["duration_ns"] / 1e9
        ),
        "transfer_seconds": seconds,
        "estimated_bytes": estimated_bytes,
        "estimated_bandwidth_bytes_per_second": bandwidth,
        "measurement_boundary": (
            "NixlWrapper.transfer() submission to first DONE observation"
        ),
        "events": selected,
    }


def main() -> int:
    args = parse_args()
    if args.iterations < 1 or args.batch_size < 1:
        raise ValueError("--iterations and --batch-size must be at least 1")
    if args.prefill_gpu == args.decode_gpu:
        raise ValueError("Prefill and decode require distinct GPUs")
    if (
        not args.skip_acoustic
        and len({args.prefill_gpu, args.decode_gpu, args.acoustic_gpu}) != 3
    ):
        raise ValueError("Acoustic decoding requires a third distinct GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.acoustic_gpu)
    if str(args.matcha_path) not in sys.path:
        sys.path.append(str(args.matcha_path))

    import torch
    from pd_exp.common import (
        DEFAULT_TEXT,
        indexed_path,
        prompt_payload,
        read_json,
        timing_summary,
        write_json,
    )
    from pd_exp.compare_tokens import compare
    text = args.text or DEFAULT_TEXT
    work = args.results_dir.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    overall_started = time.perf_counter()
    baseline = read_json(args.baseline_tokens)
    canonical_only = args.prompt_payload is not None and args.skip_acoustic
    backend = None
    if canonical_only:
        payload = torch.load(
            args.prompt_payload, map_location="cpu", weights_only=False,
        )
        spans = payload["metadata"]["spans"]
        text = baseline["text"]
        chunks = baseline["frontend_chunks"]
        frontend_seconds = 0.0
    else:
        from research.error_pattern.phase2_5_core import (
            _prepare_lm_input,
            build_model_inputs,
            load_research_backend,
        )

        backend, _ = load_research_backend(
            model_dir=args.model_dir,
            voices_json=args.voices_json,
            voice_id=args.voice,
            repository_root=ROOT,
            load_vllm=False,
            fp16=False,
        )
        frontend_started = time.perf_counter()
        chunks, model_inputs = build_model_inputs(
            backend, text=text, voice_id=args.voice, text_frontend=True,
        )
        frontend_seconds = time.perf_counter() - frontend_started
        if len(model_inputs) != 1:
            raise RuntimeError(
                "The first PD PoC requires exactly one frontend chunk"
            )
        llm = backend.model.llm
        lm_input, minimum_tokens, maximum_tokens, spans = _prepare_lm_input(
            llm, model_inputs[0],
            min_token_text_ratio=2.0,
            max_token_text_ratio=20.0,
        )
        payload = prompt_payload(
            lm_input.squeeze(0), spans=spans,
            minimum_tokens=minimum_tokens, maximum_tokens=maximum_tokens,
            seed=args.seed, stop_token_ids=list(llm.stop_token_ids),
        )
        if args.prompt_payload is not None:
            payload = torch.load(
                args.prompt_payload, map_location="cpu", weights_only=False,
            )
            spans = payload["metadata"]["spans"]
    input_path = work / "prompt_embeds.pt"
    torch.save(payload, input_path)

    prefill_output = work / "prefill.json"
    decode_output = work / "decode.json"
    prefill_ready = work / "prefill.ready.json"
    decode_ready = work / "decode.ready.json"
    release = work / "prefill.release.json"
    prefill_trace = work / "prefill_trace.jsonl"
    decode_trace = work / "decode_trace.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(args.matcha_path), env.get("PYTHONPATH", "")]
    )
    # The validated same-host path uses loopback TCP for NIXL's active-message
    # control channel and CUDA IPC/copy for KV data.  Enumerating every mlx5
    # device can exhaust the default file-descriptor limit on this host.
    env.setdefault("UCX_TLS", "tcp,cuda_ipc,cuda_copy,sm,self")
    env.setdefault("UCX_NET_DEVICES", "lo")
    common = [
        "--model-dir", str(args.model_dir / "vllm"),
        "--input", str(input_path),
        "--timeout", str(args.timeout),
        "--iterations", str(args.iterations),
        "--batch-size", str(args.batch_size),
    ]
    prefill_cmd = [
        sys.executable, str(ROOT / "pd_exp/prefill_worker.py"),
        "--gpu", str(args.prefill_gpu), *common,
        "--output", str(prefill_output), "--ready", str(prefill_ready),
        "--release", str(release),
        "--side-channel-port", str(args.prefill_side_channel_port),
        "--trace", str(prefill_trace),
    ]
    decode_cmd = [
        sys.executable, str(ROOT / "pd_exp/decode_worker.py"),
        "--gpu", str(args.decode_gpu), *common,
        "--prefill-output", str(prefill_output),
        "--output", str(decode_output), "--ready", str(decode_ready),
        "--release", str(release),
        "--side-channel-port", str(args.decode_side_channel_port),
        "--trace", str(decode_trace),
    ]
    if os.environ.get("COSY_PD_TRUNCATE_PREFILL_LAST") == "1":
        prefill_cmd.append("--truncate-last-prompt-token")
    if args.instrument_nixl:
        prefill_cmd.append("--instrument-nixl")
        decode_cmd.append("--instrument-nixl")
    if args.standard_optimized:
        prefill_cmd.append("--standard-optimized")
        decode_cmd.append("--standard-optimized")
    if args.step_trace:
        decode_cmd.extend(["--step-trace", str(work / "decode_steps.jsonl")])
    if args.graph_safe_remote_prefill:
        decode_cmd.append("--graph-safe-remote-prefill")
    if args.torch_profile:
        prefill_cmd.extend([
            "--profile-dir", str(work / "prefill_profile"),
        ])
        decode_cmd.extend([
            "--profile-dir", str(work / "decode_profile"),
        ])
    if args.decode_progress_file is not None:
        progress_index = args.decode_progress_iteration * args.batch_size
        progress_request_id = (
            "cosy-pd-decode"
            if args.iterations * args.batch_size == 1
            else f"cosy-pd-decode-{progress_index:04d}"
        )
        decode_cmd.extend([
            "--progress-file", str(args.decode_progress_file),
            "--progress-request-id", progress_request_id,
            "--progress-tokens", str(args.decode_progress_tokens),
        ])
    prefill_log = (work / "prefill.log").open("w", encoding="utf-8")
    decode_log = (work / "decode.log").open("w", encoding="utf-8")
    processes: list[subprocess.Popen[str]] = []
    workers_started = time.perf_counter()
    workers_ready_seconds = None
    try:
        decode_proc = subprocess.Popen(
            decode_cmd, env=env, stdout=decode_log, stderr=subprocess.STDOUT, text=True,
        )
        prefill_proc = subprocess.Popen(
            prefill_cmd, env=env, stdout=prefill_log, stderr=subprocess.STDOUT, text=True,
        )
        processes = [prefill_proc, decode_proc]
        _wait_process_files([prefill_ready, decode_ready], processes, args.timeout)
        workers_ready_seconds = time.perf_counter() - workers_started
        total_requests = args.iterations * args.batch_size
        prefill_outputs = [
            indexed_path(prefill_output, index, total_requests)
            for index in range(total_requests)
        ]
        decode_outputs = [
            indexed_path(decode_output, index, total_requests)
            for index in range(total_requests)
        ]
        _wait_process_files(
            prefill_outputs + decode_outputs, processes, args.timeout,
        )
        prefills = [read_json(path) for path in prefill_outputs]
        decodes = [read_json(path) for path in decode_outputs]
        for prefill, decode in zip(prefills, decodes):
            if prefill.get("status") != "ok" or decode.get("status") != "ok":
                raise RuntimeError(
                    f"PD worker failure: prefill={prefill}, decode={decode}"
                )
        worker_exit_timeout = 180 if args.torch_profile else 30
        for proc in processes:
            proc.wait(timeout=worker_exit_timeout)
    finally:
        if not release.exists():
            write_json(release, {"status": "coordinator_cleanup"})
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        prefill_log.close()
        decode_log.close()

    selected_index = args.iterations * args.batch_size - 1
    prefill = prefills[selected_index]
    decode = decodes[selected_index]
    raw_tokens = [int(item) for item in decode["raw_speech_tokens"]]
    raw_comparison = compare(baseline["raw_speech_tokens"], raw_tokens)
    if canonical_only and raw_comparison["token_by_token_equal"]:
        filtered_tokens = list(baseline["filtered_speech_tokens"])
        dropped_indexes = list(baseline["dropped_silent_token_indexes"])
    else:
        from research.error_pattern.phase2_5_core import filter_silent_tokens

        if backend is None:
            raise RuntimeError(
                "Cannot filter a mismatched canonical-only token trajectory"
            )
        filtered_tokens, dropped_indexes = filter_silent_tokens(
            raw_tokens, backend.model.silent_tokens,
        )
    pd_tokens = {
        "text": text,
        "seed": args.seed,
        "frontend_chunks": chunks,
        "spans": spans,
        "raw_speech_tokens": raw_tokens,
        "filtered_speech_tokens": filtered_tokens,
        "dropped_silent_token_indexes": dropped_indexes,
    }
    write_json(work / "pd_tokens.json", pd_tokens)
    filtered_comparison = compare(
        baseline["filtered_speech_tokens"], filtered_tokens,
    )
    trace_records = _read_jsonl(decode_trace)
    iteration_results = []
    for index, (iteration_prefill, iteration_decode) in enumerate(
        zip(prefills, decodes)
    ):
        iteration_tokens = [
            int(item) for item in iteration_decode["raw_speech_tokens"]
        ]
        request_id = iteration_decode["metrics"]["request_id"]
        iteration_results.append({
            "linear_index": index,
            "iteration": iteration_decode["iteration"],
            "request_in_batch": iteration_decode["request_in_batch"],
            "cold_connector": iteration_decode["iteration"] == 0,
            "prefill": iteration_prefill["metrics"],
            "decode": iteration_decode["metrics"],
            "decode_tpot": timing_summary(iteration_decode["metrics"]),
            "kv_transfer": _kv_trace_summary(trace_records, request_id),
            "raw_token_correctness": compare(
                baseline["raw_speech_tokens"], iteration_tokens,
            ),
        })
    selected_kv = iteration_results[selected_index]["kv_transfer"]

    if args.skip_acoustic:
        phases = {}
        acoustic = {
            "decode_latency_seconds": 0.0,
            "audio_duration_seconds": None,
            "skipped": True,
            "reason": "LLM-only steady-state profile after Gate 2 passed",
        }
        audio = {"skipped": True}
    else:
        from research.error_pattern.phase2_5_core import (
            decode_fixed_trajectory,
            write_waveform,
        )
        from pd_exp.common import wrap_nvtx_method
        from tools.cosyvoice_phase_timing import PhaseCollector, wrap_method

        assert backend is not None
        collector = PhaseCollector()
        wrap_nvtx_method(backend.model.flow, "inference", "COSY_FLOW")
        wrap_nvtx_method(backend.model.hift, "inference", "COSY_HIFT")
        wrap_method(backend.model.flow, "inference", "flow_seconds", collector)
        wrap_method(backend.model.hift, "inference", "hift_seconds", collector)
        trajectory = {
            "text": text,
            "frontend_chunks": chunks,
            "chunks": [{
                "spans": spans,
                "filtered_speech_tokens": filtered_tokens,
            }],
        }
        mark = collector.mark()
        waveform, acoustic = decode_fixed_trajectory(
            backend, trajectory=trajectory, flow_seed=0, voice_id=args.voice,
            text_frontend=True, speed=1.0,
        )
        phases = collector.since(mark)
        audio = write_waveform(
            work / "pd.wav", waveform, int(backend.sample_rate),
        )
    e2e_seconds = time.perf_counter() - overall_started
    metrics = {
        "mode": "vllm_pd_1p1d",
        "engine_mode": (
            "standard_optimized" if args.standard_optimized else "eager"
        ),
        "batch_size": args.batch_size,
        "iteration_count": args.iterations,
        "gpu_mapping": {
            "prefill": str(args.prefill_gpu),
            "decode": str(args.decode_gpu),
            "acoustic": str(args.acoustic_gpu),
        },
        "iterations": iteration_results,
        "selected_steady_state_iteration": selected_index,
        "worker_startup_to_ready_seconds": workers_ready_seconds,
        "frontend_seconds": frontend_seconds,
        "prefill": prefill["metrics"],
        "decode": decode["metrics"],
        "decode_tpot": timing_summary(decode["metrics"]),
        "kv_transfer": {
            "transport": "NIXL/UCX GPU memory pull",
            "remote_block_count": len(prefill["kv_transfer_params"]["remote_block_ids"]),
            "decoder_external_cached_tokens": decode["metrics"]["num_cached_tokens"],
            "wait_plus_first_decode_seconds": decode["metrics"]["first_token_seconds"],
            "prepare_seconds": selected_kv["prepare_seconds"],
            "independent_transfer_seconds": selected_kv["transfer_seconds"],
            "estimated_bytes": selected_kv["estimated_bytes"],
            "estimated_bandwidth_bytes_per_second": selected_kv[
                "estimated_bandwidth_bytes_per_second"
            ],
            "measurement_note": selected_kv["measurement_boundary"],
        },
        "token_correctness": {
            "raw": raw_comparison,
            "filtered": filtered_comparison,
        },
        "flow_seconds": phases.get("flow_seconds", 0.0),
        "hift_seconds": phases.get("hift_seconds", 0.0),
        "acoustic_total_seconds": acoustic["decode_latency_seconds"],
        "e2e_seconds": e2e_seconds,
        "steady_state_e2e_estimate_seconds": (
            frontend_seconds
            + decode["metrics"]["total_seconds"]
            + acoustic["decode_latency_seconds"]
        ),
        "audio_duration_seconds": acoustic["audio_duration_seconds"],
        "rtf": (
            None
            if acoustic["audio_duration_seconds"] is None
            else e2e_seconds / acoustic["audio_duration_seconds"]
        ),
        "audio": {**acoustic, **audio},
        "torch_cuda_peak_allocated_bytes_acoustic": (
            0 if backend is None else torch.cuda.max_memory_allocated()
        ),
        "torch_cuda_peak_reserved_bytes_acoustic": (
            0 if backend is None else torch.cuda.max_memory_reserved()
        ),
    }
    write_json(work / "pd_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not raw_comparison["token_by_token_equal"]:
        print("Gate 1 failed: raw speech-token mismatch", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
