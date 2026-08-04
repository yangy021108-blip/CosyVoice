#!/usr/bin/env python3
"""Benchmark real CosyVoice vLLM HTTP latency across SDAA configurations."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SERVER_MODULE = "api_server.main"
SERVER_COMMAND = [sys.executable, "-m", SERVER_MODULE]
FLAG_NAMES = (
    "COSYVOICE_SDAA_FLOW_FLASH_ATTN",
    "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH",
    "COSYVOICE_SDAA_HIFT_REFLECTION_PAD",
    "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE",
    "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE",
    "COSYVOICE_SDAA_FLOW_STEPS",
    "COSYVOICE_VLLM_SDAA_GRAPH",
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
    "baseline": {
        "COSYVOICE_SDAA_FLOW_FLASH_ATTN": "0",
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "0",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "0",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "0",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "0",
        "COSYVOICE_SDAA_FLOW_STEPS": "10",
        "COSYVOICE_VLLM_SDAA_GRAPH": "0",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "0",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "0",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "0",
    },
    "round4": {
        "COSYVOICE_SDAA_FLOW_FLASH_ATTN": "1",
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_VLLM_SDAA_GRAPH": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "0",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "0",
    },
    "optimized": {
        "COSYVOICE_SDAA_FLOW_FLASH_ATTN": "1",
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_VLLM_SDAA_GRAPH": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
        "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "0",
    },
    "optimized_trusted_mask": {
        "COSYVOICE_SDAA_FLOW_FLASH_ATTN": "1",
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_VLLM_SDAA_GRAPH": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
        "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "1",
    },
    "optimized_trusted_mask_conv_stft": {
        "COSYVOICE_SDAA_FLOW_FLASH_ATTN": "1",
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_VLLM_SDAA_GRAPH": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
        "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "1",
        "COSYVOICE_SDAA_HIFT_CONV_STFT": "1",
    },
    "optimized_trusted_mask_conv_stft_fused_ffn": {
        "COSYVOICE_SDAA_FLOW_FLASH_ATTN": "1",
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_VLLM_SDAA_GRAPH": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_NORM": "1",
        "COSYVOICE_SDAA_FLOW_PRECOMPUTE_ROPE": "1",
        "COSYVOICE_SDAA_HIFT_F0_FP32": "1",
        "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "1",
        "COSYVOICE_SDAA_HIFT_CONV_STFT": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_FFN": "1",
        "COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS": "800",
    },
    "optimized_trusted_mask_conv_stft_fused_ffn_gemm": {
        "COSYVOICE_SDAA_FLOW_FLASH_ATTN": "1",
        "COSYVOICE_SDAA_FLOW_UNMASKED_FASTPATH": "1",
        "COSYVOICE_SDAA_HIFT_REFLECTION_PAD": "1",
        "COSYVOICE_SDAA_HIFT_NEAREST_UPSAMPLE": "1",
        "COSYVOICE_SDAA_HIFT_CONTIGUOUS_UPSAMPLE": "1",
        "COSYVOICE_SDAA_FLOW_STEPS": "8",
        "COSYVOICE_VLLM_SDAA_GRAPH": "1",
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
}
RUN_ORDER = (
    "baseline",
    "round4",
    "optimized",
    "optimized",
    "round4",
    "baseline",
)
FINAL_VARIANT = "optimized_trusted_mask_conv_stft_fused_ffn_gemm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=ROOT / "tools" / "cosyvoice_perf_prompts.txt",
    )
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=None,
    )
    parser.add_argument("--cycles", type=int, default=2)
    return parser.parse_args()


def read_process_environment(pid: int) -> dict[str, str]:
    data = Path(f"/proc/{pid}/environ").read_bytes()
    environment: dict[str, str] = {}
    for item in data.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode()] = value.decode()
    return environment


def find_api_server_pid() -> int | None:
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            command = (process_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        arguments = [item for item in command.split(b"\0") if item]
        if any(
            arguments[index:index + 2] == [b"-m", b"api_server.main"]
            for index in range(len(arguments) - 1)
        ):
            return int(process_dir.name)
    return None


def stop_process(pid: int, timeout: float = 45.0) -> None:
    process_path = Path(f"/proc/{pid}")
    if not process_path.exists():
        return
    try:
        process_group = os.getpgid(pid)
        os.killpg(process_group, signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    deadline = time.monotonic() + timeout
    while process_path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_path.exists():
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(1.0)


def wait_ready(port: int, timeout: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/ready"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                body = json.load(response)
            if body.get("status") == "ready":
                return body
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(
        f"API server did not become ready at {url}: {last_error}"
    )


def start_server(
    environment: dict[str, str],
    log_path: Path,
    port: int,
    timeout: int,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        SERVER_COMMAND,
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_ready(port, timeout)
    except Exception:
        stop_process(process.pid)
        raise
    return process


def variant_environment(
    base_environment: dict[str, str],
    variant: str,
) -> dict[str, str]:
    environment = base_environment.copy()
    for name in FLAG_NAMES:
        environment.pop(name, None)
    environment.update(VARIANTS[variant])
    return environment


def run_evalscope(
    args: argparse.Namespace,
    environment: dict[str, str],
    run_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "tools" / "cosyvoice_evalscope_perf.py"),
        "--url",
        f"http://127.0.0.1:{args.port}/v1/audio/speech",
        "--model",
        "cosyvoice3-0.5b",
        "--voice",
        "default",
        "--prompt-file",
        str(args.prompt_file.resolve()),
        "--outputs-dir",
        str(run_dir.resolve()),
        "--number",
        str(args.number),
        "--parallel",
        "1",
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--run-name",
        run_name,
    ]
    with (run_dir / "evalscope_console.log").open(
        "w", encoding="utf-8"
    ) as console:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads((run_dir / "tts_summary.json").read_text())


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        record
        for run in runs
        for record in run["summary"]["records"]
    ]
    client = [record["client_latency_seconds"] for record in records]
    server = [record["inference_latency_ms"] for record in records]
    rtf = [record["real_time_factor"] for record in records]
    total_wall = sum(run["summary"]["wall_seconds"] for run in runs)
    total_audio = sum(
        record["audio_duration_seconds"] for record in records
    )
    return {
        "run_count": len(runs),
        "request_count": len(records),
        "client_latency_seconds": {
            "mean": statistics.fmean(client),
            "median": statistics.median(client),
            "p95": percentile(client, 95),
            "min": min(client),
            "max": max(client),
        },
        "server_inference_ms": {
            "mean": statistics.fmean(server),
            "median": statistics.median(server),
            "p95": percentile(server, 95),
            "min": min(server),
            "max": max(server),
        },
        "request_throughput_per_second": len(records) / total_wall,
        "audio_throughput_realtime_x": total_audio / total_wall,
        "mean_real_time_factor": statistics.fmean(rtf),
        "audio_seconds_generated": total_audio,
        "quality_retry_count": sum(
            record["quality_retry_count"] for record in records
        ),
        "run_means": [
            {
                "run_index": run["run_index"],
                "client_latency_seconds": run["summary"][
                    "client_latency_seconds"
                ]["mean"],
                "server_inference_ms": run["summary"][
                    "server_inference_ms"
                ]["mean"],
                "request_throughput_per_second": run["summary"][
                    "request_throughput_per_second"
                ],
                "mean_real_time_factor": run["summary"][
                    "real_time_factor"
                ]["mean"],
            }
            for run in runs
        ],
    }


def confidence_interval_95(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return [mean, mean]
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    # 20 paired observations use t(19, 0.975) ~= 2.093.
    critical = 2.093 if len(values) == 20 else 1.96
    return [
        mean - critical * standard_error,
        mean + critical * standard_error,
    ]


def compare_variants(
    first: str,
    second: str,
    runs_by_variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    first_runs = sorted(
        runs_by_variant[first], key=lambda run: run["cycle"]
    )
    second_runs = sorted(
        runs_by_variant[second], key=lambda run: run["cycle"]
    )
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for first_run, second_run in zip(first_runs, second_runs):
        first_records = first_run["summary"]["records"]
        second_records = second_run["summary"]["records"]
        pairs.extend(zip(first_records, second_records))

    client_relative = [
        (candidate["client_latency_seconds"]
         / baseline["client_latency_seconds"] - 1.0)
        * 100.0
        for baseline, candidate in pairs
    ]
    server_relative = [
        (candidate["inference_latency_ms"]
         / baseline["inference_latency_ms"] - 1.0)
        * 100.0
        for baseline, candidate in pairs
    ]
    first_aggregate = aggregate_runs(runs_by_variant[first])
    second_aggregate = aggregate_runs(runs_by_variant[second])
    return {
        "first": first,
        "second": second,
        "pair_count": len(pairs),
        "client_latency_change_percent_from_aggregate_means": (
            second_aggregate["client_latency_seconds"]["mean"]
            / first_aggregate["client_latency_seconds"]["mean"]
            - 1.0
        )
        * 100.0,
        "server_latency_change_percent_from_aggregate_means": (
            second_aggregate["server_inference_ms"]["mean"]
            / first_aggregate["server_inference_ms"]["mean"]
            - 1.0
        )
        * 100.0,
        "throughput_change_percent": (
            second_aggregate["request_throughput_per_second"]
            / first_aggregate["request_throughput_per_second"]
            - 1.0
        )
        * 100.0,
        "rtf_change_percent": (
            second_aggregate["mean_real_time_factor"]
            / first_aggregate["mean_real_time_factor"]
            - 1.0
        )
        * 100.0,
        "paired_client_relative_percent": {
            "mean": statistics.fmean(client_relative),
            "median": statistics.median(client_relative),
            "confidence_interval_95": confidence_interval_95(
                client_relative
            ),
            "second_faster_count": sum(
                value < 0 for value in client_relative
            ),
        },
        "paired_server_relative_percent": {
            "mean": statistics.fmean(server_relative),
            "median": statistics.median(server_relative),
            "confidence_interval_95": confidence_interval_95(
                server_relative
            ),
            "second_faster_count": sum(
                value < 0 for value in server_relative
            ),
        },
        "audio_duration_equal_count": sum(
            first_record["audio_duration_seconds"]
            == second_record["audio_duration_seconds"]
            for first_record, second_record in pairs
        ),
        "wav_sha256_equal_count": sum(
            first_record["wav_sha256"] == second_record["wav_sha256"]
            for first_record, second_record in pairs
        ),
        "quality_retry_count": sum(
            record["quality_retry_count"]
            for pair in pairs
            for record in pair
        ),
    }


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.number <= 0:
        raise ValueError("--number must be greater than zero")
    if args.warmup < 0:
        raise ValueError("--warmup must not be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_variants = args.variants or [
        "baseline",
        "round4",
        "optimized",
    ]
    if args.cycles <= 0:
        raise ValueError("--cycles must be greater than zero")
    run_order: list[str] = []
    for cycle in range(args.cycles):
        cycle_order = (
            selected_variants
            if cycle % 2 == 0
            else list(reversed(selected_variants))
        )
        run_order.extend(cycle_order)

    original_pid = find_api_server_pid()
    if original_pid is None:
        raise RuntimeError(
            "No running api_server.main process is available to provide "
            "the validated SDAA environment and API key"
        )
    base_environment = read_process_environment(original_pid)
    if not base_environment.get("COSYVOICE_API_KEY", "").strip():
        raise RuntimeError("COSYVOICE_API_KEY is missing from server process")
    stop_process(original_pid)

    runs: list[dict[str, Any]] = []
    failure: BaseException | None = None
    server: subprocess.Popen[bytes] | None = None
    try:
        for run_index, variant in enumerate(run_order):
            cycle = run_index // len(selected_variants)
            run_dir = (
                args.output_dir
                / f"run_{run_index:02d}_{variant}_cycle_{cycle}"
            )
            environment = variant_environment(base_environment, variant)
            server = start_server(
                environment,
                run_dir / "api_server.log",
                args.port,
                args.startup_timeout,
            )
            summary = run_evalscope(
                args,
                environment,
                run_dir,
                f"cosyvoice-e2e-audit-{run_index:02d}-{variant}",
            )
            runs.append(
                {
                    "run_index": run_index,
                    "cycle": cycle,
                    "variant": variant,
                    "output_dir": str(run_dir.relative_to(ROOT)),
                    "server_pid": server.pid,
                    "summary": summary,
                }
            )
            stop_process(server.pid)
            server = None
    except BaseException as exc:
        failure = exc
        if server is not None:
            stop_process(server.pid)
            server = None
    finally:
        final_environment = variant_environment(
            base_environment, FINAL_VARIANT
        )
        final_server = start_server(
            final_environment,
            args.output_dir / "final_optimized_api_server.log",
            args.port,
            args.startup_timeout,
        )

    if failure is not None:
        raise failure

    runs_by_variant = {
        variant: [
            run for run in runs if run["variant"] == variant
        ]
        for variant in selected_variants
    }
    results = {
        "method": {
            "endpoint": (
                f"http://127.0.0.1:{args.port}/v1/audio/speech"
            ),
            "run_order": run_order,
            "requests_per_run": args.number,
            "warmup_per_run": args.warmup,
            "parallel": 1,
            "seed": args.seed,
            "service_restarted_for_every_run": True,
            "final_optimized_server_pid": final_server.pid,
            "final_restored_variant": FINAL_VARIANT,
        },
        "variant_flags": VARIANTS,
        "variants": {
            variant: aggregate_runs(variant_runs)
            for variant, variant_runs in runs_by_variant.items()
        },
        "comparisons": {
            f"{first}_to_{second}": compare_variants(
                first, second, runs_by_variant
            )
            for first, second in zip(
                selected_variants, selected_variants[1:]
            )
        },
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key != "summary"
            }
            for run in runs
        ],
    }
    output_path = args.output_dir / "results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results: {output_path}")
    for variant, summary in results["variants"].items():
        print(
            f"{variant}: requests={summary['request_count']}, "
            f"latency={summary['client_latency_seconds']['mean']:.6f}s, "
            f"p50={summary['client_latency_seconds']['median']:.6f}s, "
            f"p95={summary['client_latency_seconds']['p95']:.6f}s, "
            f"throughput={summary['request_throughput_per_second']:.6f}"
            " req/s, "
            f"rtf={summary['mean_real_time_factor']:.6f}"
        )
    for name, comparison in results["comparisons"].items():
        paired = comparison["paired_client_relative_percent"]
        aggregate_change = comparison[
            "client_latency_change_percent_from_aggregate_means"
        ]
        print(
            f"{name}: aggregate={aggregate_change:.2f}%, "
            f"paired_mean={paired['mean']:.2f}%, "
            f"paired_95ci=[{paired['confidence_interval_95'][0]:.2f}%, "
            f"{paired['confidence_interval_95'][1]:.2f}%], "
            f"faster={paired['second_faster_count']}/"
            f"{comparison['pair_count']}"
        )
    print(
        "Final optimized server: "
        f"pid={final_server.pid}, port={args.port}, status=ready"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
