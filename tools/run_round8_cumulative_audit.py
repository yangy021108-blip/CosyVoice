#!/usr/bin/env python3
"""Balanced, restart-isolated HTTP audit for CosyVoice SDAA variants."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SERVER_COMMAND = [sys.executable, "-m", "api_server.main"]
VARIANT_ORDER = (
    "baseline_sdaa",
    "round5_final",
    "round8_final",
)
RUN_ORDER = (
    "baseline_sdaa",
    "round5_final",
    "round8_final",
    "round8_final",
    "round5_final",
    "baseline_sdaa",
    "round8_final",
    "baseline_sdaa",
    "round5_final",
    "round5_final",
    "baseline_sdaa",
    "round8_final",
)
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
BASELINE_FLAGS = {
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
    "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "0",
    "COSYVOICE_SDAA_HIFT_CONV_STFT": "0",
    "COSYVOICE_SDAA_FLOW_FUSED_FFN": "0",
    "COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS": "0",
    "COSYVOICE_SDAA_FLOW_FUSED_GEMM": "0",
    "COSYVOICE_SDAA_FLOW_FUSED_GEMM_MIN_ROWS": "0",
}
ROUND5_FLAGS = {
    **BASELINE_FLAGS,
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
}
ROUND8_FLAGS = {
    **ROUND5_FLAGS,
    "COSYVOICE_SDAA_FLOW_TRUST_UNMASKED_MASK": "1",
    "COSYVOICE_SDAA_HIFT_CONV_STFT": "1",
    "COSYVOICE_SDAA_FLOW_FUSED_FFN": "1",
    "COSYVOICE_SDAA_FLOW_FUSED_FFN_MIN_ROWS": "800",
    "COSYVOICE_SDAA_FLOW_FUSED_GEMM": "1",
    "COSYVOICE_SDAA_FLOW_FUSED_GEMM_MIN_ROWS": "512",
}
VARIANTS = {
    "baseline_sdaa": BASELINE_FLAGS,
    "round5_final": ROUND5_FLAGS,
    "round8_final": ROUND8_FLAGS,
}
TOKEN_TRACE_PREFIX = "vllm token trace "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "perf_results" / "round8_cumulative_audit",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=(
            ROOT / "cosyvoice_api_outputs" / "perf" /
            "round8_cumulative_audit_20260805"
        ),
    )
    parser.add_argument(
        "--prompt-file", type=Path,
        default=ROOT / "tools" / "cosyvoice_perf_prompts.txt",
    )
    parser.add_argument("--requests-per-run", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def run_git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True
    ).strip()


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
            arguments = [
                item for item in (process_dir / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(
            arguments[index:index + 2] == [b"-m", b"api_server.main"]
            for index in range(len(arguments) - 1)
        ):
            return int(process_dir.name)
    return None


def stop_process(pid: int, timeout: float = 60.0) -> None:
    process_path = Path(f"/proc/{pid}")
    if not process_path.exists():
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGINT)
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


def wait_ready(port: int, timeout: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/ready"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                result = json.load(response)
            if result.get("status") == "ready":
                return result
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"server was not ready at {url}: {last_error}")


def start_server(
    environment: dict[str, str], log_path: Path, port: int, timeout: int
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
    base_environment: dict[str, str], variant: str
) -> dict[str, str]:
    environment = base_environment.copy()
    for name in FLAG_NAMES:
        environment.pop(name, None)
    environment.update(VARIANTS[variant])
    environment["COSYVOICE_DEBUG_TOKEN_TRACE"] = "1"
    return environment


def run_evalscope(
    args: argparse.Namespace,
    environment: dict[str, str],
    run_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "tools" / "cosyvoice_evalscope_perf.py"),
        "--url", f"http://127.0.0.1:{args.port}/v1/audio/speech",
        "--model", "cosyvoice3-0.5b",
        "--voice", "default",
        "--prompt-file", str(args.prompt_file),
        "--outputs-dir", str(run_dir),
        "--number", str(args.requests_per_run),
        "--parallel", "1",
        "--warmup", str(args.warmup),
        "--seed", str(args.seed),
        "--run-name", run_name,
    ]
    with (run_dir / "evalscope_console.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    return json.loads((run_dir / "tts_summary.json").read_text(encoding="utf-8"))


def parse_token_traces(log_path: Path) -> list[list[int]]:
    traces: list[list[int]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if TOKEN_TRACE_PREFIX not in line or " tokens=" not in line:
            continue
        raw_tokens = line.rsplit(" tokens=", 1)[1]
        tokens = ast.literal_eval(raw_tokens)
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise RuntimeError(f"invalid token trace: {line}")
        traces.append(tokens)
    return traces


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def mean_ci_bootstrap(
    values: list[float], resamples: int, seed: int
) -> list[float]:
    generator = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    return [percentile(means, 2.5), percentile(means, 97.5)]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(field: str) -> dict[str, float]:
        values = [float(record[field]) for record in records]
        return {
            "mean": statistics.fmean(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "min": min(values),
            "max": max(values),
        }

    # ``run_wall_seconds`` is repeated on every request record.  Throughput is
    # a run-level quantity, so count each restart-isolated run exactly once.
    run_walls = {
        int(record["run_index"]): float(record["run_wall_seconds"])
        for record in records
    }
    total_wall = sum(run_walls.values())
    total_audio = sum(float(record["audio_duration_seconds"]) for record in records)
    return {
        "sample_count": len(records),
        "client_latency_seconds": stats("client_latency_seconds"),
        "server_inference_ms": stats("inference_latency_ms"),
        "real_time_factor": stats("real_time_factor"),
        "queue_wait_ms": stats("queue_wait_ms"),
        "request_throughput_per_second": len(records) / total_wall,
        "audio_throughput_realtime_x": total_audio / total_wall,
        "audio_seconds_generated": total_audio,
        "quality_retry_count": sum(int(record["quality_retry_count"]) for record in records),
        "run_count": len({int(record["run_index"]) for record in records}),
    }


def pairwise(
    first: str,
    second: str,
    by_variant: dict[str, list[dict[str, Any]]],
    resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first_indexed = {record["pair_key"]: record for record in by_variant[first]}
    second_indexed = {record["pair_key"]: record for record in by_variant[second]}
    if set(first_indexed) != set(second_indexed):
        raise RuntimeError(f"pair keys differ for {first} and {second}")
    rows: list[dict[str, Any]] = []
    for pair_key in sorted(first_indexed):
        baseline = first_indexed[pair_key]
        candidate = second_indexed[pair_key]
        if baseline["input"] != candidate["input"] or baseline["seed"] != candidate["seed"]:
            raise RuntimeError(f"input/seed mismatch for pair {pair_key}")
        rows.append(
            {
                "comparison": f"{first}_to_{second}",
                "pair_key": pair_key,
                "input": baseline["input"],
                "seed": baseline["seed"],
                "first_client_latency_seconds": baseline["client_latency_seconds"],
                "second_client_latency_seconds": candidate["client_latency_seconds"],
                "first_inference_latency_ms": baseline["inference_latency_ms"],
                "second_inference_latency_ms": candidate["inference_latency_ms"],
                "client_relative_percent": (
                    float(candidate["client_latency_seconds"])
                    / float(baseline["client_latency_seconds"]) - 1.0
                ) * 100.0,
                "server_relative_percent": (
                    float(candidate["inference_latency_ms"])
                    / float(baseline["inference_latency_ms"]) - 1.0
                ) * 100.0,
                "second_faster": float(candidate["client_latency_seconds"])
                < float(baseline["client_latency_seconds"]),
            }
        )
    client_relative = [float(row["client_relative_percent"]) for row in rows]
    server_relative = [float(row["server_relative_percent"]) for row in rows]
    first_summary = aggregate(by_variant[first])
    second_summary = aggregate(by_variant[second])
    summary = {
        "first": first,
        "second": second,
        "pair_count": len(rows),
        "client_mean_change_percent": (
            second_summary["client_latency_seconds"]["mean"]
            / first_summary["client_latency_seconds"]["mean"] - 1.0
        ) * 100.0,
        "client_p50_change_percent": (
            second_summary["client_latency_seconds"]["p50"]
            / first_summary["client_latency_seconds"]["p50"] - 1.0
        ) * 100.0,
        "client_p95_change_percent": (
            second_summary["client_latency_seconds"]["p95"]
            / first_summary["client_latency_seconds"]["p95"] - 1.0
        ) * 100.0,
        "server_mean_change_percent": (
            second_summary["server_inference_ms"]["mean"]
            / first_summary["server_inference_ms"]["mean"] - 1.0
        ) * 100.0,
        "throughput_change_percent": (
            second_summary["request_throughput_per_second"]
            / first_summary["request_throughput_per_second"] - 1.0
        ) * 100.0,
        "paired_client_relative_percent": {
            "mean": statistics.fmean(client_relative),
            "median": statistics.median(client_relative),
            "bootstrap_95_ci": mean_ci_bootstrap(
                client_relative, resamples, 20260805
            ),
            "second_faster_count": sum(row["second_faster"] for row in rows),
        },
        "paired_server_relative_percent": {
            "mean": statistics.fmean(server_relative),
            "median": statistics.median(server_relative),
            "bootstrap_95_ci": mean_ci_bootstrap(
                server_relative, resamples, 20260806
            ),
            "second_faster_count": sum(value < 0 for value in server_relative),
        },
    }
    return summary, rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_pairwise_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "comparison", "pair_key", "input", "seed",
        "first_client_latency_seconds", "second_client_latency_seconds",
        "first_inference_latency_ms", "second_inference_latency_ms",
        "client_relative_percent", "server_relative_percent", "second_faster",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.diagnostics_dir = args.diagnostics_dir.resolve()
    args.prompt_file = args.prompt_file.resolve()
    if args.requests_per_run <= 0 or args.warmup < 5:
        raise ValueError("requests-per-run must be positive and warmup must be >= 5")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise FileExistsError(f"results directory is not empty: {args.results_dir}")
    if args.diagnostics_dir.exists() and any(args.diagnostics_dir.iterdir()):
        raise FileExistsError(f"diagnostics directory is not empty: {args.diagnostics_dir}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    api_pid = find_api_server_pid()
    if api_pid is None:
        raise RuntimeError("start one validated api_server.main process before auditing")
    base_environment = read_process_environment(api_pid)
    if not base_environment.get("COSYVOICE_API_KEY", "").strip():
        raise RuntimeError("COSYVOICE_API_KEY is absent from initial API process")
    if base_environment.get("COSYVOICE_LOAD_VLLM", "").lower() not in ("1", "true", "yes"):
        raise RuntimeError("initial API process is not configured for vLLM")

    metadata = {
        "runtime_commit": run_git("rev-parse", "HEAD"),
        "performance_code_commit": "8b242615e22faed849f23ba17ffd09c7ec2aa9c9",
        "initial_api_pid": api_pid,
        "device": base_environment.get("SDAA_VISIBLE_DEVICES"),
        "model_dir": base_environment.get("COSYVOICE_MODEL_DIR"),
        "model_alias": base_environment.get("COSYVOICE_MODEL_ALIAS"),
        "server_command": "python -m api_server.main",
        "benchmark_command": "python tools/cosyvoice_evalscope_perf.py",
        "run_order": list(RUN_ORDER),
        "requests_per_run": args.requests_per_run,
        "warmup_per_run": args.warmup,
        "parallel": 1,
        "seed": args.seed,
        "variant_flags": VARIANTS,
    }
    (args.results_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stop_process(api_pid)
    records: list[dict[str, Any]] = []
    occurrence = defaultdict(int)
    server: subprocess.Popen[bytes] | None = None
    try:
        for run_index, variant in enumerate(RUN_ORDER):
            run_ordinal = occurrence[variant]
            occurrence[variant] += 1
            run_dir = args.diagnostics_dir / (
                f"run_{run_index:02d}_{variant}_ordinal_{run_ordinal}"
            )
            environment = variant_environment(base_environment, variant)
            server = start_server(
                environment, run_dir / "api_server.log", args.port,
                args.startup_timeout,
            )
            run_summary = run_evalscope(
                args, environment, run_dir,
                f"round8-cumulative-{run_index:02d}-{variant}",
            )
            stop_process(server.pid)
            server = None
            measured = run_summary["records"]
            token_traces = parse_token_traces(run_dir / "api_server.log")
            expected_trace_count = args.warmup + len(measured)
            if len(token_traces) != expected_trace_count:
                raise RuntimeError(
                    f"{run_dir}: expected {expected_trace_count} token traces, "
                    f"got {len(token_traces)}"
                )
            for request, tokens in zip(measured, token_traces[args.warmup:]):
                request_index = int(request["request_index"])
                record = {
                    **request,
                    "variant": variant,
                    "run_index": run_index,
                    "run_ordinal": run_ordinal,
                    "pair_key": f"{run_ordinal:02d}:{request_index:03d}",
                    "speech_tokens": tokens,
                    "token_count": len(tokens),
                    "run_wall_seconds": run_summary["wall_seconds"],
                    "run_dir": str(run_dir),
                }
                records.append(record)
            print(
                f"completed run={run_index} variant={variant} "
                f"samples={len(measured)}"
            )
    finally:
        if server is not None:
            stop_process(server.pid)

    by_variant = {
        variant: [record for record in records if record["variant"] == variant]
        for variant in VARIANT_ORDER
    }
    expected_samples = args.requests_per_run * 4
    for variant, variant_records in by_variant.items():
        if len(variant_records) != expected_samples:
            raise RuntimeError(
                f"{variant}: expected {expected_samples} records, got {len(variant_records)}"
            )

    comparisons: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    for first, second in (
        ("baseline_sdaa", "round5_final"),
        ("round5_final", "round8_final"),
        ("baseline_sdaa", "round8_final"),
    ):
        summary, rows = pairwise(
            first, second, by_variant, args.bootstrap_resamples
        )
        comparisons[f"{first}_to_{second}"] = summary
        pair_rows.extend(rows)

    write_jsonl(args.results_dir / "raw_samples.jsonl", records)
    write_pairwise_csv(args.results_dir / "pairwise_results.csv", pair_rows)
    summary = {
        "metadata": metadata,
        "variants": {
            variant: aggregate(variant_records)
            for variant, variant_records in by_variant.items()
        },
        "comparisons": comparisons,
        "raw_samples_path": "raw_samples.jsonl",
        "pairwise_results_path": "pairwise_results.csv",
        "diagnostics_dir": str(args.diagnostics_dir),
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["comparisons"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
