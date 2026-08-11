#!/usr/bin/env python3
"""Run Phase-2.5 text-to-speech-token sweeps without acoustic decoding."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.phase2_5_core import (
    append_jsonl,
    generate_trajectory,
    load_research_backend,
    read_json,
    read_jsonl,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible partial sweep from token_trajectories.jsonl.",
    )
    return parser.parse_args()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    repository_root = Path(config["repository_root"]).resolve()
    samples_path = (repository_root / config["samples"]).resolve()
    voices_json = (repository_root / config["voices_json"]).resolve()
    model_dir = Path(config["model_dir"]).resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(samples_path)
    sweep = config.get("sweep", config.get("controls"))
    if not isinstance(sweep, dict):
        raise ValueError("Configuration requires a sweep or controls object")
    selected_sample_ids = set(sweep.get("sample_ids", []))
    if not selected_sample_ids:
        selected_sample_ids = {str(sample["sample_id"]) for sample in samples}
    selected_samples = [
        sample for sample in samples if sample["sample_id"] in selected_sample_ids
    ]
    if {sample["sample_id"] for sample in selected_samples} != selected_sample_ids:
        raise ValueError("Control configuration references an unknown sample ID")

    expected_sample_count = sweep.get("expected_sample_count")
    if expected_sample_count is not None and len(selected_samples) != int(
        expected_sample_count
    ):
        raise ValueError(
            f"Expected {expected_sample_count} samples; found {len(selected_samples)}"
        )
    if "llm_seeds" in sweep:
        llm_seeds = [int(seed) for seed in sweep["llm_seeds"]]
    else:
        seed_start = int(sweep["llm_seed_start"])
        seed_end = int(sweep["llm_seed_end"])
        if seed_end < seed_start:
            raise ValueError("llm_seed_end must be greater than or equal to start")
        llm_seeds = list(range(seed_start, seed_end + 1))
    if len(llm_seeds) != len(set(llm_seeds)):
        raise ValueError("LLM seeds must be unique")
    expected_seed_count = sweep.get("expected_seed_count")
    if expected_seed_count is not None and len(llm_seeds) != int(
        expected_seed_count
    ):
        raise ValueError(
            f"Expected {expected_seed_count} seeds; found {len(llm_seeds)}"
        )

    config_sha256 = sha256_file(args.config.resolve())
    samples_sha256 = sha256_file(samples_path)
    voices_sha256 = sha256_file(voices_json)
    manifest_path = output_dir / "token_sweep_manifest.json"
    new_manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "config": config,
        "config_sha256": config_sha256,
        "samples_sha256": samples_sha256,
        "voices_sha256": voices_sha256,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": command_output(["git", "status", "--porcelain"]),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    if args.resume and manifest_path.exists():
        manifest = read_json(manifest_path)
        for field, expected in (
            ("config_sha256", config_sha256),
            ("samples_sha256", samples_sha256),
            ("voices_sha256", voices_sha256),
        ):
            if manifest.get(field) != expected:
                raise ValueError(f"Cannot resume: {field} changed")
        manifest.setdefault("resume_events_unix", []).append(time.time())
    else:
        manifest = new_manifest
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repeat_sample = sweep.get("repeat_sample_id")
    repeat_seed_value = sweep.get("repeat_llm_seed")
    repeat_seed = int(repeat_seed_value) if repeat_seed_value is not None else None
    planned: list[tuple[dict[str, Any], int, int]] = []
    for sample in selected_samples:
        for llm_seed in llm_seeds:
            repeat_count = 2 if (
                repeat_sample is not None
                and repeat_seed is not None
                and sample["sample_id"] == repeat_sample
                and llm_seed == repeat_seed
            ) else 1
            for repeat_index in range(repeat_count):
                planned.append((sample, llm_seed, repeat_index))

    results_path = output_dir / "token_trajectories.jsonl"
    existing = read_jsonl(results_path) if results_path.exists() else []
    existing_ids = [str(record["generation_id"]) for record in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("Cannot resume: duplicate generation_id in existing output")
    planned_ids = {
        f"{sample['sample_id']}__llm_{llm_seed}__repeat_{repeat_index}"
        for sample, llm_seed, repeat_index in planned
    }
    unexpected_ids = set(existing_ids) - planned_ids
    if unexpected_ids:
        raise ValueError(
            f"Cannot resume: {len(unexpected_ids)} existing generation IDs are unplanned"
        )
    completed_ids = set(existing_ids)
    total = len(planned)
    if len(completed_ids) == total:
        summary = {
            "schema_version": 1,
            "experiment_id": config.get("experiment_id"),
            "completed": total,
            "total": total,
            "sample_count": len(selected_samples),
            "llm_seed_count": len(llm_seeds),
            "fixed_flow_seed": int(sweep["fixed_flow_seed"]),
            "status": "complete",
        }
        (output_dir / "token_sweep_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"completed": total, "total": total, "status": "already_complete"}))
        return

    backend, voice = load_research_backend(
        model_dir=model_dir,
        voices_json=voices_json,
        voice_id=config["voice_id"],
        repository_root=repository_root,
        load_vllm=bool(config["load_vllm"]),
        fp16=bool(config["fp16"]),
    )
    manifest["runtime_text_frontend"] = backend.frontend.text_frontend or "none"
    manifest["sample_rate"] = int(backend.sample_rate)
    manifest["voice_prompt_audio_sha256"] = sha256_file(voice.prompt_audio)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    completed = len(completed_ids)
    for sample, llm_seed, repeat_index in planned:
        generation_id = (
            f"{sample['sample_id']}__llm_{llm_seed}__repeat_{repeat_index}"
        )
        if generation_id in completed_ids:
            continue
        record = generate_trajectory(
            backend,
            sample=sample,
            voice_id=config["voice_id"],
            llm_seed=llm_seed,
            repeat_index=repeat_index,
            text_frontend=bool(config["text_frontend"]),
            top_k=int(config["sampling"]["top_k"]),
            logprobs=config["sampling"].get("logprobs"),
            min_token_text_ratio=float(config["sampling"]["min_token_text_ratio"]),
            max_token_text_ratio=float(config["sampling"]["max_token_text_ratio"]),
        )
        record["planned_fixed_flow_seed"] = int(sweep["fixed_flow_seed"])
        append_jsonl(results_path, record)
        completed += 1
        print(
            json.dumps(
                {
                    "completed": completed,
                    "total": total,
                    "generation_id": record["generation_id"],
                    "trajectory_sha256": record["trajectory_sha256"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id"),
        "completed": completed,
        "total": total,
        "sample_count": len(selected_samples),
        "llm_seed_count": len(llm_seeds),
        "fixed_flow_seed": int(sweep["fixed_flow_seed"]),
        "status": "complete" if completed == total else "partial",
    }
    (output_dir / "token_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
