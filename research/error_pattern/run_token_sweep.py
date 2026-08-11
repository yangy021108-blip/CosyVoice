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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(samples_path)
    controls = config["controls"]
    selected_sample_ids = set(controls["sample_ids"])
    selected_samples = [
        sample for sample in samples if sample["sample_id"] in selected_sample_ids
    ]
    if {sample["sample_id"] for sample in selected_samples} != selected_sample_ids:
        raise ValueError("Control configuration references an unknown sample ID")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "config": config,
        "config_sha256": sha256_file(args.config.resolve()),
        "samples_sha256": sha256_file(samples_path),
        "voices_sha256": sha256_file(voices_json),
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
    (output_dir / "token_sweep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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
    (output_dir / "token_sweep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results_path = output_dir / "token_trajectories.jsonl"
    llm_seeds = [int(seed) for seed in controls["llm_seeds"]]
    repeat_sample = controls["repeat_sample_id"]
    repeat_seed = int(controls["repeat_llm_seed"])
    total = len(selected_samples) * len(llm_seeds) + 1
    completed = 0
    for sample in selected_samples:
        for llm_seed in llm_seeds:
            repeat_count = 2 if (
                sample["sample_id"] == repeat_sample and llm_seed == repeat_seed
            ) else 1
            for repeat_index in range(repeat_count):
                record = generate_trajectory(
                    backend,
                    sample=sample,
                    voice_id=config["voice_id"],
                    llm_seed=llm_seed,
                    repeat_index=repeat_index,
                    text_frontend=bool(config["text_frontend"]),
                    top_k=int(config["sampling"]["top_k"]),
                    logprobs=config["sampling"].get("logprobs"),
                    min_token_text_ratio=float(
                        config["sampling"]["min_token_text_ratio"]
                    ),
                    max_token_text_ratio=float(
                        config["sampling"]["max_token_text_ratio"]
                    ),
                )
                record["planned_fixed_flow_seed"] = int(
                    controls["fixed_flow_seed"]
                )
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


if __name__ == "__main__":
    main()
