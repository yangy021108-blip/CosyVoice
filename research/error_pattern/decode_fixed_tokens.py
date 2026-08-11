#!/usr/bin/env python3
"""Decode saved Phase-2.5 speech-token trajectories with explicit Flow seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.phase2_5_core import (
    append_jsonl,
    decode_fixed_trajectory,
    load_research_backend,
    read_json,
    read_jsonl,
    write_waveform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible partial decode_results.jsonl.",
    )
    return parser.parse_args()


def decode_record(
    backend,
    trajectory,
    *,
    flow_seed: int,
    experiment: str,
    decode_repeat: int,
    config,
    audio_dir: Path,
    sequence: int,
):
    waveform, metadata = decode_fixed_trajectory(
        backend,
        trajectory=trajectory,
        flow_seed=flow_seed,
        voice_id=config["voice_id"],
        text_frontend=bool(config["text_frontend"]),
        speed=float(config["speed"]),
    )
    stem = (
        f"{sequence:04d}_{experiment}_{trajectory['generation_id']}"
        f"__flow_{flow_seed}__decode_{decode_repeat}"
    )
    audio_metadata = write_waveform(
        audio_dir / f"{stem}.wav",
        waveform,
        int(backend.sample_rate),
    )
    return {
        "schema_version": 1,
        "decode_id": stem,
        "experiment": experiment,
        "generation_id": trajectory["generation_id"],
        "sample_id": trajectory["sample_id"],
        "challenge_category": trajectory.get("challenge_category"),
        "tags": trajectory.get("tags", []),
        "text": trajectory["text"],
        "llm_seed": trajectory["llm_seed"],
        "llm_repeat_index": trajectory["repeat_index"],
        "trajectory_sha256": trajectory["trajectory_sha256"],
        "flow_seed": flow_seed,
        "decode_repeat": decode_repeat,
        "generation_status": "SUCCESS",
        "content_label": "UNKNOWN",
        "error_source": "UNKNOWN",
        **metadata,
        **audio_metadata,
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    trajectories = read_jsonl(args.trajectories.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sweep = config.get("sweep", config.get("controls"))
    if not isinstance(sweep, dict):
        raise ValueError("Configuration requires a sweep or controls object")
    fixed_flow_seed = int(sweep["fixed_flow_seed"])
    results_path = output_dir / "decode_results.jsonl"
    records = read_jsonl(results_path) if results_path.exists() else []
    existing_ids = [str(record["decode_id"]) for record in records]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("Cannot resume: duplicate decode_id in existing output")

    planned: list[tuple[dict, int, str, int]] = []
    sequence = 0
    for trajectory in trajectories:
        planned.append((trajectory, fixed_flow_seed, "A", sequence))
        sequence += 1

    acoustic_flow_seeds = [
        int(seed) for seed in sweep.get("acoustic_control_flow_seeds", [])
    ]
    if acoustic_flow_seeds:
        selected = next(
            trajectory
            for trajectory in trajectories
            if trajectory["sample_id"] == sweep["acoustic_control_sample_id"]
            and int(trajectory["llm_seed"])
            == int(sweep["acoustic_control_llm_seed"])
            and int(trajectory["repeat_index"]) == 0
        )
        for flow_seed in acoustic_flow_seeds:
            planned.append((selected, flow_seed, "B", sequence))
            sequence += 1

    planned_ids = {
        (
            f"{sequence_number:04d}_{experiment}_{trajectory['generation_id']}"
            f"__flow_{flow_seed}__decode_0"
        )
        for trajectory, flow_seed, experiment, sequence_number in planned
    }
    unexpected = set(existing_ids) - planned_ids
    if unexpected:
        raise ValueError(f"Cannot resume: {len(unexpected)} decode IDs are unplanned")
    for record in records:
        if not Path(str(record["audio_path"])).is_file():
            raise FileNotFoundError(
                f"Cannot resume: missing WAV for {record['decode_id']}"
            )

    completed_ids = set(existing_ids)
    pending = [
        item
        for item in planned
        if (
            f"{item[3]:04d}_{item[2]}_{item[0]['generation_id']}"
            f"__flow_{item[1]}__decode_0"
        )
        not in completed_ids
    ]
    if pending:
        repository_root = Path(config["repository_root"]).resolve()
        backend, _ = load_research_backend(
            model_dir=Path(config["model_dir"]).resolve(),
            voices_json=(repository_root / config["voices_json"]).resolve(),
            voice_id=config["voice_id"],
            repository_root=repository_root,
            load_vllm=bool(config["load_vllm"]),
            fp16=bool(config["fp16"]),
        )
        completed = len(records)
        total = len(planned)
        for trajectory, flow_seed, experiment, sequence_number in pending:
            record = decode_record(
                backend,
                trajectory,
                flow_seed=flow_seed,
                experiment=experiment,
                decode_repeat=0,
                config=config,
                audio_dir=audio_dir,
                sequence=sequence_number,
            )
            records.append(record)
            append_jsonl(results_path, record)
            completed += 1
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": total,
                        "decode_id": record["decode_id"],
                        "quality_acceptable": record["quality_acceptable"],
                    }
                ),
                flush=True,
            )

    successful_sorted = sorted(records, key=lambda record: record["audio_path"])
    with (output_dir / "asr_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for record in successful_sorted:
            handle.write(
                json.dumps(
                    {
                        "decode_id": record["decode_id"],
                        "audio_path": record["audio_path"],
                        "expected_text": record["text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (output_dir / "asr_expected_texts.txt").write_text(
        "".join(f"{record['text']}\n" for record in successful_sorted),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id"),
        "completed": len(records),
        "total": len(planned),
        "quality_pass_count": sum(
            bool(record["quality_acceptable"]) for record in records
        ),
        "quality_rejected_count": sum(
            not bool(record["quality_acceptable"]) for record in records
        ),
        "audio_duration_seconds": sum(
            float(record["audio_duration_seconds"]) for record in records
        ),
        "status": "complete" if len(records) == len(planned) else "partial",
    }
    (output_dir / "decode_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
