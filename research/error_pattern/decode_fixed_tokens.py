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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    repository_root = Path(config["repository_root"]).resolve()
    backend, _ = load_research_backend(
        model_dir=Path(config["model_dir"]).resolve(),
        voices_json=(repository_root / config["voices_json"]).resolve(),
        voice_id=config["voice_id"],
        repository_root=repository_root,
        load_vllm=bool(config["load_vllm"]),
        fp16=bool(config["fp16"]),
    )
    controls = config["controls"]
    fixed_flow_seed = int(controls["fixed_flow_seed"])
    results_path = output_dir / "decode_results.jsonl"
    records = []
    sequence = 0

    for trajectory in trajectories:
        record = decode_record(
            backend,
            trajectory,
            flow_seed=fixed_flow_seed,
            experiment="A",
            decode_repeat=0,
            config=config,
            audio_dir=audio_dir,
            sequence=sequence,
        )
        sequence += 1
        records.append(record)
        append_jsonl(results_path, record)
        print(json.dumps({"decode_id": record["decode_id"]}), flush=True)

    selected = next(
        trajectory
        for trajectory in trajectories
        if trajectory["sample_id"] == controls["acoustic_control_sample_id"]
        and int(trajectory["llm_seed"]) == int(controls["acoustic_control_llm_seed"])
        and int(trajectory["repeat_index"]) == 0
    )
    for flow_seed in controls["acoustic_control_flow_seeds"]:
        record = decode_record(
            backend,
            selected,
            flow_seed=int(flow_seed),
            experiment="B",
            decode_repeat=0,
            config=config,
            audio_dir=audio_dir,
            sequence=sequence,
        )
        sequence += 1
        records.append(record)
        append_jsonl(results_path, record)
        print(json.dumps({"decode_id": record["decode_id"]}), flush=True)

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


if __name__ == "__main__":
    main()
