#!/usr/bin/env python3
"""Re-decode frozen speech-token trajectories across a small Flow-seed grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.phase2_5_core import (
    append_jsonl,
    decode_fixed_trajectory,
    load_research_backend,
    read_json,
    read_jsonl,
    write_jsonl,
    write_waveform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--flow-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--published-output-dir",
        type=Path,
        help="Host-visible equivalent of output-dir for saved audio_path values.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    selection = read_jsonl(args.selection.resolve())
    if len(selection) != 10:
        raise ValueError("Pilot is frozen at exactly ten selected trajectories")
    if sorted(args.flow_seeds) != [0, 1, 2, 3] or len(set(args.flow_seeds)) != 4:
        raise ValueError("Pilot is frozen at Flow seeds 0, 1, 2 and 3")
    trajectories = [
        record
        for path in args.trajectories
        for record in read_jsonl(path.resolve())
    ]
    by_generation = {str(record["generation_id"]): record for record in trajectories}
    if len(by_generation) != len(trajectories):
        raise ValueError("Duplicate generation_id across trajectory inputs")
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in selection:
        trajectory = by_generation.get(str(item["generation_id"]))
        if trajectory is None:
            raise KeyError(f"Missing trajectory {item['generation_id']}")
        if trajectory["trajectory_sha256"] != item["trajectory_sha256"]:
            raise ValueError(f"Trajectory hash changed for {item['generation_id']}")
        selected.append((item, trajectory))

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    published_root = (
        args.published_output_dir.resolve()
        if args.published_output_dir
        else output_dir
    )
    results_path = output_dir / "decode_results.jsonl"
    records = read_jsonl(results_path) if results_path.exists() else []
    existing = {str(record["decode_id"]): record for record in records}
    if len(existing) != len(records):
        raise ValueError("Duplicate decode_id in resumable output")

    plans: list[tuple[int, dict[str, Any], dict[str, Any], int, str]] = []
    for selection_index, (item, trajectory) in enumerate(selected):
        for flow_seed in args.flow_seeds:
            decode_id = (
                f"{selection_index:02d}_{item['pilot_group'][:5]}_"
                f"{trajectory['generation_id']}__flow_{flow_seed}"
            )
            plans.append((selection_index, item, trajectory, flow_seed, decode_id))
    planned_ids = {plan[-1] for plan in plans}
    unexpected = set(existing) - planned_ids
    if unexpected:
        raise ValueError(f"Resume output contains {len(unexpected)} unplanned rows")

    pending = [plan for plan in plans if plan[-1] not in existing]
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
        for sequence, item, trajectory, flow_seed, decode_id in pending:
            waveform, metadata = decode_fixed_trajectory(
                backend,
                trajectory=trajectory,
                flow_seed=flow_seed,
                voice_id=config["voice_id"],
                text_frontend=bool(config["text_frontend"]),
                speed=float(config["speed"]),
            )
            stem = decode_id
            audio_metadata = write_waveform(
                audio_dir / f"{stem}.wav", waveform, int(backend.sample_rate)
            )
            published_audio = published_root / "audio" / f"{stem}.wav"
            record = {
                "schema_version": 1,
                "decode_id": decode_id,
                "blind_id": "pilot-" + hashlib.sha256(decode_id.encode()).hexdigest()[:12],
                "pilot_group": item["pilot_group"],
                "selection_is_human_confirmation": False,
                "source_blind_id": item["blind_id"],
                "source_decode_id": item["decode_id"],
                "generation_id": trajectory["generation_id"],
                "sample_id": trajectory["sample_id"],
                "llm_seed": int(trajectory["llm_seed"]),
                "flow_seed": flow_seed,
                "text": trajectory["text"],
                "trajectory_sha256": trajectory["trajectory_sha256"],
                "raw_speech_tokens": item["raw_speech_tokens"],
                "filtered_speech_tokens": item["filtered_speech_tokens"],
                "raw_speech_token_count": item["raw_speech_token_count"],
                "filtered_speech_token_count": item["filtered_speech_token_count"],
                "prompt_speech_token_count": item["prompt_speech_token_count"],
                "prompt_speech_token_count_per_chunk": item[
                    "prompt_speech_token_count_per_chunk"
                ],
                "prompt_speech_token_count_total_across_chunks": item[
                    "prompt_speech_token_count_total_across_chunks"
                ],
                "target_first_token_position": item["target_first_token_position"],
                "target_first_token_position_definition": item[
                    "target_first_token_position_definition"
                ],
                "target_first_10_speech_token_ids": item["target_first_10_speech_token_ids"],
                "target_first_25_speech_token_ids": item["target_first_25_speech_token_ids"],
                "target_first_50_speech_token_ids": item["target_first_50_speech_token_ids"],
                "content_label": item["content_label"],
                "quality_status": (
                    "PASS" if metadata["quality_acceptable"] else "REJECTED"
                ),
                "quality_reason": metadata["quality_reason"],
                "duration": metadata["audio_duration_seconds"],
                "original_wav_sha256": item.get("wav_sha256"),
                **metadata,
                **audio_metadata,
                "audio_path": str(published_audio),
            }
            record["matches_original_wav_at_flow_seed_zero"] = (
                audio_metadata["wav_sha256"] == item.get("wav_sha256")
                if flow_seed == 0
                else None
            )
            records.append(record)
            existing[decode_id] = record
            append_jsonl(results_path, record)
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "total": len(plans),
                        "decode_id": decode_id,
                        "flow_seed": flow_seed,
                    }
                ),
                flush=True,
            )

    records = sorted(records, key=lambda record: str(record["decode_id"]))
    selection_by_generation = {
        str(item["generation_id"]): item for item in selection
    }
    review_manifest = []
    for record in records:
        item = selection_by_generation[str(record["generation_id"])]
        review_manifest.append(
            {
                **record,
                "target_first_token_position": item[
                    "target_first_token_position"
                ],
                "target_first_token_position_definition": item[
                    "target_first_token_position_definition"
                ],
                "prompt_speech_token_count": item[
                    "prompt_speech_token_count"
                ],
                "prompt_speech_token_count_per_chunk": item[
                    "prompt_speech_token_count_per_chunk"
                ],
                "prompt_speech_token_count_total_across_chunks": item[
                    "prompt_speech_token_count_total_across_chunks"
                ],
                "review_sources": ["FIXED_TOKEN_VARIED_FLOW_PILOT"],
            }
        )
    write_jsonl(output_dir / "pilot_review_manifest.jsonl", review_manifest)
    blind_audio_dir = output_dir / "blind_audio"
    blind_audio_dir.mkdir(parents=True, exist_ok=True)
    blind_manifest = []
    for record in review_manifest:
        blind_name = f"{record['blind_id']}.wav"
        source_audio = audio_dir / Path(str(record["audio_path"])).name
        blind_audio = blind_audio_dir / blind_name
        if not blind_audio.exists():
            shutil.copyfile(source_audio, blind_audio)
        blind_manifest.append(
            {
                "schema_version": 1,
                "blind_id": record["blind_id"],
                "audio_path": str(published_root / "blind_audio" / blind_name),
                "text": record["text"],
                "duration": record["duration"],
            }
        )
    write_jsonl(output_dir / "pilot_blind_manifest.jsonl", blind_manifest)
    with (output_dir / "asr_expected_texts.txt").open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda record: str(record["audio_path"])):
            handle.write(str(record["text"]) + "\n")
    summary = {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id"),
        "selected_trajectory_count": len(selection),
        "flow_seeds": args.flow_seeds,
        "observation_count": len(records),
        "unique_trajectory_count": len({record["trajectory_sha256"] for record in records}),
        "unique_flow_noise_count": len({record["flow_noise_sha256"] for record in records}),
        "unique_wav_count": len({record["wav_sha256"] for record in records}),
        "seed_zero_original_wav_match_count": sum(
            record["matches_original_wav_at_flow_seed_zero"] is True for record in records
        ),
        "seed_zero_original_wav_mismatch_count": sum(
            record["matches_original_wav_at_flow_seed_zero"] is False for record in records
        ),
        "quality_status_counts": {
            status: sum(record["quality_status"] == status for record in records)
            for status in ("PASS", "REJECTED")
        },
        "human_articulation_labels_available": False,
        "attribution_status": "PENDING_BLIND_HUMAN_REVIEW",
    }
    (output_dir / "decode_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
