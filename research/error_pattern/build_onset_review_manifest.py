#!/usr/bin/env python3
"""Build the frozen Phase-2.75 onset-articulation review subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.phase2_5_core import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--error-dataset", type=Path, required=True)
    parser.add_argument(
        "--manual-source-error-dataset",
        type=Path,
        nargs="*",
        default=[],
        help="Additional datasets searched only for exact manual nominations.",
    )
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--manual-nominations", type=Path)
    parser.add_argument("--clear-controls", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def flatten_tokens(trajectory: dict[str, Any], field: str) -> list[int]:
    return [
        int(token)
        for chunk in trajectory["chunks"]
        for token in chunk[field]
    ]


def resolve_audio_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    parts = path.parts
    if "research" in parts:
        candidate = repository_root.joinpath(*parts[parts.index("research") :])
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Audio file does not exist: {value}")


def nomination_key(record: dict[str, Any]) -> tuple[str, int]:
    if "sample_id" not in record or "llm_seed" not in record:
        raise ValueError("Each nomination requires sample_id and llm_seed")
    return str(record["sample_id"]), int(record["llm_seed"])


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    rows = read_jsonl(args.error_dataset.resolve())
    manual_source_rows = [
        record
        for path in args.manual_source_error_dataset
        for record in read_jsonl(path.resolve())
    ]
    trajectories = [
        record
        for path in args.trajectories
        for record in read_jsonl(path.resolve())
    ]
    trajectory_by_generation = {
        str(record["generation_id"]): record for record in trajectories
    }
    if len(trajectory_by_generation) != len(trajectories):
        raise ValueError("Duplicate generation_id across trajectory inputs")

    nominations: dict[tuple[str, int], dict[str, Any]] = {}
    if args.manual_nominations:
        for record in read_jsonl(args.manual_nominations.resolve()):
            key = nomination_key(record)
            if key in nominations:
                raise ValueError(f"Duplicate manual nomination: {key}")
            nominations[key] = record

    selected: dict[str, tuple[dict[str, Any], set[str]]] = {}
    for row in rows:
        sources: set[str] = set()
        if str(row.get("content_label")) == "BORDERLINE":
            sources.add("CONTENT_BORDERLINE")
        if (
            str(row.get("quality_status")) == "REJECTED"
            and str(row.get("quality_reason")) == "too_silent"
        ):
            sources.add("ACOUSTIC_TOO_SILENT")
        key = (str(row["sample_id"]), int(row["llm_seed"]))
        if key in nominations:
            sources.add("MANUAL_NOMINATION")
        if sources:
            selected[str(row["decode_id"])] = (row, sources)

    primary_keys = {
        (str(row["sample_id"]), int(row["llm_seed"])) for row in rows
    }
    external_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manual_source_rows:
        key = (str(row["sample_id"]), int(row["llm_seed"]))
        if key in external_by_key:
            raise ValueError(f"Duplicate external manual-source row: {key}")
        external_by_key[key] = row
    for key in sorted(set(nominations) - primary_keys):
        row = external_by_key.get(key)
        if row is not None:
            selected[str(row["decode_id"])] = (row, {"MANUAL_NOMINATION"})

    unresolved = set(nominations) - primary_keys - set(external_by_key)
    if unresolved:
        rendered = ", ".join(f"{sample}/{seed}" for sample, seed in sorted(unresolved))
        raise ValueError(
            "Manual nominations are outside the supplied error dataset: " + rendered
        )

    eligible_clear = [
        row
        for row in rows
        if str(row.get("content_label")) == "GOOD"
        and str(row.get("quality_status")) == "PASS"
        and str(row["decode_id"]) not in selected
    ]
    if len(eligible_clear) < args.clear_controls:
        raise ValueError(
            f"Requested {args.clear_controls} clear controls but only "
            f"{len(eligible_clear)} are eligible"
        )
    rng = random.Random(args.random_seed)
    for row in rng.sample(eligible_clear, args.clear_controls):
        selected[str(row["decode_id"])] = (row, {"FIXED_RANDOM_CLEAR_CONTROL"})

    manifest: list[dict[str, Any]] = []
    for decode_id, (row, sources) in sorted(selected.items()):
        generation_id = str(row["generation_id"])
        trajectory = trajectory_by_generation.get(generation_id)
        if trajectory is None:
            raise KeyError(f"Missing trajectory for {generation_id}")
        raw_tokens = flatten_tokens(trajectory, "raw_speech_tokens")
        filtered_tokens = flatten_tokens(trajectory, "filtered_speech_tokens")
        prompt_tokens_by_chunk = [
            [
                int(token)
                for token in chunk["spans"].get("prompt_speech_token_ids", [])
            ]
            for chunk in trajectory["chunks"]
        ]
        blind_id = "onset-" + hashlib.sha256(
            f"{args.random_seed}:{decode_id}".encode("utf-8")
        ).hexdigest()[:12]
        nomination = nominations.get(
            (str(row["sample_id"]), int(row["llm_seed"])), {}
        )
        manifest.append(
            {
                "schema_version": 1,
                "blind_id": blind_id,
                "decode_id": decode_id,
                "generation_id": generation_id,
                "sample_id": row["sample_id"],
                "llm_seed": int(row["llm_seed"]),
                "flow_seed": int(row["flow_seed"]),
                "audio_path": str(
                    resolve_audio_path(str(row["audio_path"]), repository_root)
                ),
                "text": row["text"],
                "content_label": row["content_label"],
                "quality_status": row["quality_status"],
                "quality_reason": row.get("quality_reason"),
                "raw_speech_tokens": raw_tokens,
                "filtered_speech_tokens": filtered_tokens,
                "raw_speech_token_count": len(raw_tokens),
                "filtered_speech_token_count": len(filtered_tokens),
                "prompt_speech_token_count": len(prompt_tokens_by_chunk[0]),
                "prompt_speech_token_count_per_chunk": [
                    len(tokens) for tokens in prompt_tokens_by_chunk
                ],
                "prompt_speech_token_count_total_across_chunks": sum(
                    len(tokens) for tokens in prompt_tokens_by_chunk
                ),
                "target_first_token_position": int(
                    trajectory["chunks"][0]["spans"]["generation_start"]
                ),
                "target_first_token_position_definition": (
                    "effective LLM input position after text/task/prompt-speech context"
                ),
                "target_first_10_speech_token_ids": filtered_tokens[:10],
                "target_first_25_speech_token_ids": filtered_tokens[:25],
                "target_first_50_speech_token_ids": filtered_tokens[:50],
                "trajectory_hash": row["trajectory_sha256"],
                "trajectory_sha256": row["trajectory_sha256"],
                "duration": float(row["duration"]),
                "wav_sha256": row.get("wav_sha256"),
                "review_sources": sorted(sources),
                "manual_nomination_notes": nomination.get("notes"),
            }
        )

    write_jsonl(args.output.resolve(), manifest)
    summary = {
        "schema_version": 1,
        "output": str(args.output.resolve()),
        "random_seed": args.random_seed,
        "record_count": len(manifest),
        "unique_decode_count": len({record["decode_id"] for record in manifest}),
        "borderline_count": sum(
            "CONTENT_BORDERLINE" in record["review_sources"] for record in manifest
        ),
        "too_silent_count": sum(
            "ACOUSTIC_TOO_SILENT" in record["review_sources"] for record in manifest
        ),
        "manual_nomination_count": sum(
            "MANUAL_NOMINATION" in record["review_sources"] for record in manifest
        ),
        "fixed_random_clear_control_count": sum(
            "FIXED_RANDOM_CLEAR_CONTROL" in record["review_sources"]
            for record in manifest
        ),
    }
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
