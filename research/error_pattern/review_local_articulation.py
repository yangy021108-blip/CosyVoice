#!/usr/bin/env python3
"""Blind, resumable human review for local articulation degradation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.phase2_5_core import append_jsonl, read_jsonl

LABELS = {
    "c": "CLEAR",
    "m": "MILD_DEGRADATION",
    "s": "SEVERE_DEGRADATION",
    "u": "UNKNOWN",
}
ERROR_TYPES = {
    "1": "ONSET_MERGE",
    "2": "ONSET_BLUR",
    "3": "INITIAL_DELETION",
    "4": "INITIAL_SUBSTITUTION",
    "5": "INITIAL_REPETITION",
    "6": "EXCESSIVE_INITIAL_SILENCE",
    "7": "PROMPT_TARGET_BOUNDARY_ARTIFACT",
    "8": "OTHER",
    "0": "NONE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=20260812)
    parser.add_argument(
        "--player",
        help=(
            "Player command prefix. The WAV path is appended; ffplay is used "
            "when available."
        ),
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Validate and print blind IDs without playing or writing reviews.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Use WAVs with the same filenames from this local directory.",
    )
    return parser.parse_args()


def stable_order(record: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record['blind_id']}".encode()).hexdigest()


def player_prefix(value: str | None) -> list[str]:
    if value:
        return shlex.split(value)
    ffplay = shutil.which("ffplay")
    if ffplay:
        return [ffplay, "-nodisp", "-autoexit", "-loglevel", "error"]
    raise RuntimeError("No ffplay found; provide --player explicitly")


def play(prefix: list[str], path: Path, seconds: float | None) -> None:
    command = list(prefix)
    if seconds is not None and Path(prefix[0]).name.lower().startswith("ffplay"):
        command.extend(["-t", str(seconds)])
    command.append(str(path))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Player exited with status {completed.returncode}")


def optional_float(prompt: str) -> float | None:
    value = input(prompt).strip()
    if not value:
        return None
    parsed = float(value)
    if parsed < 0:
        raise ValueError("Timestamp cannot be negative")
    return parsed


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.manifest.resolve())
    if len({record["blind_id"] for record in records}) != len(records):
        raise ValueError("Manifest contains duplicate blind_id values")
    ordered = sorted(records, key=lambda record: stable_order(record, args.shuffle_seed))
    existing = read_jsonl(args.output.resolve()) if args.output.exists() else []
    reviewed = {str(record["blind_id"]) for record in existing}
    if len(reviewed) != len(existing):
        raise ValueError("Review output contains duplicate blind_id values")

    pending = [record for record in ordered if str(record["blind_id"]) not in reviewed]
    print(
        json.dumps(
            {
                "manifest_count": len(records),
                "already_reviewed": len(reviewed),
                "pending": len(pending),
                "shuffle_seed": args.shuffle_seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.list_only:
        for record in pending:
            print(record["blind_id"])
        return

    prefix = player_prefix(args.player)
    for index, record in enumerate(pending, start=1):
        source_path = Path(str(record["audio_path"]))
        path = (
            args.audio_root.resolve() / source_path.name
            if args.audio_root
            else source_path
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"\n[{index}/{len(pending)}] blind_id={record['blind_id']}")
        print(f"duration={float(record['duration']):.3f}s")
        print(f"reference={record['text']}")
        print("Commands: p=full, 05=first 0.5s, 1=first 1s, 2=first 2s")
        while True:
            command = input("play or label [p/05/1/2/c/m/s/u/q]: ").strip().lower()
            if command == "q":
                return
            if command in {"p", "05", "1", "2"}:
                seconds = {"p": None, "05": 0.5, "1": 1.0, "2": 2.0}[command]
                play(prefix, path, seconds)
                continue
            if command not in LABELS:
                print("Unknown command")
                continue

            label = LABELS[command]
            if label == "CLEAR":
                error_type = "NONE"
                start = end = None
            else:
                print("Error types:")
                for key, value in ERROR_TYPES.items():
                    print(f"  {key}: {value}")
                type_key = input("error type: ").strip()
                if type_key not in ERROR_TYPES:
                    print("Invalid error type; review not saved")
                    continue
                error_type = ERROR_TYPES[type_key]
                start = optional_float("error start seconds (blank=unknown): ")
                end = optional_float("error end seconds (blank=unknown): ")
                if start is not None and end is not None and end < start:
                    print("End precedes start; review not saved")
                    continue
            notes = input("notes (optional): ").strip()
            review = {
                "schema_version": 1,
                "blind_id": record["blind_id"],
                "reviewer": args.reviewer,
                "articulation_label": label,
                "review_decision": "UNCERTAIN" if label == "UNKNOWN" else label,
                "articulation_error_type": error_type,
                "error_start_seconds": start,
                "error_end_seconds": end,
                "notes": notes or None,
            }
            append_jsonl(args.output.resolve(), review)
            print("saved")
            break


if __name__ == "__main__":
    main()
