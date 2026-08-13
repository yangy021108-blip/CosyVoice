#!/usr/bin/env python3
"""Summarize Phase-2.75 pilot proxies and human fixed-token recurrence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

from research.error_pattern.phase2_5_core import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def is_degraded(review: dict[str, Any] | None) -> bool:
    return bool(
        review
        and review.get("articulation_label")
        in {"MILD_DEGRADATION", "SEVERE_DEGRADATION"}
    )


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.features.resolve())
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_trajectory[str(record["generation_id"])].append(record)

    trajectories: list[dict[str, Any]] = []
    for generation_id, rows in sorted(by_trajectory.items()):
        rows.sort(key=lambda record: int(record["flow_seed"]))
        reviews = [record.get("human_review") for record in rows]
        reviewed = [review for review in reviews if review is not None]
        degraded_count = sum(is_degraded(review) for review in reviews)
        flow_zero = next(record for record in rows if int(record["flow_seed"]) == 0)
        flow_zero_degraded = is_degraded(flow_zero.get("human_review"))
        if len(reviewed) != len(rows):
            attribution = "PENDING_BLIND_HUMAN_REVIEW"
        elif flow_zero_degraded and degraded_count >= 3:
            attribution = "TOKEN_TRAJECTORY_ASSOCIATED_CANDIDATE"
        elif flow_zero_degraded and degraded_count == 1:
            attribution = "NON_REPRODUCIBLE_CANDIDATE"
        elif 0 < degraded_count < len(rows):
            attribution = "ACOUSTIC_STOCHASTICITY_CANDIDATE"
        elif degraded_count == len(rows):
            attribution = "PERSISTENT_ACROSS_FLOW_SEEDS"
        else:
            attribution = "NO_DEGRADATION_REVIEWED"
        trajectories.append(
            {
                "generation_id": generation_id,
                "sample_id": rows[0]["sample_id"],
                "pilot_group": rows[0]["pilot_group"],
                "flow_seeds": [int(record["flow_seed"]) for record in rows],
                "quality_status_counts": dict(
                    Counter(record["quality_status"] for record in rows)
                ),
                "unique_asr_transcript_count": len(
                    {record.get("asr_text") for record in rows}
                ),
                "prefix_cer_by_flow_seed": {
                    str(length): [
                        record["prefix_transcript_metrics"][str(length)]["cer"]
                        for record in rows
                    ]
                    for length in (3, 5, 10)
                },
                "human_review_count": len(reviewed),
                "articulation_label_counts": dict(
                    Counter(
                        review["articulation_label"] for review in reviewed
                    )
                ),
                "degraded_flow_seed_count": degraded_count,
                "fixed_token_attribution": attribution,
            }
        )

    suspected = [
        trajectory
        for trajectory in trajectories
        if trajectory["pilot_group"] == "SUSPECTED_DEGRADATION_CANDIDATE"
    ]
    clear = [
        trajectory
        for trajectory in trajectories
        if trajectory["pilot_group"] == "CLEAR_CONTROL_CANDIDATE"
    ]
    all_reviewed = all(
        trajectory["human_review_count"] == 4 for trajectory in trajectories
    )
    suspected_degraded = sum(
        trajectory["degraded_flow_seed_count"] for trajectory in suspected
    )
    clear_degraded = sum(
        trajectory["degraded_flow_seed_count"] for trajectory in clear
    )
    summary = {
        "schema_version": 1,
        "observation_count": len(records),
        "trajectory_count": len(trajectories),
        "flow_seeds_per_trajectory": 4,
        "candidate_trajectory_count": len(suspected),
        "clear_control_trajectory_count": len(clear),
        "all_blind_reviews_complete": all_reviewed,
        "candidate_degradation_rate": (
            suspected_degraded / (4 * len(suspected)) if all_reviewed else None
        ),
        "clear_control_degradation_rate": (
            clear_degraded / (4 * len(clear)) if all_reviewed else None
        ),
        "phase2_75_gate": {
            "minimum_confirmed_degraded_observations": 16,
            "minimum_matched_clear_controls": 16,
            "minimum_independent_text_groups": 4,
            "pilot_can_satisfy_gate": False,
            "ready": False,
        },
        "attribution_status": (
            "PILOT_REVIEW_COMPLETE_GATE_STILL_REQUIRES_SCALE"
            if all_reviewed
            else "PENDING_BLIND_HUMAN_REVIEW"
        ),
        "asr_warning": (
            "Prefix ASR variation is a screening proxy only and cannot assign "
            "an articulation label."
        ),
        "trajectories": trajectories,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    brief = {
        key: value for key, value in summary.items() if key != "trajectories"
    }
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
