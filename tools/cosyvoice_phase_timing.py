#!/usr/bin/env python3
"""Measure LLM, Flow, HiFT, and residual request time without a profiler trace."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
MATCHA_PATH = ROOT / "third_party" / "Matcha-TTS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MATCHA_PATH) not in sys.path:
    sys.path.append(str(MATCHA_PATH))

from api_server.engine import CosyVoiceEngine
from api_server.voice_store import VoiceStore
from tools.cosyvoice_perfetto_profile import _request, _run_one, _settings


class PhaseCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[str, float]] = []

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            self.events.append((name, seconds))

    def mark(self) -> int:
        with self._lock:
            return len(self.events)

    def since(self, mark: int) -> dict[str, float]:
        values: dict[str, float] = defaultdict(float)
        with self._lock:
            for name, seconds in self.events[mark:]:
                values[name] += seconds
        return dict(values)


def wrap_method(obj: Any, name: str, label: str, collector: PhaseCollector) -> None:
    original: Callable[..., Any] = getattr(obj, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original(*args, **kwargs)
        if inspect.isgenerator(result):
            def timed_generator():
                consume_started = time.perf_counter()
                try:
                    yield from result
                finally:
                    collector.add(label, time.perf_counter() - consume_started)
            return timed_generator()
        try:
            return result
        finally:
            collector.add(label, time.perf_counter() - started)

    setattr(obj, name, wrapped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/model"))
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = [
        line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise RuntimeError("prompt file is empty")
    settings = _settings(args.model_dir)
    voice_store = VoiceStore.from_json(settings.voices_file, settings.root_dir)
    engine = CosyVoiceEngine(settings, voice_store)
    engine.load()
    voice = voice_store.get("default")
    if voice is None:
        raise RuntimeError("default voice is not configured")

    collector = PhaseCollector()
    model = engine.backend.model
    wrap_method(model.llm, "inference", "llm_seconds", collector)
    wrap_method(model.flow, "inference", "flow_seconds", collector)
    wrap_method(model.hift, "inference", "hift_seconds", collector)

    for index in range(args.warmup):
        _run_one(
            engine, voice,
            _request(prompts[index % len(prompts)], args.seed + index), None,
        )

    records: list[dict[str, Any]] = []
    for index in range(args.active):
        mark = collector.mark()
        record = _run_one(
            engine, voice,
            _request(prompts[index % len(prompts)], args.seed + index), None,
        )
        phases = collector.since(mark)
        total = float(record["inference_seconds"])
        labeled = sum(phases.values())
        records.append({
            **record,
            "llm_seconds": phases.get("llm_seconds", 0.0),
            "flow_seconds": phases.get("flow_seconds", 0.0),
            "hift_seconds": phases.get("hift_seconds", 0.0),
            "other_api_seconds": total - labeled,
        })

    def mean(field: str) -> float:
        return statistics.fmean(float(record[field]) for record in records)

    summary = {
        "warmup": args.warmup,
        "active": args.active,
        "mean_seconds": {
            "end_to_end": mean("inference_seconds"),
            "llm": mean("llm_seconds"),
            "flow": mean("flow_seconds"),
            "hift": mean("hift_seconds"),
            "other_api": mean("other_api_seconds"),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["mean_seconds"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
