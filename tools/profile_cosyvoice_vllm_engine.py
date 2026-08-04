#!/usr/bin/env python3
"""Capture a compressed profiler trace from the vLLM EngineCore process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATCHA_PATH = ROOT / "third_party" / "Matcha-TTS"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MATCHA_PATH) not in sys.path:
    sys.path.append(str(MATCHA_PATH))

from api_server.engine import CosyVoiceEngine
from api_server.voice_store import VoiceStore
from tools.cosyvoice_perfetto_profile import _request, _run_one, _settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/model"))
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--tag", default="cosyvoice_vllm_engine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    prompts = [
        line.strip()
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise RuntimeError("prompt file is empty")

    os.environ["COSYVOICE_VLLM_PROFILER_DIR"] = str(args.output_dir)
    settings = _settings(args.model_dir)
    voice_store = VoiceStore.from_json(
        settings.voices_file,
        settings.root_dir,
    )
    engine = CosyVoiceEngine(settings, voice_store)
    engine.load()
    voice = voice_store.get("default")
    if voice is None:
        raise RuntimeError("default voice is not configured")

    for index in range(args.warmup):
        prompt_index = index % len(prompts)
        _run_one(
            engine,
            voice,
            _request(prompts[prompt_index], args.seed + prompt_index),
            None,
        )

    vllm_engine = engine.backend.model.llm.vllm
    records = []
    vllm_engine.start_profile(args.tag)
    try:
        for index in range(args.active):
            prompt_index = index % len(prompts)
            records.append(
                _run_one(
                    engine,
                    voice,
                    _request(
                        prompts[prompt_index],
                        args.seed + prompt_index,
                    ),
                    None,
                )
            )
    finally:
        vllm_engine.stop_profile()

    traces = sorted(args.output_dir.glob("*.json.gz"))
    if not traces:
        raise RuntimeError("vLLM EngineCore did not produce a .json.gz trace")
    summary = {
        "tag": args.tag,
        "warmup": args.warmup,
        "active": args.active,
        "records": records,
        "traces": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in traces
        ],
    }
    (args.output_dir / f"{args.tag}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
