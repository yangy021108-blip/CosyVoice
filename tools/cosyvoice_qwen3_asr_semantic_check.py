import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any


# This is an offline inference script. Set these before vLLM/HF modules import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:
    raise SystemExit(
        "Failed to import vLLM. Run this inside the tecovllm Docker environment "
        "after installing 3rdparty/vllm and tecovllm."
    ) from exc


AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
ASR_TEXT_TAG = "<asr_text>"
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac"}

DEFAULT_MODEL = "/tecogpfs/models/Qwen/Qwen3-ASR-0___6B"
DEFAULT_AUDIO = (
    "/workspace/CosyVoice/cosyvoice_api_outputs/perf/"
    "cosyvoice_sdaa_optimization_20260726/optimized/evalscope_c1/audio"
)


def json_arg(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("JSON value must be an object.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Qwen3-ASR inference with tecovllm/vLLM."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local Qwen3-ASR model path.",
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        default=[DEFAULT_AUDIO],
        help="One or more .wav/.flac files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--expected-text-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file; non-empty lines are repeated to match audio count.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional extra instruction appended after the audio placeholder.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint, for example zh, en, ja, or None for auto.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of audio files to run.",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print raw model output before ASR tag cleanup.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--additional-config", type=json_arg, default=None)
    parser.add_argument("--compilation-config", type=json_arg, default=None)
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Optional path to save one JSON object per audio.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for aggregate semantic-regression metrics.",
    )
    return parser.parse_args()


def normalize_audio_arg(audio_arg: str) -> Path:
    expanded = os.path.expandvars(audio_arg)
    path = Path(expanded).expanduser()

    return path


def resolve_audio_paths(audio_args: list[str], limit: int | None) -> list[str]:
    audio_paths: list[Path] = []

    for audio_arg in audio_args:
        expanded = normalize_audio_arg(audio_arg)
        if expanded.is_dir():
            audio_paths.extend(
                sorted(
                    path
                    for path in expanded.rglob("*")
                    if path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
                )
            )
            continue

        matches = [
            Path(path)
            for path in sorted(glob.glob(str(expanded), recursive=True))
            if Path(path).is_file()
        ]
        if matches:
            audio_paths.extend(matches)
            continue

        audio_paths.append(expanded)

    resolved: list[str] = []
    seen: set[str] = set()
    for audio_path in audio_paths:
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Only .wav/.flac audio is supported: {audio_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        path_str = str(audio_path)
        if path_str not in seen:
            seen.add(path_str)
            resolved.append(path_str)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than 0.")
        resolved = resolved[:limit]

    if not resolved:
        raise FileNotFoundError(
            "No .wav/.flac audio files were found. Pass an audio file or a "
            "directory that contains .wav/.flac files."
        )
    return resolved


def load_audio(path: str) -> tuple[Any, int]:
    import librosa

    waveform, sample_rate = librosa.load(path, sr=None)
    return waveform, sample_rate


def build_prompt(extra_text: str, language: str | None) -> str:
    user_content = f"{AUDIO_PLACEHOLDER}\n"
    if extra_text:
        user_content += f"{extra_text.strip()}\n"

    assistant_prefix = ""
    if language:
        assistant_prefix = f"language {language}{ASR_TEXT_TAG}"

    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )


def build_inputs(
    audio_paths: list[str], prompt: str, language: str | None
) -> list[dict[str, Any]]:
    text_prompt = build_prompt(prompt, language)
    return [
        {
            "prompt": text_prompt,
            "multi_modal_data": {"audio": [load_audio(audio_path)]},
        }
        for audio_path in audio_paths
    ]


def clean_asr_output(raw_text: str) -> str:
    text = raw_text.strip()
    if ASR_TEXT_TAG in text:
        text = text.split(ASR_TEXT_TAG, 1)[1]

    for stop_marker in ("<|im_end|>", "<|endoftext|>"):
        if stop_marker in text:
            text = text.split(stop_marker, 1)[0]
    return text.strip()


def normalize_text(text: str) -> str:
    return "".join(
        re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def load_expected_texts(path: Path | None, count: int) -> list[str] | None:
    if path is None:
        return None
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No expected text found in {path}")
    return [prompts[index % len(prompts)] for index in range(count)]


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser()
    if args.model.startswith("/") and not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model}")

    audio_paths = resolve_audio_paths(args.audio, args.limit)
    expected_texts = load_expected_texts(
        args.expected_text_file, len(audio_paths)
    )

    additional_config = args.additional_config or {}
    additional_config.update({"enable_chunked_prefill": False})

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs or 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 1},
        "mm_processor_cache_gb": 0,
        "additional_config": additional_config,
        "trust_remote_code": True,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.compilation_config is not None:
        llm_kwargs["compilation_config"] = args.compilation_config

    print(f"Model: {args.model}")
    print(f"Audio files: {len(audio_paths)}")

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )

    inputs = build_inputs(audio_paths, args.prompt, args.language)
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    records = []
    for index, (audio_path, output) in enumerate(zip(audio_paths, outputs)):
        raw_text = output.outputs[0].text.strip()
        text = clean_asr_output(raw_text)
        print(f"===== {audio_path} =====")
        if args.print_raw:
            print(f"RAW: {raw_text}")
        print(text)
        record = {"audio": audio_path, "text": text, "raw_text": raw_text}
        if expected_texts is not None:
            expected = expected_texts[index]
            normalized_expected = normalize_text(expected)
            normalized_text = normalize_text(text)
            distance = edit_distance(normalized_expected, normalized_text)
            record.update(
                {
                    "expected_text": expected,
                    "normalized_expected": normalized_expected,
                    "normalized_text": normalized_text,
                    "character_errors": distance,
                    "character_count": len(normalized_expected),
                    "character_error_rate": distance
                    / max(len(normalized_expected), 1),
                }
            )
        records.append(record)

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if expected_texts is not None:
        total_errors = sum(record["character_errors"] for record in records)
        total_characters = sum(record["character_count"] for record in records)
        summary = {
            "audio_count": len(records),
            "nonempty_transcript_count": sum(
                bool(record["normalized_text"]) for record in records
            ),
            "total_character_errors": total_errors,
            "total_characters": total_characters,
            "character_error_rate": total_errors / max(total_characters, 1),
            "records": records,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
