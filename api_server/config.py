"""Environment-backed service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_int(name: str, default: int, minimum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """All service settings, with conservative single-GPU defaults."""

    root_dir: Path
    model_alias: str
    model_dir: Path
    voices_file: Path
    api_key: str | None
    host: str
    port: int
    max_text_characters: int
    max_concurrency: int
    max_queue_size: int
    request_timeout_seconds: int
    fp16: bool
    load_vllm: bool

    @classmethod
    def from_env(cls) -> "Settings":
        root_dir = Path(__file__).resolve().parents[1]
        model_dir = Path(
            os.getenv(
                "COSYVOICE_MODEL_DIR",
                str(root_dir / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
            )
        ).expanduser()
        voices_file = Path(
            os.getenv(
                "COSYVOICE_VOICES_FILE",
                str(root_dir / "api_server" / "voices.json"),
            )
        ).expanduser()
        api_key = os.getenv("COSYVOICE_API_KEY")

        return cls(
            root_dir=root_dir,
            model_alias=os.getenv("COSYVOICE_MODEL_ALIAS", "cosyvoice3-0.5b"),
            model_dir=model_dir,
            voices_file=voices_file,
            api_key=api_key if api_key else None,
            host=os.getenv("COSYVOICE_HOST", "0.0.0.0"),
            port=_env_int("COSYVOICE_PORT", 8000, 1),
            max_text_characters=_env_int(
                "COSYVOICE_MAX_TEXT_CHARACTERS", 2000, 1
            ),
            max_concurrency=_env_int("COSYVOICE_MAX_CONCURRENCY", 1, 1),
            max_queue_size=_env_int("COSYVOICE_MAX_QUEUE_SIZE", 16, 0),
            request_timeout_seconds=_env_int(
                "COSYVOICE_REQUEST_TIMEOUT_SECONDS", 600, 1
            ),
            fp16=_env_bool("COSYVOICE_FP16", False),
            load_vllm=_env_bool("COSYVOICE_LOAD_VLLM", False),
        )
