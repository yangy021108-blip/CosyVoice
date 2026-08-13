"""Environment-backed service configuration."""

from __future__ import annotations

import os
from ipaddress import ip_address
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


def _env_int_range(
    name: str, default: int, minimum: int, maximum: int
) -> int:
    value = _env_int(name, default, minimum)
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


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
    allow_unauthenticated: bool = False
    default_seed: int = 2
    quality_check_enabled: bool = True
    quality_max_retries: int = 2
    stream_queue_size: int = 4

    def __post_init__(self) -> None:
        if not 0 <= self.default_seed <= 2**32 - 1:
            raise ValueError("default_seed must be between 0 and 4294967295")
        if self.quality_max_retries < 0:
            raise ValueError("quality_max_retries must be >= 0")
        if self.stream_queue_size < 1:
            raise ValueError("stream_queue_size must be >= 1")
        if (
            self.api_key is None
            and not _is_loopback_host(self.host)
            and not self.allow_unauthenticated
        ):
            raise ValueError(
                "COSYVOICE_API_KEY is required when binding to a non-loopback "
                "address; set COSYVOICE_ALLOW_UNAUTHENTICATED=true only for "
                "an intentionally unauthenticated deployment"
            )

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
            host=os.getenv("COSYVOICE_HOST", "127.0.0.1"),
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
            allow_unauthenticated=_env_bool(
                "COSYVOICE_ALLOW_UNAUTHENTICATED", False
            ),
            default_seed=_env_int_range(
                "COSYVOICE_DEFAULT_SEED", 2, 0, 2**32 - 1
            ),
            quality_check_enabled=_env_bool(
                "COSYVOICE_QUALITY_CHECK_ENABLED", True
            ),
            quality_max_retries=_env_int(
                "COSYVOICE_QUALITY_MAX_RETRIES", 2, 0
            ),
            stream_queue_size=_env_int(
                "COSYVOICE_STREAM_QUEUE_SIZE", 4, 1
            ),
        )
