"""Adapter from stable API concepts to CosyVoice inference functions."""

from __future__ import annotations

import logging
import random
import sys
import threading
import time
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from typing import Callable

import numpy as np

from api_server.audio_codec import (
    AudioQuality,
    analyze_audio_quality,
    collect_waveform,
    encode_wav,
    float_to_pcm16,
)
from api_server.config import Settings
from api_server.schemas import SpeechRequest
from api_server.voice_store import VoiceSpec, VoiceStore


LOGGER = logging.getLogger("cosyvoice.api.quality")

PYTORCH_TRANSFORMERS_VERSION = "4.51.3"
VLLM_TRANSFORMERS_VERSIONS = ("4.57.1", "4.57.3")


@dataclass(frozen=True)
class AudioResult:
    content: bytes
    media_type: str
    sample_rate: int
    duration_seconds: float
    inference_seconds: float
    real_time_factor: float
    seed: int | None = None
    quality_retry_count: int = 0
    silent_frame_ratio: float | None = None


class AudioQualityError(RuntimeError):
    """Raised when every generated candidate is clearly degenerate."""


class CosyVoiceEngine:
    """One loaded model instance shared by a single Uvicorn worker."""

    def __init__(
        self,
        settings: Settings,
        voice_store: VoiceStore,
        backend_factory: Callable[[], object] | None = None,
        seed_setter: Callable[[int], None] | None = None,
    ) -> None:
        self.settings = settings
        self.voice_store = voice_store
        self._backend_factory = backend_factory
        self._seed_setter = seed_setter
        self._inference_lock = threading.Lock()
        self.backend = None
        self.sample_rate: int | None = None
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.backend is not None and self.sample_rate is not None

    def load(self) -> None:
        matcha_path = self.settings.root_dir / "third_party" / "Matcha-TTS"
        if str(matcha_path) not in sys.path:
            sys.path.append(str(matcha_path))

        try:
            backend = self._create_backend()
            available_spks = set(backend.list_available_spks())
            for voice in self.voice_store.all():
                if voice.mode == "sft" and voice.spk_id not in available_spks:
                    raise ValueError(
                        f"SFT speaker {voice.spk_id!r} for voice "
                        f"{voice.voice_id!r} is not available"
                    )
            for voice in self.voice_store.all():
                if voice.mode == "zero_shot":
                    backend.add_zero_shot_spk(
                        voice.prompt_text,
                        str(voice.prompt_audio),
                        voice.voice_id,
                    )
            self.backend = backend
            self.sample_rate = int(backend.sample_rate)
            self.load_error = None
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

    def _create_backend(self):
        if self._backend_factory is not None:
            return self._backend_factory()
        if self.settings.load_vllm and find_spec("vllm") is None:
            raise RuntimeError(
                "COSYVOICE_LOAD_VLLM=true requires vLLM; install "
                "api_server/requirements-vllm.txt or build the Docker image "
                "with --build-arg INSTALL_VLLM=true"
            )
        self._validate_runtime_dependencies()

        from cosyvoice.cli.cosyvoice import AutoModel

        return AutoModel(
            model_dir=str(self.settings.model_dir),
            load_vllm=self.settings.load_vllm,
            fp16=self.settings.fp16,
        )

    def _validate_runtime_dependencies(self) -> None:
        required_versions = (
            VLLM_TRANSFORMERS_VERSIONS
            if self.settings.load_vllm
            else (PYTORCH_TRANSFORMERS_VERSION,)
        )
        required_description = " or ".join(
            f"transformers=={version}" for version in required_versions
        )
        requirements_file = (
            "api_server/requirements-vllm.txt"
            if self.settings.load_vllm
            else "api_server/requirements-runtime.txt"
        )
        backend_name = "vLLM" if self.settings.load_vllm else "PyTorch"
        try:
            installed_version = metadata.version("transformers")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"{backend_name} backend requires "
                f"{required_description}; transformers is not "
                f"installed. Install {requirements_file}"
            ) from exc
        if installed_version not in required_versions:
            raise RuntimeError(
                f"{backend_name} backend requires "
                f"{required_description}; found "
                f"transformers=={installed_version}. Install "
                f"{requirements_file} in a dedicated environment. Mixing "
                "the PyTorch and vLLM dependency sets can generate garbled "
                "speech."
            )

    def synthesize(
        self, request: SpeechRequest, voice: VoiceSpec
    ) -> AudioResult:
        if not self.ready:
            raise RuntimeError("CosyVoice model is not ready")
        if request.model != self.settings.model_alias:
            raise ValueError(f"unknown model: {request.model}")

        started = time.perf_counter()
        with self._inference_lock:
            waveform, seed, retry_count, quality = self._generate_candidate(
                request, voice
            )
        inference_seconds = time.perf_counter() - started
        pcm = float_to_pcm16(waveform)
        duration_seconds = len(pcm) / int(self.sample_rate)
        real_time_factor = (
            inference_seconds / duration_seconds if duration_seconds else 0.0
        )

        if request.response_format == "wav":
            content = encode_wav(pcm, int(self.sample_rate))
            media_type = "audio/wav"
        else:
            content = pcm.tobytes()
            media_type = "audio/pcm"

        return AudioResult(
            content=content,
            media_type=media_type,
            sample_rate=int(self.sample_rate),
            duration_seconds=duration_seconds,
            inference_seconds=inference_seconds,
            real_time_factor=real_time_factor,
            seed=seed,
            quality_retry_count=retry_count,
            silent_frame_ratio=quality.silent_frame_ratio,
        )

    def _generate_candidate(
        self, request: SpeechRequest, voice: VoiceSpec
    ) -> tuple[np.ndarray, int, int, AudioQuality]:
        base_seed = (
            request.seed
            if request.seed is not None
            else self.settings.default_seed
        )
        retry_count = (
            0
            if request.seed is not None or not self.settings.quality_check_enabled
            else self.settings.quality_max_retries
        )
        last_quality: AudioQuality | None = None

        for attempt in range(retry_count + 1):
            seed = (base_seed + attempt) % (2**32)
            self._set_random_seed(seed)
            waveform = collect_waveform(self._dispatch(request, voice))
            quality = analyze_audio_quality(
                waveform,
                request.input,
                sample_rate=int(self.sample_rate),
                speed=request.speed,
            )
            last_quality = quality
            if not self.settings.quality_check_enabled or quality.acceptable:
                return waveform, seed, attempt, quality
            LOGGER.warning(
                "rejected degenerate audio seed=%d attempt=%d reason=%s "
                "duration=%.3f silent_ratio=%.4f voiced_seconds_per_unit=%.4f",
                seed,
                attempt + 1,
                quality.reason,
                quality.duration_seconds,
                quality.silent_frame_ratio,
                quality.voiced_seconds_per_text_unit,
            )

        raise AudioQualityError(
            "CosyVoice generated degenerate audio after "
            f"{retry_count + 1} attempts; last_reason={last_quality.reason}"
        )

    def _set_random_seed(self, seed: int) -> None:
        if self._seed_setter is not None:
            self._seed_setter(seed)
        else:
            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch
            except ModuleNotFoundError:
                pass
            else:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                sdaa = getattr(torch, "sdaa", None)
                if sdaa is not None and sdaa.is_available():
                    sdaa.manual_seed_all(seed)

        if self.settings.load_vllm and self.backend is not None:
            model = getattr(self.backend, "model", None)
            llm = getattr(model, "llm", None)
            seed_setter = getattr(llm, "set_inference_seed", None)
            if callable(seed_setter):
                seed_setter(seed)

    def _dispatch(self, request: SpeechRequest, voice: VoiceSpec):
        common = {
            "stream": False,
            "speed": request.speed,
            "text_frontend": True,
        }
        if request.instructions:
            if voice.mode != "zero_shot" or voice.prompt_audio is None:
                raise ValueError(
                    "instructions require a zero-shot voice on CosyVoice3"
                )
            user_instruction = " ".join(request.instructions.split())
            if not user_instruction.endswith(("。", ".", "！", "!", "？", "?")):
                punctuation = (
                    "。"
                    if any("\u3400" <= char <= "\u9fff" for char in user_instruction)
                    else "."
                )
                user_instruction += punctuation
            instruction = (
                "You are a helpful assistant. "
                f"{user_instruction}<|endofprompt|>"
            )
            return self.backend.inference_instruct2(
                tts_text=request.input,
                instruct_text=instruction,
                prompt_wav=str(voice.prompt_audio),
                zero_shot_spk_id="",
                **common,
            )

        if voice.mode == "zero_shot":
            return self.backend.inference_zero_shot(
                tts_text=request.input,
                prompt_text="",
                prompt_wav="",
                zero_shot_spk_id=voice.voice_id,
                **common,
            )
        if voice.mode == "sft":
            return self.backend.inference_sft(
                tts_text=request.input,
                spk_id=voice.spk_id,
                **common,
            )
        raise ValueError(f"unsupported voice mode: {voice.mode}")
