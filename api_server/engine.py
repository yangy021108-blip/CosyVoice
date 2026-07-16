"""Adapter from stable API concepts to CosyVoice inference functions."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from api_server.audio_codec import collect_waveform, encode_wav, float_to_pcm16
from api_server.config import Settings
from api_server.schemas import SpeechRequest
from api_server.voice_store import VoiceSpec, VoiceStore


@dataclass(frozen=True)
class AudioResult:
    content: bytes
    media_type: str
    sample_rate: int
    duration_seconds: float
    inference_seconds: float
    real_time_factor: float


class CosyVoiceEngine:
    """One loaded model instance shared by a single Uvicorn worker."""

    def __init__(self, settings: Settings, voice_store: VoiceStore) -> None:
        self.settings = settings
        self.voice_store = voice_store
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
            from cosyvoice.cli.cosyvoice import AutoModel

            backend = AutoModel(
                model_dir=str(self.settings.model_dir),
                load_vllm=self.settings.load_vllm,
                fp16=self.settings.fp16,
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

    def synthesize(
        self, request: SpeechRequest, voice: VoiceSpec
    ) -> AudioResult:
        if not self.ready:
            raise RuntimeError("CosyVoice model is not ready")
        if request.model != self.settings.model_alias:
            raise ValueError(f"unknown model: {request.model}")

        started = time.perf_counter()
        outputs = self._dispatch(request, voice)
        waveform = collect_waveform(outputs)
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
        )

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
            instruction = (
                "You are a helpful assistant.\n"
                f"{request.instructions}<|endofprompt|>"
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
