"""Public request schemas and strict parameter validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SpeechRequest(BaseModel):
    """OpenAI-style speech request with the subset supported by CosyVoice."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: str = Field(min_length=1, max_length=128)
    input: str = Field(min_length=1)
    voice: str = Field(min_length=1, max_length=128)
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    instructions: str | None = Field(default=None, max_length=1000)
    stream_format: Literal["audio"] = "audio"
