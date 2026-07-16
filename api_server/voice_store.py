"""Server-owned voice registry.

Clients only submit a voice ID. Prompt paths and inference modes remain private
service configuration and can never be supplied as request-side file paths.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SUPPORTED_MODES = {"sft", "zero_shot"}


@dataclass(frozen=True)
class VoiceSpec:
    voice_id: str
    name: str
    mode: str
    spk_id: str | None = None
    prompt_text: str | None = None
    prompt_audio: Path | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.voice_id,
            "object": "voice",
            "name": self.name,
            "mode": self.mode,
        }


class VoiceStore:
    def __init__(self, voices: Iterable[VoiceSpec]) -> None:
        self._voices = {voice.voice_id: voice for voice in voices}
        if not self._voices:
            raise ValueError("voice registry must contain at least one voice")

    @classmethod
    def from_json(cls, path: Path, root_dir: Path) -> "VoiceStore":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("voice registry must be a JSON object")

        voices: list[VoiceSpec] = []
        for voice_id, raw in payload.items():
            voices.append(cls._parse_voice(voice_id, raw, root_dir))
        return cls(voices)

    @staticmethod
    def _parse_voice(
        voice_id: str, raw: Any, root_dir: Path
    ) -> VoiceSpec:
        if not VOICE_ID_PATTERN.fullmatch(voice_id):
            raise ValueError(f"invalid voice ID: {voice_id!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"voice {voice_id!r} must be an object")

        mode = raw.get("mode")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"voice {voice_id!r} has unsupported mode {mode!r}")
        name = raw.get("name", voice_id)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"voice {voice_id!r} must have a non-empty name")

        if mode == "sft":
            spk_id = raw.get("spk_id")
            if not isinstance(spk_id, str) or not spk_id:
                raise ValueError(f"SFT voice {voice_id!r} requires spk_id")
            return VoiceSpec(
                voice_id=voice_id,
                name=name,
                mode=mode,
                spk_id=spk_id,
            )

        prompt_text = raw.get("prompt_text")
        prompt_audio_value = raw.get("prompt_audio")
        if not isinstance(prompt_text, str) or not prompt_text:
            raise ValueError(
                f"zero-shot voice {voice_id!r} requires prompt_text"
            )
        if not isinstance(prompt_audio_value, str) or not prompt_audio_value:
            raise ValueError(
                f"zero-shot voice {voice_id!r} requires prompt_audio"
            )
        prompt_audio = (root_dir / prompt_audio_value).resolve()
        root_resolved = root_dir.resolve()
        if root_resolved not in prompt_audio.parents:
            raise ValueError(
                f"voice {voice_id!r} prompt_audio must stay inside the repository"
            )
        if not prompt_audio.is_file():
            raise ValueError(
                f"voice {voice_id!r} prompt audio does not exist: {prompt_audio}"
            )

        return VoiceSpec(
            voice_id=voice_id,
            name=name,
            mode=mode,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
        )

    def get(self, voice_id: str) -> VoiceSpec | None:
        return self._voices.get(voice_id)

    def all(self) -> list[VoiceSpec]:
        return list(self._voices.values())
