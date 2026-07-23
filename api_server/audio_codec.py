"""Audio conversion helpers with no dependency on torchaudio."""

from __future__ import annotations

import io
import math
import re
import wave
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


_SILENCE_RMS = 10 ** (-50 / 20)
_MAX_SILENT_FRAME_RATIO = 0.48
_MAX_SECONDS_PER_TEXT_UNIT = 0.55
_MIN_VOICED_SECONDS_PER_TEXT_UNIT = 0.08
_CJK_CHARACTER = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
)
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class AudioQuality:
    acceptable: bool
    reason: str
    duration_seconds: float
    silent_frame_ratio: float
    voiced_seconds_per_text_unit: float


def _to_numpy(waveform: object) -> np.ndarray:
    value = waveform
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not array.size:
        raise ValueError("CosyVoice returned an empty waveform")
    if not np.isfinite(array).all():
        raise ValueError("CosyVoice returned NaN or infinite audio samples")
    return array


def collect_waveform(outputs: Iterable[Mapping[str, object]]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for output in outputs:
        if "tts_speech" not in output:
            raise ValueError("CosyVoice output is missing tts_speech")
        chunks.append(_to_numpy(output["tts_speech"]))
    if not chunks:
        raise ValueError("CosyVoice returned no audio chunks")
    return np.concatenate(chunks)


def _text_unit_count(text: str) -> int:
    """Approximate spoken units across CJK text and space-delimited words."""

    cjk_count = len(_CJK_CHARACTER.findall(text))
    latin_unit_count = sum(
        _latin_spoken_unit_count(word) for word in _LATIN_WORD.findall(text)
    )
    return max(1, cjk_count + latin_unit_count)


def _latin_spoken_unit_count(word: str) -> int:
    """Estimate duration units for words, acronyms, and mixed-case names."""

    alphanumeric = "".join(
        character for character in word if character.isalnum()
    )
    if not alphanumeric:
        return 0
    letters = [character for character in alphanumeric if character.isalpha()]
    uppercase_count = sum(character.isupper() for character in letters)
    if letters and uppercase_count / len(letters) >= 0.5:
        # Acronyms such as API and vLLM are commonly spoken letter by letter.
        return len(alphanumeric)
    # Ordinary and CamelCase words need more allowance than a single CJK
    # character. Four Latin characters per unit is deliberately conservative.
    return max(1, math.ceil(len(alphanumeric) / 4))


def analyze_audio_quality(
    waveform: np.ndarray,
    text: str,
    *,
    sample_rate: int,
    speed: float = 1.0,
) -> AudioQuality:
    """Reject obvious token-sampling collapses before serving the audio."""

    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not samples.size:
        raise ValueError("cannot analyze an empty waveform")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if speed <= 0:
        raise ValueError("speed must be greater than zero")

    duration_seconds = len(samples) / sample_rate
    frame_size = max(1, int(sample_rate * 0.02))
    frame_count = max(1, (len(samples) + frame_size - 1) // frame_size)
    padded = np.pad(samples, (0, frame_count * frame_size - len(samples)))
    frames = padded.reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    silent_frame_ratio = float(np.mean(frame_rms < _SILENCE_RMS))

    text_units = _text_unit_count(text)
    voiced_seconds = duration_seconds * (1.0 - silent_frame_ratio)
    voiced_seconds_per_text_unit = voiced_seconds / text_units
    maximum_duration = max(
        3.0 / speed,
        text_units * _MAX_SECONDS_PER_TEXT_UNIT / speed,
    )
    minimum_voiced_seconds_per_unit = (
        _MIN_VOICED_SECONDS_PER_TEXT_UNIT / speed
    )

    if duration_seconds < 0.2:
        reason = "too_short"
    elif duration_seconds > maximum_duration:
        reason = "too_long_for_text"
    elif silent_frame_ratio > _MAX_SILENT_FRAME_RATIO:
        reason = "too_silent"
    elif voiced_seconds_per_text_unit < minimum_voiced_seconds_per_unit:
        reason = "insufficient_voiced_audio"
    else:
        reason = "ok"

    return AudioQuality(
        acceptable=reason == "ok",
        reason=reason,
        duration_seconds=duration_seconds,
        silent_frame_ratio=silent_frame_ratio,
        voiced_seconds_per_text_unit=voiced_seconds_per_text_unit,
    )


def float_to_pcm16(waveform: np.ndarray) -> np.ndarray:
    clipped = np.clip(waveform, -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2")


def encode_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.astype("<i2", copy=False).tobytes())
    return buffer.getvalue()
