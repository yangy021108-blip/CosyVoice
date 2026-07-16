"""Audio conversion helpers with no dependency on torchaudio."""

from __future__ import annotations

import io
import wave
from collections.abc import Iterable, Mapping

import numpy as np


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
