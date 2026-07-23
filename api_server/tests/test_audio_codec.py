from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from api_server.audio_codec import (
    analyze_audio_quality,
    collect_waveform,
    encode_wav,
    float_to_pcm16,
)


class AudioCodecTest(unittest.TestCase):
    def test_collect_and_encode_wav(self) -> None:
        waveform = collect_waveform(
            [
                {"tts_speech": np.array([[0.0, 0.5]], dtype=np.float32)},
                {"tts_speech": np.array([[-0.5, 1.0]], dtype=np.float32)},
            ]
        )
        pcm = float_to_pcm16(waveform)
        content = encode_wav(pcm, 24000)

        with wave.open(io.BytesIO(content), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 24000)
            self.assertEqual(wav_file.getnframes(), 4)

    def test_float_to_pcm16_clips_without_overflow(self) -> None:
        pcm = float_to_pcm16(
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            pcm,
            np.array([-32767, -32767, 0, 32767, 32767], dtype="<i2"),
        )

    def test_collect_rejects_invalid_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "no audio chunks"):
            collect_waveform([])
        with self.assertRaisesRegex(ValueError, "missing tts_speech"):
            collect_waveform([{}])
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            collect_waveform(
                [{"tts_speech": np.array([np.nan], dtype=np.float32)}]
            )

    def test_quality_analysis_accepts_continuous_voiced_audio(self) -> None:
        waveform = np.full(24000 * 2, 0.2, dtype=np.float32)
        quality = analyze_audio_quality(
            waveform,
            "欢迎使用我们的语音合成服务。",
            sample_rate=24000,
        )
        self.assertTrue(quality.acceptable)
        self.assertEqual(quality.reason, "ok")
        self.assertEqual(quality.silent_frame_ratio, 0.0)

    def test_quality_analysis_rejects_silent_audio(self) -> None:
        waveform = np.zeros(24000 * 2, dtype=np.float32)
        quality = analyze_audio_quality(
            waveform,
            "欢迎使用我们的语音合成服务。",
            sample_rate=24000,
        )
        self.assertFalse(quality.acceptable)
        self.assertEqual(quality.reason, "too_silent")

    def test_quality_analysis_rejects_implausibly_long_audio(self) -> None:
        waveform = np.full(24000 * 8, 0.2, dtype=np.float32)
        quality = analyze_audio_quality(
            waveform,
            "你好，这是一次 CosyVoice API 测试。",
            sample_rate=24000,
        )
        self.assertFalse(quality.acceptable)
        self.assertEqual(quality.reason, "too_long_for_text")

    def test_quality_duration_limit_accounts_for_requested_speed(self) -> None:
        waveform = np.full(24000 * 8, 0.2, dtype=np.float32)
        quality = analyze_audio_quality(
            waveform,
            "你好，这是一次 CosyVoice API 测试。",
            sample_rate=24000,
            speed=0.5,
        )
        self.assertTrue(quality.acceptable)


if __name__ == "__main__":
    unittest.main()
