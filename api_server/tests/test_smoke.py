from __future__ import annotations

import unittest

import numpy as np

from api_server.audio_codec import encode_wav, float_to_pcm16
from api_server.smoke_test import validate_wav_response


class SmokeValidationTest(unittest.TestCase):
    def test_validates_real_audio_and_metric_headers(self) -> None:
        waveform = np.sin(np.linspace(0, 8, 2400, dtype=np.float32)) * 0.25
        content = encode_wav(float_to_pcm16(waveform), 24000)
        metrics = validate_wav_response(
            content,
            {
                "Content-Type": "audio/wav",
                "X-Audio-Duration": "0.1",
                "X-Real-Time-Factor": "0.25",
            },
            expected_sample_rate=24000,
            duration_tolerance_seconds=0.001,
            max_clipping_ratio=0.01,
        )
        self.assertEqual(metrics["frame_count"], 2400)
        self.assertGreater(metrics["peak_pcm16"], 0)
        self.assertEqual(metrics["clipping_ratio"], 0)

    def test_rejects_silent_audio(self) -> None:
        content = encode_wav(np.zeros(2400, dtype="<i2"), 24000)
        with self.assertRaisesRegex(ValueError, "entirely silent"):
            validate_wav_response(
                content,
                {
                    "Content-Type": "audio/wav",
                    "X-Audio-Duration": "0.1",
                    "X-Real-Time-Factor": "0.25",
                },
                expected_sample_rate=24000,
                duration_tolerance_seconds=0.001,
                max_clipping_ratio=0.01,
            )


if __name__ == "__main__":
    unittest.main()
