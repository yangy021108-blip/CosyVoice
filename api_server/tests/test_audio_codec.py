from __future__ import annotations

import io
import unittest
import wave

import numpy as np

from api_server.audio_codec import collect_waveform, encode_wav, float_to_pcm16


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


if __name__ == "__main__":
    unittest.main()
