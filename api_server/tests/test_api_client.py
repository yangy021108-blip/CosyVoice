import json
import tempfile
import unittest
from pathlib import Path

from tools.test_cosyvoice_api import (
    build_speech_payload,
    normalize_voice_mode,
    save_audio_response,
    select_voice,
    terminal_summary,
)


VOICES_BODY = {
    "object": "list",
    "data": [
        {
            "id": "default",
            "name": "Default zero-shot voice",
            "mode": "zero_shot",
        },
        {
            "id": "speaker-a",
            "name": "SFT speaker",
            "mode": "sft",
        },
    ],
}


class TestCosyVoiceApiClient(unittest.TestCase):
    def test_normalizes_zero_shot_alias(self) -> None:
        self.assertEqual(normalize_voice_mode("zero-shot"), "zero_shot")
        self.assertEqual(normalize_voice_mode("zero_shot"), "zero_shot")
        self.assertEqual(normalize_voice_mode("sft"), "sft")

    def test_selects_voice_by_mode_or_id(self) -> None:
        self.assertEqual(
            select_voice(VOICES_BODY, "zero_shot", None)["id"],
            "default",
        )
        self.assertEqual(
            select_voice(VOICES_BODY, "sft", "speaker-a")["id"],
            "speaker-a",
        )

    def test_rejects_voice_mode_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "not 'sft'"):
            select_voice(VOICES_BODY, "sft", "default")

    def test_reports_missing_sft_voice_before_request(self) -> None:
        zero_shot_only = {
            "object": "list",
            "data": [VOICES_BODY["data"][0]],
        }
        with self.assertRaisesRegex(
            ValueError,
            "no sft voice is registered",
        ):
            select_voice(zero_shot_only, "sft", None)

    def test_payload_omits_instructions_when_not_requested(self) -> None:
        payload = build_speech_payload(
            text="测试",
            voice="default",
            voice_mode="zero_shot",
            response_format="wav",
            seed=0,
            instructions=None,
        )
        self.assertNotIn("instructions", payload)
        self.assertEqual(payload["response_format"], "wav")

    def test_payload_supports_zero_shot_instruction_and_pcm(self) -> None:
        payload = build_speech_payload(
            text="测试",
            voice="default",
            voice_mode="zero_shot",
            response_format="pcm",
            seed=1,
            instructions="  请开心地说。  ",
        )
        self.assertEqual(payload["instructions"], "请开心地说。")
        self.assertEqual(payload["response_format"], "pcm")

    def test_rejects_instructions_with_sft(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "only with a zero_shot voice",
        ):
            build_speech_payload(
                text="测试",
                voice="speaker-a",
                voice_mode="sft",
                response_format="wav",
                seed=0,
                instructions="请开心地说。",
            )

    def test_http_error_is_never_written_as_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = save_audio_response(
                output_dir,
                "failed_case",
                503,
                {"content-type": "application/json"},
                json.dumps(
                    {"error": {"code": "audio_quality_failed"}}
                ).encode("utf-8"),
                expected_format="wav",
            )
            self.assertFalse(result["ok"])
            self.assertTrue(
                (output_dir / "failed_case.error.json").is_file()
            )
            self.assertFalse((output_dir / "failed_case.wav").exists())

    def test_default_terminal_summary_is_curl_style(self) -> None:
        summary = {
            "speech_requests": [
                {
                    "ok": True,
                    "request_bytes": 212,
                    "response_bytes": 170924,
                    "elapsed_seconds": 2.8,
                }
            ]
        }
        output = terminal_summary(summary, verbose=False)
        self.assertIn("% Total", output)
        self.assertIn("100", output)
        self.assertNotIn("speech_requests", output)

    def test_verbose_terminal_summary_prints_json(self) -> None:
        summary = {"ok": True, "speech_requests": []}
        output = terminal_summary(summary, verbose=True)
        self.assertEqual(json.loads(output), summary)


if __name__ == "__main__":
    unittest.main()
