from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api_server.app import create_app
from api_server.config import Settings
from api_server.engine import CosyVoiceEngine
from api_server.schemas import SpeechRequest
from api_server.voice_store import VoiceSpec, VoiceStore


ROOT_DIR = Path(__file__).resolve().parents[2]


def make_settings() -> Settings:
    return Settings(
        root_dir=ROOT_DIR,
        model_alias="cosyvoice3-0.5b",
        model_dir=ROOT_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B",
        voices_file=ROOT_DIR / "api_server" / "voices.json",
        api_key=None,
        host="127.0.0.1",
        port=8000,
        max_text_characters=2000,
        max_concurrency=1,
        max_queue_size=1,
        request_timeout_seconds=10,
        fp16=False,
        load_vllm=False,
    )


def make_store() -> VoiceStore:
    return VoiceStore(
        [
            VoiceSpec(
                voice_id="default",
                name="Zero-shot voice",
                mode="zero_shot",
                prompt_text="prompt text<|endofprompt|>",
                prompt_audio=ROOT_DIR / "asset" / "zero_shot_prompt.wav",
            ),
            VoiceSpec(
                voice_id="sft-test",
                name="SFT voice",
                mode="sft",
                spk_id="speaker-a",
            ),
        ]
    )


class RecordingBackend:
    sample_rate = 24000

    def __init__(self, speakers=("speaker-a",)) -> None:
        self.speakers = list(speakers)
        self.calls: list[tuple[str, dict]] = []

    def list_available_spks(self):
        self.calls.append(("list_available_spks", {}))
        return self.speakers

    def add_zero_shot_spk(self, prompt_text, prompt_wav, voice_id):
        self.calls.append(
            (
                "add_zero_shot_spk",
                {
                    "prompt_text": prompt_text,
                    "prompt_wav": prompt_wav,
                    "voice_id": voice_id,
                },
            )
        )
        return True

    def _result(self, name, kwargs):
        self.calls.append((name, kwargs))
        return [{"tts_speech": np.array([[0.25, -0.25]], dtype=np.float32)}]

    def inference_zero_shot(self, **kwargs):
        return self._result("inference_zero_shot", kwargs)

    def inference_instruct2(self, **kwargs):
        return self._result("inference_instruct2", kwargs)

    def inference_sft(self, **kwargs):
        return self._result("inference_sft", kwargs)


def request(**overrides) -> SpeechRequest:
    payload = {
        "model": "cosyvoice3-0.5b",
        "input": "hello",
        "voice": "default",
        "response_format": "wav",
        "speed": 1.0,
    }
    payload.update(overrides)
    return SpeechRequest(**payload)


class EngineTest(unittest.TestCase):
    def test_dispatches_all_supported_inference_modes(self) -> None:
        store = make_store()
        backend = RecordingBackend()
        engine = CosyVoiceEngine(
            make_settings(), store, backend_factory=lambda: backend
        )
        engine.load()

        zero_voice = store.get("default")
        sft_voice = store.get("sft-test")
        self.assertIsNotNone(zero_voice)
        self.assertIsNotNone(sft_voice)

        engine.synthesize(request(), zero_voice)
        engine.synthesize(
            request(instructions="speak happily"), zero_voice
        )
        engine.synthesize(request(voice="sft-test"), sft_voice)

        calls = dict(backend.calls)
        self.assertEqual(calls["add_zero_shot_spk"]["voice_id"], "default")
        self.assertEqual(
            calls["inference_zero_shot"]["zero_shot_spk_id"], "default"
        )
        self.assertEqual(calls["inference_zero_shot"]["prompt_wav"], "")
        self.assertEqual(
            calls["inference_instruct2"]["prompt_wav"],
            str(zero_voice.prompt_audio),
        )
        self.assertTrue(
            calls["inference_instruct2"]["instruct_text"].endswith(
                "<|endofprompt|>"
            )
        )
        self.assertEqual(calls["inference_sft"]["spk_id"], "speaker-a")

    def test_invalid_sft_speaker_prevents_ready_state(self) -> None:
        backend = RecordingBackend(speakers=())
        engine = CosyVoiceEngine(
            make_settings(), make_store(), backend_factory=lambda: backend
        )
        with self.assertRaisesRegex(ValueError, "speaker-a.*not available"):
            engine.load()
        self.assertFalse(engine.ready)
        self.assertIn("speaker-a", engine.load_error)

    def test_invalid_sft_speaker_keeps_ready_endpoint_at_503(self) -> None:
        store = make_store()
        engine = CosyVoiceEngine(
            make_settings(),
            store,
            backend_factory=lambda: RecordingBackend(speakers=()),
        )
        app = create_app(make_settings(), engine, store)
        with self.assertLogs("cosyvoice.api", level="ERROR"):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "model_not_ready")

    def test_missing_optional_vllm_has_actionable_error(self) -> None:
        settings = replace(make_settings(), load_vllm=True)
        engine = CosyVoiceEngine(settings, make_store())
        with patch("api_server.engine.find_spec", return_value=None):
            with self.assertRaisesRegex(
                RuntimeError, "requirements-vllm.txt"
            ):
                engine._create_backend()


if __name__ == "__main__":
    unittest.main()
