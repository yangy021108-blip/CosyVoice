from __future__ import annotations

import io
import unittest
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api_server.app import create_app
from api_server.audio_codec import encode_wav, float_to_pcm16
from api_server.config import Settings
from api_server.engine import AudioResult
from api_server.voice_store import VoiceSpec, VoiceStore


ROOT_DIR = Path(__file__).resolve().parents[2]


def make_settings(api_key: str | None = None, max_text: int = 2000) -> Settings:
    return Settings(
        root_dir=ROOT_DIR,
        model_alias="cosyvoice3-0.5b",
        model_dir=ROOT_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B",
        voices_file=ROOT_DIR / "api_server" / "voices.json",
        api_key=api_key,
        host="127.0.0.1",
        port=8000,
        max_text_characters=max_text,
        max_concurrency=1,
        max_queue_size=1,
        request_timeout_seconds=10,
        fp16=False,
        load_vllm=False,
    )


class FakeEngine:
    def __init__(self, voice_store: VoiceStore, ready: bool = True) -> None:
        self.voice_store = voice_store
        self.ready = ready
        self.sample_rate = 24000 if ready else None
        self.load_error = None

    def load(self) -> None:
        return None

    def synthesize(self, payload, voice) -> AudioResult:
        del voice
        samples = np.zeros(2400, dtype=np.float32)
        pcm = float_to_pcm16(samples)
        if payload.response_format == "wav":
            content = encode_wav(pcm, 24000)
            media_type = "audio/wav"
        else:
            content = pcm.tobytes()
            media_type = "audio/pcm"
        return AudioResult(
            content=content,
            media_type=media_type,
            sample_rate=24000,
            duration_seconds=0.1,
            inference_seconds=0.02,
            real_time_factor=0.2,
        )


def make_store() -> VoiceStore:
    return VoiceStore(
        [
            VoiceSpec(
                voice_id="default",
                name="Test voice",
                mode="zero_shot",
                prompt_text="prompt",
                prompt_audio=ROOT_DIR / "asset" / "zero_shot_prompt.wav",
            )
        ]
    )


class ApiTest(unittest.TestCase):
    def make_client(
        self,
        *,
        api_key: str | None = None,
        max_text: int = 2000,
        ready: bool = True,
    ) -> TestClient:
        store = make_store()
        app = create_app(
            settings=make_settings(api_key=api_key, max_text=max_text),
            engine=FakeEngine(store, ready=ready),
            voice_store=store,
        )
        return TestClient(app, raise_server_exceptions=False)

    def request_body(self, **overrides):
        body = {
            "model": "cosyvoice3-0.5b",
            "input": "你好，这是一次接口测试。",
            "voice": "default",
            "response_format": "wav",
            "speed": 1.0,
        }
        body.update(overrides)
        return body

    def test_health_ready_models_and_voices(self) -> None:
        with self.make_client() as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            ready = client.get("/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["sample_rate"], 24000)
            self.assertEqual(
                client.get("/v1/models").json()["data"][0]["id"],
                "cosyvoice3-0.5b",
            )
            self.assertEqual(
                client.get("/v1/audio/voices").json()["data"][0]["id"],
                "default",
            )

    def test_wav_speech_response(self) -> None:
        with self.make_client() as client:
            response = client.post(
                "/v1/audio/speech", json=self.request_body()
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("audio/wav"))
        self.assertEqual(response.headers["x-audio-sample-rate"], "24000")
        self.assertIn("x-request-id", response.headers)
        with wave.open(io.BytesIO(response.content), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 24000)
            self.assertGreater(wav_file.getnframes(), 0)

    def test_pcm_speech_response(self) -> None:
        with self.make_client() as client:
            response = client.post(
                "/v1/audio/speech",
                json=self.request_body(response_format="pcm"),
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("audio/pcm"))
        self.assertEqual(len(response.content), 4800)

    def test_authentication(self) -> None:
        with self.make_client(api_key="secret") as client:
            health = client.get("/health")
            unauthorized = client.get("/v1/models")
            authorized = client.get(
                "/v1/models", headers={"Authorization": "Bearer secret"}
            )
        self.assertEqual(health.status_code, 200)
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized.json()["error"]["code"], "invalid_api_key")
        self.assertEqual(authorized.status_code, 200)

    def test_parameter_errors_are_stable(self) -> None:
        cases = [
            ({"input": ""}, 400, "invalid_parameter"),
            ({"voice": "missing"}, 404, "voice_not_found"),
            ({"model": "unknown"}, 404, "model_not_found"),
            ({"response_format": "mp3"}, 400, "invalid_parameter"),
            ({"speed": 3.0}, 400, "invalid_parameter"),
            ({"pitch": 1.2}, 400, "invalid_parameter"),
        ]
        with self.make_client() as client:
            for overrides, status, code in cases:
                with self.subTest(overrides=overrides):
                    response = client.post(
                        "/v1/audio/speech",
                        json=self.request_body(**overrides),
                    )
                    self.assertEqual(response.status_code, status)
                    self.assertEqual(response.json()["error"]["code"], code)
                    self.assertIn("x-request-id", response.headers)

    def test_text_length_limit(self) -> None:
        with self.make_client(max_text=4) as client:
            response = client.post(
                "/v1/audio/speech", json=self.request_body(input="12345")
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "input_too_large")

    def test_not_ready(self) -> None:
        with self.make_client(ready=False) as client:
            response = client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "model_not_ready")


if __name__ == "__main__":
    unittest.main()
