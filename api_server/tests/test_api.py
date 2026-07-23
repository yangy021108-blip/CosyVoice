from __future__ import annotations

import asyncio
import io
import threading
import time
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api_server.app import _await_inference_task, create_app
from api_server.audio_codec import encode_wav, float_to_pcm16
from api_server.config import Settings
from api_server.engine import AudioResult
from api_server.voice_store import VoiceSpec, VoiceStore


ROOT_DIR = Path(__file__).resolve().parents[2]


def make_settings(
    api_key: str | None = None,
    max_text: int = 2000,
    max_queue_size: int = 1,
    timeout: float = 10,
) -> Settings:
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
        max_queue_size=max_queue_size,
        request_timeout_seconds=timeout,
        fp16=False,
        load_vllm=False,
    )


class FakeEngine:
    def __init__(self, voice_store: VoiceStore, ready: bool = True) -> None:
        self.voice_store = voice_store
        self.ready = ready
        self.sample_rate = 24000 if ready else None
        self.load_error = None
        self.calls = 0

    def load(self) -> None:
        return None

    def synthesize(self, payload, voice) -> AudioResult:
        del voice
        self.calls += 1
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
            seed=2,
            quality_retry_count=1,
            silent_frame_ratio=0.1,
        )


class BlockingEngine(FakeEngine):
    def __init__(self, voice_store: VoiceStore) -> None:
        super().__init__(voice_store)
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize(self, payload, voice) -> AudioResult:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test engine was not released")
        return super().synthesize(payload, voice)


class RaisingEngine(FakeEngine):
    def synthesize(self, payload, voice) -> AudioResult:
        raise RuntimeError("backend failed")


class QualityFailingEngine(FakeEngine):
    def synthesize(self, payload, voice) -> AudioResult:
        del payload, voice
        from api_server.engine import AudioQualityError

        raise AudioQualityError("generated audio was degenerate")


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


def make_store_with_sft() -> VoiceStore:
    return VoiceStore(
        [
            VoiceSpec(
                voice_id="default",
                name="Test voice",
                mode="zero_shot",
                prompt_text="prompt",
                prompt_audio=ROOT_DIR / "asset" / "zero_shot_prompt.wav",
            ),
            VoiceSpec(
                voice_id="sft-test",
                name="SFT test voice",
                mode="sft",
                spk_id="speaker-a",
            ),
        ]
    )


class ApiTest(unittest.TestCase):
    def make_client(
        self,
        *,
        api_key: str | None = None,
        max_text: int = 2000,
        ready: bool = True,
        store: VoiceStore | None = None,
        engine: FakeEngine | None = None,
        max_queue_size: int = 1,
        timeout: float = 10,
    ) -> TestClient:
        store = store or make_store()
        engine = engine or FakeEngine(store, ready=ready)
        app = create_app(
            settings=make_settings(
                api_key=api_key,
                max_text=max_text,
                max_queue_size=max_queue_size,
                timeout=timeout,
            ),
            engine=engine,
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
        self.assertEqual(response.headers["x-generation-seed"], "2")
        self.assertEqual(response.headers["x-quality-retry-count"], "1")
        self.assertEqual(response.headers["x-silent-frame-ratio"], "0.100000")
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
            ({"stream_format": "audio"}, 400, "invalid_parameter"),
            ({"seed": -1}, 400, "invalid_parameter"),
            ({"seed": 2**32}, 400, "invalid_parameter"),
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

    def test_instructions_with_sft_voice_returns_400_before_engine(self) -> None:
        store = make_store_with_sft()
        engine = FakeEngine(store)
        with self.make_client(store=store, engine=engine) as client:
            response = client.post(
                "/v1/audio/speech",
                json=self.request_body(
                    voice="sft-test", instructions="speak happily"
                ),
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "unsupported_parameter")
        self.assertEqual(response.json()["error"]["param"], "instructions")
        self.assertEqual(engine.calls, 0)

    def test_backend_exception_maps_to_stable_500(self) -> None:
        store = make_store()
        with self.make_client(
            store=store, engine=RaisingEngine(store)
        ) as client:
            with self.assertLogs("cosyvoice.api", level="ERROR"):
                response = client.post(
                    "/v1/audio/speech", json=self.request_body()
                )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_error")

    def test_quality_failure_maps_to_retryable_503(self) -> None:
        store = make_store()
        with self.make_client(
            store=store, engine=QualityFailingEngine(store)
        ) as client:
            response = client.post(
                "/v1/audio/speech", json=self.request_body()
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["error"]["code"], "audio_quality_failed"
        )

    def test_concurrency_and_queue_limit_rejects_third_request(self) -> None:
        store = make_store()
        engine = BlockingEngine(store)
        client = self.make_client(
            store=store,
            engine=engine,
            max_queue_size=1,
            timeout=5,
        )
        with client, ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                client.post,
                "/v1/audio/speech",
                json=self.request_body(input="first"),
            )
            self.assertTrue(engine.started.wait(timeout=1))
            second = pool.submit(
                client.post,
                "/v1/audio/speech",
                json=self.request_body(input="second"),
            )

            deadline = time.monotonic() + 1
            while (
                client.app.state.admission_slots._value != 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertEqual(client.app.state.admission_slots._value, 0)

            third = client.post(
                "/v1/audio/speech", json=self.request_body(input="third")
            )
            self.assertEqual(third.status_code, 429)

            engine.release.set()
            self.assertEqual(first.result(timeout=2).status_code, 200)
            self.assertEqual(second.result(timeout=2).status_code, 200)

    def test_timed_out_tasks_continue_to_fill_bounded_capacity(self) -> None:
        store = make_store()
        engine = BlockingEngine(store)
        client = self.make_client(
            store=store,
            engine=engine,
            max_queue_size=1,
            timeout=0.05,
        )
        with client:
            first = client.post(
                "/v1/audio/speech", json=self.request_body(input="first")
            )
            self.assertTrue(engine.started.is_set())
            self.assertEqual(first.status_code, 504)

            second = client.post(
                "/v1/audio/speech", json=self.request_body(input="second")
            )
            self.assertEqual(second.status_code, 504)

            third = client.post(
                "/v1/audio/speech", json=self.request_body(input="third")
            )
            self.assertEqual(third.status_code, 429)
            self.assertEqual(third.json()["error"]["code"], "queue_full")

            engine.release.set()
            deadline = time.monotonic() + 2
            while engine.calls < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(engine.calls, 2)

            deadline = time.monotonic() + 2
            while (
                client.app.state.admission_slots._value < 2
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertEqual(client.app.state.admission_slots._value, 2)

            fourth = client.post(
                "/v1/audio/speech", json=self.request_body(input="fourth")
            )
            self.assertEqual(fourth.status_code, 200)


class AdmissionCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_client_keeps_slot_until_background_finishes(self):
        admission = asyncio.BoundedSemaphore(1)
        await admission.acquire()
        backend_done = asyncio.Event()

        async def background():
            await backend_done.wait()
            return "complete"

        inference_task = asyncio.create_task(background())
        request_task = asyncio.create_task(
            _await_inference_task(inference_task, admission, 10)
        )
        await asyncio.sleep(0)
        request_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request_task

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(admission.acquire(), timeout=0.01)

        backend_done.set()
        await inference_task
        await asyncio.sleep(0)
        await asyncio.wait_for(admission.acquire(), timeout=0.1)
        admission.release()


if __name__ == "__main__":
    unittest.main()
