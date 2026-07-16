"""FastAPI application factory and OpenAI-style HTTP routes."""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from api_server.config import Settings
from api_server.engine import CosyVoiceEngine
from api_server.errors import ServiceError
from api_server.schemas import SpeechRequest
from api_server.voice_store import VoiceStore


LOGGER = logging.getLogger("cosyvoice.api")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error_response(request: Request, error: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {
                "message": error.message,
                "type": error.error_type,
                "param": error.param,
                "code": error.code,
            },
            "request_id": _request_id(request),
        },
    )


def create_app(
    settings: Settings | None = None,
    engine: CosyVoiceEngine | None = None,
    voice_store: VoiceStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    voice_store = voice_store or (
        engine.voice_store
        if engine is not None
        else VoiceStore.from_json(settings.voices_file, settings.root_dir)
    )
    engine = engine or CosyVoiceEngine(settings, voice_store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.voice_store = voice_store
        app.state.engine = engine
        app.state.admission_slots = asyncio.BoundedSemaphore(
            settings.max_concurrency + settings.max_queue_size
        )
        app.state.inference_slots = asyncio.Semaphore(settings.max_concurrency)
        if not engine.ready:
            try:
                await asyncio.to_thread(engine.load)
            except Exception:
                LOGGER.exception("CosyVoice model failed to load")
        yield

    app = FastAPI(
        title="CosyVoice API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "").strip()
        request.state.request_id = (
            supplied[:128] if supplied else str(uuid.uuid4())
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, error: ServiceError):
        return _error_response(request, error)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ):
        first = error.errors()[0] if error.errors() else {}
        location = first.get("loc", ())
        param = ".".join(str(part) for part in location if part != "body") or None
        message = first.get("msg", "Invalid request")
        return _error_response(
            request,
            ServiceError(
                400,
                message,
                code="invalid_parameter",
                param=param,
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception):
        LOGGER.exception("Unhandled API error", exc_info=error)
        return _error_response(
            request,
            ServiceError(
                500,
                "Internal server error",
                code="internal_error",
                error_type="server_error",
            ),
        )

    async def require_api_key(
        authorization: str | None = Header(default=None),
    ) -> None:
        if settings.api_key is None:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token, settings.api_key
        ):
            raise ServiceError(
                401,
                "Invalid or missing API key",
                code="invalid_api_key",
                error_type="authentication_error",
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        if not request.app.state.engine.ready:
            raise ServiceError(
                503,
                "CosyVoice model is not ready",
                code="model_not_ready",
                error_type="server_error",
            )
        return {
            "status": "ready",
            "model": settings.model_alias,
            "sample_rate": request.app.state.engine.sample_rate,
        }

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def list_models(request: Request):
        sample_rate = request.app.state.engine.sample_rate
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_alias,
                    "object": "model",
                    "owned_by": "local",
                    "ready": request.app.state.engine.ready,
                    "sample_rate": sample_rate,
                    "response_formats": ["wav", "pcm"],
                }
            ],
        }

    @app.get("/v1/audio/voices", dependencies=[Depends(require_api_key)])
    async def list_voices(request: Request):
        return {
            "object": "list",
            "data": [
                voice.public_dict()
                for voice in request.app.state.voice_store.all()
            ],
        }

    @app.post("/v1/audio/speech", dependencies=[Depends(require_api_key)])
    async def create_speech(payload: SpeechRequest, request: Request):
        if not request.app.state.engine.ready:
            raise ServiceError(
                503,
                "CosyVoice model is not ready",
                code="model_not_ready",
                error_type="server_error",
            )
        if payload.model != settings.model_alias:
            raise ServiceError(
                404,
                f"Model {payload.model!r} was not found",
                code="model_not_found",
                param="model",
            )
        if len(payload.input) > settings.max_text_characters:
            raise ServiceError(
                413,
                f"input exceeds {settings.max_text_characters} characters",
                code="input_too_large",
                param="input",
            )
        voice = request.app.state.voice_store.get(payload.voice)
        if voice is None:
            raise ServiceError(
                404,
                f"Voice {payload.voice!r} was not found",
                code="voice_not_found",
                param="voice",
            )

        admission = request.app.state.admission_slots
        try:
            await asyncio.wait_for(admission.acquire(), timeout=0.01)
        except TimeoutError as exc:
            raise ServiceError(
                429,
                "The inference queue is full",
                code="queue_full",
                error_type="rate_limit_error",
            ) from exc

        queued_at = time.perf_counter()
        try:
            async def run_inference():
                async with request.app.state.inference_slots:
                    queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
                    result = await asyncio.to_thread(
                        request.app.state.engine.synthesize, payload, voice
                    )
                    return result, queue_wait_ms

            try:
                result, queue_wait_ms = await asyncio.wait_for(
                    run_inference(), timeout=settings.request_timeout_seconds
                )
            except TimeoutError as exc:
                raise ServiceError(
                    504,
                    "Speech synthesis timed out",
                    code="inference_timeout",
                    error_type="server_error",
                ) from exc
        finally:
            admission.release()

        LOGGER.info(
            "request_id=%s model=%s voice=%s chars=%d queue_wait_ms=%.2f "
            "inference_ms=%.2f audio_ms=%.2f rtf=%.4f",
            _request_id(request),
            payload.model,
            payload.voice,
            len(payload.input),
            queue_wait_ms,
            result.inference_seconds * 1000.0,
            result.duration_seconds * 1000.0,
            result.real_time_factor,
        )
        suffix = "wav" if payload.response_format == "wav" else "pcm"
        return Response(
            content=result.content,
            media_type=result.media_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{suffix}"',
                "X-Audio-Sample-Rate": str(result.sample_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Duration": f"{result.duration_seconds:.6f}",
                "X-Queue-Wait-Ms": f"{queue_wait_ms:.3f}",
                "X-Inference-Latency-Ms": (
                    f"{result.inference_seconds * 1000.0:.3f}"
                ),
                "X-Real-Time-Factor": f"{result.real_time_factor:.6f}",
            },
        )

    return app
