"""FastAPI application and Roboflow-compatible endpoint."""

from __future__ import annotations

import base64
import binascii
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request

from .backend import UltralyticsBackend
from .config import Settings
from .engine import InferenceEngine, QueueFullError, QueueWaitTimeoutError

BackendFactory = Callable[[Settings], Any]


def default_backend_factory(settings: Settings) -> UltralyticsBackend:
    return UltralyticsBackend(
        weight_path=settings.weight_path,
        confidence=settings.confidence,
        iou=settings.iou,
        image_size=settings.image_size,
    )


def create_app(
    settings: Settings | None = None,
    backend_factory: BackendFactory = default_backend_factory,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        backend = backend_factory(settings)
        engine = InferenceEngine(
            backend=backend,
            max_queue_size=settings.max_queue_size,
            wait_timeout=settings.queue_timeout_seconds,
            batch_size=settings.batch_size,
            batch_wait_ms=settings.batch_wait_ms,
        )
        await engine.start()
        app.state.engine = engine
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(title="HC Vision API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/")
    async def index() -> dict[str, Any]:
        engine: InferenceEngine | None = getattr(app.state, "engine", None)
        return {
            "service": "HC Vision API",
            "status": "ready" if engine is not None else "starting",
            "model_id": settings.model_id,
            "health": "/healthz",
            "docs": "/docs",
        }

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        engine: InferenceEngine | None = getattr(app.state, "engine", None)
        return {
            "status": "ok",
            "model_id": settings.model_id,
            "ready": engine is not None,
            "inference": engine.metrics() if engine is not None else None,
        }

    @app.get("/readyz")
    async def ready() -> dict[str, str]:
        if getattr(app.state, "engine", None) is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return {"status": "ready"}

    @app.post("/{model_id:path}")
    async def infer(
        model_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        request_started = perf_counter()
        if model_id.strip("/") != settings.model_id:
            raise HTTPException(status_code=404, detail="Unknown model")
        if settings.api_key and authorization != f"Bearer {settings.api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")

        body = await request.body()
        if len(body) > settings.max_body_bytes:
            raise HTTPException(status_code=413, detail="Image payload is too large")
        try:
            encoded = body.strip()
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(status_code=400, detail="Body must be a base64 image") from error

        decode_started = perf_counter()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        decode_ms = (perf_counter() - decode_started) * 1000.0
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        engine: InferenceEngine = app.state.engine
        try:
            result = await engine.infer(image)
            timing = result.setdefault("timing", {})
            timing["decode_ms"] = round(decode_ms, 2)
            timing["server_total_ms"] = round(
                (perf_counter() - request_started) * 1000.0,
                2,
            )
            return result
        except QueueFullError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        except QueueWaitTimeoutError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


app = create_app()

