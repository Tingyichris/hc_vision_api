"""Runtime settings loaded from constants.py."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from . import constants as const


@dataclass(frozen=True, slots=True)
class Settings:
    weight_path: Path
    model_id: str
    api_key: str
    host: str
    port: int
    confidence: float
    iou: float
    image_size: int
    max_queue_size: int
    queue_timeout_seconds: float
    batch_size: int
    batch_wait_ms: float
    max_body_bytes: int

    @classmethod
    def from_constants(cls) -> Settings:
        return cls(
            weight_path=Path(const.WEIGHT_PATH), model_id=str(const.MODEL_ID).strip("/"),
            api_key=str(const.API_KEY), host=str(const.HOST), port=int(const.PORT),
            confidence=float(const.CONFIDENCE), iou=float(const.IOU),
            image_size=int(const.IMAGE_SIZE), max_queue_size=int(const.MAX_QUEUE_SIZE),
            queue_timeout_seconds=float(const.QUEUE_TIMEOUT_SECONDS),
            batch_size=max(1, int(const.BATCH_SIZE)),
            batch_wait_ms=max(0.0, float(const.BATCH_WAIT_MS)),
            max_body_bytes=int(const.MAX_BODY_BYTES),
        )

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls.from_constants()
        return replace(
            settings,
            max_queue_size=max(
                1,
                int(os.getenv("HC_VISION_MAX_QUEUE_SIZE", settings.max_queue_size)),
            ),
            queue_timeout_seconds=max(
                0.1,
                float(
                    os.getenv(
                        "HC_VISION_QUEUE_TIMEOUT_SECONDS",
                        settings.queue_timeout_seconds,
                    )
                ),
            ),
            batch_size=max(
                1,
                int(os.getenv("HC_VISION_BATCH_SIZE", settings.batch_size)),
            ),
            batch_wait_ms=max(
                0.0,
                float(os.getenv("HC_VISION_BATCH_WAIT_MS", settings.batch_wait_ms)),
            ),
        )

