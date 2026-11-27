from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from hc_vision_api.app import create_app
from hc_vision_api.config import Settings


class FakeBackend:
    def predict(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        return {
            "predictions": [
                {
                    "x": width / 2,
                    "y": height / 2,
                    "width": width / 2,
                    "height": height / 2,
                    "confidence": 0.9,
                    "class": "plate",
                    "class_id": 0,
                    "points": [{"x": 1.0, "y": 1.0}],
                }
            ],
            "image": {"width": width, "height": height},
        }


def settings(**overrides) -> Settings:
    values = {
        "weight_path": Path("unused.pt"),
        "model_id": "workspace/plate-model",
        "api_key": "",
        "host": "127.0.0.1",
        "port": 9001,
        "confidence": 0.35,
        "iou": 0.5,
        "image_size": 640,
        "max_queue_size": 2,
        "queue_timeout_seconds": 1.0,
        "batch_size": 4,
        "batch_wait_ms": 5.0,
        "max_body_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return Settings(**values)


def encoded_image() -> bytes:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(encoded.tobytes())


def test_roboflow_compatible_request_and_response() -> None:
    app = create_app(settings(), backend_factory=lambda _: FakeBackend())
    with TestClient(app) as client:
        response = client.post(
            "/workspace/plate-model",
            content=encoded_image(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["image"] == {"width": 60, "height": 40}
    assert body["predictions"][0]["class"] == "plate"
    assert body["predictions"][0]["points"]


def test_index_shows_service_links() -> None:
    app = create_app(settings(), backend_factory=lambda _: FakeBackend())
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "HC Vision API"
    assert response.json()["docs"] == "/docs"


def test_unknown_model_is_rejected() -> None:
    app = create_app(settings(), backend_factory=lambda _: FakeBackend())
    with TestClient(app) as client:
        response = client.post("/wrong/model", content=encoded_image())
    assert response.status_code == 404


def test_api_key_is_optional_but_enforced_when_configured() -> None:
    app = create_app(settings(api_key="secret"), backend_factory=lambda _: FakeBackend())
    with TestClient(app) as client:
        denied = client.post("/workspace/plate-model", content=encoded_image())
        allowed = client.post(
            "/workspace/plate-model",
            content=encoded_image(),
            headers={"Authorization": "Bearer secret"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_invalid_base64_is_rejected() -> None:
    app = create_app(settings(), backend_factory=lambda _: FakeBackend())
    with TestClient(app) as client:
        response = client.post("/workspace/plate-model", content=b"not base64!")
    assert response.status_code == 400


def test_scalability_settings_can_be_overridden_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("HC_VISION_BATCH_SIZE", "16")
    monkeypatch.setenv("HC_VISION_BATCH_WAIT_MS", "7.5")
    monkeypatch.setenv("HC_VISION_MAX_QUEUE_SIZE", "96")
    monkeypatch.setenv("HC_VISION_QUEUE_TIMEOUT_SECONDS", "8.0")

    configured = Settings.from_env()

    assert configured.batch_size == 16
    assert configured.batch_wait_ms == 7.5
    assert configured.max_queue_size == 96
    assert configured.queue_timeout_seconds == 8.0

