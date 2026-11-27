from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from hc_vision_api.engine import InferenceEngine, QueueFullError


class ConcurrencyBackend:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def predict(self, image: np.ndarray) -> dict:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return {"predictions": [], "image": {"width": 1, "height": 1}}


class BlockingBackend:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def predict(self, image: np.ndarray) -> dict:
        self.started.set()
        assert self.release.wait(timeout=2)
        return {"predictions": [], "image": {"width": 1, "height": 1}}


class BatchBackend:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, image: np.ndarray) -> dict:
        raise AssertionError("batch path should be used")

    def predict_batch(self, images: list[np.ndarray]) -> list[dict]:
        self.batch_sizes.append(len(images))
        return [
            {"predictions": [], "image": {"width": int(image.shape[1]), "height": int(image.shape[0])}}
            for image in images
        ]


def test_worker_serializes_simultaneous_requests() -> None:
    async def scenario() -> None:
        backend = ConcurrencyBackend()
        engine = InferenceEngine(backend, max_queue_size=8, wait_timeout=2)
        await engine.start()
        try:
            image = np.zeros((1, 1, 3), dtype=np.uint8)
            await asyncio.gather(*(engine.infer(image) for _ in range(4)))
            assert backend.max_active == 1
            assert engine.processed == 4
            assert engine.failed == 0
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_worker_batches_simultaneous_requests_and_preserves_results() -> None:
    async def scenario() -> None:
        backend = BatchBackend()
        engine = InferenceEngine(
            backend,
            max_queue_size=8,
            wait_timeout=2,
            batch_size=4,
            batch_wait_ms=50,
        )
        await engine.start()
        try:
            images = [np.zeros((index + 1, index + 2, 3), dtype=np.uint8) for index in range(4)]
            results = await asyncio.gather(*(engine.infer(image) for image in images))
            assert backend.batch_sizes == [4]
            assert [result["image"]["height"] for result in results] == [1, 2, 3, 4]
            assert all(result["timing"]["batch_size"] == 4 for result in results)
            assert engine.processed == 4
            assert engine.metrics()["average_batch_size"] == 4.0
            assert engine.metrics()["max_observed_batch_size"] == 4
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_full_queue_rejects_instead_of_growing_unbounded() -> None:
    async def scenario() -> None:
        backend = BlockingBackend()
        engine = InferenceEngine(backend, max_queue_size=1, wait_timeout=2)
        await engine.start()
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        first = asyncio.create_task(engine.infer(image))
        assert await asyncio.to_thread(backend.started.wait, 1)
        second = asyncio.create_task(engine.infer(image))
        await asyncio.sleep(0)
        try:
            with pytest.raises(QueueFullError):
                await engine.infer(image)
        finally:
            backend.release.set()
            await first
            await second
            await engine.close()

    asyncio.run(scenario())
