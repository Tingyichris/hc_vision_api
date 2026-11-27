"""Bounded single-worker queue with short-window batch inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np


class Backend(Protocol):
    def predict(self, image: np.ndarray) -> dict[str, Any]: ...


@dataclass(slots=True)
class WorkItem:
    image: np.ndarray
    future: asyncio.Future[dict[str, Any]]
    queued_at: float


class QueueFullError(RuntimeError):
    pass


class QueueWaitTimeoutError(RuntimeError):
    pass


class InferenceEngine:
    def __init__(
        self,
        backend: Backend,
        max_queue_size: int,
        wait_timeout: float,
        batch_size: int = 1,
        batch_wait_ms: float = 0.0,
    ) -> None:
        self.backend = backend
        self.queue: asyncio.Queue[WorkItem | None] = asyncio.Queue(maxsize=max_queue_size)
        self.wait_timeout = wait_timeout
        self.batch_size = max(1, batch_size)
        self.batch_wait_seconds = max(0.0, batch_wait_ms) / 1000.0
        self.worker_task: asyncio.Task[None] | None = None
        self.processed = 0
        self.failed = 0
        self.last_inference_ms = 0.0
        self.last_queue_wait_ms = 0.0
        self.last_batch_size = 0
        self.batches_processed = 0
        self.total_batch_items = 0
        self.max_observed_batch_size = 0

    async def start(self) -> None:
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker(), name="inference-worker")

    async def close(self) -> None:
        if self.worker_task is None:
            return
        await self.queue.put(None)
        await self.worker_task
        self.worker_task = None

    async def infer(self, image: np.ndarray) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        item = WorkItem(image=image, future=future, queued_at=perf_counter())
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull as error:
            raise QueueFullError("Inference queue is full") from error

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=self.wait_timeout)
        except TimeoutError as error:
            future.cancel()
            raise QueueWaitTimeoutError("Inference queue wait timed out") from error

    def _predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        predict_batch = getattr(self.backend, "predict_batch", None)
        if callable(predict_batch):
            return list(predict_batch(images))
        return [self.backend.predict(image) for image in images]

    async def _collect_batch(self, first: WorkItem) -> tuple[list[WorkItem], bool]:
        batch = [first]
        stop_after_batch = False
        deadline = asyncio.get_running_loop().time() + self.batch_wait_seconds
        while len(batch) < self.batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0.0 and self.queue.empty():
                break
            try:
                if self.queue.empty():
                    item = await asyncio.wait_for(self.queue.get(), timeout=max(0.0, remaining))
                else:
                    item = self.queue.get_nowait()
            except (asyncio.QueueEmpty, TimeoutError):
                break
            if item is None:
                self.queue.task_done()
                stop_after_batch = True
                break
            batch.append(item)
        return batch, stop_after_batch

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            batch, stop_after_batch = await self._collect_batch(item)
            try:
                active = [entry for entry in batch if not entry.future.cancelled()]
                if not active:
                    if stop_after_batch:
                        return
                    continue
                queue_wait_by_item = {
                    id(entry): (perf_counter() - entry.queued_at) * 1000.0
                    for entry in active
                }
                self.last_queue_wait_ms = max(queue_wait_by_item.values())
                self.last_batch_size = len(active)
                started = perf_counter()
                try:
                    results = await asyncio.to_thread(
                        self._predict_batch,
                        [entry.image for entry in active],
                    )
                    if len(results) != len(active):
                        raise RuntimeError(
                            f"Batch backend returned {len(results)} results "
                            f"for {len(active)} images"
                        )
                    inference_ms = (perf_counter() - started) * 1000.0
                    self.last_inference_ms = inference_ms
                    self.batches_processed += 1
                    self.total_batch_items += len(active)
                    self.max_observed_batch_size = max(
                        self.max_observed_batch_size,
                        len(active),
                    )
                    completed_at = perf_counter()
                    for entry, raw_result in zip(active, results, strict=True):
                        result = dict(raw_result)
                        existing_timing = result.get("timing")
                        timing = dict(existing_timing) if isinstance(existing_timing, dict) else {}
                        timing.update(
                            {
                                "queue_wait_ms": round(queue_wait_by_item[id(entry)], 2),
                                "inference_ms": round(inference_ms, 2),
                                "engine_total_ms": round(
                                    (completed_at - entry.queued_at) * 1000.0,
                                    2,
                                ),
                                "batch_size": len(active),
                            }
                        )
                        result["timing"] = timing
                        if not entry.future.done():
                            entry.future.set_result(result)
                    self.processed += len(active)
                except Exception as error:
                    self.failed += len(active)
                    for entry in active:
                        if not entry.future.done():
                            entry.future.set_exception(error)
            finally:
                for _entry in batch:
                    self.queue.task_done()
            if stop_after_batch:
                return

    def metrics(self) -> dict[str, int | float]:
        average_batch_size = (
            self.total_batch_items / self.batches_processed
            if self.batches_processed
            else 0.0
        )
        return {
            "queue_size": self.queue.qsize(),
            "queue_capacity": self.queue.maxsize,
            "processed": self.processed,
            "failed": self.failed,
            "last_queue_wait_ms": round(self.last_queue_wait_ms, 2),
            "last_inference_ms": round(self.last_inference_ms, 2),
            "last_batch_size": self.last_batch_size,
            "batch_capacity": self.batch_size,
            "batch_wait_ms": round(self.batch_wait_seconds * 1000.0, 2),
            "batches_processed": self.batches_processed,
            "average_batch_size": round(average_batch_size, 2),
            "max_observed_batch_size": self.max_observed_batch_size,
        }

