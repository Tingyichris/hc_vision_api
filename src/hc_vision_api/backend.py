"""Ultralytics model adapter producing Roboflow-compatible predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class UltralyticsBackend:
    def __init__(
        self,
        weight_path: Path,
        confidence: float,
        iou: float,
        image_size: int,
    ) -> None:
        if not weight_path.is_file():
            raise FileNotFoundError(f"Model weight not found: {weight_path}")

        from ultralytics import YOLO

        self.weight_path = weight_path
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.model = YOLO(str(weight_path), task="segment")
        # Complete graph setup before the service reports ready. This keeps the
        # first Raspberry Pi request from paying the several-second cold-start
        # cost observed on CPU.
        self.model.predict(
            source=np.zeros((image_size, image_size, 3), dtype=np.uint8),
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            device="cpu",
            verbose=False,
        )

    @staticmethod
    def _format_result(image: np.ndarray, result: Any | None) -> dict[str, Any]:
        height, width = image.shape[:2]
        if result is None:
            return {"predictions": [], "image": {"width": width, "height": height}}

        boxes = result.boxes
        polygons = result.masks.xy if result.masks is not None else []
        predictions: list[dict[str, Any]] = []

        if boxes is not None:
            xywh = boxes.xywh.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for index, (box, confidence, class_id) in enumerate(
                zip(xywh, confidences, class_ids, strict=True)
            ):
                polygon = polygons[index] if index < len(polygons) else np.empty((0, 2))
                points = [
                    {"x": float(point[0]), "y": float(point[1])}
                    for point in np.asarray(polygon)
                ]
                class_name = str(result.names.get(int(class_id), class_id))
                predictions.append(
                    {
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "width": float(box[2]),
                        "height": float(box[3]),
                        "confidence": float(confidence),
                        "class": class_name,
                        "class_id": int(class_id),
                        "points": points,
                    }
                )

        return {
            "predictions": predictions,
            "image": {"width": width, "height": height},
        }

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )
        return self._format_result(image, results[0] if results else None)

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        if not images:
            return []
        results = self.model.predict(
            source=images,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )
        if len(results) != len(images):
            raise RuntimeError(
                f"Batch inference returned {len(results)} results for {len(images)} images"
            )
        return [
            self._format_result(image, result)
            for image, result in zip(images, results, strict=True)
        ]

