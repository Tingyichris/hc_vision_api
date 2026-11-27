"""Local deployment configuration. Edit this file for the target Linux host."""

from pathlib import Path

HOST = "0.0.0.0"
PORT = 7000
MODEL_ID = "-v4qph/-j7pjz-4-yolo26n-seg-t1"
API_KEY = ""
WEIGHT_PATH = Path.home() / "hc_vision_api" / "weights" / "weights.pt"
CONFIDENCE = 0.35
IOU = 0.50
IMAGE_SIZE = 640
MAX_QUEUE_SIZE = 32
QUEUE_TIMEOUT_SECONDS = 5.0
# Defaults only. Production can override these scalability controls with
# HC_VISION_BATCH_SIZE, HC_VISION_BATCH_WAIT_MS,
# HC_VISION_MAX_QUEUE_SIZE, and HC_VISION_QUEUE_TIMEOUT_SECONDS.
BATCH_SIZE = 11
BATCH_WAIT_MS = 10.0
MAX_BODY_BYTES = 16 * 1024 * 1024

