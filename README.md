# HC Vision API

Local CPU inference service for the `plate_detect` Raspberry Pi clients. The
HTTP request and response format intentionally matches the existing Roboflow
client, so the device-side detection pipeline does not need to change.

## Setup

Use Python 3.11 on the inference computer:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Start the service:

```powershell
.\.venv\Scripts\hc-vision-api.exe
```

The defaults use `C:/Users/desig/Downloads/weights.pt` and listen on port
`9001`. Configuration is read from environment variables; `.env` can be
loaded by setting the variables in the shell or by using the supplied start
script.

```powershell
.\start.ps1
```

Health endpoints:

- `GET /healthz`: process and queue health.
- `GET /readyz`: whether the model is loaded and ready.

## Roboflow-compatible inference

The existing client sends a base64-encoded JPEG as the raw request body:

```http
POST /-v4qph/-j7pjz-4-yolo26n-seg-t1
Content-Type: application/x-www-form-urlencoded
Authorization: Bearer
```

Successful responses contain `predictions`, with center-based bounding boxes
and instance mask polygons in `points`. Requests wait in a bounded FIFO queue.
The service returns HTTP 429 when the queue is full and HTTP 503 when a request
waits too long.

The model performs one blank CPU warm-up inference during application startup.
`/readyz` becomes available only after that warm-up has completed, so the first
camera request does not pay the model initialization cost.


