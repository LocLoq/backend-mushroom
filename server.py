import asyncio
import hashlib
from io import BytesIO
import imghdr
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4
import time
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

app = FastAPI(title="ML Image Queue API")

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"jpeg", "png", "gif", "bmp", "webp", "tiff"}
MODEL_PATH = os.getenv("MODEL_PATH", "nammushroom_efficientnet_b0.pth")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")

job_queue: asyncio.Queue[str] = asyncio.Queue()
jobs: dict[str, dict[str, Any]] = {}
job_payloads: dict[str, bytes] = {}
jobs_lock = asyncio.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _detect_image_type(image_bytes: bytes) -> str | None:
    return imghdr.what(None, h=image_bytes)


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._connections)

        if not connections:
            return

        dead_connections: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        if dead_connections:
            async with self._lock:
                for connection in dead_connections:
                    self._connections.discard(connection)


ws_manager = WebSocketManager()


async def _build_queue_snapshot() -> dict[str, Any]:
    async with jobs_lock:
        queued = sum(1 for item in jobs.values() if item["status"] == "queued")
        processing = sum(1 for item in jobs.values() if item["status"] == "processing")
        completed = sum(1 for item in jobs.values() if item["status"] == "completed")
        failed = sum(1 for item in jobs.values() if item["status"] == "failed")

    return {
        "queue_size": job_queue.qsize(),
        "queued": queued,
        "processing": processing,
        "completed": completed,
        "failed": failed,
        "total": queued + processing + completed + failed,
    }


async def _broadcast_event(event: str, data: dict[str, Any]) -> None:
    await ws_manager.broadcast({"event": event, "timestamp": _now_iso(), "data": data})


def _load_class_names_from_file(labels_path: str) -> list[str]:
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Khong tim thay file label tai duong dan: {labels_path}")

    class_names: list[str] = []
    with open(labels_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Ho tro dinh dang co index: "0,label", "0:label", "0\tlabel"
            parsed = line
            for separator in (",", ":", "\t"):
                if separator in line:
                    left, right = line.split(separator, 1)
                    if left.strip().isdigit() and right.strip():
                        parsed = right.strip()
                    break

            class_names.append(parsed)

    if not class_names:
        raise ValueError(
            f"File label rong hoac khong hop le: {labels_path}. Moi dong nen la ten lop, hoac dang 'index,label'."
        )

    return class_names


def _extract_state_dict_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict

        if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()) and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            return checkpoint

    raise ValueError(
        "Khong doc duoc state_dict tu file .pth. Hay luu model theo state_dict hoac checkpoint co key state_dict/model_state_dict."
    )


def _infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    weight_key = "classifier.1.weight"
    weight = state_dict.get(weight_key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])

    bias_key = "classifier.1.bias"
    bias = state_dict.get(bias_key)
    if isinstance(bias, torch.Tensor) and bias.ndim == 1:
        return int(bias.shape[0])

    return None


def _load_model_and_labels(model_path: str, device: torch.device) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    class_names = _load_class_names_from_file(LABELS_PATH)
    state_dict = _extract_state_dict_from_checkpoint(checkpoint)

    expected_num_classes = _infer_num_classes_from_state_dict(state_dict)
    if expected_num_classes is not None and len(class_names) != expected_num_classes:
        raise ValueError(
            f"So label trong file ({len(class_names)}) khong khop so lop cua model ({expected_num_classes})."
            f" Hay cap nhat file label: {LABELS_PATH}"
        )

    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model, class_names


async def _run_model_inference(image_bytes: bytes, image_type: str) -> dict[str, Any]:
    start_time = time.perf_counter()

    model_path = MODEL_PATH
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Khong tim thay model .pth tai duong dan: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = _load_model_and_labels(model_path=model_path, device=device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(BytesIO(image_bytes)).convert('RGB')


    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)
        
        probabilities = F.softmax(outputs[0], dim=0)

        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_label = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return {
        "prediction": predicted_label,
        "confidence": confidence_percent,
        "image_type": image_type,
        "size_bytes": len(image_bytes),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "inference_time_seconds": elapsed_time,
    }


async def _process_job(job_id: str) -> None:
    async with jobs_lock:
        job = jobs[job_id]
        job["status"] = "processing"
        job["updated_at"] = _now_iso()

    await _broadcast_event(
        "job.status",
        {
            "job_id": job_id,
            "status": "processing",
        },
    )
    await _broadcast_event("queue.status", await _build_queue_snapshot())

    image_bytes = job_payloads.get(job_id)
    if image_bytes is None:
        async with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Không tìm thấy dữ liệu ảnh để xử lý."
            jobs[job_id]["updated_at"] = _now_iso()

        await _broadcast_event(
            "job.result",
            {
                "job_id": job_id,
                "status": "failed",
                "error": "Không tìm thấy dữ liệu ảnh để xử lý.",
            },
        )
        return

    try:
        async with jobs_lock:
            image_type = jobs[job_id]["image_info"]["image_type"]

        result = await _run_model_inference(image_bytes=image_bytes, image_type=image_type)

        async with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result
            jobs[job_id]["updated_at"] = _now_iso()

        await _broadcast_event(
            "job.result",
            {
                "job_id": job_id,
                "status": "completed",
                "result": result,
            },
        )
    except Exception as exc:
        async with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)
            jobs[job_id]["updated_at"] = _now_iso()

        await _broadcast_event(
            "job.result",
            {
                "job_id": job_id,
                "status": "failed",
                "error": str(exc),
            },
        )
    finally:
        job_payloads.pop(job_id, None)
        await _broadcast_event("queue.status", await _build_queue_snapshot())


async def _queue_worker() -> None:
    while True:
        job_id = await job_queue.get()
        try:
            await _process_job(job_id)
        finally:
            job_queue.task_done()


@app.on_event("startup")
async def startup_event() -> None:
    app.state.queue_worker_task = asyncio.create_task(_queue_worker())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    task: asyncio.Task | None = getattr(app.state, "queue_worker_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@app.get("/")
def default_route() -> dict[str, str]:
    return {"status": "running"}


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải là ảnh hợp lệ (MIME type).")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Tệp ảnh rỗng.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Kích thước ảnh vượt quá giới hạn {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    image_type = _detect_image_type(image_bytes)
    if image_type is None or image_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Nội dung tệp không phải định dạng ảnh được hỗ trợ hoặc tệp bị hỏng.",
        )

    job_id = str(uuid4())
    created_at = _now_iso()

    async with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": file.filename,
            "created_at": created_at,
            "updated_at": created_at,
            "result": None,
            "error": None,
            "image_info": {
                "image_type": image_type,
                "content_type": file.content_type,
                "size_bytes": len(image_bytes),
            },
        }

    job_payloads[job_id] = image_bytes
    await job_queue.put(job_id)

    await _broadcast_event(
        "job.status",
        {
            "job_id": job_id,
            "status": "queued",
        },
    )
    await _broadcast_event("queue.status", await _build_queue_snapshot())

    return {
        "job_id": job_id,
        "status": "queued",
        "image_info": {
            "filename": file.filename,
            "content_type": file.content_type,
            "detected_image_type": image_type,
            "size_bytes": len(image_bytes),
        },
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    async with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job.")

    return job


@app.websocket("/ws/queue")
async def queue_websocket(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    await websocket.send_json(
        {
            "event": "queue.snapshot",
            "timestamp": _now_iso(),
            "data": await _build_queue_snapshot(),
        }
    )

    try:
        while True:
            message = await websocket.receive_text()
            if message.strip().lower() == "ping":
                await websocket.send_json({"event": "pong", "timestamp": _now_iso()})
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)

