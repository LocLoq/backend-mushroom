import asyncio
import hashlib
import imghdr
import json
import os
import unicodedata
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
QUEUE_IMAGE_DIR = os.getenv("QUEUE_IMAGE_DIR", "queue_images")
MUSHROOM_CATALOG_PATH = os.getenv("MUSHROOM_CATALOG_PATH", "mushroom.json")

POISONOUS_CONFIDENCE_THRESHOLD = 60.0
NON_POISONOUS_CONFIDENCE_THRESHOLD = 80.0
UNKNOWN_PREDICTION_LABEL = "unknown"

job_queue: asyncio.Queue[str] = asyncio.Queue()
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = asyncio.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _detect_image_type(image_bytes: bytes) -> str | None:
    return imghdr.what(None, h=image_bytes)


def _build_temp_image_path(job_id: str, image_type: str) -> str:
    extension = "jpg" if image_type == "jpeg" else image_type
    return os.path.join(QUEUE_IMAGE_DIR, f"{job_id}.{extension}")


def _build_staging_image_path(job_id: str) -> str:
    return os.path.join(QUEUE_IMAGE_DIR, f"{job_id}.upload")


def _delete_temp_image(temp_path: str) -> None:
    try:
        os.remove(temp_path)
    except FileNotFoundError:
        pass


def _cleanup_stale_temp_images() -> None:
    if not os.path.isdir(QUEUE_IMAGE_DIR):
        return

    for entry in os.scandir(QUEUE_IMAGE_DIR):
        if not entry.is_file():
            continue
        try:
            os.remove(entry.path)
        except OSError:
            continue


async def _save_upload_to_temp_file(file: UploadFile, job_id: str) -> tuple[str, str, int]:
    os.makedirs(QUEUE_IMAGE_DIR, exist_ok=True)
    staging_path = _build_staging_image_path(job_id)

    total_size = 0
    sniff_buffer = bytearray()

    try:
        with open(staging_path, "wb") as output_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_size += len(chunk)
                if total_size > MAX_IMAGE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Kích thước ảnh vượt quá giới hạn {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB.",
                    )

                if len(sniff_buffer) < 4096:
                    remaining = 4096 - len(sniff_buffer)
                    sniff_buffer.extend(chunk[:remaining])

                output_file.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Tệp ảnh rỗng.")

        image_type = _detect_image_type(bytes(sniff_buffer))
        if image_type is None or image_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Nội dung tệp không phải định dạng ảnh được hỗ trợ hoặc tệp bị hỏng.",
            )

        final_path = _build_temp_image_path(job_id, image_type)
        os.replace(staging_path, final_path)
        return final_path, image_type, total_size
    except Exception:
        _delete_temp_image(staging_path)
        raise


def _compute_file_sha256_and_size(file_path: str) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total_size = 0
    with open(file_path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)
            total_size += len(chunk)
    return hasher.hexdigest(), total_size


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    ascii_only = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(ascii_only.split())


def _normalize_catalog_key(key: str) -> str:
    return _normalize_text(key).replace(" ", "_").replace("-", "_")


def _parse_bool_field(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None

    if isinstance(value, str):
        parsed = _normalize_text(value)
        if parsed in {"true", "1", "yes", "y", "co", "doc", "poisonous", "toxic"}:
            return True
        if parsed in {"false", "0", "no", "n", "khong", "safe", "non-toxic", "nontoxic"}:
            return False

    return None


def _first_non_empty_string(item: dict[str, Any], accepted_keys: set[str]) -> str | None:
    for key, value in item.items():
        if _normalize_catalog_key(key) not in accepted_keys:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_value(item: dict[str, Any], accepted_keys: set[str]) -> Any:
    for key, value in item.items():
        if _normalize_catalog_key(key) in accepted_keys:
            return value
    return None


def _load_mushroom_catalog(catalog_path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(catalog_path):
        raise FileNotFoundError(f"Khong tim thay file danh muc nam tai duong dan: {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as file:
        raw_catalog = json.load(file)

    if not isinstance(raw_catalog, list):
        raise ValueError("File danh muc nam phai co dinh dang JSON array.")

    name_keys = {"ten_nam", "name", "mushroom_name", "label"}
    scientific_name_keys = {"ten_khoa_hoc", "scientific_name", "scientificname", "latin_name"}
    poisonous_keys = {"co_doc", "is_poisonous", "poisonous", "toxic"}

    parsed_catalog: list[dict[str, Any]] = []
    for index, item in enumerate(raw_catalog):
        if not isinstance(item, dict):
            raise ValueError(f"Muc thu {index} trong danh muc khong phai object.")

        mushroom_name = _first_non_empty_string(item, name_keys)
        scientific_name = _first_non_empty_string(item, scientific_name_keys)
        poisonous_value = _first_value(item, poisonous_keys)
        is_poisonous = _parse_bool_field(poisonous_value)

        if mushroom_name is None:
            raise ValueError(f"Muc thu {index} thieu truong ten nam.")
        if scientific_name is None:
            raise ValueError(f"Muc thu {index} thieu truong ten khoa hoc.")
        if is_poisonous is None:
            raise ValueError(f"Muc thu {index} co truong co doc khong hop le.")

        parsed_catalog.append(
            {
                "name": mushroom_name,
                "scientific_name": scientific_name,
                "is_poisonous": is_poisonous,
            }
        )

    if not parsed_catalog:
        raise ValueError("File danh muc nam rong.")

    return parsed_catalog


def _build_poisonous_lookup(catalog: list[dict[str, Any]]) -> dict[str, bool]:
    lookup: dict[str, bool] = {}
    for item in catalog:
        name_key = _normalize_text(item["name"])
        scientific_name_key = _normalize_text(item["scientific_name"])
        lookup[name_key] = bool(item["is_poisonous"])
        lookup[scientific_name_key] = bool(item["is_poisonous"])
    return lookup


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


async def _run_model_inference(image_path: str, image_type: str) -> dict[str, Any]:
    start_time = time.perf_counter()

    model_path = MODEL_PATH
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Khong tim thay model .pth tai duong dan: {model_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Khong tim thay tep anh tam thoi: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = _load_model_and_labels(model_path=model_path, device=device)
    catalog = _load_mushroom_catalog(MUSHROOM_CATALOG_PATH)
    poisonous_lookup = _build_poisonous_lookup(catalog)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with Image.open(image_path) as input_image:
        image = input_image.convert("RGB")


    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)
        
        probabilities = F.softmax(outputs[0], dim=0)

        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_label = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100
    label_key = _normalize_text(predicted_label)
    # get mushroom name by label key
    name = catalog[predicted_idx.item()]["name"] if predicted_idx.item() < len(catalog) else None
    is_poisonous = poisonous_lookup.get(label_key)

    accepted_prediction = False
    required_confidence: float | None = None
    decision_reason = ""
    if is_poisonous is None:
        final_prediction = UNKNOWN_PREDICTION_LABEL
        decision_reason = "Không tìm thấy thông tin độc tính cho nhãn dự đoán trong danh mục JSON."
    else:
        required_confidence = (
            POISONOUS_CONFIDENCE_THRESHOLD if is_poisonous else NON_POISONOUS_CONFIDENCE_THRESHOLD
        )
        if confidence_percent >= required_confidence:
            final_prediction = predicted_label
            accepted_prediction = True
            decision_reason = "Độ tin cậy đạt ngưỡng nên chấp nhận kết quả dự đoán."
        else:
            final_prediction = UNKNOWN_PREDICTION_LABEL
            decision_reason = "Độ tin cậy thấp hơn ngưỡng cần thiết nên không chấp nhận kết quả dự đoán."

    sha256, size_bytes = _compute_file_sha256_and_size(image_path)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return {
        "mushroom_name": name,
        "prediction": final_prediction,
        "raw_prediction": predicted_label,
        "accepted_prediction": accepted_prediction,
        "confidence": confidence_percent,
        "confidence_threshold": required_confidence,
        "is_poisonous": is_poisonous,
        "decision_reason": decision_reason,
        "image_type": image_type,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "inference_time_seconds": elapsed_time,
    }


async def _process_job(job_id: str) -> None:
    async with jobs_lock:
        job = jobs[job_id]
        job["status"] = "processing"
        job["updated_at"] = _now_iso()
        image_type = job["image_info"]["image_type"]
        temp_path = job["image_info"].get("temp_path")

    await _broadcast_event(
        "job.status",
        {
            "job_id": job_id,
            "status": "processing",
        },
    )
    await _broadcast_event("queue.status", await _build_queue_snapshot())

    if temp_path is None:
        async with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Không tìm thấy tệp ảnh tạm để xử lý."
            jobs[job_id]["updated_at"] = _now_iso()

        await _broadcast_event(
            "job.result",
            {
                "job_id": job_id,
                "status": "failed",
                "error": "Không tìm thấy tệp ảnh tạm để xử lý.",
            },
        )
        return

    try:
        result = await _run_model_inference(image_path=temp_path, image_type=image_type)

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
        _delete_temp_image(temp_path)
        async with jobs_lock:
            job = jobs.get(job_id)
            if job is not None and "image_info" in job:
                job["image_info"]["temp_path"] = None
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
    os.makedirs(QUEUE_IMAGE_DIR, exist_ok=True)
    _cleanup_stale_temp_images()
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


@app.get("/api/mushrooms/catalog")
async def get_mushroom_catalog() -> dict[str, Any]:
    try:
        catalog = _load_mushroom_catalog(MUSHROOM_CATALOG_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Khong the doc danh muc nam: {exc}") from exc

    poisonous_mushrooms = [item for item in catalog if item["is_poisonous"]]

    return {
        "source": MUSHROOM_CATALOG_PATH,
        "total": len(catalog),
        "poisonous_count": len(poisonous_mushrooms),
        "safe_count": len(catalog) - len(poisonous_mushrooms),
        "mushrooms": catalog,
        "poisonous_mushrooms": poisonous_mushrooms,
    }


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải là ảnh hợp lệ (MIME type).")

    job_id = str(uuid4())
    created_at = _now_iso()
    try:
        temp_path, image_type, size_bytes = await _save_upload_to_temp_file(file=file, job_id=job_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể lưu ảnh tạm thời: {exc}") from exc

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
                "size_bytes": size_bytes,
                "temp_path": temp_path,
            },
        }

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
            "size_bytes": size_bytes,
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

