import argparse
import asyncio
import contextlib
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx
import websockets
from websockets.client import WebSocketClientProtocol


async def upload_image(base_url: str, image_path: Path) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        with image_path.open("rb") as image_file:
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if mime_type is None:
                mime_type = "image/jpeg"

            files = {
                "file": (image_path.name, image_file, mime_type),
            }
            response = await client.post(f"{base_url}/api/images/upload", files=files)

    if response.is_error:
        detail: Any
        with contextlib.suppress(ValueError):
            detail = response.json()
            raise RuntimeError(
                f"Upload lỗi HTTP {response.status_code}. API trả về: {json.dumps(detail, ensure_ascii=False)}"
            )

        raise RuntimeError(f"Upload lỗi HTTP {response.status_code}. API trả về: {response.text}")

    data = response.json()

    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Upload response không có job_id: {data}")

    print("[UPLOAD] Thành công:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return job_id


async def poll_job_status(base_url: str, job_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout_seconds

    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            response = await client.get(f"{base_url}/api/jobs/{job_id}")
            response.raise_for_status()
            data = response.json()

            status = data.get("status")
            print(f"[POLL] job={job_id} status={status}")

            if status in {"completed", "failed"}:
                return data

            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Hết thời gian chờ job {job_id} hoàn tất.")

            await asyncio.sleep(1.0)


async def websocket_listener(
    ws_url: str,
    state: dict[str, Any],
    stop_event: asyncio.Event,
) -> None:
    async with websockets.connect(ws_url) as websocket:
        websocket = websocket  # type: WebSocketClientProtocol
        print(f"[WS] Đã kết nối: {ws_url}")

        ping_task = asyncio.create_task(_ping_loop(websocket, stop_event))

        try:
            while not stop_event.is_set():
                message = await websocket.recv()
                print(f"[WS] {message}")

                with contextlib.suppress(json.JSONDecodeError):
                    payload = json.loads(message)
                    event = payload.get("event")
                    data = payload.get("data", {})

                    target_job_id = state.get("target_job_id")
                    if target_job_id and event == "job.result" and data.get("job_id") == target_job_id:
                        stop_event.set()
                        return
        finally:
            ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ping_task


async def _ping_loop(websocket: WebSocketClientProtocol, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(15)
        await websocket.send("ping")


async def run_test(server_ip: str, server_port: int, image_path: Path, timeout_seconds: int) -> None:
    base_url = f"http://{server_ip}:{server_port}"
    ws_url = f"ws://{server_ip}:{server_port}/ws/queue"

    stop_event = asyncio.Event()
    state: dict[str, Any] = {"target_job_id": None}

    ws_task = asyncio.create_task(websocket_listener(ws_url, state=state, stop_event=stop_event))

    try:
        await asyncio.sleep(0.3)
        job_id = await upload_image(base_url=base_url, image_path=image_path)
        state["target_job_id"] = job_id

        final_status = await poll_job_status(base_url=base_url, job_id=job_id, timeout_seconds=timeout_seconds)

        if not stop_event.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=5)

            if not stop_event.is_set():
                print("[WS] Không nhận được job.result trong 5s, tiếp tục bằng kết quả polling.")
                stop_event.set()

        print("\n[FINAL] Kết quả job:")
        print(json.dumps(final_status, indent=2, ensure_ascii=False))
    finally:
        ws_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ws_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Client test upload image + websocket queue/result cho ML Image Queue API"
    )
    parser.add_argument(
        "--server-ip",
        "--host",
        dest="server_ip",
        default="127.0.0.1",
        help="IP hoặc hostname của backend server",
    )
    parser.add_argument(
        "--server-port",
        "--port",
        dest="server_port",
        type=int,
        default=8000,
        help="Port của backend server",
    )
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh cần upload")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Số giây tối đa chờ job completed/failed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

    asyncio.run(
        run_test(
            server_ip=args.server_ip,
            server_port=args.server_port,
            image_path=image_path,
            timeout_seconds=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
