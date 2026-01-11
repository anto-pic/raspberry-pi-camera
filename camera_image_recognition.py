#!/usr/bin/env python3
import io
import threading
import time
from typing import List, Tuple

from fastapi import FastAPI, Response, HTTPException, Query
from fastapi.responses import StreamingResponse

app = FastAPI(title="Raspberry Pi 5 Camera API (Snapshot + MJPEG + YOLOv8n)")

_cam_lock = threading.Lock()
_picam2 = None
_started = False

# YOLO
_model = None
_model_lock = threading.Lock()

# cache ultimi box per stream fluido
_last_boxes_lock = threading.Lock()
_last_boxes: List[Tuple[int, int, int, int, str, float]] = []
_last_infer_ts: float = 0.0


def init_camera():
    """
    IDENTICO al tuo script originale (questo è fondamentale per i colori).
    """
    global _picam2, _started
    if _started:
        return
    try:
        from picamera2 import Picamera2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Picamera2 non disponibile. Installa: sudo apt-get install -y python3-picamera2"
        ) from e

    _picam2 = Picamera2()

    video_config = _picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls={"FrameRate": 15},
    )
    _picam2.configure(video_config)
    _picam2.start()
    time.sleep(0.2)
    _started = True


def init_yolo(weights_path: str = "./yolov8n.pt"):
    global _model
    if _model is not None:
        return
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError("ultralytics non installato nel venv. Fai: pip install ultralytics") from e

    print(f"[YOLO] Loading: {weights_path}")
    _model = YOLO(weights_path)
    print("[YOLO] Loaded OK")


def capture_jpeg() -> bytes:
    """
    JPEG da Picamera2 come nello script originale -> colori corretti.
    """
    init_camera()
    assert _picam2 is not None
    buf = io.BytesIO()
    _picam2.capture_file(buf, format="jpeg")
    return buf.getvalue()


def decode_jpeg_to_bgr(jpeg: bytes):
    """
    JPEG -> BGR (OpenCV standard)
    """
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img  # BGR


def encode_jpeg_from_bgr(img_bgr, quality: int) -> bytes:
    import cv2  # type: ignore

    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def run_inference_rgb(frame_rgb, conf: float, imgsz: int):
    """
    frame_rgb: numpy array RGB
    returns: list of (x1,y1,x2,y2,label,score)
    """
    assert _model is not None

    with _model_lock:
        results = _model.predict(frame_rgb, conf=conf, imgsz=imgsz, verbose=False)

    r = results[0]
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out

    names = r.names
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        score = float(b.conf[0].item()) if b.conf is not None else 0.0
        label = names.get(cls_id, str(cls_id))
        out.append((int(x1), int(y1), int(x2), int(y2), label, score))

    return out


def draw_boxes_bgr(img_bgr, boxes):
    """
    Disegno deterministico su BGR (niente plot() per evitare ambiguità colore).
    """
    import cv2  # type: ignore

    for x1, y1, x2, y2, label, score in boxes:
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text_top = max(0, y1 - th - baseline - 6)
        cv2.rectangle(img_bgr, (x1, y_text_top), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(
            img_bgr,
            text,
            (x1 + 3, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return img_bgr


@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/camera", "/camera/stream", "/health"]}


@app.get("/health")
def health():
    return {
        "camera_started": _started,
        "yolo_loaded": _model is not None,
        "last_infer_seconds_ago": None if _last_infer_ts == 0 else round(time.time() - _last_infer_ts, 2),
        "last_boxes_count": len(_last_boxes),
    }


@app.get("/camera")
def get_camera_snapshot(
    yolo: bool = Query(default=False),
    conf: float = Query(default=0.25, ge=0.01, le=0.99),
    imgsz: int = Query(default=640, ge=320, le=1280),
    q: int = Query(default=85, ge=10, le=95),
):
    import cv2  # type: ignore

    with _cam_lock:
        try:
            jpeg = capture_jpeg()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Se YOLO è disattivo -> IDENTICO al tuo script originale
    if not yolo:
        return Response(content=jpeg, media_type="image/jpeg")

    # YOLO ON
    try:
        init_yolo("./yolov8n.pt")
        img_bgr = decode_jpeg_to_bgr(jpeg)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        boxes = run_inference_rgb(img_rgb, conf=conf, imgsz=imgsz)
        if boxes:
            img_bgr = draw_boxes_bgr(img_bgr, boxes)

        out = encode_jpeg_from_bgr(img_bgr, quality=q)
        return Response(content=out, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO failed: {e}")


def mjpeg_generator(
    fps: int,
    quality: int,
    yolo: bool,
    conf: float,
    imgsz: int,
    infer_every: int,
):
    import cv2  # type: ignore

    boundary = b"--frame"
    delay = 1.0 / max(1, fps)

    frame_idx = 0
    global _last_boxes, _last_infer_ts

    while True:
        start = time.time()

        with _cam_lock:
            jpeg = capture_jpeg()

        if not yolo:
            frame = jpeg
        else:
            init_yolo("./yolov8n.pt")

            img_bgr = decode_jpeg_to_bgr(jpeg)

            frame_idx += 1
            if infer_every <= 1 or (frame_idx % infer_every == 0):
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                boxes = run_inference_rgb(img_rgb, conf=conf, imgsz=imgsz)
                with _last_boxes_lock:
                    _last_boxes = boxes
                    _last_infer_ts = time.time()

            with _last_boxes_lock:
                boxes_to_draw = list(_last_boxes)

            if boxes_to_draw:
                img_bgr = draw_boxes_bgr(img_bgr, boxes_to_draw)

            frame = encode_jpeg_from_bgr(img_bgr, quality=quality)

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8")
        yield frame + b"\r\n"

        elapsed = time.time() - start
        sleep_time = delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


@app.get("/camera/stream")
def camera_stream(
    fps: int = Query(default=10, ge=1, le=30),
    q: int = Query(default=80, ge=10, le=95),
    yolo: bool = Query(default=True),
    conf: float = Query(default=0.25, ge=0.01, le=0.99),
    imgsz: int = Query(default=640, ge=320, le=1280),
    infer_every: int = Query(default=2, ge=1, le=10),
):
    return StreamingResponse(
        mjpeg_generator(fps=fps, quality=q, yolo=yolo, conf=conf, imgsz=imgsz, infer_every=infer_every),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_yolo_colorsafe:app", host="0.0.0.0", port=8000, reload=False)
