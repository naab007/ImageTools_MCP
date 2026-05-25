"""YOLOv8/v11 segmentation via Ultralytics.

Best perf/latency tradeoff in the 2026-05 benchmark: yolov8l-seg hit 0.77
median IoU at ~20 ms/call on a 4090, beating SAM 2.1 large at 8× the speed.

Variants:
- ``yolov8n-seg``  ~3 MB,  fastest
- ``yolov8s-seg``  ~13 MB
- ``yolov8m-seg``  ~28 MB
- ``yolov8l-seg``  ~46 MB, **default — recommended balance**
- ``yolov8x-seg``  ~71 MB, best quality
- ``yolo11n-seg`` / -s / -m / -l / -x — newer architecture, same speed tier

Idle eviction mirrors the SD/Qwen/SAM modules.
"""
from __future__ import annotations

import gc
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

_lock = threading.RLock()
_model: Any = None
_loaded_weights: str | None = None

_last_used: float | None = None
_idle_timeout_s: float = 3600.0
_SWEEP_INTERVAL_S: float = 300.0
_sweeper_thread: threading.Thread | None = None

DEFAULT_WEIGHTS = "yolov8l-seg.pt"


def _check_available() -> None:
    try:
        import ultralytics  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "YOLO segmentation is unavailable. Install with "
            "`pip install image-tools-mcp[yolo]`."
        ) from e


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _touch() -> None:
    global _last_used
    with _lock:
        _last_used = time.monotonic()


def _start_sweeper_if_needed() -> None:
    global _sweeper_thread
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return
    _sweeper_thread = threading.Thread(
        target=_sweep_loop, daemon=True, name="yolo-idle-sweeper",
    )
    _sweeper_thread.start()


def _sweep_loop() -> None:
    while True:
        time.sleep(_SWEEP_INTERVAL_S)
        try:
            _sweep_once()
        except Exception:
            pass


def _sweep_once() -> None:
    if _idle_timeout_s <= 0:
        return
    with _lock:
        if _model is None or _last_used is None:
            return
        if time.monotonic() - _last_used <= _idle_timeout_s:
            return
    unload()


def set_idle_timeout(seconds: float) -> dict[str, Any]:
    global _idle_timeout_s
    _idle_timeout_s = max(0.0, float(seconds))
    return {
        "idle_timeout_s": _idle_timeout_s,
        "sweep_interval_s": _SWEEP_INTERVAL_S,
        "auto_evict_enabled": _idle_timeout_s > 0,
    }


def _get_model(weights: str = DEFAULT_WEIGHTS):
    global _model, _loaded_weights
    with _lock:
        if _model is not None and _loaded_weights == weights:
            _touch()
            return _model
        if _model is not None:
            _model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        from ultralytics import YOLO
        _model = YOLO(weights)
        if _device() == "cuda":
            _model.to("cuda")
        _loaded_weights = weights
        _touch()
        active = _model
    _start_sweeper_if_needed()
    return active


def status() -> dict[str, Any]:
    try:
        _check_available()
    except RuntimeError as e:
        return {"available": False, "reason": str(e)}
    with _lock:
        idle_s = None
        if _model is not None and _last_used is not None:
            idle_s = round(time.monotonic() - _last_used, 1)
        return {
            "available": True,
            "device": _device(),
            "loaded": _model is not None,
            "weights": _loaded_weights,
            "idle_s": idle_s,
            "idle_timeout_s": _idle_timeout_s,
            "auto_evict_enabled": _idle_timeout_s > 0,
        }


def load_model(weights: str = DEFAULT_WEIGHTS) -> dict[str, Any]:
    _check_available()
    was_loaded = _model is not None and _loaded_weights == weights
    t0 = time.perf_counter()
    _get_model(weights)
    elapsed = time.perf_counter() - t0
    return {
        "weights": weights, "device": _device(),
        "was_already_loaded": was_loaded,
        "load_time_s": round(elapsed, 3),
    }


def unload() -> dict[str, Any]:
    global _model, _loaded_weights, _last_used
    with _lock:
        had = _model is not None
        _model = None
        _loaded_weights = None
        _last_used = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": had}


# ---- inference -------------------------------------------------------------

def _infer(image: Image.Image, *, weights: str, conf: float = 0.25,
           iou: float = 0.5):
    model = _get_model(weights)
    rgb = np.asarray(image.convert("RGB"))
    results = model(rgb, retina_masks=True, conf=conf, iou=iou, verbose=False)
    _touch()
    if not results or results[0].masks is None:
        return np.zeros((0, *rgb.shape[:2]), dtype=bool), \
               np.zeros((0, 4), dtype=np.float32), \
               np.zeros((0,), dtype=np.float32), []
    masks = results[0].masks.data.cpu().numpy().astype(bool)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
    names = [results[0].names[c] for c in cls_ids]
    return masks, boxes, scores, names


def segment_at_point(image: Image.Image, x: float, y: float, *,
                     weights: str = DEFAULT_WEIGHTS,
                     conf: float = 0.25) -> tuple[Image.Image, float, str]:
    """Return the highest-confidence detection whose bbox contains (x, y).
    Returns ``(mask_L_mode, score, class_name)``."""
    _check_available()
    masks, boxes, scores, names = _infer(image, weights=weights, conf=conf)
    best, best_score = -1, -1.0
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        if x1 <= x <= x2 and y1 <= y <= y2 and scores[i] > best_score:
            best, best_score = i, float(scores[i])
    if best < 0:
        raise RuntimeError(f"YOLO found no detection containing ({x}, {y})")
    return _mask_to_image(masks[best]), best_score, names[best]


def segment_in_box(image: Image.Image, x1: float, y1: float,
                   x2: float, y2: float, *,
                   weights: str = DEFAULT_WEIGHTS,
                   conf: float = 0.25) -> tuple[Image.Image, float, str]:
    """Return the detection whose mask best overlaps the given box."""
    _check_available()
    masks, boxes, scores, names = _infer(image, weights=weights, conf=conf)
    target = np.array([x1, y1, x2, y2], dtype=np.float32)
    best, best_iou = -1, 0.0
    for i, b in enumerate(boxes):
        ix1, iy1 = max(b[0], target[0]), max(b[1], target[1])
        ix2, iy2 = min(b[2], target[2]), min(b[3], target[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b[2] - b[0]) * (b[3] - b[1])
        a2 = (target[2] - target[0]) * (target[3] - target[1])
        iou = inter / max(a1 + a2 - inter, 1e-6)
        if iou > best_iou:
            best_iou, best = iou, i
    if best < 0:
        raise RuntimeError(
            f"YOLO found no detection overlapping ({x1},{y1})-({x2},{y2})"
        )
    return _mask_to_image(masks[best]), float(scores[best]), names[best]


def segment_everything(image: Image.Image, *,
                       weights: str = DEFAULT_WEIGHTS,
                       conf: float = 0.25,
                       iou: float = 0.5,
                       max_results: int | None = None
                       ) -> list[dict[str, Any]]:
    """Return every detection as ``{mask, bbox, score, class_name}``."""
    _check_available()
    masks, boxes, scores, names = _infer(image, weights=weights,
                                          conf=conf, iou=iou)
    out = []
    n = len(masks) if max_results is None else min(len(masks), max_results)
    for i in range(n):
        out.append({
            "mask": _mask_to_image(masks[i]),
            "bbox": [float(v) for v in boxes[i]],
            "score": float(scores[i]),
            "class_name": names[i],
        })
    return out


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
