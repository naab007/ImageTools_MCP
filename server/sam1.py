"""Original SAM (Meta, 2023) via HuggingFace transformers.

Beat SAM 2.1 on COCO box prompts in the 2026-05 benchmark (median IoU 0.78
vs 0.05 for SAM 2.1 large). Different mask granularity — SAM 1 sticks closer
to individual instances, SAM 2.1 sometimes returns whole-scene parts.

Variants (HuggingFace repo → param count):
- ``facebook/sam-vit-base``   90 M params,  fastest
- ``facebook/sam-vit-large``  308 M, **default — best perf/quality**
- ``facebook/sam-vit-huge``   636 M, marginal gain over Large

Same idle-eviction pattern as the other model modules.
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
_processor: Any = None
_loaded_model: str | None = None
_device_used: str = "cpu"

_last_used: float | None = None
_idle_timeout_s: float = 3600.0
_SWEEP_INTERVAL_S: float = 300.0
_sweeper_thread: threading.Thread | None = None

DEFAULT_MODEL = "facebook/sam-vit-large"


def _check_available() -> None:
    try:
        from transformers import SamModel, SamProcessor  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "SAM 1 is unavailable. Install with "
            "`pip install image-tools-mcp[seg]`."
        ) from e


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
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
        target=_sweep_loop, daemon=True, name="sam1-idle-sweeper",
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


def _get_model(model: str = DEFAULT_MODEL) -> tuple[Any, Any, str]:
    """Return ``(model, processor, device)`` — snapshot under the lock so a
    concurrent model switch can't change the device underneath inference."""
    global _model, _processor, _loaded_model, _device_used
    with _lock:
        if _model is not None and _loaded_model == model:
            _touch()
            return _model, _processor, _device_used
        if _model is not None:
            _model = None
            _processor = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        from transformers import SamModel, SamProcessor
        _device_used = _device()
        _model = SamModel.from_pretrained(model).to(_device_used).eval()
        _processor = SamProcessor.from_pretrained(model)
        _loaded_model = model
        _touch()
        active_m, active_p, active_d = _model, _processor, _device_used
    _start_sweeper_if_needed()
    return active_m, active_p, active_d


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
            "model": _loaded_model,
            "idle_s": idle_s,
            "idle_timeout_s": _idle_timeout_s,
            "auto_evict_enabled": _idle_timeout_s > 0,
        }


def load_model(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    _check_available()
    was_loaded = _model is not None and _loaded_model == model
    t0 = time.perf_counter()
    _get_model(model)
    elapsed = time.perf_counter() - t0
    return {
        "model": model, "device": _device(),
        "was_already_loaded": was_loaded,
        "load_time_s": round(elapsed, 3),
    }


def unload() -> dict[str, Any]:
    global _model, _processor, _loaded_model, _last_used
    with _lock:
        had = _model is not None
        _model = None
        _processor = None
        _loaded_model = None
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

def _run(image: Image.Image, *, model: str,
         point_coords: list[list[float]] | None = None,
         point_labels: list[int] | None = None,
         box: list[float] | None = None) -> tuple[np.ndarray, float]:
    import torch
    m, proc, device = _get_model(model)
    rgb = image.convert("RGB")

    kwargs: dict[str, Any] = {"images": rgb, "return_tensors": "pt"}
    if point_coords is not None:
        kwargs["input_points"] = [[point_coords]]
        kwargs["input_labels"] = [[point_labels or [1] * len(point_coords)]]
    if box is not None:
        kwargs["input_boxes"] = [[box]]

    inputs = proc(**kwargs).to(device)
    with torch.inference_mode():
        outputs = m(**inputs)
    masks = proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores.detach().cpu().numpy().squeeze()
    _touch()
    # masks[0] shape: (1, 3, H, W) — pick best of the 3 candidates.
    m0 = masks[0]
    if m0.ndim == 4:
        m0 = m0[0]
    scores = np.atleast_1d(scores)
    best = int(np.argmax(scores))
    return m0[best].numpy().astype(bool), float(scores[best])


def segment_point(image: Image.Image, x: float, y: float, *,
                  label: int = 1, model: str = DEFAULT_MODEL
                  ) -> tuple[Image.Image, float]:
    _check_available()
    mask, score = _run(image, model=model,
                       point_coords=[[float(x), float(y)]],
                       point_labels=[int(label)])
    return _mask_to_image(mask), score


def segment_points(image: Image.Image, points: list[list[float]],
                   labels: list[int] | None = None, *,
                   model: str = DEFAULT_MODEL
                   ) -> tuple[Image.Image, float]:
    _check_available()
    if not points:
        raise ValueError("need at least one point")
    if labels is not None and len(labels) != len(points):
        raise ValueError("labels length must match points length")
    mask, score = _run(image, model=model,
                       point_coords=[[float(p[0]), float(p[1])] for p in points],
                       point_labels=labels or [1] * len(points))
    return _mask_to_image(mask), score


def segment_box(image: Image.Image, x1: float, y1: float,
                x2: float, y2: float, *,
                model: str = DEFAULT_MODEL) -> tuple[Image.Image, float]:
    _check_available()
    mask, score = _run(image, model=model,
                       box=[float(x1), float(y1), float(x2), float(y2)])
    return _mask_to_image(mask), score


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
