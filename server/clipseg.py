"""CLIPSeg — text-prompted segmentation via CLIP feature decoding.

Lightweight (~50 MB, 654 MB VRAM), fast (~25 ms/call). 2026-05 benchmark:
0.47 median IoU on COCO text prompts — middling but extremely cheap. Good
default for "vaguely mask anything I describe in text" when you don't need
boundary precision.

Output is a soft confidence map (0..1) — already feathered by design. The
``threshold`` parameter controls where you cut the mask if you need binary.
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
_loaded_repo: str | None = None
_device_used: str = "cpu"

_last_used: float | None = None
_idle_timeout_s: float = 3600.0
_SWEEP_INTERVAL_S: float = 300.0
_sweeper_thread: threading.Thread | None = None

DEFAULT_MODEL = "CIDAS/clipseg-rd64-refined"


def _check_available() -> None:
    try:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "CLIPSeg is unavailable. Install with "
            "`pip install image-tools-mcp[seg]`."
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
        target=_sweep_loop, daemon=True, name="clipseg-idle-sweeper",
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
    global _model, _processor, _loaded_repo, _device_used
    with _lock:
        if _model is not None and _loaded_repo == model:
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
        from transformers import (
            CLIPSegProcessor, CLIPSegForImageSegmentation,
        )
        _device_used = _device()
        _processor = CLIPSegProcessor.from_pretrained(model)
        _model = (
            CLIPSegForImageSegmentation.from_pretrained(model)
            .to(_device_used).eval()
        )
        _loaded_repo = model
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
            "model": _loaded_repo,
            "idle_s": idle_s,
            "idle_timeout_s": _idle_timeout_s,
            "auto_evict_enabled": _idle_timeout_s > 0,
        }


def load_model(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    _check_available()
    was_loaded = _model is not None and _loaded_repo == model
    t0 = time.perf_counter()
    _get_model(model)
    elapsed = time.perf_counter() - t0
    return {
        "model": model, "device": _device(),
        "was_already_loaded": was_loaded,
        "load_time_s": round(elapsed, 3),
    }


def unload() -> dict[str, Any]:
    global _model, _processor, _loaded_repo, _last_used
    with _lock:
        had = _model is not None
        _model = None
        _processor = None
        _loaded_repo = None
        _last_used = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": had}


def segment_text(image: Image.Image, text: str, *,
                 model: str = DEFAULT_MODEL,
                 threshold: float | None = 0.5
                 ) -> tuple[Image.Image, float]:
    """Return ``(mask, max_confidence)``. If ``threshold`` is None the mask is
    a soft 0..255 confidence map (feathered by design); pass a float to
    binarize."""
    _check_available()
    import torch
    m, proc, device = _get_model(model)
    rgb = image.convert("RGB")
    inputs = proc(text=[text], images=[rgb], return_tensors="pt",
                  padding=True).to(device)
    with torch.inference_mode():
        out = m(**inputs)
    prob = out.logits.sigmoid()[0].cpu().float().numpy()
    _touch()
    # Upscale to original image dims (CLIPSeg outputs 352² by default).
    soft = (prob * 255).astype(np.uint8)
    soft_img = Image.fromarray(soft).resize(rgb.size, Image.BILINEAR)
    max_conf = float(prob.max())
    if threshold is None:
        return soft_img, max_conf
    arr = np.asarray(soft_img)
    binary = (arr >= int(threshold * 255)).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L"), max_conf
