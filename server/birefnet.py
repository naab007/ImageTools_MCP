"""BiRefNet — Bilateral Reference Network for high-resolution dichotomous
image segmentation. Best-in-class background removal (2025-2026).

Won the background-removal track in the 2026-05 benchmark: 0.66 median IoU
on COCO scenes at 96 ms/call, beating InspyreNet (0.63 / 206 ms) and rembg-
isnet (0.57 / 124 ms).

Variants:
- ``general``      — default, 1024 px, balanced
- ``portrait``     — fine-tuned for people / portraits
- ``hr``           — 2048 px, slower, sharper edges
- ``matting``      — produces soft alpha mattes for compositing
- ``dis5k``        — fine-tuned on DIS5K (small / intricate subjects)

Output is a soft (0..255) grayscale mask, perfect for feathering.
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
_loaded_repo: str | None = None
_device_used: str = "cpu"
_input_size: int = 1024

_last_used: float | None = None
_idle_timeout_s: float = 3600.0
_SWEEP_INTERVAL_S: float = 300.0
_sweeper_thread: threading.Thread | None = None

_VARIANT_REPOS = {
    "general":  ("ZhengPeng7/BiRefNet",          1024),
    "portrait": ("ZhengPeng7/BiRefNet-portrait", 1024),
    "hr":       ("ZhengPeng7/BiRefNet_HR",       2048),
    "matting":  ("ZhengPeng7/BiRefNet-matting",  1024),
    "dis5k":    ("ZhengPeng7/BiRefNet-DIS5K",    1024),
}

DEFAULT_VARIANT = "general"


def _check_available() -> None:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import timm  # noqa: F401
        import einops  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "BiRefNet is unavailable. Install with "
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
        target=_sweep_loop, daemon=True, name="birefnet-idle-sweeper",
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


def _get_model(variant: str = DEFAULT_VARIANT) -> tuple[Any, str, int]:
    """Load (or fetch from cache) the BiRefNet model for ``variant``.

    Returns ``(model, device, input_size)`` — a *snapshot* captured under
    the lock. Callers should use these locals, not the module globals,
    because a concurrent variant switch can rotate the globals while
    inference runs.
    """
    global _model, _loaded_repo, _device_used, _input_size
    if variant not in _VARIANT_REPOS:
        raise ValueError(
            f"unknown BiRefNet variant {variant!r}. "
            f"Choose from {list(_VARIANT_REPOS)}."
        )
    repo, size = _VARIANT_REPOS[variant]
    with _lock:
        if _model is not None and _loaded_repo == repo:
            _touch()
            return _model, _device_used, _input_size
        if _model is not None:
            _model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        from transformers import AutoModelForImageSegmentation
        _device_used = _device()
        _model = AutoModelForImageSegmentation.from_pretrained(
            repo, trust_remote_code=True,
        ).to(_device_used).eval()
        if _device_used == "cuda":
            _model = _model.half()
        _loaded_repo = repo
        _input_size = size
        _touch()
        # Capture the consistent snapshot before releasing the lock.
        active_model, active_device, active_size = _model, _device_used, _input_size
    _start_sweeper_if_needed()
    return active_model, active_device, active_size


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
            "repo": _loaded_repo,
            "input_size": _input_size,
            "idle_s": idle_s,
            "idle_timeout_s": _idle_timeout_s,
            "auto_evict_enabled": _idle_timeout_s > 0,
        }


def load_model(variant: str = DEFAULT_VARIANT) -> dict[str, Any]:
    _check_available()
    repo = _VARIANT_REPOS[variant][0]
    was_loaded = _model is not None and _loaded_repo == repo
    t0 = time.perf_counter()
    _get_model(variant)
    elapsed = time.perf_counter() - t0
    return {
        "variant": variant, "repo": repo, "device": _device(),
        "was_already_loaded": was_loaded,
        "load_time_s": round(elapsed, 3),
    }


def unload() -> dict[str, Any]:
    global _model, _loaded_repo, _last_used, _input_size
    with _lock:
        had = _model is not None
        _model = None
        _loaded_repo = None
        _last_used = None
        _input_size = 1024
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": had}


# ---- inference -------------------------------------------------------------

def remove_background(image: Image.Image, *,
                      variant: str = DEFAULT_VARIANT) -> Image.Image:
    """Return a soft (0..255) L-mode foreground mask. The input image is
    resized to the model's working resolution internally; the returned mask
    is rescaled back to the original image size."""
    _check_available()
    import torch
    from torchvision import transforms as T

    # Snapshot model + device + input_size under the lock so a concurrent
    # variant switch can't rotate them mid-call.
    model, device, size = _get_model(variant)
    rgb = image.convert("RGB")

    tx = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    inp = tx(rgb).unsqueeze(0).to(device)
    if device == "cuda":
        inp = inp.half()

    with torch.inference_mode():
        pred = model(inp)[-1].sigmoid().cpu().float()
    _touch()

    mask_small = (pred[0, 0].numpy() * 255).astype(np.uint8)
    mask = Image.fromarray(mask_small, mode="L").resize(rgb.size, Image.BILINEAR)
    return mask
