"""SAM 2.1 (Segment Anything 2) integration.

Wraps Meta's ``sam2`` package (https://pypi.org/project/sam2/) for promptable
segmentation. Two engines:

- :class:`SAM2ImagePredictor` — given an image plus prompts (points / box),
  emits one or more candidate masks with confidence scores. We pick the
  top-scoring mask by default.
- :class:`SAM2AutomaticMaskGenerator` — grid-samples points across the
  image and produces *every* discoverable mask (cars, faces, signs, etc.).

Model weights are pulled from HuggingFace via ``from_pretrained``. Four
sizes ship:

- ``facebook/sam2.1-hiera-tiny``       — ~38M params, fastest
- ``facebook/sam2.1-hiera-small``      — ~46M, fast
- ``facebook/sam2.1-hiera-base-plus``  — ~80M, balanced
- ``facebook/sam2.1-hiera-large``      — ~224M, best quality (default)

Idle eviction mirrors :mod:`sd` and :mod:`qwen` — a background sweeper
unloads the model after ``_idle_timeout_s`` seconds of inactivity. Set the
timeout to 0 to disable.
"""
from __future__ import annotations

import gc
import threading
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

# Re-entrant so the lock-locked ``_touch`` can be called from inside other
# lock-held sections without deadlocking.
_lock = threading.RLock()
_predictor: Any = None         # SAM2ImagePredictor instance
_mask_gen: Any = None          # SAM2AutomaticMaskGenerator instance (lazy)
_loaded_model: str | None = None

# Idle eviction.
_last_used: float | None = None
_idle_timeout_s: float = 3600.0
_SWEEP_INTERVAL_S: float = 300.0
_sweeper_thread: threading.Thread | None = None


# ---- availability ----------------------------------------------------------

def _check_available() -> None:
    try:
        import sam2  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "SAM 2 is unavailable. Install the optional extra: "
            "`pip install image-tools-mcp[sam]`. This pulls the sam2 package "
            "and torch."
        ) from e


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---- timeout sweeper -------------------------------------------------------

def _touch() -> None:
    """Mark the model as recently used. Holds the lock so the sweeper's
    timestamp check can't race with the write."""
    global _last_used
    with _lock:
        _last_used = time.monotonic()


def _start_sweeper_if_needed() -> None:
    global _sweeper_thread
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return
    _sweeper_thread = threading.Thread(
        target=_sweep_loop, daemon=True, name="sam-idle-sweeper",
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
    """Evict the model atomically if and only if it's been idle past the
    limit at the moment we hold the lock."""
    if _idle_timeout_s <= 0:
        return
    with _lock:
        if _predictor is None or _last_used is None:
            return
        if time.monotonic() - _last_used <= _idle_timeout_s:
            return
    unload()


def set_idle_timeout(seconds: float) -> dict[str, Any]:
    """Set how long the SAM 2 model stays resident after its last use.
    Pass 0 to disable auto-eviction (manual unload only)."""
    global _idle_timeout_s
    _idle_timeout_s = max(0.0, float(seconds))
    return {
        "idle_timeout_s": _idle_timeout_s,
        "sweep_interval_s": _SWEEP_INTERVAL_S,
        "auto_evict_enabled": _idle_timeout_s > 0,
    }


# ---- pipeline management ---------------------------------------------------

DEFAULT_MODEL = "facebook/sam2.1-hiera-large"


def _get_predictor(model: str = DEFAULT_MODEL):
    """Return the cached predictor, loading it on first call or after a
    model switch. Holds the global lock for the entire load operation since
    SAM 2 weight downloads can take a while."""
    global _predictor, _mask_gen, _loaded_model
    with _lock:
        if _predictor is not None and _loaded_model == model:
            _touch()
            active = _predictor
            return active
        # Switching models: drop everything tied to the old one.
        if _predictor is not None:
            _predictor = None
            _mask_gen = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        _predictor = SAM2ImagePredictor.from_pretrained(model, device=_device())
        _loaded_model = model
        _touch()
        active = _predictor
    _start_sweeper_if_needed()
    return active


def _get_mask_generator(model: str = DEFAULT_MODEL):
    """Return the auto-mask generator, building it lazily on top of the
    cached predictor's underlying SAM2 model so we don't double-load weights.

    Acquires + holds the lock for the whole build/cache decision; ``_lock``
    is an RLock so the nested ``_get_predictor`` call re-enters safely.
    Returns a local reference so callers can run inference outside the
    lock without racing a concurrent unload.
    """
    global _mask_gen
    with _lock:
        _get_predictor(model)  # nested RLock acquire, same-thread safe
        if _mask_gen is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
            # Build on top of the predictor's model so weights are shared.
            _mask_gen = SAM2AutomaticMaskGenerator(_predictor.model)
        _touch()
        return _mask_gen


def status() -> dict[str, Any]:
    """Report whether SAM 2 is installed, the device, and what's loaded."""
    try:
        _check_available()
    except RuntimeError as e:
        return {"available": False, "reason": str(e)}
    with _lock:
        idle_s = None
        if _predictor is not None and _last_used is not None:
            idle_s = round(time.monotonic() - _last_used, 1)
        info = {
            "available": True,
            "device": _device(),
            "predictor_loaded": _predictor is not None,
            "mask_generator_loaded": _mask_gen is not None,
            "model": _loaded_model,
            "idle_s": idle_s,
            "idle_timeout_s": _idle_timeout_s,
            "auto_evict_enabled": _idle_timeout_s > 0,
        }
    return info


def load_model(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Pre-load a SAM 2 model so segmentation calls don't pay load latency.
    Idempotent — reports ``was_already_loaded: True`` if the requested model
    is already resident."""
    _check_available()
    was_loaded = _predictor is not None and _loaded_model == model
    t0 = time.perf_counter()
    _get_predictor(model)
    elapsed = time.perf_counter() - t0
    return {
        "model": model,
        "device": _device(),
        "was_already_loaded": was_loaded,
        "load_time_s": round(elapsed, 3),
    }


def unload() -> dict[str, Any]:
    """Drop the predictor + mask generator and free GPU memory."""
    global _predictor, _mask_gen, _loaded_model, _last_used
    with _lock:
        had = _predictor is not None
        _predictor = None
        _mask_gen = None
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


# ---- segmentation ----------------------------------------------------------

def _mask_array_to_image(mask: np.ndarray) -> Image.Image:
    """Convert a boolean / 0-1 mask array to an L-mode 0/255 image."""
    if mask.dtype == bool:
        arr = mask.astype(np.uint8) * 255
    elif mask.dtype.kind == "f":
        arr = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        arr = (mask > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def _run_predictor(image: Image.Image, *,
                   point_coords: np.ndarray | None = None,
                   point_labels: np.ndarray | None = None,
                   box: np.ndarray | None = None,
                   model: str = DEFAULT_MODEL,
                   multimask: bool = True) -> tuple[np.ndarray, float]:
    """Set the image on the predictor and run inference. Returns
    (best_mask_HxW_bool, best_iou_score)."""
    import torch

    predictor = _get_predictor(model)
    rgb = np.asarray(image.convert("RGB"))
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask,
        )
    _touch()
    # masks shape: (n_masks, H, W). Pick highest score when multimask=True.
    best = int(np.argmax(scores))
    return masks[best], float(scores[best])


def segment_with_points(image: Image.Image,
                        points: Sequence[Sequence[float]],
                        labels: Sequence[int] | None = None, *,
                        model: str = DEFAULT_MODEL,
                        multimask: bool = True
                        ) -> tuple[Image.Image, float]:
    """Promptable segmentation by point clicks.

    Each entry in ``points`` is ``[x, y]``. ``labels[i]`` is 1 for a
    foreground point (include) or 0 for background (exclude). Default
    labels are all 1.

    Returns ``(mask_image_L_mode, iou_score)``. The mask is the
    highest-IoU candidate when ``multimask=True``.
    """
    _check_available()
    if not points:
        raise ValueError("segment_with_points: need at least one point")
    coords = np.asarray(points, dtype=np.float32)
    if labels is None:
        labels = [1] * len(points)
    if len(labels) != len(points):
        raise ValueError(
            f"labels length ({len(labels)}) must match points length ({len(points)})"
        )
    lab = np.asarray(labels, dtype=np.int32)
    mask, score = _run_predictor(
        image, point_coords=coords, point_labels=lab,
        model=model, multimask=multimask,
    )
    return _mask_array_to_image(mask), score


def segment_with_box(image: Image.Image,
                     x1: float, y1: float, x2: float, y2: float, *,
                     model: str = DEFAULT_MODEL,
                     multimask: bool = False
                     ) -> tuple[Image.Image, float]:
    """Promptable segmentation by bounding box. Returns the mask plus IoU."""
    _check_available()
    box = np.array([float(x1), float(y1), float(x2), float(y2)],
                   dtype=np.float32)
    mask, score = _run_predictor(
        image, box=box, model=model, multimask=multimask,
    )
    return _mask_array_to_image(mask), score


def segment_everything(image: Image.Image, *,
                       model: str = DEFAULT_MODEL,
                       points_per_side: int | None = None,
                       min_mask_region_area: int = 0
                       ) -> list[dict[str, Any]]:
    """Auto-mask generation across the whole image — finds every salient
    region without prompts.

    Returns a list of ``{"mask": Image, "score": float, "area": int,
    "bbox": [x, y, w, h]}`` entries sorted by score (highest first).

    ``points_per_side`` controls grid density (default 32, more = more masks
    + slower). ``min_mask_region_area`` filters out tiny masks below this
    pixel count.
    """
    _check_available()
    import torch

    # Build the generator with overridden settings if requested. SAM2's
    # constructor accepts these kwargs. Snapshot the generator under the
    # lock so a concurrent unload can't null it during inference.
    if points_per_side is not None or min_mask_region_area:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
        global _mask_gen
        with _lock:
            _get_predictor(model)  # nested re-entrant acquire
            kwargs: dict[str, Any] = {}
            if points_per_side is not None:
                kwargs["points_per_side"] = int(points_per_side)
            if min_mask_region_area:
                kwargs["min_mask_region_area"] = int(min_mask_region_area)
            _mask_gen = SAM2AutomaticMaskGenerator(_predictor.model, **kwargs)
            mask_gen = _mask_gen
    else:
        mask_gen = _get_mask_generator(model)

    rgb = np.asarray(image.convert("RGB"))
    with torch.inference_mode():
        results = mask_gen.generate(rgb)
    with _lock:
        _touch()

    out: list[dict[str, Any]] = []
    for r in results:
        out.append({
            "mask": _mask_array_to_image(r["segmentation"]),
            "score": float(r.get("predicted_iou", r.get("stability_score", 0.0))),
            "area": int(r.get("area", 0)),
            "bbox": [int(v) for v in r.get("bbox", [0, 0, 0, 0])],
        })
    out.sort(key=lambda m: m["score"], reverse=True)
    return out
