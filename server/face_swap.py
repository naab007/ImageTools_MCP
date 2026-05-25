"""Face swap + face restoration — slim port of ComfyUI-ReActor.

Two things, no more:
1. Face swap via InsightFace's ``buffalo_l`` detector + ``inswapper_128.onnx``.
2. Face restoration via GFPGAN (1.3 or 1.4).

What's intentionally NOT included (vs upstream ReActor):
- NSFW / safe-for-work classifier
- Gender filtering
- Multiple swap-model dispatch (reswapper / hyperswap / hififace) — only inswapper
- Face boost
- Multi-image batch helpers (handle one image at a time)

Models (auto-located in order):
- detector: ``<insightface_root>/models/buffalo_l/`` — defaults to the ComfyUI convention
  ``B:\\-AI-Stuff-\\ComfyUI\\models\\insightface``, falls back to ``~/.insightface``
- swapper: ``inswapper_128.onnx`` — defaults to ``<insightface_root>/inswapper_128.onnx``
- restorer: ``GFPGANv1.4.pth`` (or 1.3) — defaults to
  ``B:\\-AI-Stuff-\\ComfyUI\\models\\facerestore_models/`` or `~/.cache/gfpgan/weights/`

Lifecycle: status / load / unload / set_idle_timeout, same pattern as the other
model loaders. The detector + swapper + restorer share idle timeout.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Re-entrant — module reuses _get_* helpers from inside other lock-takers.
_lock = threading.RLock()
_face_analyser: Any = None
_face_analyser_det_size: tuple[int, int] | None = None
_face_swapper: Any = None
_face_swapper_path: str | None = None
_restorer: Any = None
_restorer_model: str | None = None

_last_used: float | None = None
_idle_timeout_s: float = 3600.0
_SWEEP_INTERVAL_S: float = 300.0
_sweeper_thread: threading.Thread | None = None


# ---- model resolution helpers --------------------------------------------

_COMFYUI_INSIGHTFACE_ROOT = Path(r"B:\-AI-Stuff-\ComfyUI\models\insightface")
_COMFYUI_FACERESTORE_DIR = Path(r"B:\-AI-Stuff-\ComfyUI\models\facerestore_models")


def _resolve_insightface_root() -> Path:
    """Where insightface looks for ``<root>/models/<name>``. Prefer the
    ComfyUI store if it exists (so we don't re-download multi-hundred-MB
    weights when ComfyUI already has them), else the default ``~/.insightface``."""
    if _COMFYUI_INSIGHTFACE_ROOT.exists():
        return _COMFYUI_INSIGHTFACE_ROOT
    return Path.home() / ".insightface"


def _resolve_swapper_path(explicit: str | None) -> Path:
    """Path to the inswapper_128.onnx weights."""
    if explicit:
        return Path(explicit)
    for candidate in (
        _COMFYUI_INSIGHTFACE_ROOT / "inswapper_128.onnx",
        _resolve_insightface_root() / "models" / "inswapper_128.onnx",
    ):
        if candidate.exists():
            return candidate
    # Fall back to insightface's auto-download convention — model_zoo.get_model
    # will pull it.
    return Path("inswapper_128.onnx")


def _resolve_restorer_path(model_name: str) -> Path | None:
    """Path to a GFPGAN weights file (``GFPGANv1.4.pth`` or ``GFPGANv1.3.pth``)."""
    for d in (
        _COMFYUI_FACERESTORE_DIR,
        Path.home() / ".cache" / "gfpgan" / "weights",
    ):
        p = d / model_name
        if p.exists():
            return p
    return None


# ---- idle eviction sweeper ------------------------------------------------

def _touch() -> None:
    global _last_used
    with _lock:
        _last_used = time.monotonic()


def _start_sweeper_if_needed() -> None:
    global _sweeper_thread
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return
    _sweeper_thread = threading.Thread(
        target=_sweep_loop, daemon=True, name="face-idle-sweeper",
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
        if _last_used is None:
            return
        if (time.monotonic() - _last_used) > _idle_timeout_s and (
            _face_analyser is not None or _face_swapper is not None
            or _restorer is not None
        ):
            _drop_locked()
    gc.collect()


def _drop_locked() -> None:
    """Free all face models. Caller holds ``_lock``."""
    global _face_analyser, _face_analyser_det_size, _face_swapper
    global _face_swapper_path, _restorer, _restorer_model, _last_used
    _face_analyser = None
    _face_analyser_det_size = None
    _face_swapper = None
    _face_swapper_path = None
    _restorer = None
    _restorer_model = None
    _last_used = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def set_idle_timeout(seconds: float) -> dict[str, Any]:
    """How long an unused face model stays resident before the sweeper unloads it.
    Pass 0 to disable auto-eviction (manual unload only)."""
    global _idle_timeout_s
    _idle_timeout_s = max(0.0, float(seconds))
    return {
        "idle_timeout_s": _idle_timeout_s,
        "sweep_interval_s": _SWEEP_INTERVAL_S,
        "auto_evict_enabled": _idle_timeout_s > 0,
    }


# ---- availability check ---------------------------------------------------

def _check_available() -> None:
    try:
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Face swap is unavailable. Install the optional extra: "
            "`pip install image-tools-mcp[face]`. Pulls insightface, "
            "onnxruntime, and gfpgan."
        ) from e


def _onnx_providers() -> list[str]:
    """CUDA when available, else CPU. Matches ReActor's selection."""
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                pass
    except Exception:
        pass
    return ["CPUExecutionProvider"]


_basicsr_patched = False


def _patch_basicsr_torchvision_compat() -> None:
    """basicsr 1.4 imports ``torchvision.transforms.functional_tensor`` which
    torchvision deleted in 0.17+. Shim it to the new ``functional`` module
    BEFORE basicsr / gfpgan is imported. Idempotent.
    """
    global _basicsr_patched
    if _basicsr_patched:
        return
    import sys
    try:
        import torchvision.transforms.functional as _tvf
    except Exception:
        return
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _tvf)
    _basicsr_patched = True


# ---- model loaders --------------------------------------------------------

def _get_face_analyser(det_size: tuple[int, int] = (640, 640)):
    """Return ``insightface.app.FaceAnalysis`` with the ``buffalo_l`` pack
    (detection + alignment + ArcFace embedding + age/sex)."""
    global _face_analyser, _face_analyser_det_size
    with _lock:
        if _face_analyser is not None and _face_analyser_det_size == det_size:
            _touch()
            return _face_analyser
        from insightface.app import FaceAnalysis
        root = _resolve_insightface_root()
        analyser = FaceAnalysis(
            name="buffalo_l",
            root=str(root),
            providers=_onnx_providers(),
        )
        try:
            import torch
            ctx = 0 if torch.cuda.is_available() else -1
        except Exception:
            ctx = -1
        analyser.prepare(ctx_id=ctx, det_size=det_size)
        _face_analyser = analyser
        _face_analyser_det_size = det_size
        _touch()
        _start_sweeper_if_needed()
        return analyser


def _get_face_swapper(explicit_path: str | None = None):
    """Return the inswapper_128 model wrapped as ``insightface.model_zoo.INSwapper``."""
    global _face_swapper, _face_swapper_path
    desired = str(_resolve_swapper_path(explicit_path))
    with _lock:
        if _face_swapper is not None and _face_swapper_path == desired:
            _touch()
            return _face_swapper
        import insightface
        # ``get_model`` accepts an absolute path; it'll wrap the onnx in the
        # right inference class based on the input name.
        swapper = insightface.model_zoo.get_model(
            desired, providers=_onnx_providers(),
        )
        _face_swapper = swapper
        _face_swapper_path = desired
        _touch()
        _start_sweeper_if_needed()
        return swapper


def _get_restorer(model_name: str = "GFPGANv1.4.pth"):
    """Return a ``GFPGANer`` for face restoration. Falls back to auto-download
    via gfpgan's mechanism if the weights aren't locally cached."""
    global _restorer, _restorer_model
    with _lock:
        if _restorer is not None and _restorer_model == model_name:
            _touch()
            return _restorer
        try:
            _patch_basicsr_torchvision_compat()
            from gfpgan import GFPGANer
        except ImportError as e:
            raise RuntimeError(
                "Face restoration needs gfpgan. "
                "`pip install gfpgan` (or install the [face] extra)."
            ) from e
        local = _resolve_restorer_path(model_name)
        # GFPGAN auto-downloads if model_path is a URL; locally-cached path
        # also works. arch="clean" matches the .pth files we ship; v1.4 weights
        # use clean arch with channel_multiplier=2 (same as 1.3).
        model_path = str(local) if local else f"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/{model_name}"
        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _restorer = restorer
        _restorer_model = model_name
        _touch()
        _start_sweeper_if_needed()
        return restorer


# ---- public API ----------------------------------------------------------

def status() -> dict[str, Any]:
    """Report what's loaded + the search paths used for model resolution."""
    try:
        _check_available()
    except RuntimeError as e:
        return {"available": False, "reason": str(e)}
    with _lock:
        last_used = _last_used
        info = {
            "available": True,
            "detector_loaded": _face_analyser is not None,
            "detector_det_size": _face_analyser_det_size,
            "swapper_loaded": _face_swapper is not None,
            "swapper_path": _face_swapper_path,
            "restorer_loaded": _restorer is not None,
            "restorer_model": _restorer_model,
            "providers": _onnx_providers(),
            "insightface_root": str(_resolve_insightface_root()),
            "idle_timeout_s": _idle_timeout_s,
            "auto_evict_enabled": _idle_timeout_s > 0,
        }
    if last_used is not None:
        info["idle_s"] = round(time.monotonic() - last_used, 1)
    return info


def load(*, detector_det_size: int = 640,
         swapper_path: str | None = None,
         restorer_model: str | None = "GFPGANv1.4.pth") -> dict[str, Any]:
    """Pre-warm the face stack. ``restorer_model`` may be set to None to
    skip GFPGAN (saves ~340 MB)."""
    _check_available()
    t0 = time.perf_counter()
    _get_face_analyser((detector_det_size, detector_det_size))
    _get_face_swapper(swapper_path)
    if restorer_model:
        _get_restorer(restorer_model)
    return {
        "loaded": True,
        "load_time_s": round(time.perf_counter() - t0, 3),
        "detector_det_size": detector_det_size,
        "swapper_path": _face_swapper_path,
        "restorer_model": _restorer_model,
    }


def unload() -> dict[str, Any]:
    """Drop all face models. Frees ~1.2 GB of memory (~280 + 554 + 340)."""
    with _lock:
        had = (_face_analyser is not None or _face_swapper is not None
               or _restorer is not None)
        _drop_locked()
    gc.collect()
    return {"unloaded": had}


def detect_faces(image: Image.Image, *,
                 det_size: int = 640) -> list[dict[str, Any]]:
    """Run face detection + embedding. Returns one dict per face with:
      - ``bbox``: [x1, y1, x2, y2] floats
      - ``area``: bbox area in pixels (float)
      - ``score``: detector confidence
      - ``age``: estimated age (int) — buffalo_l includes this
      - ``sex``: 'M' or 'F' (best-effort)
      - ``kps``: 5-point landmarks (list of 5 [x, y])

    Faces are sorted by bbox area, largest first — matching the default
    ``large-small`` order in ReActor.
    """
    _check_available()
    analyser = _get_face_analyser((det_size, det_size))
    bgr = _pil_to_bgr(image)
    faces = analyser.get(bgr)
    if not faces:
        return []
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    out = []
    for f in faces_sorted:
        bx = f.bbox.tolist()
        out.append({
            "bbox": [float(v) for v in bx],
            "area": float((bx[2] - bx[0]) * (bx[3] - bx[1])),
            "score": float(getattr(f, "det_score", 0.0)),
            "age": int(getattr(f, "age", -1)) if hasattr(f, "age") else None,
            "sex": getattr(f, "sex", None),
            "kps": [[float(p[0]), float(p[1])] for p in f.kps]
                   if getattr(f, "kps", None) is not None else None,
        })
    return out


def swap_face(source_image: Image.Image, target_image: Image.Image, *,
              source_face_index: int = 0,
              target_face_indices: list[int] | None = None,
              swapper_path: str | None = None,
              restore: bool = False,
              restorer_model: str = "GFPGANv1.4.pth",
              restore_weight: float = 0.5) -> Image.Image:
    """Swap face(s) from ``source_image`` onto ``target_image``.

    - ``source_face_index`` picks which face from the source (faces are
      sorted by bbox area, 0 = largest).
    - ``target_face_indices`` is the list of face indices in ``target_image``
      to overwrite. ``None`` = all detected faces. ``[0]`` = just the
      largest. ``[0, 1]`` = the two largest, etc.
    - ``restore=True`` runs GFPGAN on the result. ``restore_weight`` ∈ 0-1
      blends restoration intensity (lower keeps more of the swapped look).

    Returns the result as an RGB PIL Image.
    """
    _check_available()
    analyser = _get_face_analyser()
    swapper = _get_face_swapper(swapper_path)

    src_bgr = _pil_to_bgr(source_image)
    src_faces = analyser.get(src_bgr)
    if not src_faces:
        raise ValueError("no face detected in source_image")
    src_faces = sorted(
        src_faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    if source_face_index >= len(src_faces):
        raise ValueError(
            f"source_face_index={source_face_index} out of range; "
            f"only {len(src_faces)} face(s) in source"
        )
    source_face = src_faces[source_face_index]

    tgt_bgr = _pil_to_bgr(target_image)
    tgt_faces = analyser.get(tgt_bgr)
    if not tgt_faces:
        raise ValueError("no face detected in target_image")
    tgt_faces_sorted = sorted(
        tgt_faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    if target_face_indices is None:
        target_face_indices = list(range(len(tgt_faces_sorted)))
    bad = [i for i in target_face_indices if i >= len(tgt_faces_sorted)]
    if bad:
        raise ValueError(
            f"target_face_indices contains out-of-range entries {bad}; "
            f"only {len(tgt_faces_sorted)} face(s) in target"
        )

    result = tgt_bgr
    for i in target_face_indices:
        result = swapper.get(result, tgt_faces_sorted[i], source_face,
                             paste_back=True)
    _touch()

    if restore:
        result = _restore_bgr(result, restorer_model, restore_weight)
        _touch()

    return _bgr_to_pil(result)


def restore_faces(image: Image.Image, *,
                  restorer_model: str = "GFPGANv1.4.pth",
                  weight: float = 0.5) -> Image.Image:
    """Run GFPGAN face restoration on ``image``. Detects faces internally,
    aligns + enhances each, and pastes back. ``weight`` 0-1 blends the
    restoration vs the original (lower = subtler).

    Returns the restored RGB PIL Image. Best for upscaling small faces or
    cleaning up swap artefacts."""
    _check_available()
    bgr = _pil_to_bgr(image)
    out = _restore_bgr(bgr, restorer_model, weight)
    _touch()
    return _bgr_to_pil(out)


def _restore_bgr(bgr: np.ndarray, restorer_model: str,
                 weight: float) -> np.ndarray:
    """GFPGAN enhance on a BGR uint8 numpy image. Returns the restored
    BGR image (same shape). Caller does conversion."""
    restorer = _get_restorer(restorer_model)
    # GFPGANer.enhance returns (cropped_faces, restored_faces, restored_img).
    # paste_back=True puts faces back; weight blends with original.
    _, _, out = restorer.enhance(
        bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=float(weight),
    )
    return out


# ---- PIL <-> BGR helpers --------------------------------------------------

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """PIL RGB(A) → BGR uint8 numpy for OpenCV / InsightFace."""
    import cv2
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """BGR uint8 numpy → PIL RGB."""
    import cv2
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
