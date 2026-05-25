"""Photoshop-style blur effects beyond the basic Gaussian / box filters
already exposed via ``apply_filter``.

Five tools:
- ``motion_blur`` — directional linear streak (PS Filter > Blur > Motion Blur)
- ``radial_blur`` — spin (rotation) or zoom (radial) variant
  (PS Filter > Blur > Radial Blur)
- ``lens_blur`` — bokeh approximation via a disc / hex aperture kernel
  (PS Filter > Blur > Lens Blur)
- ``tilt_shift`` — horizontal-band-in-focus DOF effect
  (PS Filter > Blur > Tilt-Shift)
- ``box_blur`` — fast mean kernel (cheap alternative to gaussian)

All require OpenCV (already a transitive dep via segmentation / face /
warp).
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _check_cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError as e:
        raise RuntimeError(
            "Blur effects need opencv-python. "
            "`pip install opencv-python` (or install [seg]/[face]/[yolo]/[sd])."
        ) from e


def _arr(image: Image.Image) -> tuple[np.ndarray, str]:
    """Return (RGB or RGBA numpy uint8, mode). Used to preserve alpha
    through the blur pipeline."""
    has_alpha = image.mode == "RGBA"
    src = image.convert("RGBA" if has_alpha else "RGB")
    return np.asarray(src), src.mode


def _to_pil(arr: np.ndarray, mode: str) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=mode)


# ---- motion blur ---------------------------------------------------------

def motion_blur(image: Image.Image, *,
                angle: float = 0.0,
                distance: int = 20) -> Image.Image:
    """Directional motion blur — linear streak of length ``distance`` pixels
    at ``angle`` degrees (0 = horizontal, 90 = vertical, CCW). Matches PS
    Filter > Blur > Motion Blur."""
    cv2 = _check_cv2()
    if distance < 2:
        return image.copy()
    arr, mode = _arr(image)
    # Build a line kernel of length ``distance`` rotated to ``angle``.
    k = int(distance)
    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.line(kernel, (0, k // 2), (k - 1, k // 2), 1.0, thickness=1)
    if abs(angle) > 1e-3:
        M = cv2.getRotationMatrix2D((k / 2, k / 2), float(angle), 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s < 1e-6:
        return image.copy()
    kernel /= s
    out = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return _to_pil(out, mode)


# ---- radial blur (spin / zoom) ------------------------------------------

def radial_blur(image: Image.Image, *,
                mode: str = "spin",
                amount: float = 0.05,
                center_x: float | None = None,
                center_y: float | None = None,
                samples: int = 20) -> Image.Image:
    """Radial blur. PS Filter > Blur > Radial Blur.

    - ``mode='spin'``: rotational blur around ``(center_x, center_y)``.
      ``amount`` is the max rotation in radians at the image edge.
    - ``mode='zoom'``: radial blur outward from centre. ``amount`` is the
      max scale offset (0.1 = 10 %).
    - ``samples`` controls quality (higher = smoother but slower).

    Centre defaults to the image centre.
    """
    cv2 = _check_cv2()
    if amount <= 0 or samples < 2:
        return image.copy()
    arr, mode_str = _arr(image)
    h, w = arr.shape[:2]
    cx = float(center_x) if center_x is not None else (w - 1) / 2.0
    cy = float(center_y) if center_y is not None else (h - 1) / 2.0
    acc = np.zeros_like(arr, dtype=np.float64)
    samples = int(samples)
    for i in range(samples):
        # Sample fraction in [-amount/2, amount/2] for symmetric spread.
        t = (i / (samples - 1) - 0.5) * 2.0
        if mode == "spin":
            angle = t * float(amount) * (180.0 / np.pi)
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        elif mode == "zoom":
            scale = 1.0 + t * float(amount)
            M = cv2.getRotationMatrix2D((cx, cy), 0.0, scale)
        else:
            raise ValueError(
                f"unknown radial mode {mode!r}; choose spin or zoom"
            )
        warped = cv2.warpAffine(arr, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        acc += warped
    acc /= samples
    return _to_pil(acc, mode_str)


# ---- lens blur (bokeh) --------------------------------------------------

def _disc_kernel(radius: int) -> np.ndarray:
    """Circular disc kernel, normalised."""
    r = max(1, int(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    mask = (xx * xx + yy * yy) <= (r * r)
    k = mask.astype(np.float32)
    k /= k.sum()
    return k


def _hex_kernel(radius: int) -> np.ndarray:
    """Hexagonal-ish bokeh kernel — closer to a real iris aperture. Built
    by sampling whether each cell is inside a regular hexagon."""
    r = max(1, int(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    # Hexagon defined by 3 line constraints; here using horizontal-flat.
    h = float(r) * np.sqrt(3) / 2.0  # half-height
    inside = (np.abs(xx) <= r) & (np.abs(yy) <= h) & \
             ((np.abs(yy) + np.abs(xx) * np.sqrt(3) / 3.0) <= h * 2.0 / np.sqrt(3) * 1.0)
    # Looser polygon test — use the analytical inequality for a flat-top
    # hexagon (inradius = r * sqrt(3) / 2):
    inside = (
        (np.abs(yy) <= h) &
        ((np.abs(xx) * np.sqrt(3) + np.abs(yy)) <= 2 * h)
    )
    k = inside.astype(np.float32)
    s = k.sum()
    if s < 1:  # degenerate (very small radius)
        return _disc_kernel(radius)
    k /= s
    return k


def lens_blur(image: Image.Image, *, radius: int = 12,
              shape: str = "disc") -> Image.Image:
    """Bokeh-style lens blur using a disc or hexagonal aperture kernel —
    bright highlights bloom into shapes, unlike Gaussian which just
    softens. PS Filter > Blur > Lens Blur (without the depth-map input —
    pair with ``displace_by_map`` if you want depth-aware blur).

    - ``radius`` is the kernel radius in pixels (PS calls this "Iris Radius")
    - ``shape`` ∈ ``disc`` (circular, most common) | ``hex`` (six-bladed
      aperture)
    """
    cv2 = _check_cv2()
    if radius < 1:
        return image.copy()
    if shape == "disc":
        k = _disc_kernel(radius)
    elif shape == "hex":
        k = _hex_kernel(radius)
    else:
        raise ValueError(f"unknown shape {shape!r}; choose disc or hex")
    arr, mode = _arr(image)
    # filter2D works per-channel for multi-channel uint8.
    out = cv2.filter2D(arr, -1, k, borderType=cv2.BORDER_REPLICATE)
    return _to_pil(out, mode)


# ---- tilt-shift ---------------------------------------------------------

def tilt_shift(image: Image.Image, *,
               focus_y: float | None = None,
               focus_height: float = 0.25,
               max_blur: float = 12.0,
               falloff: float = 2.0) -> Image.Image:
    """Tilt-shift fake-miniature effect — a horizontal band stays in focus,
    everything above and below blurs increasingly. PS Filter > Blur >
    Tilt-Shift.

    - ``focus_y`` (0-1 vertical position): centre of the focus band.
      ``None`` = image centre.
    - ``focus_height`` (0-1): fraction of the image height that stays
      sharp.
    - ``max_blur`` (px): Gaussian radius at the top + bottom edges.
    - ``falloff`` (≥ 1): steepness of the focus → blur transition. 2 ≈
      smooth, 4 = sharp boundary.

    Implementation: blur once at ``max_blur``, then alpha-blend with the
    sharp original using a vertical weight ramp.
    """
    from PIL import ImageFilter
    cv2 = _check_cv2()  # ensures the OpenCV path the rest of the stack uses
    h, w = image.size[1], image.size[0]
    fy = float(focus_y) if focus_y is not None else 0.5
    fy_px = fy * h
    half = float(focus_height) * h * 0.5
    arr_sharp = np.asarray(
        image.convert("RGBA") if image.mode == "RGBA" else image.convert("RGB"),
        dtype=np.float32,
    )
    blurred_img = image.filter(ImageFilter.GaussianBlur(radius=float(max_blur)))
    arr_blur = np.asarray(
        blurred_img.convert("RGBA") if image.mode == "RGBA"
        else blurred_img.convert("RGB"),
        dtype=np.float32,
    )
    # Per-row weight: 0 inside focus band, → 1 at the edges (max_blur).
    y = np.arange(h, dtype=np.float32)
    dist = np.maximum(0.0, np.abs(y - fy_px) - half)
    far = max(h - fy_px - half, fy_px - half, 1e-3)
    weight = np.clip(dist / far, 0.0, 1.0) ** float(falloff)
    w_full = weight[:, None, None]  # broadcast over W and channels
    out = arr_sharp * (1.0 - w_full) + arr_blur * w_full
    return _to_pil(out, "RGBA" if image.mode == "RGBA" else "RGB")


# ---- box blur -----------------------------------------------------------

def box_blur(image: Image.Image, *, radius: int = 4) -> Image.Image:
    """Box (mean) blur — fast alternative to Gaussian when slightly harder
    edges are acceptable. ``radius`` is half the kernel size; kernel is
    ``(2*radius+1)`` square.
    """
    cv2 = _check_cv2()
    if radius < 1:
        return image.copy()
    arr, mode = _arr(image)
    k = int(radius) * 2 + 1
    out = cv2.boxFilter(arr, -1, (k, k),
                        borderType=cv2.BORDER_REPLICATE)
    return _to_pil(out, mode)
