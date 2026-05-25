"""Photoshop painting brushes: clone stamp, dodge, burn, blur, sharpen.

Each tool takes a stroke (a list of points) and applies a brush effect at
each step along it. Brushes use a soft round falloff so strokes feather
nicely at the edges.

Mutates the target image in place and returns it (matching the convention of
:mod:`drawing`). For all stroke-based tools, density is one stamp per
``brush_spacing`` pixels — at the default this produces a smooth line for
brushes ≥ 4 px.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageFilter


# ---- stroke helpers --------------------------------------------------------

def _densify(points: Sequence[Sequence[int]], spacing: float) -> list[tuple[int, int]]:
    """Walk the polyline and emit one (x, y) every ``spacing`` pixels."""
    if not points:
        return []
    out: list[tuple[int, int]] = [(int(points[0][0]), int(points[0][1]))]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1][0], points[i - 1][1]
        x1, y1 = points[i][0], points[i][1]
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        n = max(1, int(dist / max(spacing, 0.5)))
        for k in range(1, n + 1):
            t = k / n
            out.append((int(x0 + dx * t), int(y0 + dy * t)))
    return out


def _make_round_brush(radius: int, hardness: float = 0.5) -> np.ndarray:
    """A (2r+1)x(2r+1) brush mask with soft falloff. Values in 0..1.

    ``hardness`` 0 → smoothest, 1 → hard-edged disc. The transition zone is
    ``radius * (1 - hardness)``.
    """
    r = max(1, int(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    dist = np.sqrt(xx * xx + yy * yy)
    inner = r * float(np.clip(hardness, 0.0, 1.0))
    outer = r
    mask = np.where(
        dist <= inner, 1.0,
        np.where(dist >= outer, 0.0, (outer - dist) / max(outer - inner, 1e-6)),
    )
    return mask.astype(np.float32)


def _stamp(image_arr: np.ndarray, brush: np.ndarray, cx: int, cy: int,
           fn) -> None:
    """Apply ``fn(region_view, brush_window)`` to the brush footprint at
    (cx, cy). ``fn`` mutates ``region_view`` in place — caller decides what
    the brush does there (paint, dodge, blur, ...)."""
    h, w = image_arr.shape[:2]
    br = brush.shape[0] // 2
    x0, y0 = cx - br, cy - br
    x1, y1 = cx + br + 1, cy + br + 1
    # Clip against the image bounds
    bx0 = max(0, -x0); by0 = max(0, -y0)
    ix0 = max(0, x0); iy0 = max(0, y0)
    ix1 = min(w, x1); iy1 = min(h, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    bx1 = bx0 + (ix1 - ix0)
    by1 = by0 + (iy1 - iy0)
    region = image_arr[iy0:iy1, ix0:ix1]
    bwin = brush[by0:by1, bx0:bx1, np.newaxis]
    fn(region, bwin)


# ---- clone stamp -----------------------------------------------------------

def clone_stamp(img: Image.Image, points: Sequence[Sequence[int]], *,
                source_x: int, source_y: int,
                size: int = 20, hardness: float = 0.5,
                opacity: float = 1.0,
                source_img: Image.Image | None = None) -> Image.Image:
    """Photoshop Clone Stamp: copy pixels from a source location to the stroke.

    The source point ``(source_x, source_y)`` lines up with the first stroke
    point; subsequent stamps move the source along with the stroke (Photoshop's
    "Aligned" mode). Optionally sample from a different image via ``source_img``.
    """
    if not points:
        return img
    target = np.array(img.convert("RGBA"), dtype=np.float32)
    src = (np.array(source_img.convert("RGBA"), dtype=np.float32)
           if source_img is not None else target.copy())

    brush = _make_round_brush(size // 2, hardness)
    op = float(np.clip(opacity, 0.0, 1.0))

    pts = _densify(points, spacing=max(1.0, size * 0.25))
    if not pts:
        return img
    sx0, sy0 = int(source_x), int(source_y)
    x0, y0 = pts[0]
    dx_src = sx0 - x0
    dy_src = sy0 - y0

    h, w = target.shape[:2]
    sh, sw = src.shape[:2]
    br = brush.shape[0] // 2

    for (cx, cy) in pts:
        # Sampling center in the source image
        scx = cx + dx_src
        scy = cy + dy_src

        # Compute the visible destination window — same calculation _stamp
        # uses internally, repeated here so we can map matching source coords.
        dx_lo = max(0, cx - br); dy_lo = max(0, cy - br)
        dx_hi = min(w, cx + br + 1); dy_hi = min(h, cy + br + 1)
        if dx_lo >= dx_hi or dy_lo >= dy_hi:
            continue
        # Translate destination pixels into source coords by adding (dx_src, dy_src).
        # Clip the source window to the source image too; if any side falls
        # outside, narrow the destination window symmetrically so source and
        # destination sub-windows stay aligned.
        sx_lo = dx_lo + dx_src
        sy_lo = dy_lo + dy_src
        sx_hi = dx_hi + dx_src
        sy_hi = dy_hi + dy_src
        # Clip source vs. its image bounds; transfer any clip back to dest.
        left_clip = max(0, -sx_lo)
        top_clip = max(0, -sy_lo)
        right_clip = max(0, sx_hi - sw)
        bot_clip = max(0, sy_hi - sh)
        dx_lo += left_clip; dy_lo += top_clip
        dx_hi -= right_clip; dy_hi -= bot_clip
        sx_lo += left_clip; sy_lo += top_clip
        sx_hi -= right_clip; sy_hi -= bot_clip
        if dx_lo >= dx_hi or dy_lo >= dy_hi:
            continue

        # Brush sub-window matching the aligned dst/src windows.
        b_dx_lo = dx_lo - (cx - br); b_dy_lo = dy_lo - (cy - br)
        b_dx_hi = dx_hi - (cx - br); b_dy_hi = dy_hi - (cy - br)

        alpha = brush[b_dy_lo:b_dy_hi, b_dx_lo:b_dx_hi, np.newaxis] * op
        sample = src[sy_lo:sy_hi, sx_lo:sx_hi]
        region = target[dy_lo:dy_hi, dx_lo:dx_hi]
        region[:] = region * (1 - alpha) + sample * alpha

    return Image.fromarray(np.clip(target, 0, 255).astype(np.uint8), mode="RGBA")


# ---- dodge / burn ----------------------------------------------------------

def _dodge_burn_stroke(img: Image.Image, points: Sequence[Sequence[int]], *,
                       size: int, hardness: float, exposure: float,
                       direction: int) -> Image.Image:
    """Shared implementation: direction=+1 dodge (lighten), -1 burn (darken)."""
    if not points:
        return img
    arr = np.array(img.convert("RGBA"), dtype=np.float32) / 255.0
    brush = _make_round_brush(size // 2, hardness)
    strength = float(np.clip(exposure, 0.0, 1.0))

    pts = _densify(points, spacing=max(1.0, size * 0.25))
    for (cx, cy) in pts:

        def adj(region, bwin):
            amount = bwin[..., 0:1] * strength
            if direction > 0:
                # Dodge: push toward white
                region[..., :3] = region[..., :3] + (1.0 - region[..., :3]) * amount
            else:
                # Burn: push toward black
                region[..., :3] = region[..., :3] * (1.0 - amount)

        _stamp(arr, brush, cx, cy, adj)

    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def dodge_brush(img: Image.Image, points: Sequence[Sequence[int]], *,
                size: int = 20, hardness: float = 0.5,
                exposure: float = 0.3) -> Image.Image:
    """Brighten pixels along a stroke. ``exposure`` 0..1 controls strength per
    stamp (Photoshop's "Exposure" slider). Multiple stamps build up further."""
    return _dodge_burn_stroke(img, points, size=size, hardness=hardness,
                              exposure=exposure, direction=+1)


def burn_brush(img: Image.Image, points: Sequence[Sequence[int]], *,
               size: int = 20, hardness: float = 0.5,
               exposure: float = 0.3) -> Image.Image:
    """Darken pixels along a stroke."""
    return _dodge_burn_stroke(img, points, size=size, hardness=hardness,
                              exposure=exposure, direction=-1)


# ---- blur / sharpen brushes ------------------------------------------------

def blur_brush(img: Image.Image, points: Sequence[Sequence[int]], *,
               size: int = 20, hardness: float = 0.5,
               strength: float = 0.5,
               radius: float = 2.0) -> Image.Image:
    """Locally blur along a stroke. ``radius`` is the gaussian blur radius
    applied within the stamp footprint; ``strength`` 0..1 blends the blurred
    pixels onto the original."""
    if not points:
        return img
    rgba = img.convert("RGBA")
    blurred = rgba.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    target = np.array(rgba, dtype=np.float32)
    bsrc = np.array(blurred, dtype=np.float32)

    brush = _make_round_brush(size // 2, hardness)
    s = float(np.clip(strength, 0.0, 1.0))

    pts = _densify(points, spacing=max(1.0, size * 0.25))
    for (cx, cy) in pts:

        def mix(region, bwin):
            br = brush.shape[0] // 2
            h, w = bsrc.shape[:2]
            x0 = max(0, cx - br); y0 = max(0, cy - br)
            x1 = min(w, cx + br + 1); y1 = min(h, cy + br + 1)
            sample = bsrc[y0:y1, x0:x1]
            alpha = bwin * s
            region[:] = region * (1 - alpha) + sample * alpha

        _stamp(target, brush, cx, cy, mix)

    return Image.fromarray(np.clip(target, 0, 255).astype(np.uint8), mode="RGBA")


def sharpen_brush(img: Image.Image, points: Sequence[Sequence[int]], *,
                  size: int = 20, hardness: float = 0.5,
                  strength: float = 0.5) -> Image.Image:
    """Locally sharpen via unsharp mask along a stroke."""
    if not points:
        return img
    rgba = img.convert("RGBA")
    sharpened = rgba.filter(ImageFilter.UnsharpMask(radius=2.0, percent=200))
    target = np.array(rgba, dtype=np.float32)
    src_arr = np.array(sharpened, dtype=np.float32)

    brush = _make_round_brush(size // 2, hardness)
    s = float(np.clip(strength, 0.0, 1.0))

    pts = _densify(points, spacing=max(1.0, size * 0.25))
    for (cx, cy) in pts:

        def mix(region, bwin):
            br = brush.shape[0] // 2
            h, w = src_arr.shape[:2]
            x0 = max(0, cx - br); y0 = max(0, cy - br)
            x1 = min(w, cx + br + 1); y1 = min(h, cy + br + 1)
            sample = src_arr[y0:y1, x0:x1]
            alpha = bwin * s
            region[:] = region * (1 - alpha) + sample * alpha

        _stamp(target, brush, cx, cy, mix)

    return Image.fromarray(np.clip(target, 0, 255).astype(np.uint8), mode="RGBA")
