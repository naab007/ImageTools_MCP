"""Gradient generation: linear / radial / angular / reflected / diamond.

A ``stops`` list defines the color ramp:

    [{"position": 0.0, "color": "#000000"},
     {"position": 1.0, "color": "#ffffff"}]

Positions are in 0..1. Colors accept anything :func:`colors.parse_color`
understands. ``make_gradient`` returns an RGBA image of the requested size.

Internally we precompute a 1024-entry color LUT once, then look it up per
pixel using a per-mode parameter ``t(x, y) ∈ [0, 1]``. This keeps even big
canvases fast (a 4K linear gradient is a single LUT walk in numpy).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from .colors import parse_color


GRADIENT_TYPES = ["linear", "radial", "angular", "reflected", "diamond"]

# All gradient lookups use a 1024-entry LUT — small enough to stay L1-resident,
# wide enough that 8-bit output across a 4K canvas hits ~every entry.
_LUT_SIZE = 1024


def _build_lut(stops: list[dict[str, Any]], n: int = _LUT_SIZE) -> np.ndarray:
    """Expand the stops list into an Nx4 uint8 color table."""
    if not stops or len(stops) < 2:
        raise ValueError("gradient needs at least two stops")
    # Normalize + sort by position
    norm = []
    for s in stops:
        p = float(s["position"])
        c = parse_color(s["color"])
        norm.append((max(0.0, min(1.0, p)), c))
    norm.sort(key=lambda x: x[0])

    # Build LUT by linear interp between consecutive stops
    lut = np.zeros((n, 4), dtype=np.float32)
    positions = np.linspace(0.0, 1.0, n)
    i = 0
    for k, t in enumerate(positions):
        # Advance i so norm[i].pos <= t <= norm[i+1].pos
        while i < len(norm) - 2 and t > norm[i + 1][0]:
            i += 1
        p0, c0 = norm[i]
        p1, c1 = norm[i + 1] if i + 1 < len(norm) else norm[-1]
        span = max(p1 - p0, 1e-9)
        frac = max(0.0, min(1.0, (t - p0) / span))
        lut[k] = (1 - frac) * np.array(c0) + frac * np.array(c1)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _t_linear(w: int, h: int, x1: float, y1: float,
              x2: float, y2: float) -> np.ndarray:
    """Per-pixel parameter for a linear gradient from (x1,y1) → (x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom < 1e-9:
        raise ValueError("linear gradient: start and end points coincide")
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    t = ((X - x1) * dx + (Y - y1) * dy) / denom
    return t


def _t_radial(w: int, h: int, cx: float, cy: float,
              radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radial gradient: radius must be > 0")
    xs = np.arange(w, dtype=np.float32) - cx
    ys = np.arange(h, dtype=np.float32) - cy
    X, Y = np.meshgrid(xs, ys)
    return np.sqrt(X * X + Y * Y) / radius


def _t_angular(w: int, h: int, cx: float, cy: float,
               start_angle_deg: float) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32) - cx
    ys = np.arange(h, dtype=np.float32) - cy
    X, Y = np.meshgrid(xs, ys)
    # atan2 returns -pi..pi → shift to 0..1
    angle = np.arctan2(Y, X)  # radians, -pi..pi
    rotated = angle - math.radians(start_angle_deg)
    t = (rotated % (2 * math.pi)) / (2 * math.pi)
    return t


def _t_reflected(w: int, h: int, x1: float, y1: float,
                 x2: float, y2: float) -> np.ndarray:
    """Reflected = linear mirrored around the midpoint."""
    t = _t_linear(w, h, x1, y1, x2, y2)
    return 1.0 - np.abs(2.0 * t - 1.0)


def _t_diamond(w: int, h: int, cx: float, cy: float,
               radius: float) -> np.ndarray:
    if radius <= 0:
        raise ValueError("diamond gradient: radius must be > 0")
    xs = np.arange(w, dtype=np.float32) - cx
    ys = np.arange(h, dtype=np.float32) - cy
    X, Y = np.meshgrid(xs, ys)
    return (np.abs(X) + np.abs(Y)) / radius


def make_gradient(size: tuple[int, int], gradient_type: str,
                  stops: list[dict[str, Any]], *,
                  x1: float = 0, y1: float = 0,
                  x2: float | None = None, y2: float | None = None,
                  center_x: float | None = None,
                  center_y: float | None = None,
                  radius: float | None = None,
                  start_angle: float = 0.0,
                  repeat: bool = False) -> Image.Image:
    """Generate a gradient as an RGBA image.

    Geometry args depend on ``gradient_type``:
    - ``linear`` / ``reflected``: ``x1,y1`` → ``x2,y2`` (default: left-to-right)
    - ``radial`` / ``diamond``:   ``center_x,center_y`` + ``radius``
                                  (default: canvas center, radius = half min dim)
    - ``angular``:                ``center_x,center_y`` + ``start_angle`` deg
                                  (default: canvas center, start = 0)
    """
    w, h = int(size[0]), int(size[1])
    if x2 is None: x2 = float(w - 1)
    if y2 is None: y2 = 0.0
    if center_x is None: center_x = (w - 1) / 2
    if center_y is None: center_y = (h - 1) / 2
    if radius is None: radius = min(w, h) / 2

    if gradient_type == "linear":
        t = _t_linear(w, h, x1, y1, x2, y2)
    elif gradient_type == "radial":
        t = _t_radial(w, h, center_x, center_y, radius)
    elif gradient_type == "angular":
        t = _t_angular(w, h, center_x, center_y, start_angle)
    elif gradient_type == "reflected":
        t = _t_reflected(w, h, x1, y1, x2, y2)
    elif gradient_type == "diamond":
        t = _t_diamond(w, h, center_x, center_y, radius)
    else:
        raise ValueError(
            f"unknown gradient_type {gradient_type!r}. Choose from {GRADIENT_TYPES}."
        )

    if repeat:
        t = t % 1.0
    else:
        t = np.clip(t, 0.0, 1.0)

    lut = _build_lut(stops)
    idx = (t * (len(lut) - 1) + 0.5).astype(np.int32)
    np.clip(idx, 0, len(lut) - 1, out=idx)
    rgba = lut[idx]
    return Image.fromarray(rgba, mode="RGBA")


def gradient_map(image: Image.Image, stops: list[dict[str, Any]]) -> Image.Image:
    """Photoshop's Gradient Map: remap each pixel's luminance through a gradient.

    Preserves the original alpha channel; ignores hue/saturation and keys only
    on grayscale luminance (BT.601 weights).
    """
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    lum = (0.299 * rgba[..., 0] + 0.587 * rgba[..., 1]
           + 0.114 * rgba[..., 2]) / 255.0
    lut = _build_lut(stops)
    idx = (np.clip(lum, 0, 1) * (len(lut) - 1) + 0.5).astype(np.int32)
    mapped = lut[idx]
    mapped[..., 3] = rgba[..., 3].astype(np.uint8)
    return Image.fromarray(mapped, mode="RGBA")
