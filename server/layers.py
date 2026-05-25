"""Layer model + Photoshop-style blend mode compositor.

A ``Layer`` is an image with metadata: position offset within the canvas,
opacity 0..1, visibility, a blend mode, and an optional grayscale mask.
``compose_layers`` flattens a stack into a single RGBA image of the canvas
dimensions.

Compositing math is done in 0..1 float space via numpy. The blend formula
combines two stages:

1. A per-channel blend function ``B(Cs, Cb)`` (one of 14 PS-compatible modes).
2. Porter-Duff source-over with alpha so partially-transparent layers behave
   correctly: ``co = Cs·αs + Cb·αb·(1-αs)``, plus the alpha-weighted blend.

The exact formula used (W3C Compositing-1 spec):

    αo  = αs + αb·(1-αs)
    co  = (1-αb)·αs·Cs + (1-αs)·αb·Cb + αs·αb·B(Cs, Cb)
    Co  = co / αo                   when αo > 0

This matches Photoshop's compositing behaviour for non-pass-through groups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


BLEND_MODES = [
    "normal", "multiply", "screen", "overlay",
    "darken", "lighten",
    "color_dodge", "color_burn",
    "hard_light", "soft_light",
    "difference", "exclusion",
    "add", "subtract",
]


@dataclass
class Layer:
    name: str
    image: Image.Image  # RGBA, any size (offset within canvas)
    offset: tuple[int, int] = (0, 0)
    opacity: float = 1.0
    visible: bool = True
    blend_mode: str = "normal"
    mask: Image.Image | None = None  # L mode, same size as image, or None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.image.width,
            "height": self.image.height,
            "offset": list(self.offset),
            "opacity": self.opacity,
            "visible": self.visible,
            "blend_mode": self.blend_mode,
            "has_mask": self.mask is not None,
        }

    def copy(self) -> "Layer":
        return Layer(
            name=self.name,
            image=self.image.copy(),
            offset=self.offset,
            opacity=self.opacity,
            visible=self.visible,
            blend_mode=self.blend_mode,
            mask=self.mask.copy() if self.mask is not None else None,
        )


# ---- compositing -----------------------------------------------------------

def _to_canvas_array(layer: Layer, canvas_size: tuple[int, int]) -> np.ndarray:
    """Render ``layer`` onto a canvas-sized HxWx4 float array, applying offset,
    layer mask, and per-layer opacity to the alpha channel.

    Returns zeros outside the layer's footprint so downstream blending treats
    those pixels as fully transparent.
    """
    w, h = canvas_size
    canvas = np.zeros((h, w, 4), dtype=np.float32)
    if not layer.visible or layer.opacity <= 0:
        return canvas

    lw, lh = layer.image.size
    ox, oy = layer.offset

    # Intersection of [0,lw)x[0,lh) shifted by (ox,oy) with [0,w)x[0,h).
    sx1 = max(0, -ox)
    sy1 = max(0, -oy)
    sx2 = min(lw, w - ox)
    sy2 = min(lh, h - oy)
    if sx1 >= sx2 or sy1 >= sy2:
        return canvas

    src = np.asarray(layer.image.convert("RGBA"), dtype=np.float32) / 255.0
    region = src[sy1:sy2, sx1:sx2].copy()

    if layer.mask is not None:
        mask_arr = np.asarray(layer.mask.convert("L"), dtype=np.float32) / 255.0
        if mask_arr.shape != src.shape[:2]:
            # Mismatched mask: resize to match layer image so it always applies.
            # Happens after layer-image transforms that didn't touch the mask.
            mask_resized = layer.mask.convert("L").resize(
                (lw, lh), Image.LANCZOS,
            )
            mask_arr = np.asarray(mask_resized, dtype=np.float32) / 255.0
        region[..., 3] *= mask_arr[sy1:sy2, sx1:sx2]

    region[..., 3] *= float(layer.opacity)

    dx1 = sx1 + ox
    dy1 = sy1 + oy
    canvas[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = region
    return canvas


def _blend_rgb(src_rgb: np.ndarray, dst_rgb: np.ndarray, mode: str) -> np.ndarray:
    """Per-channel blend function ``B(Cs, Cb)`` in 0..1 space."""
    if mode == "normal":
        return src_rgb
    if mode == "multiply":
        return src_rgb * dst_rgb
    if mode == "screen":
        return 1.0 - (1.0 - src_rgb) * (1.0 - dst_rgb)
    if mode == "darken":
        return np.minimum(src_rgb, dst_rgb)
    if mode == "lighten":
        return np.maximum(src_rgb, dst_rgb)
    if mode == "overlay":
        return np.where(
            dst_rgb < 0.5,
            2.0 * src_rgb * dst_rgb,
            1.0 - 2.0 * (1.0 - src_rgb) * (1.0 - dst_rgb),
        )
    if mode == "hard_light":
        return np.where(
            src_rgb < 0.5,
            2.0 * src_rgb * dst_rgb,
            1.0 - 2.0 * (1.0 - src_rgb) * (1.0 - dst_rgb),
        )
    if mode == "soft_light":
        # Pegtop's formula — continuous and a good visual match to PS.
        return (1.0 - 2.0 * src_rgb) * dst_rgb * dst_rgb + 2.0 * src_rgb * dst_rgb
    if mode == "color_dodge":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                src_rgb >= 1.0,
                1.0,
                np.minimum(1.0, dst_rgb / np.maximum(1.0 - src_rgb, 1e-10)),
            )
    if mode == "color_burn":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                src_rgb <= 0.0,
                0.0,
                1.0 - np.minimum(1.0, (1.0 - dst_rgb) / np.maximum(src_rgb, 1e-10)),
            )
    if mode == "difference":
        return np.abs(src_rgb - dst_rgb)
    if mode == "exclusion":
        return src_rgb + dst_rgb - 2.0 * src_rgb * dst_rgb
    if mode == "add":
        return np.minimum(1.0, src_rgb + dst_rgb)
    if mode == "subtract":
        return np.maximum(0.0, dst_rgb - src_rgb)
    # Fall back to normal for unknown modes rather than raise — keeps compositing
    # robust if a PSD ships an exotic blend the loader didn't normalize.
    return src_rgb


def _composite_over(dst: np.ndarray, src: np.ndarray, mode: str) -> np.ndarray:
    """Source-over composite of ``src`` onto ``dst`` using blend ``mode``.

    Both arrays are HxWx4 float in 0..1. The output is the same shape; the
    result alpha follows Porter-Duff source-over.
    """
    dst_rgb = dst[..., :3]
    src_rgb = src[..., :3]
    dst_a = dst[..., 3:4]
    src_a = src[..., 3:4]

    blended = _blend_rgb(src_rgb, dst_rgb, mode)

    # W3C Compositing-1: alpha-weighted mix of three regions
    # (source-only, dest-only, intersection-blended).
    co = (
        (1.0 - dst_a) * src_a * src_rgb
        + (1.0 - src_a) * dst_a * dst_rgb
        + src_a * dst_a * blended
    )
    ao = src_a + dst_a * (1.0 - src_a)

    safe = np.where(ao < 1e-10, 1.0, ao)
    Co = co / safe
    Co = np.where(ao < 1e-10, 0.0, Co)

    out = np.empty_like(dst)
    out[..., :3] = np.clip(Co, 0.0, 1.0)
    out[..., 3:4] = np.clip(ao, 0.0, 1.0)
    return out


def compose_layers(layers: list[Layer], canvas_size: tuple[int, int]) -> Image.Image:
    """Flatten ``layers`` (bottom-to-top) onto a fresh canvas. Returns RGBA."""
    w, h = canvas_size
    acc = np.zeros((h, w, 4), dtype=np.float32)
    for layer in layers:
        if not layer.visible or layer.opacity <= 0:
            continue
        src = _to_canvas_array(layer, canvas_size)
        acc = _composite_over(acc, src, layer.blend_mode)
    rgba = (np.clip(acc, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


# ---- mask helpers ----------------------------------------------------------

def make_white_mask(size: tuple[int, int]) -> Image.Image:
    """A fully-revealing layer mask."""
    return Image.new("L", size, 255)


def invert_mask(mask: Image.Image) -> Image.Image:
    """Invert a grayscale mask."""
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    return Image.fromarray(255 - arr, mode="L")
