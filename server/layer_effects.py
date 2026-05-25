"""Layer effects rendered as new layers (drop shadow, stroke, outer glow).

Photoshop layer styles are non-destructive vector-rendered effects that
re-render at composite time. We don't have a styling layer; instead we
rasterize each effect as a new pixel layer inserted relative to the source.

All three effects key off the source layer's alpha mask — black-where-empty,
white-where-opaque. The shadow/glow then blurs that mask and tints it; the
stroke does an edge-detect-and-thicken.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .colors import parse_color
from .layers import Layer


def _alpha_mask(layer: Layer) -> Image.Image:
    """Return an L-mode image of the layer's alpha (after any layer mask)."""
    alpha = layer.image.split()[-1]
    if layer.mask is None:
        return alpha
    mask = layer.mask.convert("L").resize(alpha.size, Image.LANCZOS)
    arr = (np.asarray(alpha, dtype=np.float32)
           * np.asarray(mask, dtype=np.float32) / 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_drop_shadow_layer(source: Layer, *,
                           offset_x: int = 6, offset_y: int = 6,
                           blur: float = 8.0,
                           color: str = "#000000",
                           opacity: float = 0.6,
                           name: str | None = None) -> Layer:
    """Build a drop-shadow layer keyed on ``source``'s alpha.

    The returned layer is sized to the source image and positioned to align
    with it (the offset is applied to the alpha mask interior, not the
    layer's canvas-level offset, so the shadow stays attached to the source
    when the source moves).
    """
    src_alpha = _alpha_mask(source)
    w, h = src_alpha.size
    # Padding: enough for the shifted alpha to stay inside, plus 3σ of the
    # gaussian on each side so the blur tail isn't clipped.
    pad = int(max(abs(offset_x), abs(offset_y)) + blur * 3 + 2)
    big = Image.new("L", (w + pad * 2, h + pad * 2), 0)
    big.paste(src_alpha, (pad + offset_x, pad + offset_y))
    big = big.filter(ImageFilter.GaussianBlur(float(blur)))

    cr, cg, cb, _ = parse_color(color)
    op = max(0.0, min(1.0, float(opacity)))
    arr = np.asarray(big, dtype=np.float32) * op
    rgba_arr = np.zeros((*big.size[::-1], 4), dtype=np.uint8)
    rgba_arr[..., 0] = cr
    rgba_arr[..., 1] = cg
    rgba_arr[..., 2] = cb
    rgba_arr[..., 3] = arr.clip(0, 255).astype(np.uint8)
    rgba = Image.fromarray(rgba_arr, mode="RGBA")

    # Place so the shadow's reference point aligns with the source's.
    offset = (source.offset[0] - pad, source.offset[1] - pad)
    return Layer(
        name=name or f"{source.name} shadow",
        image=rgba,
        offset=offset,
    )


def make_outer_glow_layer(source: Layer, *,
                          blur: float = 12.0,
                          color: str = "#ffff80",
                          opacity: float = 0.8,
                          intensity: float = 1.0,
                          name: str | None = None) -> Layer:
    """Outer glow keyed on the source's alpha. ``intensity`` multiplies the
    pre-blur alpha (>1 makes the glow extend further before fading)."""
    src_alpha = _alpha_mask(source)
    w, h = src_alpha.size
    pad = int(blur * 3 + 4)
    big = Image.new("L", (w + pad * 2, h + pad * 2), 0)
    arr = (np.asarray(src_alpha, dtype=np.float32) * float(intensity)).clip(0, 255)
    big.paste(Image.fromarray(arr.astype(np.uint8), mode="L"), (pad, pad))
    big = big.filter(ImageFilter.GaussianBlur(float(blur)))

    cr, cg, cb, _ = parse_color(color)
    op = max(0.0, min(1.0, float(opacity)))
    arr2 = np.asarray(big, dtype=np.float32) * op
    rgba_arr = np.zeros((big.size[1], big.size[0], 4), dtype=np.uint8)
    rgba_arr[..., 0] = cr
    rgba_arr[..., 1] = cg
    rgba_arr[..., 2] = cb
    rgba_arr[..., 3] = arr2.clip(0, 255).astype(np.uint8)
    rgba = Image.fromarray(rgba_arr, mode="RGBA")
    return Layer(
        name=name or f"{source.name} glow",
        image=rgba,
        offset=(source.offset[0] - pad, source.offset[1] - pad),
        blend_mode="screen",
    )


def make_stroke_layer(source: Layer, *,
                      width: int = 4,
                      color: str = "#000000",
                      position: str = "outside",
                      name: str | None = None) -> Layer:
    """Render a stroke around the source's opaque region.

    ``position``:
      - ``outside``: stroke grows outward from the alpha edge
      - ``inside``: stroke is contained within the alpha
      - ``center``: half outside, half inside

    Works by dilating/eroding the alpha mask (via MaxFilter / MinFilter at
    multiple steps) and XOR-ing to isolate the band.
    """
    src_alpha = _alpha_mask(source)
    w, h = src_alpha.size
    pad = max(width + 2, 1)
    big = Image.new("L", (w + pad * 2, h + pad * 2), 0)
    big.paste(src_alpha, (pad, pad))

    # Build inner (eroded) and outer (dilated) masks.
    if position == "outside":
        inner = big
        outer = _morph(big, +width)
    elif position == "inside":
        inner = _morph(big, -width)
        outer = big
    else:  # center
        half_in = max(0, width // 2)
        half_out = width - half_in
        inner = _morph(big, -half_in)
        outer = _morph(big, +half_out)

    band = _ring_mask(outer, inner)

    cr, cg, cb, ca = parse_color(color)
    band_arr = np.asarray(band, dtype=np.uint8)
    rgba_arr = np.zeros((band_arr.shape[0], band_arr.shape[1], 4), dtype=np.uint8)
    rgba_arr[..., 0] = cr
    rgba_arr[..., 1] = cg
    rgba_arr[..., 2] = cb
    rgba_arr[..., 3] = (band_arr.astype(np.float32) * (ca / 255.0)).astype(np.uint8)
    rgba = Image.fromarray(rgba_arr, mode="RGBA")
    return Layer(
        name=name or f"{source.name} stroke",
        image=rgba,
        offset=(source.offset[0] - pad, source.offset[1] - pad),
    )


def _morph(mask: Image.Image, radius: int) -> Image.Image:
    """Positive radius dilates (MaxFilter); negative erodes (MinFilter).
    Done in single-pixel steps so kernel size stays small."""
    if radius == 0:
        return mask.copy()
    out = mask
    step = ImageFilter.MaxFilter(3) if radius > 0 else ImageFilter.MinFilter(3)
    for _ in range(abs(int(radius))):
        out = out.filter(step)
    return out


def _ring_mask(outer: Image.Image, inner: Image.Image) -> Image.Image:
    """Compute ``outer ∧ ¬inner`` — the band between two nested masks. Used to
    isolate a stroke from a dilated/eroded alpha. (Despite reading like XOR it
    isn't: since ``inner`` is always a subset of ``outer`` here, the symmetric
    difference reduces to one-sided subtraction.)"""
    arr_outer = np.asarray(outer, dtype=np.int16)
    arr_inner = np.asarray(inner, dtype=np.int16)
    band = np.clip(arr_outer - arr_inner, 0, 255).astype(np.uint8)
    return Image.fromarray(band, mode="L")
