"""Photoshop-style pattern tools — define, fill, stamp, overlay, make seamless.

Patterns are PIL images stored in a process-local registry, keyed by user-chosen
name. They survive across tool calls within the same MCP-server process but
not across restarts (you'd reload via ``define_pattern_from_file`` on demand).

Tools:
- ``define_pattern`` / ``define_pattern_from_file`` / ``list_patterns`` /
  ``delete_pattern`` — library management.
- ``fill_pattern`` — tile a pattern across a region (or whole canvas), with
  scale / rotation / opacity / blend-mode controls. Mirrors PS Edit > Fill >
  Pattern.
- ``pattern_stamp`` — paint with the pattern as a brush along a stroke,
  sampling from the canvas-aligned tile. PS Pattern Stamp tool.
- ``pattern_overlay`` — apply the pattern as a new layer with blend mode +
  opacity, like the layer style of the same name.
- ``make_seamless`` — wrap-offset-and-feather to make any image a seamless
  tile.
"""
from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

# Pattern registry: in-process, name -> PIL.Image. Names are case-sensitive
# strings — no path-style restrictions (the registry is internal, not a
# filesystem). Concurrent tool calls are guarded by ``_lock``.
_lock = threading.Lock()
_patterns: dict[str, Image.Image] = {}


# ---- library management ---------------------------------------------------

def define_pattern(name: str, image: Image.Image) -> dict[str, Any]:
    """Store ``image`` under ``name``. Overwrites any existing pattern with
    the same name. Returns the registry's resulting size."""
    if not name:
        raise ValueError("pattern name must be non-empty")
    img = image.convert("RGBA").copy()
    with _lock:
        _patterns[name] = img
    return {
        "stored": name,
        "size": list(img.size),
        "mode": img.mode,
        "library_size": len(_patterns),
    }


def get_pattern(name: str) -> Image.Image:
    with _lock:
        if name not in _patterns:
            raise KeyError(
                f"unknown pattern {name!r}. Known: {sorted(_patterns)}"
            )
        # Return a copy so a caller mutating doesn't poison the cache.
        return _patterns[name].copy()


def list_patterns() -> dict[str, Any]:
    """Library contents — names + tile sizes."""
    with _lock:
        return {
            "patterns": [
                {"name": n, "size": list(img.size), "mode": img.mode}
                for n, img in sorted(_patterns.items())
            ],
        }


def delete_pattern(name: str) -> dict[str, Any]:
    """Drop a pattern from the registry. Returns ``{"deleted": true/false}``."""
    with _lock:
        had = name in _patterns
        _patterns.pop(name, None)
    return {"deleted": had, "name": name}


# ---- tile core ------------------------------------------------------------

def _tile(pattern: Image.Image, target_size: tuple[int, int], *,
          scale: float = 1.0, rotation: float = 0.0,
          offset_x: int = 0, offset_y: int = 0) -> Image.Image:
    """Tile ``pattern`` to fill ``target_size``. Scale resamples the pattern
    first; rotation rotates the resampled tile (with edge fill) BEFORE
    tiling; offset shifts the tile origin so seams can be re-aligned.

    Returns an RGBA image of exactly ``target_size``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    tw, th = pattern.size
    # 1. Scale the tile.
    if abs(scale - 1.0) > 1e-3:
        tw_s = max(1, int(round(tw * scale)))
        th_s = max(1, int(round(th * scale)))
        tile = pattern.resize((tw_s, th_s), Image.LANCZOS)
    else:
        tile = pattern
    # 2. Optionally rotate the tile.
    if abs(rotation) > 1e-3:
        # ``expand=True`` keeps the rotated content; transparent fill.
        tile = tile.rotate(rotation, resample=Image.BICUBIC, expand=True)
    # 3. Allocate target + paste copies.
    tw, th = tile.size
    target_w, target_h = target_size
    out = Image.new("RGBA", target_size, (0, 0, 0, 0))
    # Negative origin so partial tiles cover left/top edges.
    ox = -((offset_x % tw) if tw > 0 else 0)
    oy = -((offset_y % th) if th > 0 else 0)
    y = oy
    while y < target_h:
        x = ox
        while x < target_w:
            out.paste(tile, (x, y), tile if tile.mode == "RGBA" else None)
            x += tw
        y += th
    return out


# ---- blend helpers --------------------------------------------------------

def _blend_normal(base: Image.Image, overlay: Image.Image,
                  *, opacity: float = 1.0) -> Image.Image:
    """Normal compositing (overlay on top of base) at ``opacity`` 0-1.
    Both inputs RGBA, same size. Returns RGBA."""
    if abs(opacity - 1.0) > 1e-3:
        # Reduce overlay's alpha channel by ``opacity``.
        ov = overlay.copy()
        a = ov.split()[-1].point(lambda v: int(v * opacity))
        ov.putalpha(a)
    else:
        ov = overlay
    return Image.alpha_composite(base, ov)


def _blend_multiply(base: Image.Image, overlay: Image.Image,
                    *, opacity: float = 1.0) -> Image.Image:
    """``base * overlay / 255`` per channel, opacity-modulated by overlay alpha."""
    b = base.convert("RGBA")
    o = overlay.convert("RGBA")
    blended = ImageChops.multiply(b, o)
    return _blend_normal(b, _retint_alpha(blended, o, opacity), opacity=1.0)


def _blend_screen(base: Image.Image, overlay: Image.Image,
                  *, opacity: float = 1.0) -> Image.Image:
    b = base.convert("RGBA")
    o = overlay.convert("RGBA")
    blended = ImageChops.screen(b, o)
    return _blend_normal(b, _retint_alpha(blended, o, opacity), opacity=1.0)


def _blend_overlay(base: Image.Image, overlay: Image.Image,
                   *, opacity: float = 1.0) -> Image.Image:
    """PS Overlay = (a < 0.5) ? 2*a*b : 1 - 2*(1-a)*(1-b). Applied per channel."""
    b = np.asarray(base.convert("RGBA"), dtype=np.float32) / 255.0
    o = np.asarray(overlay.convert("RGBA"), dtype=np.float32) / 255.0
    rgb_b, alpha_b = b[..., :3], b[..., 3:4]
    rgb_o, alpha_o = o[..., :3], o[..., 3:4]
    mask = rgb_b < 0.5
    out = np.where(mask, 2 * rgb_b * rgb_o, 1.0 - 2.0 * (1.0 - rgb_b) * (1.0 - rgb_o))
    a = alpha_o * opacity
    rgb = rgb_b * (1.0 - a) + out * a
    out_alpha = alpha_b + a * (1.0 - alpha_b)
    arr = np.concatenate([rgb, out_alpha], axis=-1)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA")


def _retint_alpha(blended_rgba: Image.Image, ref_overlay: Image.Image,
                  opacity: float) -> Image.Image:
    """Multiply image's RGB by 1, but reset alpha to ref_overlay's alpha
    × opacity. Used so blend-mode results carry the overlay's coverage,
    not the result of blending the alpha channels."""
    out = blended_rgba.copy()
    a = ref_overlay.split()[-1].point(lambda v: int(v * opacity))
    out.putalpha(a)
    return out


_BLENDS = {
    "normal": _blend_normal,
    "multiply": _blend_multiply,
    "screen": _blend_screen,
    "overlay": _blend_overlay,
}


def _blend(base: Image.Image, overlay: Image.Image, *,
           mode: str = "normal", opacity: float = 1.0) -> Image.Image:
    fn = _BLENDS.get(mode.lower())
    if fn is None:
        raise ValueError(
            f"unknown blend mode {mode!r}. Supported: {sorted(_BLENDS)}"
        )
    return fn(base, overlay, opacity=opacity)


# ---- fill / overlay / stamp -----------------------------------------------

def fill_with_pattern(image: Image.Image, pattern: Image.Image, *,
                      bbox: tuple[int, int, int, int] | None = None,
                      scale: float = 1.0, rotation: float = 0.0,
                      offset_x: int = 0, offset_y: int = 0,
                      opacity: float = 1.0,
                      blend_mode: str = "normal") -> Image.Image:
    """Fill ``image`` (or the ``bbox`` region) with the tiled ``pattern``.
    Returns a new RGBA image the same size as ``image``."""
    base = image.convert("RGBA")
    target_size = base.size
    tiled = _tile(pattern, target_size, scale=scale, rotation=rotation,
                  offset_x=offset_x, offset_y=offset_y)
    if bbox is not None:
        # Restrict the tile to the bbox by zeroing alpha outside.
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(target_size[0], int(x2)); y2 = min(target_size[1], int(y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"empty bbox: {bbox}")
        mask = Image.new("L", target_size, 0)
        ImageDraw.Draw(mask).rectangle([x1, y1, x2 - 1, y2 - 1], fill=255)
        tiled.putalpha(ImageChops.multiply(tiled.split()[-1], mask))
    return _blend(base, tiled, mode=blend_mode, opacity=opacity)


def make_pattern_overlay_layer(pattern: Image.Image,
                               target_size: tuple[int, int], *,
                               scale: float = 1.0,
                               rotation: float = 0.0,
                               offset_x: int = 0,
                               offset_y: int = 0,
                               opacity: float = 1.0) -> Image.Image:
    """Just the tiled overlay layer (no compositing) so the caller can add
    it to a layered canvas with its preferred blend mode."""
    tiled = _tile(pattern, target_size, scale=scale, rotation=rotation,
                  offset_x=offset_x, offset_y=offset_y)
    if abs(opacity - 1.0) > 1e-3:
        a = tiled.split()[-1].point(lambda v: int(v * opacity))
        tiled.putalpha(a)
    return tiled


def _round_brush_mask(size: int, hardness: float = 0.5) -> Image.Image:
    """Round soft brush as an L-mode image. ``hardness`` 0-1: 0 = fully
    gradient (Gaussian-ish edge), 1 = hard edge."""
    s = max(1, int(size))
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    cx = cy = (s - 1) / 2.0
    r = (s - 1) / 2.0
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0)
    # Smoothstep from hardness to 1 — alpha=1 inside hardness*r, falls to 0 at r.
    h = max(0.0, min(1.0, float(hardness)))
    edge_start = h
    edge_end = 1.0
    t = np.clip((d - edge_start) / max(edge_end - edge_start, 1e-3), 0.0, 1.0)
    alpha = 1.0 - (t * t * (3.0 - 2.0 * t))  # smoothstep
    alpha[d > 1.0] = 0.0
    return Image.fromarray((alpha * 255).astype(np.uint8), mode="L")


def stamp_pattern(image: Image.Image, pattern: Image.Image,
                  points: list[tuple[int, int]], *,
                  brush_size: int = 64, hardness: float = 0.5,
                  scale: float = 1.0, rotation: float = 0.0,
                  opacity: float = 1.0) -> Image.Image:
    """Paint with ``pattern`` as a soft brush along ``points``. Each point
    deposits a circular sample of the canvas-aligned tile.

    The pattern is sampled IN CANVAS SPACE — moving the brush over the
    canvas reveals different parts of the tiling, the way Photoshop's
    Pattern Stamp tool works. This makes overlapping strokes line up.

    ``brush_size`` is the diameter in pixels; ``hardness`` 0-1 softens the
    edge; ``opacity`` 0-1 modulates the deposited alpha.
    """
    if not points:
        return image.convert("RGBA")
    base = image.convert("RGBA")
    target_size = base.size
    # Pre-tile once at canvas size + scale/rotation. Stamp samples from this.
    tiled = _tile(pattern, target_size, scale=scale, rotation=rotation)
    # Per-stamp brush alpha mask.
    brush = _round_brush_mask(brush_size, hardness=hardness)
    if abs(opacity - 1.0) > 1e-3:
        brush = brush.point(lambda v: int(v * opacity))
    out = base
    radius = brush_size // 2
    for p in points:
        x, y = int(p[0]) - radius, int(p[1]) - radius
        # Crop the tile region under the brush.
        crop_box = (x, y, x + brush_size, y + brush_size)
        sample = tiled.crop(crop_box).convert("RGBA")
        # Multiply the sample's alpha by the brush mask.
        sample.putalpha(ImageChops.multiply(sample.split()[-1], brush))
        # Paste back at the same location.
        out.paste(sample, (x, y), sample)
    return out


def make_seamless(image: Image.Image, *,
                  blend_width: int | None = None) -> Image.Image:
    """Turn ``image`` into a seamless tile.

    Classic offset-and-feather: roll the image by (W/2, H/2) so the original
    seams sit interior, then linearly cross-fade a strip of width
    ``blend_width`` along each seam line.

    ``blend_width`` defaults to 1/16 of the shorter edge (capped between 4
    and 64 px). Result tile is the same size as ``image``.
    """
    src = image.convert("RGBA")
    arr = np.asarray(src, dtype=np.float32)
    h, w = arr.shape[:2]
    bw = blend_width if blend_width is not None else max(4, min(64, min(w, h) // 16))
    bw = max(1, int(bw))
    # 1. Wrap-shift so seams move from edges to centre.
    shifted = np.roll(np.roll(arr, w // 2, axis=1), h // 2, axis=0)
    seam_x = w // 2
    seam_y = h // 2
    # 2. Horizontal seam: blend a strip of width 2*bw centred on seam_x with
    #    the wrapped mirror (i.e. the pre-shift content from the other side).
    # We achieve this by blending ``shifted`` with itself rolled half-width
    # only around the seam.
    def _crossfade_axis(buf: np.ndarray, seam: int, bw: int,
                        axis: int) -> np.ndarray:
        # Build a 1-D linear weight that peaks at the seam and falls to 0
        # at ±bw, then broadcast to the right shape.
        idx = np.arange(buf.shape[axis], dtype=np.float32)
        dist = np.abs(idx - seam)
        w = np.clip(1.0 - dist / float(bw), 0.0, 1.0)  # triangular kernel
        # Mirror buffer: average around the seam by half-rolling along axis.
        mirror = np.roll(buf, buf.shape[axis] // 2, axis=axis)
        shape = [1] * buf.ndim
        shape[axis] = -1
        w = w.reshape(shape)
        return buf * (1.0 - w) + mirror * w

    blended = _crossfade_axis(shifted, seam_x, bw, axis=1)
    blended = _crossfade_axis(blended, seam_y, bw, axis=0)
    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGBA")
    return out
