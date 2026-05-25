"""ImageTools MCP server entry point.

Wires the @mcp.tool() surface and delegates to the helper modules:

- ``canvas``     in-memory image store with layer stack + undo/redo
- ``layers``     blend modes + compositing
- ``io_formats`` load / save / info (PNG/JPEG/WebP/HEIC/AVIF/SVG/RAW/PSD/...)
- ``drawing``    MS Paint primitives (line/rect/ellipse/text/brush/flood/eyedropper)
- ``transforms`` crop/resize/rotate/flip + filters + adjustments
- ``sd``         optional Stable Diffusion (txt2img/img2img/inpaint)

All drawing/transform tools snapshot the canvas before mutating so undo/redo
works. Drawing and filter ops target the **active layer**; canvas-wide
transforms (resize/crop/rotate/flip/add_border) flatten the stack and operate
on the composite. ``save_canvas`` and previews always show the composite of
visible layers. PSD save preserves the layer stack.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Sequence

from mcp.server.fastmcp import FastMCP, Image as MCPImage
from PIL import Image as PILImage

from . import (
    adjustments, birefnet, blurs, channels, clipseg, conversions, drawing,
    face_swap, gradients, gguf_io, io_formats, layer_effects,
    layers as layers_mod, mask_ops, painting, palette, patterns,
    preprocessors, qwen, sam, sam1, sd, stateless, transforms, utilities,
    warp, yolo_seg,
)
from .canvas import store
from .colors import parse_color

log = logging.getLogger("image-tools-mcp")
mcp = FastMCP("image-tools-mcp")


# ============================================================ helpers

def _img_to_png_bytes(img, *, max_size: int | None = None) -> bytes:
    """Encode for MCP transport. Optionally downscale for previews."""
    out = img
    if max_size and (out.width > max_size or out.height > max_size):
        ratio = max_size / max(out.width, out.height)
        out = out.resize(
            (max(1, int(out.width * ratio)), max(1, int(out.height * ratio))),
            resample=2,  # BILINEAR; fast for previews
        )
    if out.mode not in ("RGB", "RGBA", "L", "LA", "P"):
        out = out.convert("RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _summary(canvas_id: str) -> dict[str, Any]:
    """Standard response shape after a mutating op."""
    e = store.entry(canvas_id)
    return {
        "ok": True,
        "canvas_id": canvas_id,
        "width": e.width,
        "height": e.height,
        "n_layers": len(e.layers),
        "active_index": e.active_index,
    }


# ============================================================ canvas mgmt

@mcp.tool()
def new_canvas(width: int, height: int, *, background: str = "white",
               canvas_id: str | None = None) -> dict:
    """Create a blank single-layer canvas and return its ``canvas_id``.

    Background accepts CSS-style colors (``#ffcc00``, ``red``, ``transparent``)
    or ``[r, g, b]`` / ``[r, g, b, a]``.
    """
    bg = parse_color(background)
    img = PILImage.new("RGBA", (int(width), int(height)), bg)
    cid = store.put_image(img, canvas_id=canvas_id, layer_name="Background")
    return _summary(cid)


@mcp.tool()
def open_canvas(path: str, *, canvas_id: str | None = None,
                svg_width: int | None = None,
                svg_height: int | None = None) -> dict:
    """Load an image from disk into a new canvas.

    Supports PNG/JPEG/GIF/BMP/TIFF/WebP/ICO/PCX/TGA/PPM/DDS + HEIC/HEIF/AVIF
    (via pillow-heif) + SVG (via cairosvg, rasterized — pass svg_width/height
    to control size) + camera RAW (CR2/NEF/ARW/DNG/..., requires `[raw]` extra) +
    PSD (layered, via psd-tools — preserves layer stack with opacity/blend modes).
    """
    resolved = str(Path(path).resolve())
    if path.lower().endswith(".psd"):
        from .psd_io import load_psd

        width, height, lyrs = load_psd(path)
        cid = store.put_layers(width, height, lyrs, canvas_id=canvas_id,
                               path=resolved)
        return _summary(cid)
    img = io_formats.load_image(path, svg_width=svg_width, svg_height=svg_height)
    cid = store.put_image(img, canvas_id=canvas_id, path=resolved)
    return _summary(cid)


@mcp.tool()
def save_canvas(canvas_id: str, path: str, *, format: str | None = None,
                quality: int | None = None) -> dict:
    """Write the canvas to disk. Format is inferred from extension unless given.

    For .psd: layer stack is preserved (names, opacity, blend modes, masks,
    visibility, offsets). All other formats flatten visible layers into a
    composite first. JPEG additionally flattens alpha onto white.
    """
    if path.lower().endswith(".psd"):
        from .psd_io import save_psd

        e = store.entry(canvas_id)
        return save_psd(e.layers, (e.width, e.height), path)

    composite = store.compose(canvas_id)
    return io_formats.save_image(composite, path, format=format, quality=quality)


@mcp.tool()
def close_canvas(canvas_id: str) -> dict:
    """Drop a canvas from memory."""
    store.close(canvas_id)
    return {"ok": True, "canvas_id": canvas_id}


@mcp.tool()
def list_canvases() -> dict:
    """List all open canvases with their dimensions and layer counts."""
    return {"canvases": store.summary()}


@mcp.tool()
def get_canvas_info(canvas_id: str) -> dict:
    """Document-level info: dimensions, layer count, active layer, path."""
    e = store.entry(canvas_id)
    return {
        "canvas_id": canvas_id,
        "width": e.width,
        "height": e.height,
        "n_layers": len(e.layers),
        "active_index": e.active_index,
        "path": e.path,
        "undo_depth": len(e.undo_stack),
        "redo_depth": len(e.redo_stack),
    }


@mcp.tool()
def get_canvas_preview(canvas_id: str, *, max_size: int = 512) -> MCPImage:
    """Return a downscaled PNG preview of the composited canvas. Useful for
    inline display. Default max edge is 512 px."""
    png = _img_to_png_bytes(store.compose(canvas_id), max_size=max_size)
    return MCPImage(data=png, format="png")


@mcp.tool()
def screenshot_canvas(canvas_id: str, path: str | None = None, *,
                      format: str = "png", quality: int | None = None) -> dict:
    """Snapshot the current visible-layer composite at full resolution.

    Composites every layer whose ``visible`` flag is true (hidden layers are
    skipped — the result reflects exactly what `get_canvas_preview` would
    show, but at the canvas's native dimensions instead of downscaled).

    ``path`` may be omitted to save to a tempfile; the resolved path is
    always returned so the caller can surface the file."""
    import tempfile
    from pathlib import Path

    composite = store.compose(canvas_id)
    if path is None:
        ext = format.lower().lstrip(".") or "png"
        fd, tmp_path = tempfile.mkstemp(prefix=f"canvas_{canvas_id}_",
                                        suffix="." + ext)
        import os
        os.close(fd)
        path = tmp_path

    return io_formats.save_image(composite, path, format=format, quality=quality)


@mcp.tool()
def duplicate_canvas(canvas_id: str, *, new_canvas_id: str | None = None) -> dict:
    """Deep-copy a canvas (all layers, masks, history) into a new one."""
    src = store.entry(canvas_id)
    cloned = [l.copy() for l in src.layers]
    cid = store.put_layers(src.width, src.height, cloned, canvas_id=new_canvas_id)
    return _summary(cid)


@mcp.tool()
def undo(canvas_id: str) -> dict:
    ok = store.undo(canvas_id)
    return {"ok": ok, **_summary(canvas_id)} if ok else {"ok": False, "reason": "nothing to undo"}


@mcp.tool()
def redo(canvas_id: str) -> dict:
    ok = store.redo(canvas_id)
    return {"ok": ok, **_summary(canvas_id)} if ok else {"ok": False, "reason": "nothing to redo"}


# ============================================================ drawing

def _draw(canvas_id: str, op) -> dict:
    """Snapshot, run op on the active layer's image, store result, summarize.

    The active layer is always RGBA, so drawing/filter helpers see a uniform
    mode regardless of how the canvas was created.
    """
    store.snapshot(canvas_id)
    img = store.get_active_image(canvas_id)
    new_img = op(img)
    if new_img is not img:
        store.replace_active_image(canvas_id, new_img)
    return _summary(canvas_id)


@mcp.tool()
def draw_pixel(canvas_id: str, x: int, y: int, color) -> dict:
    """Set a single pixel."""
    return _draw(canvas_id, lambda img: drawing.draw_pixel(img, x, y, color))


@mcp.tool()
def draw_line(canvas_id: str, x1: int, y1: int, x2: int, y2: int,
              color, *, width: int = 1) -> dict:
    """Straight line from (x1,y1) to (x2,y2)."""
    return _draw(canvas_id, lambda img: drawing.draw_line(img, x1, y1, x2, y2, color, width))


@mcp.tool()
def draw_rectangle(canvas_id: str, x1: int, y1: int, x2: int, y2: int, *,
                   outline=None, fill=None, width: int = 1) -> dict:
    """Rectangle. Provide ``outline``, ``fill``, or both."""
    return _draw(canvas_id, lambda img: drawing.draw_rectangle(
        img, x1, y1, x2, y2, outline=outline, fill=fill, width=width))


@mcp.tool()
def draw_ellipse(canvas_id: str, x1: int, y1: int, x2: int, y2: int, *,
                 outline=None, fill=None, width: int = 1) -> dict:
    """Ellipse inscribed in the bounding box (x1,y1)-(x2,y2). Use a square
    bbox for a circle."""
    return _draw(canvas_id, lambda img: drawing.draw_ellipse(
        img, x1, y1, x2, y2, outline=outline, fill=fill, width=width))


@mcp.tool()
def draw_polygon(canvas_id: str, points: Sequence[Sequence[int]], *,
                 outline=None, fill=None) -> dict:
    """Polygon through a list of [x, y] points (>=3)."""
    return _draw(canvas_id, lambda img: drawing.draw_polygon(
        img, points, outline=outline, fill=fill))


@mcp.tool()
def draw_arc(canvas_id: str, x1: int, y1: int, x2: int, y2: int,
             start: float, end: float, color, *, width: int = 1) -> dict:
    """Arc of an ellipse from ``start`` to ``end`` degrees (0 = 3 o'clock,
    clockwise)."""
    return _draw(canvas_id, lambda img: drawing.draw_arc(
        img, x1, y1, x2, y2, start, end, color, width))


@mcp.tool()
def draw_text(canvas_id: str, x: int, y: int, text: str, *,
              font: str | None = None, size: int = 16, color="black",
              anchor: str = "la", stroke_width: int = 0,
              stroke_color=None) -> dict:
    """Render text. ``font`` is a TTF filename or absolute path
    (e.g. ``arial.ttf``); omit for default. ``anchor`` is Pillow's 2-letter
    anchor (la=left-ascender, mm=middle-middle, etc.)."""
    return _draw(canvas_id, lambda img: drawing.draw_text(
        img, x, y, text, font=font, size=size, color=color, anchor=anchor,
        stroke_width=stroke_width, stroke_color=stroke_color))


@mcp.tool()
def draw_brush(canvas_id: str, points: Sequence[Sequence[int]], color, *,
               size: int = 4) -> dict:
    """Freehand stroke through the given points using a round brush."""
    return _draw(canvas_id, lambda img: drawing.draw_brush(img, points, color, size))


@mcp.tool()
def eraser(canvas_id: str, points: Sequence[Sequence[int]], *,
           size: int = 8, background=None) -> dict:
    """Erase along a stroke. RGBA canvases erase to transparent; RGB canvases
    paint with ``background`` (default white)."""
    return _draw(canvas_id, lambda img: drawing.eraser(
        img, points, size=size, background=background))


@mcp.tool()
def flood_fill(canvas_id: str, x: int, y: int, color, *,
               tolerance: int = 0) -> dict:
    """Paint-bucket starting at (x, y). ``tolerance`` is per-channel diff
    (0 = exact match only)."""
    return _draw(canvas_id, lambda img: drawing.flood_fill(
        img, x, y, color, tolerance=tolerance))


@mcp.tool()
def pick_color(canvas_id: str, x: int, y: int) -> dict:
    """Eyedropper: read the composited pixel at (x, y) as RGBA + hex."""
    r, g, b, a = drawing.pick_color(store.compose(canvas_id), x, y)
    return {
        "rgba": [r, g, b, a],
        "hex": f"#{r:02x}{g:02x}{b:02x}" + (f"{a:02x}" if a != 255 else ""),
    }


# ============================================================ transforms

def _layer_filter(canvas_id: str, op) -> dict:
    """Apply ``op(image) -> image`` to the active layer only (filters,
    adjustments). Does not change canvas dimensions."""
    store.snapshot(canvas_id)
    layer = store.active_layer(canvas_id)
    layer.image = op(layer.image).convert("RGBA")
    return _summary(canvas_id)


def _canvas_transform(canvas_id: str, op) -> dict:
    """Apply ``op(composite) -> new_image`` to the whole canvas. Flattens
    layers — the canvas is replaced with a single Background layer holding
    the transformed composite. Canvas dimensions update to the result.
    """
    store.snapshot(canvas_id)
    flat = store.compose(canvas_id)
    new_img = op(flat).convert("RGBA")
    e = store.entry(canvas_id)
    e.width, e.height = new_img.size
    e.layers = [layers_mod.Layer(name="Background", image=new_img)]
    e.active_index = 0
    return _summary(canvas_id)


@mcp.tool()
def crop(canvas_id: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    """Crop the canvas to (x1,y1)-(x2,y2). Flattens layers."""
    return _canvas_transform(canvas_id, lambda img: transforms.crop(img, x1, y1, x2, y2))


@mcp.tool()
def resize(canvas_id: str, width: int, height: int, *,
           resample: str = "lanczos") -> dict:
    """Resize the canvas. ``resample`` ∈ nearest|box|bilinear|hamming|bicubic|lanczos.
    Flattens layers."""
    return _canvas_transform(canvas_id, lambda img: transforms.resize(
        img, width, height, resample=resample))


@mcp.tool()
def rotate(canvas_id: str, angle: float, *, expand: bool = True,
           background="transparent") -> dict:
    """Rotate the canvas CCW by ``angle`` degrees. Flattens layers."""
    return _canvas_transform(canvas_id, lambda img: transforms.rotate(
        img, angle, expand=expand, background=background))


@mcp.tool()
def flip(canvas_id: str, axis: str) -> dict:
    """Mirror the canvas. ``axis`` is 'horizontal' or 'vertical'. Flattens layers."""
    return _canvas_transform(canvas_id, lambda img: transforms.flip(img, axis))


@mcp.tool()
def copy_region(canvas_id: str, x1: int, y1: int, x2: int, y2: int, *,
                new_canvas_id: str | None = None) -> dict:
    """Copy a rectangle of the composite into a new single-layer canvas."""
    region = transforms.copy_region(store.compose(canvas_id), x1, y1, x2, y2)
    cid = store.put_image(region, canvas_id=new_canvas_id, layer_name="Background")
    return _summary(cid)


@mcp.tool()
def paste_canvas(dst_canvas_id: str, src_canvas_id: str, x: int, y: int, *,
                 as_new_layer: bool = True) -> dict:
    """Paste the source canvas onto the destination at (x, y).

    Default behavior (``as_new_layer=True``) adds the source as a new layer
    above the active layer, with offset (x, y) — non-destructive. Set
    ``as_new_layer=False`` to flatten the source onto the destination's
    composite (destructive)."""
    store.snapshot(dst_canvas_id)
    src_composite = store.compose(src_canvas_id)
    if as_new_layer:
        e = store.entry(dst_canvas_id)
        new_layer = layers_mod.Layer(
            name=f"Pasted {len(e.layers) + 1}",
            image=src_composite.convert("RGBA"),
            offset=(int(x), int(y)),
        )
        idx = e.active_index + 1
        e.layers.insert(idx, new_layer)
        e.active_index = idx
    else:
        flat = store.compose(dst_canvas_id)
        merged = transforms.paste_region(flat, src_composite, x, y)
        e = store.entry(dst_canvas_id)
        e.layers = [layers_mod.Layer(name="Background",
                                     image=merged.convert("RGBA"))]
        e.active_index = 0
    return _summary(dst_canvas_id)


@mcp.tool()
def clear_region(canvas_id: str, x1: int, y1: int, x2: int, y2: int, *,
                 color: str = "transparent") -> dict:
    """Fill a rectangle on the active layer."""
    return _layer_filter(canvas_id, lambda img: transforms.clear_region(
        img, x1, y1, x2, y2, color=color))


@mcp.tool()
def apply_filter(canvas_id: str, filter_name: str, *,
                 radius: float | None = None) -> dict:
    """Apply a Pillow filter to the active layer.

    Names: blur, sharpen, smooth, smooth_more, edge_enhance, edge_enhance_more,
    find_edges, contour, emboss, detail, gaussian_blur (uses radius),
    box_blur (uses radius), unsharp_mask (uses radius)."""
    return _layer_filter(canvas_id, lambda img: transforms.apply_filter(
        img, filter_name, radius=radius))


@mcp.tool()
def adjust(canvas_id: str, *, brightness: float = 1.0, contrast: float = 1.0,
           color: float = 1.0, sharpness: float = 1.0) -> dict:
    """Multiplicative adjustments on the active layer. 1.0 = identity."""
    return _layer_filter(canvas_id, lambda img: transforms.adjust(
        img, brightness=brightness, contrast=contrast, color=color,
        sharpness=sharpness))


@mcp.tool()
def invert(canvas_id: str) -> dict:
    """Color-invert the active layer. Preserves alpha."""
    return _layer_filter(canvas_id, transforms.invert)


@mcp.tool()
def grayscale(canvas_id: str) -> dict:
    """Desaturate the active layer. Preserves alpha."""
    return _layer_filter(canvas_id, transforms.grayscale)


@mcp.tool()
def posterize(canvas_id: str, bits: int) -> dict:
    """Reduce bits-per-channel on the active layer (1-8)."""
    return _layer_filter(canvas_id, lambda img: transforms.posterize(img, bits))


@mcp.tool()
def add_border(canvas_id: str, width: int, *, color: str = "black") -> dict:
    """Add a solid border around the canvas. Flattens layers."""
    return _canvas_transform(canvas_id, lambda img: transforms.add_border(img, width, color))


# ============================================================ layer management

@mcp.tool()
def list_layers(canvas_id: str) -> dict:
    """List all layers bottom-to-top with their properties."""
    e = store.entry(canvas_id)
    return {
        "canvas_id": canvas_id,
        "active_index": e.active_index,
        "layers": [{"index": i, **l.summary()} for i, l in enumerate(e.layers)],
    }


@mcp.tool()
def add_layer(canvas_id: str, *, name: str | None = None,
              fill: str = "transparent", above: int | None = None) -> dict:
    """Insert a new layer above ``above`` (default: top of stack). The new
    layer is transparent unless ``fill`` is given. Returns its index in
    ``active_index`` of the canvas summary."""
    store.snapshot(canvas_id)
    fill_rgba = parse_color(fill)
    store.add_layer(canvas_id, name=name, fill=fill_rgba, above=above)
    return _summary(canvas_id)


@mcp.tool()
def remove_layer(canvas_id: str, index: int) -> dict:
    """Delete a layer. The canvas must keep at least one layer."""
    store.snapshot(canvas_id)
    store.remove_layer(canvas_id, index)
    return _summary(canvas_id)


@mcp.tool()
def duplicate_layer(canvas_id: str, index: int) -> dict:
    """Clone a layer (including mask, blend mode, opacity). The clone becomes
    the active layer."""
    store.snapshot(canvas_id)
    store.duplicate_layer(canvas_id, index)
    return _summary(canvas_id)


@mcp.tool()
def rename_layer(canvas_id: str, index: int, name: str) -> dict:
    """Set a layer's name."""
    store.snapshot(canvas_id)
    store.entry(canvas_id).layers[index].name = name
    return _summary(canvas_id)


@mcp.tool()
def reorder_layer(canvas_id: str, src: int, dst: int) -> dict:
    """Move a layer in the z-order. ``src`` is its current index, ``dst`` the
    target index. 0 = bottom of stack."""
    store.snapshot(canvas_id)
    store.reorder_layer(canvas_id, src, dst)
    return _summary(canvas_id)


@mcp.tool()
def set_active_layer(canvas_id: str, index: int) -> dict:
    """Choose which layer drawing/filter ops target."""
    store.set_active(canvas_id, index)
    return _summary(canvas_id)


@mcp.tool()
def set_layer_visibility(canvas_id: str, index: int, visible: bool) -> dict:
    """Toggle a layer's visibility (the eyeball icon in PS)."""
    store.snapshot(canvas_id)
    store.entry(canvas_id).layers[index].visible = bool(visible)
    return _summary(canvas_id)


@mcp.tool()
def set_layer_opacity(canvas_id: str, index: int, opacity: float) -> dict:
    """Set a layer's opacity (0.0 fully transparent, 1.0 fully opaque)."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be in [0, 1], got {opacity}")
    store.snapshot(canvas_id)
    store.entry(canvas_id).layers[index].opacity = float(opacity)
    return _summary(canvas_id)


@mcp.tool()
def set_layer_blend_mode(canvas_id: str, index: int, blend_mode: str) -> dict:
    """Set a layer's blend mode. Choices: normal, multiply, screen, overlay,
    darken, lighten, color_dodge, color_burn, hard_light, soft_light,
    difference, exclusion, add, subtract."""
    if blend_mode not in layers_mod.BLEND_MODES:
        raise ValueError(
            f"unknown blend_mode {blend_mode!r}. Choose from {layers_mod.BLEND_MODES}."
        )
    store.snapshot(canvas_id)
    store.entry(canvas_id).layers[index].blend_mode = blend_mode
    return _summary(canvas_id)


@mcp.tool()
def set_layer_offset(canvas_id: str, index: int, x: int, y: int) -> dict:
    """Move a layer within the canvas. The layer's image itself doesn't
    change; only where it composites. Negative offsets are allowed."""
    store.snapshot(canvas_id)
    store.entry(canvas_id).layers[index].offset = (int(x), int(y))
    return _summary(canvas_id)


@mcp.tool()
def merge_down(canvas_id: str, index: int) -> dict:
    """Composite layer ``index`` onto the one below it and remove the top
    one. The result inherits the bottom layer's name/position."""
    store.snapshot(canvas_id)
    store.merge_down(canvas_id, index)
    return _summary(canvas_id)


@mcp.tool()
def flatten_canvas(canvas_id: str) -> dict:
    """Collapse all visible layers into a single Background layer."""
    store.snapshot(canvas_id)
    store.flatten(canvas_id)
    return _summary(canvas_id)


@mcp.tool()
def get_layer_preview(canvas_id: str, index: int, *,
                      max_size: int = 512) -> MCPImage:
    """Return an inline preview of a single layer's image (no compositing,
    no mask applied — shows the layer's raw pixels)."""
    layer = store.entry(canvas_id).layers[index]
    png = _img_to_png_bytes(layer.image, max_size=max_size)
    return MCPImage(data=png, format="png")


# ============================================================ layer masks

@mcp.tool()
def add_layer_mask(canvas_id: str, index: int | None = None, *,
                   fill: str = "white") -> dict:
    """Attach a grayscale mask to a layer (default: active layer).

    Mask convention: white = fully visible, black = fully hidden. Pass
    ``fill="black"`` to start hidden. The mask is sized to the layer's image."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if index is None else index
    layer = e.layers[i]
    if layer.mask is not None:
        raise ValueError(f"layer {i} already has a mask; delete it first")
    fill_rgba = parse_color(fill)
    layer.mask = PILImage.new("L", layer.image.size, fill_rgba[0])
    return _summary(canvas_id)


@mcp.tool()
def delete_layer_mask(canvas_id: str, index: int | None = None) -> dict:
    """Remove the mask from a layer without applying it."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if index is None else index
    e.layers[i].mask = None
    return _summary(canvas_id)


@mcp.tool()
def apply_layer_mask(canvas_id: str, index: int | None = None) -> dict:
    """Bake the mask into the layer's alpha channel and remove the mask.
    Destructive — undo to recover."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if index is None else index
    layer = e.layers[i]
    if layer.mask is None:
        raise ValueError(f"layer {i} has no mask to apply")
    import numpy as np

    rgba = np.asarray(layer.image.convert("RGBA"), dtype=np.float32)
    mask = np.asarray(layer.mask.convert("L"), dtype=np.float32) / 255.0
    rgba[..., 3] = rgba[..., 3] * mask
    layer.image = PILImage.fromarray(rgba.astype("uint8"), mode="RGBA")
    layer.mask = None
    return _summary(canvas_id)


@mcp.tool()
def invert_layer_mask(canvas_id: str, index: int | None = None) -> dict:
    """Invert the mask (white ↔ black)."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if index is None else index
    layer = e.layers[i]
    if layer.mask is None:
        raise ValueError(f"layer {i} has no mask")
    layer.mask = layers_mod.invert_mask(layer.mask)
    return _summary(canvas_id)


@mcp.tool()
def fill_layer_mask(canvas_id: str, index: int | None = None, *,
                    color: str = "white") -> dict:
    """Replace the entire mask with a single value. ``color`` may be any
    color form; only the luminance is used (white = visible, black = hidden)."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if index is None else index
    layer = e.layers[i]
    if layer.mask is None:
        raise ValueError(f"layer {i} has no mask; call add_layer_mask first")
    rgba = parse_color(color)
    # ITU-R BT.601 luminance.
    lum = int(round(0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]))
    layer.mask = PILImage.new("L", layer.mask.size, lum)
    return _summary(canvas_id)


@mcp.tool()
def set_layer_mask_from_canvas(canvas_id: str, index: int,
                               mask_canvas_id: str) -> dict:
    """Use another canvas's composited grayscale as this layer's mask.

    The mask canvas should be the same size as the target layer; if not it
    is resized via lanczos before being applied."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    layer = e.layers[index]
    src = store.compose(mask_canvas_id).convert("L")
    if src.size != layer.image.size:
        src = src.resize(layer.image.size, PILImage.LANCZOS)
    layer.mask = src
    return _summary(canvas_id)


# ============================================================ file ops (stateless)

@mcp.tool()
def image_info(path: str) -> dict:
    """Read width/height/format/mode/EXIF from a file without opening a canvas."""
    return io_formats.image_info(path)


@mcp.tool()
def convert_image(src_path: str, dst_path: str, *,
                  format: str | None = None, quality: int | None = None) -> dict:
    """Read ``src_path`` and write it to ``dst_path`` in a different format.

    Format is inferred from the destination extension unless given. Useful for
    one-shot HEIC→JPEG, PNG→WebP, RAW→TIFF, SVG→PNG, etc.
    """
    img = io_formats.load_image(src_path)
    return io_formats.save_image(img, dst_path, format=format, quality=quality)


@mcp.tool()
def batch_convert(src_dir: str, dst_dir: str, target_format: str, *,
                  quality: int | None = None, recursive: bool = False) -> dict:
    """Convert every image under ``src_dir`` to ``target_format`` in ``dst_dir``.

    Skips non-image files. Returns counts and per-file results so the caller
    can see what failed.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not src.is_dir():
        raise ValueError(f"src_dir is not a directory: {src_dir}")
    dst.mkdir(parents=True, exist_ok=True)
    ext = target_format.lower().lstrip(".")
    pattern = "**/*" if recursive else "*"

    results: list[dict[str, Any]] = []
    n_ok = n_err = 0
    known = io_formats.supported_formats()["load"]
    for p in src.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in known:
            continue
        rel = p.relative_to(src).with_suffix("." + ext)
        out = dst / rel
        try:
            img = io_formats.load_image(str(p))
            info = io_formats.save_image(img, str(out), quality=quality)
            results.append({"src": str(p), "dst": info["path"], "ok": True})
            n_ok += 1
        except Exception as e:
            results.append({"src": str(p), "ok": False,
                            "error": f"{type(e).__name__}: {e}"})
            n_err += 1
    return {"converted": n_ok, "failed": n_err, "results": results}


@mcp.tool()
def supported_formats() -> dict:
    """List what extensions can be loaded vs saved, and which optional deps
    are present."""
    return io_formats.supported_formats()


# ============================================================ stateless editing

@mcp.tool()
def edit_image(src_path: str, dst_path: str,
               operations: list[dict[str, Any]], *,
               format: str | None = None,
               quality: int | None = None) -> dict:
    """Apply a pipeline of operations to ``src_path`` and write the result to
    ``dst_path``. No canvas, no undo — purely stateless.

    Each operation is ``{"op": <name>, ...kwargs}``. Available ops:
    ``resize, crop, rotate, flip, thumbnail, add_border, clear_region,``
    ``apply_filter, adjust, invert, grayscale, posterize,``
    ``draw_pixel, draw_line, draw_rectangle, draw_ellipse, draw_polygon,``
    ``draw_arc, draw_text, draw_brush, eraser, flood_fill``.

    Example::

        [
          {"op": "resize", "width": 1200, "height": 800},
          {"op": "adjust", "brightness": 1.05, "contrast": 1.1},
          {"op": "draw_text", "x": 20, "y": 20, "text": "© 2026",
           "size": 28, "color": "white", "stroke_width": 1, "stroke_color": "black"}
        ]
    """
    img = io_formats.load_image(src_path)
    out = stateless.apply_pipeline(img, operations)
    return io_formats.save_image(out, dst_path, format=format, quality=quality)


@mcp.tool()
def resize_image(src_path: str, dst_path: str, width: int, height: int, *,
                 resample: str = "lanczos", quality: int | None = None) -> dict:
    """One-shot stateless resize. Output format inferred from dst extension."""
    img = io_formats.load_image(src_path)
    out = transforms.resize(img, width, height, resample=resample)
    return io_formats.save_image(out, dst_path, quality=quality)


@mcp.tool()
def thumbnail_image(src_path: str, dst_path: str, max_size: int = 512, *,
                    quality: int | None = None) -> dict:
    """Stateless thumbnail: scale longest edge to ``max_size`` preserving aspect.
    Never upscales."""
    img = io_formats.load_image(src_path)
    out = stateless.apply_pipeline(img, [{"op": "thumbnail", "max_size": max_size}])
    return io_formats.save_image(out, dst_path, quality=quality)


@mcp.tool()
def crop_image(src_path: str, dst_path: str, x1: int, y1: int, x2: int, y2: int, *,
               quality: int | None = None) -> dict:
    """Stateless crop to the rectangle (x1,y1)-(x2,y2)."""
    img = io_formats.load_image(src_path)
    out = transforms.crop(img, x1, y1, x2, y2)
    return io_formats.save_image(out, dst_path, quality=quality)


@mcp.tool()
def rotate_image(src_path: str, dst_path: str, angle: float, *,
                 expand: bool = True, background: str = "transparent",
                 quality: int | None = None) -> dict:
    """Stateless counter-clockwise rotation."""
    img = io_formats.load_image(src_path)
    out = transforms.rotate(img, angle, expand=expand, background=background)
    return io_formats.save_image(out, dst_path, quality=quality)


@mcp.tool()
def flip_image(src_path: str, dst_path: str, axis: str, *,
               quality: int | None = None) -> dict:
    """Stateless flip. ``axis`` is 'horizontal' or 'vertical'."""
    img = io_formats.load_image(src_path)
    out = transforms.flip(img, axis)
    return io_formats.save_image(out, dst_path, quality=quality)


@mcp.tool()
def grayscale_image(src_path: str, dst_path: str, *,
                    quality: int | None = None) -> dict:
    """Stateless desaturate. Preserves alpha if the input has it."""
    img = io_formats.load_image(src_path)
    out = transforms.grayscale(img)
    return io_formats.save_image(out, dst_path, quality=quality)


@mcp.tool()
def list_ops() -> dict:
    """Enumerate the operation names that ``edit_image`` accepts in its
    ``operations`` list."""
    return {"ops": stateless.ALL_OPS}


# ============================================================ format conversions

@mcp.tool()
def extract_frames(src_path: str, dst_dir: str, *,
                   format: str = "png",
                   filename_template: str = "frame_{i:04d}") -> dict:
    """Save every frame of an animated GIF/WebP/APNG as individual files.

    For single-frame inputs writes one file. ``filename_template`` uses
    ``{i}`` for the 0-based frame index. Returns the list of written paths."""
    return conversions.extract_frames(src_path, dst_dir, format=format,
                                      filename_template=filename_template)


@mcp.tool()
def build_animation(frame_paths: list[str], dst_path: str, *,
                    fps: float = 10.0, loop: int = 0,
                    quality: int | None = None,
                    optimize: bool = True) -> dict:
    """Build an animated GIF/WebP/APNG from frame files. Output format from
    ``dst_path`` extension. ``loop=0`` loops forever; mismatched-size frames
    are resized to the first frame's dimensions."""
    return conversions.build_animation(frame_paths, dst_path, fps=fps,
                                       loop=loop, quality=quality,
                                       optimize=optimize)


@mcp.tool()
def build_ico(src_paths: list[str], dst_path: str, *,
              sizes: list[int] | None = None) -> dict:
    """Pack images into a multi-resolution Windows .ico.

    With multiple ``src_paths``: each becomes one embedded size. With a single
    path + ``sizes``: re-render that image at every requested square size
    (defaults to 16/24/32/48/64/128/256)."""
    return conversions.build_ico(src_paths, dst_path, sizes=sizes)


@mcp.tool()
def split_ico(src_path: str, dst_dir: str, *,
              format: str = "png") -> dict:
    """Extract every embedded resolution from a .ico to separate files
    named ``icon_{w}x{h}.{ext}``."""
    return conversions.split_ico(src_path, dst_dir, format=format)


@mcp.tool()
def pdf_to_images(src_path: str, dst_dir: str, *,
                  dpi: int = 200, format: str = "png",
                  pages: str | None = None) -> dict:
    """Rasterize PDF pages to images (via pypdfium2 — no external deps).

    ``pages`` accepts ranges like ``"1-3,7,10-12"`` (1-based). ``dpi`` 200
    is a good default for screen; bump to 300+ for print."""
    return conversions.pdf_to_images(src_path, dst_dir, dpi=dpi,
                                     format=format, pages=pages)


@mcp.tool()
def images_to_pdf(src_paths: list[str], dst_path: str, *,
                  quality: int = 92) -> dict:
    """Combine images into a multi-page PDF (one image per page). Alpha is
    flattened onto white."""
    return conversions.images_to_pdf(src_paths, dst_path, quality=quality)


@mcp.tool()
def convert_mode(src_path: str, dst_path: str, mode: str, *,
                 dither: bool = True, palette_size: int = 256,
                 quality: int | None = None) -> dict:
    """Change color mode and save. ``mode`` ∈ 1, L, LA, P, RGB, RGBA, CMYK, I, F.

    - ``P`` honors ``palette_size`` (2-256) for indexed-color output.
    - ``1`` / ``P`` honor ``dither`` (Floyd-Steinberg when true)."""
    return conversions.convert_mode(src_path, dst_path, mode, dither=dither,
                                    palette_size=palette_size, quality=quality)


@mcp.tool()
def strip_metadata(src_path: str, dst_path: str, *,
                   quality: int | None = None) -> dict:
    """Re-encode the image dropping EXIF, ICC profile, XMP, and any other
    metadata. Pixel data is preserved."""
    return conversions.strip_metadata(src_path, dst_path, quality=quality)


@mcp.tool()
def copy_metadata(meta_src: str, image_src: str, dst_path: str, *,
                  quality: int | None = None) -> dict:
    """Transplant the EXIF block from ``meta_src`` onto ``image_src`` and save.

    Useful after destructive edits (crop/rotate) where you want to retain
    camera capture metadata. Only JPEG/TIFF/WebP/HEIF carry EXIF."""
    return conversions.copy_metadata(meta_src, image_src, dst_path,
                                     quality=quality)


# ============================================================ AI generation

@mcp.tool()
def sd_status() -> dict:
    """Report whether Stable Diffusion is installed and what's loaded.

    Install with ``pip install image-tools-mcp[sd]`` (downloads torch +
    diffusers; large).
    """
    return sd.sd_status()


@mcp.tool()
def sd_load(kind: str, *,
            model: str = "runwayml/stable-diffusion-v1-5",
            gguf_path: str | None = None) -> dict:
    """Pre-load an SD pipeline into memory so subsequent inference calls
    don't pay the (tens-of-seconds) model-load latency. ``kind`` ∈
    txt2img|img2img|inpaint. Idempotent — reports
    ``was_already_loaded: true`` if the requested ``(kind, model, gguf_path)``
    is already cached.

    ``gguf_path`` swaps in a GGUF-quantized UNet/transformer for the heavy
    component. Best supported for FLUX and SD3-class models (raises a clear
    error for older SD 1.x/2.x). Drops VRAM by 50-70% vs fp16. Use
    ``download_gguf`` to fetch one from HuggingFace."""
    return sd.load_pipeline(kind, model, gguf_path=gguf_path)


@mcp.tool()
def sd_unload(*, kind: str | None = None) -> dict:
    """Free SD pipeline memory. ``kind`` ∈ txt2img|img2img|inpaint or omit
    to unload all."""
    return sd.sd_unload(kind)


@mcp.tool()
def sd_set_idle_timeout(seconds: float) -> dict:
    """Set the idle timeout for SD pipelines (default 3600s = 1 hour). A
    background sweeper unloads any pipeline that hasn't been used in this
    long. Pass 0 to disable auto-eviction (manual unload only)."""
    return sd.set_idle_timeout(seconds)


@mcp.tool()
def sd_generate(prompt: str, *, negative_prompt: str | None = None,
                width: int = 512, height: int = 512, steps: int = 25,
                guidance: float = 7.5, seed: int | None = None,
                model: str | None = None,
                gguf_path: str | None = None,
                control_canvas_id: str | None = None,
                controlnet_conditioning_scale: float | None = None,
                canvas_id: str | None = None) -> dict:
    """Stable Diffusion text-to-image. First call downloads the model
    (~4 GB for SD1.5). Result is stored as a new canvas; pass ``canvas_id``
    to choose its id.

    Omit ``model`` and ``gguf_path`` to reuse whatever was pre-loaded via
    ``sd_load(kind="txt2img", ...)``; passing the default literal would
    silently evict any non-default pre-load and re-download.

    If a ControlNet is loaded via ``sd_load_controlnet``, pass
    ``control_canvas_id`` (a canvas containing the conditioning image —
    canny edges, depth map, etc., produced via ``canny_edges`` or similar)
    and an optional ``controlnet_conditioning_scale`` (typically 0.4-1.5;
    1.0 is the model's nominal strength)."""
    control = store.compose(control_canvas_id) if control_canvas_id else None
    img = sd.sd_txt2img(prompt, negative_prompt=negative_prompt,
                       width=width, height=height, steps=steps,
                       guidance=guidance, seed=seed, model=model,
                       gguf_path=gguf_path,
                       control_image=control,
                       controlnet_conditioning_scale=controlnet_conditioning_scale)
    cid = store.put_image(img, canvas_id=canvas_id, layer_name="SD Generated")
    return _summary(cid)


@mcp.tool()
def sd_img2img(canvas_id: str, prompt: str, *,
               strength: float = 0.6, negative_prompt: str | None = None,
               steps: int = 25, guidance: float = 7.5,
               seed: int | None = None,
               model: str | None = None,
               gguf_path: str | None = None,
               control_canvas_id: str | None = None,
               controlnet_conditioning_scale: float | None = None,
               new_canvas_id: str | None = None) -> dict:
    """Stable Diffusion image-to-image. ``strength`` ∈ 0.0-1.0; higher
    deviates more from the source. Result is a NEW canvas; original is
    untouched.

    Same model-default rule as ``sd_generate`` — omit ``model``/``gguf_path``
    to reuse the pre-loaded pipeline. ``control_canvas_id`` /
    ``controlnet_conditioning_scale`` activate ControlNet (see
    ``sd_load_controlnet`` + ``canny_edges``)."""
    src = store.compose(canvas_id)
    control = store.compose(control_canvas_id) if control_canvas_id else None
    img = sd.sd_img2img(src, prompt, strength=strength,
                        negative_prompt=negative_prompt, steps=steps,
                        guidance=guidance, seed=seed, model=model,
                        gguf_path=gguf_path,
                        control_image=control,
                        controlnet_conditioning_scale=controlnet_conditioning_scale)
    cid = store.put_image(img, canvas_id=new_canvas_id, layer_name="SD img2img")
    return _summary(cid)


@mcp.tool()
def sd_inpaint(canvas_id: str, mask_canvas_id: str, prompt: str, *,
               negative_prompt: str | None = None,
               steps: int = 25, guidance: float = 7.5,
               seed: int | None = None,
               model: str | None = None,
               gguf_path: str | None = None,
               control_canvas_id: str | None = None,
               controlnet_conditioning_scale: float | None = None,
               new_canvas_id: str | None = None) -> dict:
    """Stable Diffusion inpainting. ``mask_canvas_id`` is a canvas where
    white = repaint, black = keep. Result is a NEW canvas.

    Same model-default rule as ``sd_generate``. ``control_canvas_id`` /
    ``controlnet_conditioning_scale`` activate ControlNet."""
    src = store.compose(canvas_id)
    mask = store.compose(mask_canvas_id)
    control = store.compose(control_canvas_id) if control_canvas_id else None
    img = sd.sd_inpaint(src, mask, prompt, negative_prompt=negative_prompt,
                        steps=steps, guidance=guidance, seed=seed,
                        model=model, gguf_path=gguf_path,
                        control_image=control,
                        controlnet_conditioning_scale=controlnet_conditioning_scale)
    cid = store.put_image(img, canvas_id=new_canvas_id, layer_name="SD inpaint")
    return _summary(cid)


# ---- ControlNet management ------------------------------------------------

@mcp.tool()
def sd_load_controlnet(repo: str, *, family: str | None = None) -> dict:
    """Load a ControlNet model. Future ``sd_generate`` / ``sd_img2img`` /
    ``sd_inpaint`` calls that pass ``control_canvas_id`` will run the
    ControlNet-aware pipeline.

    ``repo`` is a HuggingFace ControlNet repo id, e.g.:
        - SD1.5 canny:  ``lllyasviel/sd-controlnet-canny``
        - SD1.5 depth:  ``lllyasviel/sd-controlnet-depth``
        - SD1.5 pose:   ``lllyasviel/sd-controlnet-openpose``
        - SDXL canny:   ``diffusers/controlnet-canny-sdxl-1.0``
        - SDXL depth:   ``diffusers/controlnet-depth-sdxl-1.0``
        - FLUX canny:   ``InstantX/FLUX.1-dev-Controlnet-Canny``

    ``family`` is one of sd15/sdxl/sd3/flux and must match the base model
    you'll use at inference. Omit to auto-detect from the repo id.

    Loading a new ControlNet evicts any cached SD pipelines bound to the
    previous one. Use ``sd_unload_controlnet`` to revert to plain SD."""
    return sd.load_controlnet(repo, family=family)


@mcp.tool()
def sd_unload_controlnet() -> dict:
    """Detach the loaded ControlNet. Subsequent inference calls fall back
    to plain SD pipelines."""
    return sd.unload_controlnet()


@mcp.tool()
def sd_controlnet_status() -> dict:
    """Report whether a ControlNet is loaded, which repo, and which family."""
    return sd.controlnet_status()


# ---- Preprocessors --------------------------------------------------------

@mcp.tool()
def canny_edges(canvas_id: str, *, low: int = 100, high: int = 200,
                blur_radius: int = 0,
                new_canvas_id: str | None = None) -> dict:
    """Canny edge detection — produces the conditioning image for a Canny
    ControlNet. Defaults match the ``lllyasviel/sd-controlnet-canny``
    reference. Result is a NEW canvas (black background, white edges).

    ``low`` / ``high`` are the hysteresis thresholds (0-255). Lower
    ``low`` → more edges. ``blur_radius`` optionally pre-smooths the input
    (0 disables, 1-3 typical for noisy / high-frequency images)."""
    src = store.compose(canvas_id)
    out = preprocessors.canny_edges(src, low=low, high=high,
                                    blur_radius=blur_radius)
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Canny")
    return _summary(cid)


@mcp.tool()
def depth_from_grayscale(canvas_id: str, *, invert: bool = False,
                         new_canvas_id: str | None = None) -> dict:
    """Cheap depth-like map (luminance, optionally inverted). Stand-in for
    a real depth model — works OK for ControlNet on flat-lit subjects;
    swap to a Marigold / ZoeDepth output when you need accurate depth."""
    src = store.compose(canvas_id)
    out = preprocessors.depth_from_grayscale(src, invert=invert)
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Depth")
    return _summary(cid)


# ============================================================ Photoshop adjustments

@mcp.tool()
def hue_saturation(canvas_id: str, *, hue: float = 0.0,
                   saturation: float = 0.0,
                   lightness: float = 0.0) -> dict:
    """PS Hue/Saturation on the active layer. All inputs are -100..+100.
    Hue shifts ±180° at the extremes; saturation -100 desaturates fully,
    +100 doubles; lightness pushes toward black or white."""
    return _layer_filter(canvas_id, lambda img: adjustments.hue_saturation_lightness(
        img, hue=hue, saturation=saturation, lightness=lightness))


@mcp.tool()
def levels(canvas_id: str, *, in_black: int = 0, in_white: int = 255,
           gamma: float = 1.0, out_black: int = 0,
           out_white: int = 255) -> dict:
    """PS Levels on the active layer. ``in_black``/``in_white`` clip the
    histogram (0-255), ``gamma`` reshapes midtones (>1 darkens), ``out_*``
    rescale to a narrower range."""
    return _layer_filter(canvas_id, lambda img: adjustments.levels(
        img, in_black=in_black, in_white=in_white, gamma=gamma,
        out_black=out_black, out_white=out_white))


@mcp.tool()
def curves(canvas_id: str, *,
           rgb_curve: list[list[float]] | None = None,
           r_curve: list[list[float]] | None = None,
           g_curve: list[list[float]] | None = None,
           b_curve: list[list[float]] | None = None) -> dict:
    """PS Curves on the active layer. Each curve is a list of ``[input, output]``
    points in 0-255 (identity = ``[[0,0],[255,255]]``). ``rgb_curve`` runs
    first on all channels; per-channel curves stack on top."""
    def to_tuples(c):
        return [(float(p[0]), float(p[1])) for p in c] if c else None
    return _layer_filter(canvas_id, lambda img: adjustments.curves(
        img,
        rgb_curve=to_tuples(rgb_curve),
        r_curve=to_tuples(r_curve),
        g_curve=to_tuples(g_curve),
        b_curve=to_tuples(b_curve),
    ))


@mcp.tool()
def color_balance(canvas_id: str, *,
                  cyan_red: float = 0.0,
                  magenta_green: float = 0.0,
                  yellow_blue: float = 0.0,
                  tonal_range: str = "midtones") -> dict:
    """PS Color Balance on the active layer. Sliders -100..+100. ``tonal_range``
    ∈ shadows | midtones | highlights — applies the shift weighted by a bell
    over luminance centered on that range."""
    return _layer_filter(canvas_id, lambda img: adjustments.color_balance(
        img, cyan_red=cyan_red, magenta_green=magenta_green,
        yellow_blue=yellow_blue, tonal_range=tonal_range))


@mcp.tool()
def threshold(canvas_id: str, level: int = 128) -> dict:
    """PS Threshold on the active layer. Pixels with luminance ≥ ``level``
    (0-255) become white, the rest become black. Preserves alpha."""
    return _layer_filter(canvas_id, lambda img: adjustments.threshold(img, level))


@mcp.tool()
def vibrance(canvas_id: str, amount: float = 0.0) -> dict:
    """PS Vibrance on the active layer. -100..+100. Boosts saturation of
    less-saturated pixels more than already-saturated ones — preserves skin
    tones better than a flat saturation increase."""
    return _layer_filter(canvas_id, lambda img: adjustments.vibrance(img, amount))


@mcp.tool()
def channel_mixer(canvas_id: str, *,
                  r_mix: list[float] = [1.0, 0.0, 0.0],
                  g_mix: list[float] = [0.0, 1.0, 0.0],
                  b_mix: list[float] = [0.0, 0.0, 1.0],
                  constant: list[float] = [0.0, 0.0, 0.0]) -> dict:
    """PS Channel Mixer on the active layer. Each output channel is a weighted
    sum of the three input channels plus a constant (in 0..1 space)."""
    return _layer_filter(canvas_id, lambda img: adjustments.channel_mixer(
        img,
        r_mix=tuple(r_mix), g_mix=tuple(g_mix), b_mix=tuple(b_mix),
        constant=tuple(constant)))


@mcp.tool()
def gradient_map(canvas_id: str, stops: list[dict[str, Any]]) -> dict:
    """PS Gradient Map on the active layer. Maps each pixel's luminance through
    the gradient defined by ``stops`` (list of ``{position, color}``)."""
    return _layer_filter(canvas_id, lambda img: gradients.gradient_map(img, stops))


@mcp.tool()
def auto_levels(canvas_id: str, *, clip: float = 0.01) -> dict:
    """Per-channel histogram stretch. ``clip`` is the fraction of pixels to
    clip at each end (0.01 = 1%)."""
    return _layer_filter(canvas_id, lambda img: adjustments.auto_levels(img, clip=clip))


@mcp.tool()
def auto_contrast(canvas_id: str, *, clip: float = 0.01) -> dict:
    """Luminance-driven stretch — preserves color balance unlike auto_levels."""
    return _layer_filter(canvas_id, lambda img: adjustments.auto_contrast(img, clip=clip))


@mcp.tool()
def equalize(canvas_id: str) -> dict:
    """Histogram equalize on luminance, preserving hue/saturation roughly."""
    return _layer_filter(canvas_id, adjustments.equalize)


# ============================================================ Photoshop gradient fills

@mcp.tool()
def gradient_fill(canvas_id: str, gradient_type: str,
                  stops: list[dict[str, Any]], *,
                  x1: float = 0, y1: float = 0,
                  x2: float | None = None, y2: float | None = None,
                  center_x: float | None = None,
                  center_y: float | None = None,
                  radius: float | None = None,
                  start_angle: float = 0.0,
                  repeat: bool = False,
                  as_new_layer: bool = False,
                  layer_name: str = "Gradient") -> dict:
    """Fill the canvas with a gradient. ``gradient_type`` ∈ linear, radial,
    angular, reflected, diamond. ``stops`` is a list of
    ``{"position": 0.0-1.0, "color": ...}``.

    Geometry args:
    - linear / reflected: ``x1,y1`` → ``x2,y2`` (defaults to left-to-right edge)
    - radial / diamond:   ``center_x,center_y`` + ``radius`` (defaults centered)
    - angular:            ``center_x,center_y`` + ``start_angle`` (deg)

    ``as_new_layer=True`` adds the gradient above the active layer instead of
    overwriting the active layer's pixels."""
    e = store.entry(canvas_id)
    grad = gradients.make_gradient(
        e.size, gradient_type, stops,
        x1=x1, y1=y1, x2=x2, y2=y2,
        center_x=center_x, center_y=center_y, radius=radius,
        start_angle=start_angle, repeat=repeat,
    )
    store.snapshot(canvas_id)
    if as_new_layer:
        layer = layers_mod.Layer(name=layer_name, image=grad)
        idx = e.active_index + 1
        e.layers.insert(idx, layer)
        e.active_index = idx
    else:
        e.active_layer.image = grad
    return _summary(canvas_id)


# ============================================================ Photoshop channels

@mcp.tool()
def extract_channel(canvas_id: str, channel: str, *,
                    new_canvas_id: str | None = None) -> dict:
    """Pull a channel (R, G, B, A, L) from the composited canvas into a new
    grayscale canvas. ``L`` is BT.601 luminance."""
    composite = store.compose(canvas_id)
    extracted = channels.extract_channel(composite, channel)
    cid = store.put_image(extracted, canvas_id=new_canvas_id,
                          layer_name=f"Channel {channel}")
    return _summary(cid)


@mcp.tool()
def merge_channels(r_canvas_id: str, g_canvas_id: str, b_canvas_id: str, *,
                   a_canvas_id: str | None = None,
                   new_canvas_id: str | None = None) -> dict:
    """Combine grayscale canvases into one RGB/RGBA canvas. All must be the
    same size; each contributes its luminance to one channel."""
    r = store.compose(r_canvas_id)
    g = store.compose(g_canvas_id)
    b = store.compose(b_canvas_id)
    a = store.compose(a_canvas_id) if a_canvas_id else None
    merged = channels.merge_channels(r, g, b, a)
    cid = store.put_image(merged, canvas_id=new_canvas_id,
                          layer_name="Merged channels")
    return _summary(cid)


@mcp.tool()
def split_channels_to_layers(canvas_id: str) -> dict:
    """Replace the canvas with one layer per channel (R, G, B, A as grayscale
    layers stacked bottom-to-top). Useful for channel-specific editing."""
    composite = store.compose(canvas_id)
    chs = channels.split_to_grayscales(composite)
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    new_layers = []
    for name in ("R", "G", "B", "A"):
        ch_img = chs[name].convert("RGBA")
        new_layers.append(layers_mod.Layer(name=f"Channel {name}", image=ch_img))
    e.layers = new_layers
    e.active_index = 0
    return _summary(canvas_id)


# ============================================================ Photoshop painting brushes

@mcp.tool()
def clone_stamp(canvas_id: str, points: Sequence[Sequence[int]],
                source_x: int, source_y: int, *,
                size: int = 20, hardness: float = 0.5,
                opacity: float = 1.0,
                source_canvas_id: str | None = None) -> dict:
    """PS Clone Stamp on the active layer. Samples from ``(source_x, source_y)``
    and paints along the stroke (Aligned mode — source moves with the brush).
    If ``source_canvas_id`` is given, samples from that canvas's composite
    instead of the same layer."""
    src_img = None
    if source_canvas_id:
        src_img = store.compose(source_canvas_id)
    return _draw(canvas_id, lambda img: painting.clone_stamp(
        img, points, source_x=source_x, source_y=source_y,
        size=size, hardness=hardness, opacity=opacity, source_img=src_img))


@mcp.tool()
def dodge_brush(canvas_id: str, points: Sequence[Sequence[int]], *,
                size: int = 20, hardness: float = 0.5,
                exposure: float = 0.3) -> dict:
    """PS Dodge tool on the active layer — brightens along the stroke."""
    return _draw(canvas_id, lambda img: painting.dodge_brush(
        img, points, size=size, hardness=hardness, exposure=exposure))


@mcp.tool()
def burn_brush(canvas_id: str, points: Sequence[Sequence[int]], *,
               size: int = 20, hardness: float = 0.5,
               exposure: float = 0.3) -> dict:
    """PS Burn tool on the active layer — darkens along the stroke."""
    return _draw(canvas_id, lambda img: painting.burn_brush(
        img, points, size=size, hardness=hardness, exposure=exposure))


@mcp.tool()
def blur_brush(canvas_id: str, points: Sequence[Sequence[int]], *,
               size: int = 20, hardness: float = 0.5,
               strength: float = 0.5,
               radius: float = 2.0) -> dict:
    """PS Blur tool — locally blurs along the stroke."""
    return _draw(canvas_id, lambda img: painting.blur_brush(
        img, points, size=size, hardness=hardness, strength=strength, radius=radius))


@mcp.tool()
def sharpen_brush(canvas_id: str, points: Sequence[Sequence[int]], *,
                  size: int = 20, hardness: float = 0.5,
                  strength: float = 0.5) -> dict:
    """PS Sharpen tool — locally sharpens along the stroke via unsharp mask."""
    return _draw(canvas_id, lambda img: painting.sharpen_brush(
        img, points, size=size, hardness=hardness, strength=strength))


# ============================================================ Photoshop layer effects

@mcp.tool()
def add_drop_shadow(canvas_id: str, *, index: int | None = None,
                    offset_x: int = 6, offset_y: int = 6,
                    blur: float = 8.0,
                    color: str = "#000000",
                    opacity: float = 0.6) -> dict:
    """Insert a drop-shadow layer below ``index`` (default: active layer)
    keyed on its alpha. The shadow is RGBA at the source's footprint plus
    padding for the offset and blur."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if index is None else index
    source = e.layers[i]
    shadow = layer_effects.make_drop_shadow_layer(
        source, offset_x=offset_x, offset_y=offset_y,
        blur=blur, color=color, opacity=opacity,
    )
    e.layers.insert(i, shadow)
    if e.active_index >= i:
        e.active_index += 1
    return _summary(canvas_id)


@mcp.tool()
def add_outer_glow(canvas_id: str, *, index: int | None = None,
                   blur: float = 12.0,
                   color: str = "#ffff80",
                   opacity: float = 0.8,
                   intensity: float = 1.0) -> dict:
    """Insert an outer-glow layer below ``index`` (default: active layer).
    Uses screen blend mode so the glow lightens what's behind it."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if index is None else index
    source = e.layers[i]
    glow = layer_effects.make_outer_glow_layer(
        source, blur=blur, color=color, opacity=opacity, intensity=intensity,
    )
    e.layers.insert(i, glow)
    if e.active_index >= i:
        e.active_index += 1
    return _summary(canvas_id)


@mcp.tool()
def add_layer_stroke(canvas_id: str, *, index: int | None = None,
                     width: int = 4,
                     color: str = "#000000",
                     position: str = "outside") -> dict:
    """Insert a stroke layer above ``index`` (default: active layer) traced
    along the source's alpha edge. ``position`` ∈ outside | inside | center."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if index is None else index
    source = e.layers[i]
    stroke = layer_effects.make_stroke_layer(
        source, width=width, color=color, position=position,
    )
    e.layers.insert(i + 1, stroke)
    if e.active_index > i:
        e.active_index += 1
    return _summary(canvas_id)


# ============================================================ Qwen Image Edit

@mcp.tool()
def qwen_status() -> dict:
    """Report whether Qwen-Image-Edit is installed, the device it would use,
    and what's currently loaded (pipeline + LoRA adapters with weights).

    Install with ``pip install image-tools-mcp[qwen]`` (pulls diffusers,
    transformers, accelerate, torch — large)."""
    return qwen.status()


@mcp.tool()
def qwen_load(*, model: str = "Qwen/Qwen-Image-Edit",
              gguf_path: str | None = None) -> dict:
    """Pre-load the Qwen pipeline into memory so subsequent edits don't pay
    model-load latency. Idempotent — reports ``was_already_loaded: true`` if
    the same ``(model, gguf_path)`` is already resident.

    ``gguf_path`` swaps in a GGUF-quantized transformer (drops VRAM from
    ~12 GB to ~3-6 GB depending on quant level). Use ``download_gguf`` to
    fetch one from HuggingFace, e.g.
    ``download_gguf("city96/Qwen-Image-Edit-gguf", "qwen-image-edit-Q4_K_S.gguf")``.

    Loaded pipelines stay resident until either ``qwen_unload``, a different
    ``(model, gguf_path)`` is loaded, or — by default — 1 hour of inactivity."""
    return qwen.load_pipeline(model, gguf_path=gguf_path)


@mcp.tool()
def qwen_unload() -> dict:
    """Drop the Qwen pipeline and all loaded LoRAs. Frees GPU memory."""
    return qwen.unload()


@mcp.tool()
def qwen_set_idle_timeout(seconds: float) -> dict:
    """Set the idle timeout for the Qwen pipeline (default 3600s = 1 hour).
    The sweeper unloads the pipeline if it's been idle this long. Pass 0 to
    disable auto-eviction."""
    return qwen.set_idle_timeout(seconds)


@mcp.tool()
def qwen_edit_image(canvas_id: str, prompt: str, *,
                    additional_canvas_ids: list[str] | None = None,
                    negative_prompt: str | None = None,
                    steps: int = 30,
                    guidance: float = 4.0,
                    true_cfg_scale: float | None = None,
                    seed: int | None = None,
                    model: str | None = None,
                    gguf_path: str | None = None,
                    new_canvas_id: str | None = None) -> dict:
    """Edit the composited canvas with Qwen-Image-Edit and save the result
    as a NEW canvas (the source is untouched).

    ``additional_canvas_ids`` enables multi-image input: every listed canvas
    is composited and passed to the pipeline alongside the primary one.
    Qwen-Image-Edit-2509 / 2511 condition on all of them (fusion /
    reference-driven edits).

    ``model`` and ``gguf_path``: omit both to reuse whatever was pre-loaded
    via ``qwen_load`` (recommended — supplying the literal default model
    cache-misses any non-default pre-load and silently re-downloads the
    bf16 base model).

    LoRA adapters loaded via ``qwen_load_lora`` apply automatically with
    their current weights."""
    images = [store.compose(canvas_id)]
    if additional_canvas_ids:
        images.extend(store.compose(cid) for cid in additional_canvas_ids)
    payload = images[0] if len(images) == 1 else images
    result = qwen.edit_image(payload, prompt, negative_prompt=negative_prompt,
                             steps=steps, guidance=guidance,
                             true_cfg_scale=true_cfg_scale, seed=seed,
                             model=model, gguf_path=gguf_path)
    cid = store.put_image(result, canvas_id=new_canvas_id,
                          layer_name="Qwen edit")
    return _summary(cid)


@mcp.tool()
def qwen_load_lora(name: str, source: str, *, weight: float = 1.0,
                   weight_name: str | None = None) -> dict:
    """Attach a LoRA adapter to the Qwen pipeline.

    - ``name``: unique handle to reference this LoRA later.
    - ``source``: HuggingFace repo id or local ``.safetensors`` path.
    - ``weight``: initial blend weight (typically 0.0-1.5; 1.0 is full strength).
    - ``weight_name``: filename within a multi-file LoRA repo.

    Multiple LoRAs can stack — diffusers combines them at inference using
    each adapter's weight. Tweak weights later with ``qwen_set_lora_weights``."""
    return qwen.load_lora(name, source, weight=weight, weight_name=weight_name)


@mcp.tool()
def qwen_set_lora_weights(weights: dict[str, float]) -> dict:
    """Adjust the weight of one or more loaded LoRAs in a single call.

    Example: ``{"style": 0.8, "character": 0.5}``. Pass ``0.0`` to disable
    a LoRA without unloading it. Unmentioned LoRAs keep their current weight."""
    return qwen.set_lora_weights(weights)


@mcp.tool()
def qwen_unload_lora(*, name: str | None = None) -> dict:
    """Detach a LoRA. Pass ``name`` to drop one, or omit to drop all."""
    return qwen.unload_lora(name)


@mcp.tool()
def qwen_list_loras() -> dict:
    """List currently attached LoRAs with their sources and weights."""
    return qwen.list_loras()


# ---- Qwen ControlNet ------------------------------------------------------

@mcp.tool()
def qwen_load_controlnet(repo: str) -> dict:
    """Load a Qwen-Image ControlNet model. After loading, call
    ``qwen_controlnet_generate`` (text-to-image with structural conditioning)
    or ``qwen_controlnet_inpaint``.

    Note Qwen ControlNet is a separate workflow from ``qwen_edit_image``:
    Edit conditions on an INPUT image's content (img2img with prompt
    understanding); ControlNet conditions on a STRUCTURE cue (canny edges,
    depth map, pose) while the rest is text-generated.

    Public repos (verified 2026-05-25):
      - ``InstantX/Qwen-Image-ControlNet-Union`` — multi-control (canny,
        depth, pose, soft_edge)
      - ``InstantX/Qwen-Image-ControlNet-Inpainting`` — inpaint
      - ``DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny`` — canny
      - ``DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth`` — depth"""
    return qwen.load_controlnet(repo)


@mcp.tool()
def qwen_unload_controlnet() -> dict:
    """Detach the loaded Qwen ControlNet. Plain ``qwen_edit_image`` is
    unaffected (it doesn't use ControlNet)."""
    return qwen.unload_controlnet()


@mcp.tool()
def qwen_controlnet_status() -> dict:
    """Report whether a Qwen ControlNet is loaded and which repo."""
    return qwen.controlnet_status()


@mcp.tool()
def qwen_controlnet_generate(prompt: str, control_canvas_id: str, *,
                             negative_prompt: str | None = None,
                             width: int = 1024, height: int = 1024,
                             steps: int = 30, guidance: float = 4.0,
                             true_cfg_scale: float | None = None,
                             controlnet_conditioning_scale: float = 1.0,
                             seed: int | None = None,
                             canvas_id: str | None = None) -> dict:
    """Qwen-Image text-to-image WITH ControlNet conditioning. Result is a
    NEW canvas.

    ``control_canvas_id`` is the structural cue (canny edges, depth, pose,
    etc.) — produce it with the ``canny_edges`` preprocessor or similar.
    ``controlnet_conditioning_scale`` ∈ 0.0-2.0 controls how strictly the
    output follows the cue (1.0 is the model's nominal strength).

    Requires ``qwen_load_controlnet(repo=...)`` first."""
    control = store.compose(control_canvas_id)
    img = qwen.controlnet_generate(
        prompt, control,
        negative_prompt=negative_prompt,
        width=width, height=height, steps=steps, guidance=guidance,
        true_cfg_scale=true_cfg_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
    )
    cid = store.put_image(img, canvas_id=canvas_id,
                          layer_name="Qwen CN generate")
    return _summary(cid)


@mcp.tool()
def qwen_controlnet_inpaint(prompt: str, control_canvas_id: str,
                            mask_canvas_id: str, *,
                            negative_prompt: str | None = None,
                            width: int | None = None,
                            height: int | None = None,
                            steps: int = 30, guidance: float = 4.0,
                            true_cfg_scale: float | None = None,
                            controlnet_conditioning_scale: float = 1.0,
                            seed: int | None = None,
                            canvas_id: str | None = None) -> dict:
    """Qwen-Image inpaint with ControlNet conditioning. Result is a NEW
    canvas. ``mask_canvas_id`` is white in the region to paint, black to
    keep. Requires ``qwen_load_controlnet`` first."""
    control = store.compose(control_canvas_id)
    mask = store.compose(mask_canvas_id)
    img = qwen.controlnet_inpaint(
        prompt, control, mask,
        negative_prompt=negative_prompt,
        width=width, height=height, steps=steps, guidance=guidance,
        true_cfg_scale=true_cfg_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
    )
    cid = store.put_image(img, canvas_id=canvas_id,
                          layer_name="Qwen CN inpaint")
    return _summary(cid)


# ============================================================ GGUF helper

@mcp.tool()
def download_gguf(repo_id: str, filename: str, *,
                  dest_dir: str | None = None) -> dict:
    """Pull a GGUF-quantized model file from HuggingFace into a local cache.
    Returns the absolute path you can then pass as ``gguf_path`` to
    ``qwen_load`` / ``sd_load``.

    Examples (popular GGUF repos as of 2026):

    - ``city96/Qwen-Image-Edit-gguf`` — quantized Qwen-Image-Edit transformer
      (``qwen-image-edit-Q4_K_S.gguf`` is a popular balance of size/quality)
    - ``city96/FLUX.1-dev-gguf`` — quantized FLUX.1 dev transformer
    - ``city96/stable-diffusion-3.5-large-gguf`` — quantized SD3.5

    The Q4_K_S level lands around 4-5 GB and runs comfortably in 6 GB VRAM.
    Q8_0 is near-lossless but ~2× the size. See ``gguf_io.QUANT_LEVELS`` for
    the full set."""
    path = gguf_io.download_gguf(repo_id, filename, dest_dir=dest_dir)
    return {
        "path": path,
        "quantization": gguf_io.detect_quant_level(path),
        "size_bytes": __import__("os").path.getsize(path),
    }


# ============================================================ Segment Anything 2

@mcp.tool()
def sam_status() -> dict:
    """Report whether SAM 2 is installed, the device it would use, what's
    loaded, and idle/timeout config.

    Install with ``pip install image-tools-mcp[sam]`` (pulls the sam2
    package + torch)."""
    return sam.status()


@mcp.tool()
def sam_load(*, model: str = sam.DEFAULT_MODEL) -> dict:
    """Pre-load a SAM 2 model so the first segmentation call doesn't pay
    model-load latency. Idempotent.

    Model sizes: ``facebook/sam2.1-hiera-tiny`` (fastest), ``-small``,
    ``-base-plus``, ``-large`` (default, best quality)."""
    return sam.load_model(model)


@mcp.tool()
def sam_unload() -> dict:
    """Drop the SAM 2 predictor + auto-mask generator. Frees GPU memory."""
    return sam.unload()


@mcp.tool()
def sam_set_idle_timeout(seconds: float) -> dict:
    """Set how long SAM 2 stays resident after its last use (default 1 h).
    Pass 0 to disable auto-eviction."""
    return sam.set_idle_timeout(seconds)


@mcp.tool()
def sam_segment_point(canvas_id: str, x: float, y: float, *,
                      label: int = 1,
                      model: str = sam.DEFAULT_MODEL,
                      new_canvas_id: str | None = None) -> dict:
    """Segment with a single click. ``label=1`` is a foreground point
    (include this region), ``label=0`` is background (exclude).

    The result is a grayscale mask (white = selected, black = not) saved
    as a new canvas you can feed to ``set_layer_mask_from_canvas``,
    ``clear_region`` via a layer mask workflow, etc."""
    composite = store.compose(canvas_id)
    mask_img, score = sam.segment_with_points(
        composite, [[float(x), float(y)]], [int(label)], model=model,
    )
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"SAM mask (score {score:.2f})")
    return {**_summary(cid), "iou_score": round(score, 3)}


@mcp.tool()
def sam_segment_points(canvas_id: str,
                       points: list[list[float]],
                       labels: list[int] | None = None, *,
                       model: str = sam.DEFAULT_MODEL,
                       new_canvas_id: str | None = None) -> dict:
    """Segment with multiple point prompts. ``points`` is a list of
    ``[x, y]``; ``labels[i]`` is 1 (foreground, include) or 0 (background,
    exclude). When omitted, all points are treated as foreground.

    Multiple foreground points refine the mask to span the indicated region;
    background points punch holes in the candidate. Result: mask as new canvas."""
    composite = store.compose(canvas_id)
    mask_img, score = sam.segment_with_points(
        composite, points, labels, model=model,
    )
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"SAM mask (score {score:.2f})")
    return {**_summary(cid), "iou_score": round(score, 3)}


@mcp.tool()
def sam_segment_box(canvas_id: str,
                    x1: float, y1: float, x2: float, y2: float, *,
                    model: str = sam.DEFAULT_MODEL,
                    new_canvas_id: str | None = None) -> dict:
    """Segment whatever's inside the bounding box (x1,y1)-(x2,y2). Result:
    mask as new canvas."""
    composite = store.compose(canvas_id)
    mask_img, score = sam.segment_with_box(
        composite, x1, y1, x2, y2, model=model,
    )
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"SAM mask (score {score:.2f})")
    return {**_summary(cid), "iou_score": round(score, 3)}


@mcp.tool()
def sam_segment_everything(canvas_id: str, *,
                           model: str = sam.DEFAULT_MODEL,
                           points_per_side: int | None = None,
                           min_mask_region_area: int = 0,
                           max_masks: int = 100) -> dict:
    """Auto-mask generation — find every salient region without prompts.

    Returns a list of mask canvas IDs sorted by score (highest first).
    ``points_per_side`` controls grid density (more = more masks + slower).
    ``min_mask_region_area`` filters out masks smaller than this many pixels.
    ``max_masks`` caps the number returned (-1 = no cap)."""
    composite = store.compose(canvas_id)
    results = sam.segment_everything(
        composite, model=model,
        points_per_side=points_per_side,
        min_mask_region_area=min_mask_region_area,
    )
    if max_masks > 0:
        results = results[:max_masks]

    mask_canvas_ids: list[str] = []
    summaries: list[dict] = []
    for i, r in enumerate(results):
        cid = store.put_image(
            r["mask"], layer_name=f"SAM mask {i} (score {r['score']:.2f})",
        )
        mask_canvas_ids.append(cid)
        summaries.append({
            "canvas_id": cid,
            "score": round(r["score"], 3),
            "area": r["area"],
            "bbox": r["bbox"],
        })
    return {
        "count": len(mask_canvas_ids),
        "masks": summaries,
    }


@mcp.tool()
def sam_apply_to_layer(canvas_id: str, mask_canvas_id: str, *,
                       layer_index: int | None = None) -> dict:
    """Convenience: take a SAM-generated mask canvas and apply it as a
    layer mask on ``canvas_id`` (default: active layer). Replaces any
    existing mask on that layer.

    Equivalent to ``set_layer_mask_from_canvas`` but explicit about the
    SAM workflow."""
    store.snapshot(canvas_id)
    e = store.entry(canvas_id)
    i = e.active_index if layer_index is None else layer_index
    layer = e.layers[i]
    src = store.compose(mask_canvas_id).convert("L")
    if src.size != layer.image.size:
        from PIL import Image as PILImage
        src = src.resize(layer.image.size, PILImage.LANCZOS)
    layer.mask = src
    return _summary(canvas_id)


# ============================================================ mask ops (feathering)

def _feather_if_requested(mask_img, feather: float, feather_method: str,
                          source_image=None):
    """Apply feathering to a mask if requested. Returns the (possibly
    modified) mask."""
    if feather and feather > 0:
        return mask_ops.feather_mask(
            mask_img, radius=float(feather),
            method=feather_method, source_image=source_image,
        )
    return mask_img


@mcp.tool()
def feather_mask(canvas_id: str, *, radius: float = 4.0,
                 method: str = "gaussian",
                 source_canvas_id: str | None = None,
                 new_canvas_id: str | None = None) -> dict:
    """Soften a mask canvas's edges.

    - ``radius``: blur radius in pixels (0 = no-op).
    - ``method``: ``gaussian`` (symmetric blur — classic Photoshop feather),
      ``inside`` (blur only inward, outer edge stays put),
      ``outside`` (blur only outward, halo effect),
      ``matte`` (guided-filter edge-aware refine — needs ``source_canvas_id``).
    - ``source_canvas_id``: required for ``matte`` mode; the image the mask
      was generated from, used as the guided-filter reference.
    """
    mask = store.compose(canvas_id).convert("L")
    src_img = store.compose(source_canvas_id) if source_canvas_id else None
    out = mask_ops.feather_mask(
        mask, radius=float(radius), method=method, source_image=src_img,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Feathered (r={radius:.1f})")
    return _summary(cid)


@mcp.tool()
def expand_mask(canvas_id: str, *, pixels: int = 4,
                new_canvas_id: str | None = None) -> dict:
    """Dilate a mask outward by ``pixels`` pixels."""
    mask = store.compose(canvas_id).convert("L")
    out = mask_ops.expand_mask(mask, pixels=int(pixels))
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Expanded (+{pixels}px)")
    return _summary(cid)


@mcp.tool()
def contract_mask(canvas_id: str, *, pixels: int = 4,
                  new_canvas_id: str | None = None) -> dict:
    """Erode a mask inward by ``pixels`` pixels."""
    mask = store.compose(canvas_id).convert("L")
    out = mask_ops.contract_mask(mask, pixels=int(pixels))
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Contracted (-{pixels}px)")
    return _summary(cid)


@mcp.tool()
def refine_mask(canvas_id: str, *,
                fill_holes_px: int = 0, remove_islands_px: int = 0,
                feather_radius: float = 0.0, feather_method: str = "gaussian",
                source_canvas_id: str | None = None,
                new_canvas_id: str | None = None) -> dict:
    """One-shot mask cleanup: close small holes, remove small islands, feather."""
    mask = store.compose(canvas_id).convert("L")
    src_img = store.compose(source_canvas_id) if source_canvas_id else None
    out = mask_ops.refine_mask(
        mask,
        fill_holes_px=int(fill_holes_px),
        remove_islands_px=int(remove_islands_px),
        feather_radius=float(feather_radius),
        feather_method=feather_method,
        source_image=src_img,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name="Refined mask")
    return _summary(cid)


@mcp.tool()
def apply_mask_as_alpha(canvas_id: str, mask_canvas_id: str, *,
                        new_canvas_id: str | None = None) -> dict:
    """Combine an image canvas with a mask canvas, producing an RGBA cutout
    where the mask becomes the alpha channel. Best paired with a feathered
    mask for natural-edge compositing."""
    img = store.compose(canvas_id)
    mask = store.compose(mask_canvas_id).convert("L")
    out = mask_ops.apply_mask_as_alpha(img, mask)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name="Cutout (RGBA)")
    return _summary(cid)


@mcp.tool()
def mask_overlay(canvas_id: str, mask_canvas_ids: list[str], *,
                 colors: list[str] | None = None,
                 alpha: float = 0.5,
                 show_bbox: bool = True,
                 bbox_width: int = 2,
                 labels: list[str] | None = None,
                 label_size: int = 14,
                 new_canvas_id: str | None = None) -> dict:
    """Render a "screenshot" of segmentation results: composite one or more
    masks onto ``canvas_id`` as tinted overlays with bbox outlines and
    optional labels. Result is a NEW canvas — pair with
    ``get_canvas_preview`` to view inline.

    - ``mask_canvas_ids``: list of mask canvases (composited and treated as L)
    - ``colors``: per-mask hex strings (e.g. ``["#FF00FF", "#00FF00"]``).
      Default cycles a 10-color palette.
    - ``alpha`` ∈ 0-1: overlay intensity. 0.5 keeps the original visible
      under the tint.
    - ``show_bbox``: draws a 2-px rectangle around each mask's tight bbox.
    - ``labels``: optional list of strings drawn near each bbox corner
      (e.g. ``["person", "dog"]``).

    Use after ``sam_*`` / ``yolo_*`` / ``birefnet_*`` / ``clipseg_*`` to
    visually verify mask quality before feathering or compositing."""
    if isinstance(mask_canvas_ids, str):
        mask_canvas_ids = [mask_canvas_ids]
    base = store.compose(canvas_id)
    masks = [store.compose(mid).convert("L") for mid in mask_canvas_ids]
    out = mask_ops.overlay_masks(
        base, masks,
        colors=colors, alpha=alpha,
        show_bbox=show_bbox, bbox_width=bbox_width,
        labels=labels, label_size=label_size,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name="Mask overlay")
    return _summary(cid)


@mcp.tool()
def mask_preview(canvas_id: str, mask_canvas_ids: list[str], *,
                 colors: list[str] | None = None,
                 alpha: float = 0.5,
                 show_bbox: bool = True,
                 labels: list[str] | None = None,
                 max_size: int = 800) -> MCPImage:
    """Inline-PNG screenshot of segmentation results (the "look at this mask"
    quick-view). Same args as ``mask_overlay`` but returns the PNG directly
    instead of creating a canvas — use this in interactive workflows where
    you want a visual without polluting the canvas store.

    ``max_size`` caps the longer edge for compact transport."""
    if isinstance(mask_canvas_ids, str):
        mask_canvas_ids = [mask_canvas_ids]
    base = store.compose(canvas_id)
    masks = [store.compose(mid).convert("L") for mid in mask_canvas_ids]
    out = mask_ops.overlay_masks(
        base, masks,
        colors=colors, alpha=alpha,
        show_bbox=show_bbox, labels=labels,
    )
    png = _img_to_png_bytes(out, max_size=int(max_size))
    return MCPImage(data=png, format="png")


# ============================================================ YOLO segmentation

@mcp.tool()
def yolo_status() -> dict:
    """Whether YOLOv8/v11-seg is installed + idle state. Install with
    ``pip install image-tools-mcp[yolo]``."""
    return yolo_seg.status()


@mcp.tool()
def yolo_load(*, weights: str = yolo_seg.DEFAULT_WEIGHTS) -> dict:
    """Pre-warm a YOLO segmentation model. Weights: ``yolov8n-seg.pt`` (smallest)
    .. ``yolov8x-seg.pt`` (largest); ``yolo11{n,s,m,l,x}-seg.pt`` for the v11
    family. Default is ``yolov8l-seg.pt`` — the best perf/latency tradeoff per
    the 2026-05 benchmark."""
    return yolo_seg.load_model(weights)


@mcp.tool()
def yolo_unload() -> dict:
    """Free YOLO GPU memory."""
    return yolo_seg.unload()


@mcp.tool()
def yolo_set_idle_timeout(seconds: float) -> dict:
    """Set YOLO idle timeout (default 1 h; 0 disables auto-evict)."""
    return yolo_seg.set_idle_timeout(seconds)


@mcp.tool()
def yolo_segment_at_point(canvas_id: str, x: float, y: float, *,
                          weights: str = yolo_seg.DEFAULT_WEIGHTS,
                          conf: float = 0.25,
                          feather: float = 0.0,
                          feather_method: str = "gaussian",
                          new_canvas_id: str | None = None) -> dict:
    """Find the highest-confidence YOLO detection containing (x, y) and
    return its mask. ``feather > 0`` softens the mask edges."""
    composite = store.compose(canvas_id)
    mask_img, score, class_name = yolo_seg.segment_at_point(
        composite, x, y, weights=weights, conf=conf,
    )
    mask_img = _feather_if_requested(mask_img, feather, feather_method,
                                     source_image=composite)
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"YOLO {class_name} ({score:.2f})")
    return {**_summary(cid), "score": round(score, 3), "class": class_name}


@mcp.tool()
def yolo_segment_in_box(canvas_id: str, x1: float, y1: float,
                        x2: float, y2: float, *,
                        weights: str = yolo_seg.DEFAULT_WEIGHTS,
                        conf: float = 0.25,
                        feather: float = 0.0,
                        feather_method: str = "gaussian",
                        new_canvas_id: str | None = None) -> dict:
    """Find the YOLO detection whose mask best overlaps the given bounding box."""
    composite = store.compose(canvas_id)
    mask_img, score, class_name = yolo_seg.segment_in_box(
        composite, x1, y1, x2, y2, weights=weights, conf=conf,
    )
    mask_img = _feather_if_requested(mask_img, feather, feather_method,
                                     source_image=composite)
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"YOLO {class_name} ({score:.2f})")
    return {**_summary(cid), "score": round(score, 3), "class": class_name}


@mcp.tool()
def yolo_segment_everything(canvas_id: str, *,
                            weights: str = yolo_seg.DEFAULT_WEIGHTS,
                            conf: float = 0.25, iou: float = 0.5,
                            max_results: int = 50,
                            feather: float = 0.0,
                            feather_method: str = "gaussian") -> dict:
    """Run YOLO seg + return every detection as a separate mask canvas. Each
    entry includes the class name, bbox, score, and canvas_id."""
    composite = store.compose(canvas_id)
    results = yolo_seg.segment_everything(
        composite, weights=weights, conf=conf, iou=iou, max_results=max_results,
    )
    out = []
    for i, r in enumerate(results):
        m = _feather_if_requested(r["mask"], feather, feather_method,
                                  source_image=composite)
        cid = store.put_image(m, layer_name=f"YOLO {r['class_name']} ({r['score']:.2f})")
        out.append({
            "canvas_id": cid,
            "class": r["class_name"],
            "score": round(r["score"], 3),
            "bbox": r["bbox"],
        })
    return {"count": len(out), "masks": out}


# ============================================================ SAM 1 (original)

@mcp.tool()
def sam1_status() -> dict:
    """Whether SAM 1 (Meta original) is installed + idle state."""
    return sam1.status()


@mcp.tool()
def sam1_load(*, model: str = sam1.DEFAULT_MODEL) -> dict:
    """Pre-warm SAM 1. Models: ``facebook/sam-vit-base``,
    ``facebook/sam-vit-large`` (default — best perf/quality on COCO),
    ``facebook/sam-vit-huge``."""
    return sam1.load_model(model)


@mcp.tool()
def sam1_unload() -> dict:
    """Free SAM 1 GPU memory."""
    return sam1.unload()


@mcp.tool()
def sam1_set_idle_timeout(seconds: float) -> dict:
    """Set SAM 1 idle timeout (default 1 h)."""
    return sam1.set_idle_timeout(seconds)


@mcp.tool()
def sam1_segment_point(canvas_id: str, x: float, y: float, *,
                       label: int = 1,
                       model: str = sam1.DEFAULT_MODEL,
                       feather: float = 0.0,
                       feather_method: str = "gaussian",
                       new_canvas_id: str | None = None) -> dict:
    """Single-click prompt segmentation via SAM 1. ``label`` is 1 (include)
    or 0 (exclude). ``feather > 0`` softens the edge."""
    composite = store.compose(canvas_id)
    mask_img, score = sam1.segment_point(composite, x, y,
                                          label=int(label), model=model)
    mask_img = _feather_if_requested(mask_img, feather, feather_method,
                                     source_image=composite)
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"SAM1 mask ({score:.2f})")
    return {**_summary(cid), "iou_score": round(score, 3)}


@mcp.tool()
def sam1_segment_points(canvas_id: str,
                        points: list[list[float]],
                        labels: list[int] | None = None, *,
                        model: str = sam1.DEFAULT_MODEL,
                        feather: float = 0.0,
                        feather_method: str = "gaussian",
                        new_canvas_id: str | None = None) -> dict:
    """Multi-point refinement: foreground (label=1) and background (label=0)
    clicks can be combined to narrow the mask."""
    composite = store.compose(canvas_id)
    mask_img, score = sam1.segment_points(composite, points, labels, model=model)
    mask_img = _feather_if_requested(mask_img, feather, feather_method,
                                     source_image=composite)
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"SAM1 mask ({score:.2f})")
    return {**_summary(cid), "iou_score": round(score, 3)}


@mcp.tool()
def sam1_segment_box(canvas_id: str, x1: float, y1: float,
                     x2: float, y2: float, *,
                     model: str = sam1.DEFAULT_MODEL,
                     feather: float = 0.0,
                     feather_method: str = "gaussian",
                     new_canvas_id: str | None = None) -> dict:
    """Bounding-box prompt. The 2026-05 benchmark winner on COCO boxes
    (0.78 median IoU)."""
    composite = store.compose(canvas_id)
    mask_img, score = sam1.segment_box(composite, x1, y1, x2, y2, model=model)
    mask_img = _feather_if_requested(mask_img, feather, feather_method,
                                     source_image=composite)
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"SAM1 mask ({score:.2f})")
    return {**_summary(cid), "iou_score": round(score, 3)}


# ============================================================ BiRefNet

@mcp.tool()
def birefnet_status() -> dict:
    """Whether BiRefNet is installed + idle state."""
    return birefnet.status()


@mcp.tool()
def birefnet_load(*, variant: str = birefnet.DEFAULT_VARIANT) -> dict:
    """Pre-warm BiRefNet. Variants: ``general`` (default), ``portrait``,
    ``hr`` (2048 px, sharpest), ``matting`` (soft alpha), ``dis5k`` (small/
    intricate subjects)."""
    return birefnet.load_model(variant)


@mcp.tool()
def birefnet_unload() -> dict:
    """Free BiRefNet GPU memory."""
    return birefnet.unload()


@mcp.tool()
def birefnet_set_idle_timeout(seconds: float) -> dict:
    """Set BiRefNet idle timeout (default 1 h)."""
    return birefnet.set_idle_timeout(seconds)


@mcp.tool()
def birefnet_remove_background(canvas_id: str, *,
                               variant: str = birefnet.DEFAULT_VARIANT,
                               feather: float = 0.0,
                               feather_method: str = "gaussian",
                               new_canvas_id: str | None = None) -> dict:
    """Best-in-class background removal. Returns a soft (0..255) foreground
    mask. ``feather`` can further smooth the edges; BiRefNet's natural output
    already has soft edges, so feather is usually unnecessary."""
    composite = store.compose(canvas_id)
    mask = birefnet.remove_background(composite, variant=variant)
    mask = _feather_if_requested(mask, feather, feather_method,
                                 source_image=composite)
    cid = store.put_image(mask, canvas_id=new_canvas_id,
                          layer_name=f"BiRefNet {variant}")
    return _summary(cid)


# ============================================================ CLIPSeg

@mcp.tool()
def clipseg_status() -> dict:
    """Whether CLIPSeg is installed + idle state."""
    return clipseg.status()


@mcp.tool()
def clipseg_load(*, model: str = clipseg.DEFAULT_MODEL) -> dict:
    """Pre-warm CLIPSeg. Default model is ~50 MB."""
    return clipseg.load_model(model)


@mcp.tool()
def clipseg_unload() -> dict:
    """Free CLIPSeg GPU memory."""
    return clipseg.unload()


@mcp.tool()
def clipseg_set_idle_timeout(seconds: float) -> dict:
    """Set CLIPSeg idle timeout (default 1 h)."""
    return clipseg.set_idle_timeout(seconds)


@mcp.tool()
def clipseg_segment_text(canvas_id: str, text: str, *,
                         model: str = clipseg.DEFAULT_MODEL,
                         threshold: float | None = 0.5,
                         feather: float = 0.0,
                         feather_method: str = "gaussian",
                         new_canvas_id: str | None = None) -> dict:
    """Text-prompted segmentation. Pass ``threshold=None`` to get a soft
    confidence mask (already feathered by design). Pass a float (default 0.5)
    to binarize. ``feather > 0`` adds further smoothing."""
    composite = store.compose(canvas_id)
    mask_img, max_conf = clipseg.segment_text(
        composite, text, model=model, threshold=threshold,
    )
    mask_img = _feather_if_requested(mask_img, feather, feather_method,
                                     source_image=composite)
    cid = store.put_image(mask_img, canvas_id=new_canvas_id,
                          layer_name=f"CLIPSeg '{text}' ({max_conf:.2f})")
    return {**_summary(cid), "max_confidence": round(max_conf, 3)}


# ============================================================ Blur effects (PS-style)

@mcp.tool()
def motion_blur(canvas_id: str, *, angle: float = 0.0,
                distance: int = 20) -> dict:
    """Directional motion blur on the active layer (PS Filter > Blur >
    Motion Blur). ``angle`` is in degrees (0 = horizontal, 90 = vertical),
    ``distance`` is the streak length in pixels."""
    return _layer_filter(canvas_id, lambda img: blurs.motion_blur(
        img, angle=angle, distance=distance))


@mcp.tool()
def radial_blur(canvas_id: str, *, mode: str = "spin",
                amount: float = 0.05,
                center_x: float | None = None,
                center_y: float | None = None,
                samples: int = 20) -> dict:
    """Radial blur on the active layer (PS Filter > Blur > Radial Blur).

    - ``mode='spin'``: rotational blur. ``amount`` is the max rotation
      in radians at the image edge.
    - ``mode='zoom'``: radial blur outward from centre. ``amount`` is the
      max scale offset (0.1 = 10%).

    ``samples`` is the number of intermediate steps averaged (higher =
    smoother, slower). Centre defaults to image centre."""
    return _layer_filter(canvas_id, lambda img: blurs.radial_blur(
        img, mode=mode, amount=amount,
        center_x=center_x, center_y=center_y, samples=samples))


@mcp.tool()
def lens_blur(canvas_id: str, *, radius: int = 12,
              shape: str = "disc") -> dict:
    """Bokeh-style lens blur on the active layer (PS Filter > Blur > Lens
    Blur). Bright highlights bloom into ``shape``-aperture forms — unlike
    Gaussian which just softens.

    - ``radius`` is the aperture radius in pixels.
    - ``shape`` ∈ ``disc`` (circular, default) | ``hex`` (six-blade
      aperture). Hex gives more "cinematic" bokeh."""
    return _layer_filter(canvas_id, lambda img: blurs.lens_blur(
        img, radius=radius, shape=shape))


@mcp.tool()
def tilt_shift(canvas_id: str, *,
               focus_y: float | None = None,
               focus_height: float = 0.25,
               max_blur: float = 12.0,
               falloff: float = 2.0) -> dict:
    """Tilt-shift "fake miniature" effect on the active layer (PS Filter >
    Blur > Tilt-Shift). A horizontal band stays sharp; everything above
    and below blurs progressively.

    - ``focus_y`` ∈ 0-1: vertical centre of the focus band (None = image
      centre).
    - ``focus_height`` ∈ 0-1: fraction of image height kept sharp.
    - ``max_blur``: Gaussian radius (px) at the top + bottom edges.
    - ``falloff`` ≥ 1: steepness of focus→blur transition (2 = smooth,
      4 = sharp)."""
    return _layer_filter(canvas_id, lambda img: blurs.tilt_shift(
        img, focus_y=focus_y, focus_height=focus_height,
        max_blur=max_blur, falloff=falloff))


@mcp.tool()
def box_blur(canvas_id: str, *, radius: int = 4) -> dict:
    """Box (mean) blur on the active layer — fast alternative to gaussian
    when slightly harder edges are acceptable. ``radius`` is half the
    kernel size (kernel = ``2*radius + 1`` square)."""
    return _layer_filter(canvas_id, lambda img: blurs.box_blur(
        img, radius=radius))


# ============================================================ Utilities (watermark, QR, hash, compare, histogram, annotate)

@mcp.tool()
def add_watermark(canvas_id: str, *,
                  text: str | None = None,
                  watermark_canvas_id: str | None = None,
                  position: str = "bottom_right",
                  padding: int = 20,
                  opacity: float = 0.5,
                  scale: float = 1.0,
                  text_color: str = "#FFFFFFFF",
                  text_size: int = 24,
                  font_path: str | None = None,
                  new_canvas_id: str | None = None) -> dict:
    """Add a text or image watermark to a canvas. Provide EXACTLY ONE of
    ``text`` or ``watermark_canvas_id``.

    - ``position`` ∈ ``top_left | top_right | bottom_left | bottom_right |
      center | top | bottom | left | right``
    - ``padding`` is the edge inset in pixels
    - ``opacity`` 0-1 multiplies the watermark's alpha
    - ``scale`` resizes an image watermark (ignored for text)
    - ``text_color`` is a hex / RGBA value (default solid white)
    - ``font_path`` optionally points to a TTF; default tries arial.ttf
      then falls back to PIL bitmap"""
    base = store.compose(canvas_id)
    wm_img = store.compose(watermark_canvas_id) if watermark_canvas_id else None
    tc = parse_color(text_color)
    out = utilities.add_watermark(
        base, text=text, watermark_image=wm_img,
        position=position, padding=padding,
        opacity=opacity, scale=scale,
        text_color=(tc[0], tc[1], tc[2], tc[3]),
        text_size=text_size, font_path=font_path,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Watermark")
    return _summary(cid)


@mcp.tool()
def make_qr_code(text: str, *, size: int = 256,
                 error_correction: str = "M",
                 fill_color: str = "#000000",
                 back_color: str = "#FFFFFF",
                 border: int = 4,
                 canvas_id: str | None = None) -> dict:
    """Generate a QR code as a new canvas of ``size``×``size`` pixels.

    - ``error_correction`` ∈ ``L | M | Q | H`` (7/15/25/30 % recoverable)
    - ``border`` is the quiet-zone width in modules (spec: 4)"""
    img = utilities.make_qr_code(
        text, size=size, error_correction=error_correction,
        fill_color=fill_color, back_color=back_color, border=border,
    )
    cid = store.put_image(img, canvas_id=canvas_id, layer_name="QR code")
    return _summary(cid)


@mcp.tool()
def perceptual_hash(canvas_id: str, *, kind: str = "phash",
                    size: int = 8) -> dict:
    """Compute a perceptual hash of a canvas for duplicate / near-duplicate
    detection. ``kind`` ∈ ``phash`` (DCT-based, robust to compression and
    minor edits), ``ahash`` (average — fast, brittle), ``dhash``
    (difference — fast, OK). Returns hex string + bit length.

    Use ``hamming_distance(a, b)`` to compare two hashes — < 5 bits ≈
    visually identical, 5-10 similar, > 12 different."""
    img = store.compose(canvas_id)
    return utilities.perceptual_hash(img, kind=kind, size=size)


@mcp.tool()
def hamming_distance(hash_a: str, hash_b: str) -> dict:
    """Bit-difference count between two perceptual hashes of the same
    length. Cheap interpretation guide: < 5 = same image, 5-10 = similar,
    > 12 = different."""
    return {"hamming_distance": utilities.hamming_distance(hash_a, hash_b)}


@mcp.tool()
def compare_images(canvas_a_id: str, canvas_b_id: str, *,
                   metrics: list[str] | None = None) -> dict:
    """Quantitative comparison between two canvases. Smaller is resized to
    match larger if dimensions differ.

    ``metrics`` may include ``mse`` (mean squared error), ``rmse``,
    ``psnr`` (peak signal-to-noise, dB — higher is better, > 40 ≈
    indistinguishable), ``ssim`` (structural similarity, 0-1 — closer to
    1 is better), ``diff_pct`` (% of pixels differing by > 5 grey
    levels). Default runs all five."""
    a = store.compose(canvas_a_id)
    b = store.compose(canvas_b_id)
    return utilities.compare_images(a, b, metrics=metrics)


@mcp.tool()
def diff_image(canvas_a_id: str, canvas_b_id: str, *,
               highlight_color: str = "#FF00FF",
               alpha: float = 0.6,
               new_canvas_id: str | None = None) -> dict:
    """Visualise the difference between two canvases: pixels that differ
    by > 3 grey levels are tinted ``highlight_color`` over the first
    canvas. Result lands on a NEW canvas. Pair with ``compare_images`` for
    numeric metrics."""
    a = store.compose(canvas_a_id)
    b = store.compose(canvas_b_id)
    c = parse_color(highlight_color)
    out = utilities.diff_image(a, b, highlight_color=(c[0], c[1], c[2]),
                                alpha=alpha)
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Diff")
    return _summary(cid)


@mcp.tool()
def histogram(canvas_id: str, *, bins: int = 256,
              channels: list[str] | None = None) -> dict:
    """Return histogram data for inspection — counts per bin per channel.
    ``channels`` ∈ subset of ``r``, ``g``, ``b``, ``a``, ``l`` (luminance).
    Default: ``[r, g, b, l]``. Use to diagnose exposure / clipping issues
    or feed into UIs."""
    img = store.compose(canvas_id)
    return utilities.histogram(img, bins=bins, channels=channels)


@mcp.tool()
def color_replace(canvas_id: str, from_color: str, to_color: str, *,
                  tolerance: int = 16, feather: float = 0.0,
                  new_canvas_id: str | None = None) -> dict:
    """Swap pixels within ``tolerance`` of ``from_color`` to ``to_color``
    (which may carry alpha, e.g. ``"#00000000"`` to make a colour
    transparent). ``feather`` > 0 softens the selection mask before
    applying. Result is a NEW canvas (RGBA)."""
    img = store.compose(canvas_id)
    fc = parse_color(from_color)
    tc = parse_color(to_color)
    out = utilities.color_replace(
        img, from_color=(fc[0], fc[1], fc[2]),
        to_color=(tc[0], tc[1], tc[2], tc[3]),
        tolerance=tolerance, feather=feather,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name="Colour replace")
    return _summary(cid)


@mcp.tool()
def rounded_corners(canvas_id: str, *, radius: int = 20,
                    corners: str = "all",
                    new_canvas_id: str | None = None) -> dict:
    """Round the corners of a canvas, producing an RGBA result where
    rounded-off pixels are transparent. ``corners`` ∈ ``all | top |
    bottom | left | right | top_left | top_right | bottom_left |
    bottom_right``."""
    img = store.compose(canvas_id)
    out = utilities.rounded_corners(img, radius=radius, corners=corners)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Rounded {radius}px")
    return _summary(cid)


@mcp.tool()
def letterbox(canvas_id: str, *, aspect: float,
              fill_color: str = "#000000",
              new_canvas_id: str | None = None) -> dict:
    """Extend the canvas to ``aspect = width/height`` by adding bars on
    the short axis. Opposite of ``smart_crop_to_aspect`` — preserves all
    of the source content."""
    img = store.compose(canvas_id)
    c = parse_color(fill_color)
    out = utilities.letterbox(img, aspect=aspect,
                              fill_color=(c[0], c[1], c[2], c[3]))
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Letterbox {aspect:.3f}")
    return _summary(cid)


@mcp.tool()
def white_balance(canvas_id: str, *, method: str = "gray_world",
                  strength: float = 1.0) -> dict:
    """Auto white-balance / colour-cast removal on the active layer.

    Methods:
      - ``gray_world`` (default): scale channels so the scene average is
        neutral. Best general-purpose.
      - ``white_patch``: scale so the brightest pixel becomes white.
      - ``simplest_cb``: stretch each channel's 1-99% quantile to full
        range. Most aggressive — handles heavy casts.

    ``strength`` 0-1 blends with the original."""
    return _layer_filter(canvas_id, lambda img: utilities.white_balance(
        img, method=method, strength=strength))


@mcp.tool()
def annotate(canvas_id: str, items: list[dict], *,
             font_size: int = 14, default_color: str = "#FF00FF",
             new_canvas_id: str | None = None) -> dict:
    """Draw a batch of annotations on a copy of the canvas. ``items`` is a
    list of dicts; each has a ``kind`` and per-kind args:

    - ``{"kind":"rect","bbox":[x1,y1,x2,y2],"label":"...","color":"#..","width":2}``
    - ``{"kind":"arrow","x1":..,"y1":..,"x2":..,"y2":..,"color":"#..","width":2}``
    - ``{"kind":"label","x":..,"y":..,"text":"...","color":"#..","bg":[r,g,b,a]}``
    - ``{"kind":"circle","x":..,"y":..,"radius":..,"color":"#..","width":2}``

    Useful for labelling AI outputs, building demos, marking up
    screenshots."""
    img = store.compose(canvas_id)
    out = utilities.annotate(img, items, font_size=font_size,
                              default_color=default_color)
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Annotated")
    return _summary(cid)


@mcp.tool()
def glitch_effect(canvas_id: str, *, intensity: float = 0.5,
                  seed: int | None = None,
                  new_canvas_id: str | None = None) -> dict:
    """Quick stylised glitch — RGB channel misalignment + random row
    shifts. ``intensity`` 0-1; ``seed`` for reproducibility."""
    img = store.compose(canvas_id)
    out = utilities.glitch_effect(img, intensity=intensity, seed=seed)
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Glitch")
    return _summary(cid)


# ============================================================ Extras (palette, crop, sharpen, vignette, noise, wand, blend)

@mcp.tool()
def auto_crop_to_content(canvas_id: str, *,
                         alpha_threshold: int = 1,
                         bg_color: list[int] | None = None,
                         tolerance: int = 5,
                         padding: int = 0,
                         new_canvas_id: str | None = None) -> dict:
    """Trim transparent / uniform-background borders. For RGBA inputs,
    trims pixels with alpha < ``alpha_threshold``. For opaque inputs,
    trims pixels within ``tolerance`` of ``bg_color`` (defaults to the
    median of the four corner pixels). ``padding`` adds margin around the
    detected content."""
    img = store.compose(canvas_id)
    out = transforms.auto_crop_to_content(
        img, alpha_threshold=alpha_threshold,
        bg_color=tuple(bg_color) if bg_color else None,
        tolerance=tolerance, padding=padding,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Auto crop")
    return _summary(cid)


@mcp.tool()
def smart_crop_to_aspect(canvas_id: str, *,
                         aspect: float,
                         anchor: str = "center",
                         new_canvas_id: str | None = None) -> dict:
    """Crop to the largest rectangle of the given ``aspect = width/height``
    that fits inside the canvas. ``anchor`` ∈ ``center | top | bottom |
    left | right | top_left | top_right | bottom_left | bottom_right``.

    Example: ``aspect=1.0`` for square, ``aspect=16/9`` ≈ 1.777..., ``9/16``
    ≈ 0.5625, ``4/3`` ≈ 1.333. Use before AI generation tools to match
    your target output ratio."""
    img = store.compose(canvas_id)
    out = transforms.smart_crop_to_aspect(img, aspect=aspect, anchor=anchor)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Smart crop {aspect:.3f}")
    return _summary(cid)


@mcp.tool()
def pixelate(canvas_id: str, *, block_size: int = 16,
             x1: int | None = None, y1: int | None = None,
             x2: int | None = None, y2: int | None = None,
             mask_canvas_id: str | None = None,
             new_canvas_id: str | None = None) -> dict:
    """Mosaic / pixelate. ``block_size`` is the block size in pixels. Pass
    a bbox (``x1, y1, x2, y2``) or a ``mask_canvas_id`` to restrict to a
    region — without either, the whole canvas is pixelated.

    Common uses: face / text anonymisation (combine with ``face_detect``
    or ``sam_segment_*`` to get the mask), retro pixel-art style."""
    img = store.compose(canvas_id)
    bbox = None
    if any(v is not None for v in (x1, y1, x2, y2)):
        w, h = img.size
        bbox = (max(0, int(x1 or 0)),
                max(0, int(y1 or 0)),
                min(w, int(x2 if x2 is not None else w)),
                min(h, int(y2 if y2 is not None else h)))
    mask = store.compose(mask_canvas_id).convert("L") if mask_canvas_id else None
    out = transforms.pixelate(img, block_size=block_size, bbox=bbox, mask=mask)
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Pixelate")
    return _summary(cid)


@mcp.tool()
def extract_palette(canvas_id: str, *, n_colors: int = 8,
                    method: str = "kmeans") -> dict:
    """Return the dominant ``n_colors`` colours of ``canvas_id`` as a list
    sorted by frequency (most common first). Each entry: ``hex``, ``rgb``,
    ``ratio`` (fraction of pixels).

    ``method``: ``kmeans`` (pure-NumPy clustering, default), ``median_cut``
    (Pillow's quantiser — fast), ``mode`` (6³-cell binning, fastest but
    crude)."""
    img = store.compose(canvas_id)
    return {
        "canvas_id": canvas_id,
        "n_colors": int(n_colors),
        "method": method,
        "palette": palette.extract_palette(img, n_colors=n_colors, method=method),
    }


@mcp.tool()
def color_quantize(canvas_id: str, *, n_colors: int = 16,
                   method: str = "median_cut",
                   dither: bool = True,
                   new_canvas_id: str | None = None) -> dict:
    """Reduce the canvas to ``n_colors`` total colours. Different from
    ``posterize`` (which clips bit-depth per channel) — this finds the
    best palette via the chosen algorithm.

    ``method`` ∈ ``median_cut`` (default), ``maxcoverage`` (alternative
    Pillow algo), ``fastoctree``. ``dither=True`` applies Floyd-Steinberg
    dithering for smoother gradients."""
    img = store.compose(canvas_id)
    out = palette.quantize_to_palette(
        img, n_colors=n_colors, method=method, dither=dither,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Quantize {n_colors}")
    return _summary(cid)


@mcp.tool()
def unsharp_mask(canvas_id: str, *, radius: float = 2.0,
                 amount: float = 1.5, threshold: int = 0) -> dict:
    """Photoshop-style Unsharp Mask sharpening on the active layer.
    ``radius`` controls blur extent (pixels), ``amount`` is the strength
    multiplier (1.0 ≈ PS 100%), ``threshold`` 0-255 suppresses sharpening
    where the local diff is below this many gray levels (keep skin smooth,
    sharpen edges)."""
    return _layer_filter(canvas_id, lambda img: adjustments.unsharp_mask(
        img, radius=radius, amount=amount, threshold=threshold))


@mcp.tool()
def high_pass(canvas_id: str, *, radius: float = 4.0) -> dict:
    """High-pass filter on the active layer: image − blur(image), shifted
    to mid-gray. Pair with ``set_layer_blend_mode("soft_light")`` or
    ``"linear_light"`` for frequency-separation retouching."""
    return _layer_filter(canvas_id, lambda img: adjustments.high_pass(
        img, radius=radius))


@mcp.tool()
def add_vignette(canvas_id: str, *, strength: float = 0.6,
                 radius: float = 1.0, falloff: float = 2.0,
                 color: str = "#000000") -> dict:
    """Photo vignette — darken (or tint) the corners radially on the
    active layer. ``strength`` 0-1 = corner intensity. ``radius`` 0-2 =
    where the unaffected centre ends (1 = halfway to edge). ``falloff``
    ≥ 1 = gradient steepness. ``color`` (default black) is the corner
    tint — try ``"#3a1a00"`` for a warm sepia vignette."""
    c = parse_color(color)
    return _layer_filter(canvas_id, lambda img: adjustments.add_vignette(
        img, strength=strength, radius=radius, falloff=falloff,
        color=(c[0], c[1], c[2])))


@mcp.tool()
def add_noise(canvas_id: str, *, amount: float = 0.1,
              kind: str = "gaussian",
              seed: int | None = None) -> dict:
    """Add noise to the active layer. ``kind``: ``gaussian`` (default,
    zero-mean), ``uniform`` (±amount), ``salt_pepper`` (random
    black/white pixels — ``amount`` = proportion). Typical ``amount`` 0.02-0.2
    for grain effect."""
    return _layer_filter(canvas_id, lambda img: adjustments.add_noise(
        img, amount=amount, kind=kind, seed=seed))


@mcp.tool()
def bilateral_filter(canvas_id: str, *, diameter: int = 9,
                     sigma_color: float = 75.0,
                     sigma_space: float = 75.0) -> dict:
    """Edge-preserving smoothing on the active layer (denoise without
    blurring edges, OpenCV's ``bilateralFilter``). ``diameter`` 5-15
    typical; higher ``sigma_color`` = more aggressive cross-edge smoothing."""
    return _layer_filter(canvas_id, lambda img: adjustments.bilateral_filter(
        img, diameter=diameter, sigma_color=sigma_color,
        sigma_space=sigma_space))


@mcp.tool()
def magic_wand(canvas_id: str, x: int, y: int, *,
               tolerance: int = 16, contiguous: bool = True,
               new_canvas_id: str | None = None) -> dict:
    """Photoshop Magic Wand — select all pixels within ``tolerance`` of the
    colour at ``(x, y)``. Returns a NEW mask canvas (L mode, 255 = selected).

    - ``tolerance`` 0-255: max per-channel difference.
    - ``contiguous=true`` (default) restricts to the connected region from
      the seed point. ``false`` selects every matching pixel image-wide
      (PS Select > Colour Range behaviour)."""
    img = store.compose(canvas_id)
    mask = mask_ops.magic_wand(img, x=x, y=y, tolerance=tolerance,
                               contiguous=contiguous)
    cid = store.put_image(mask, canvas_id=new_canvas_id, layer_name="Magic wand")
    return _summary(cid)


@mcp.tool()
def blend_canvases(base_canvas_id: str, overlay_canvas_id: str, *,
                   mode: str = "normal", opacity: float = 1.0,
                   new_canvas_id: str | None = None) -> dict:
    """Composite two canvases into a NEW canvas with the chosen blend mode
    + opacity. Overlay is resized to match base if dimensions differ.

    ``mode`` ∈ ``normal | multiply | screen | overlay | darken | lighten |
    color_dodge | color_burn | hard_light | soft_light | difference |
    exclusion | add | subtract``. Use this when you want a one-shot blend
    without managing layers (vs ``add_layer`` + ``set_layer_blend_mode``)."""
    if mode not in layers_mod.BLEND_MODES:
        raise ValueError(
            f"unknown blend mode {mode!r}. Choose from {layers_mod.BLEND_MODES}."
        )
    base = store.compose(base_canvas_id)
    ov = store.compose(overlay_canvas_id)
    out = mask_ops.blend_two(base, ov, mode=mode, opacity=opacity)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Blend ({mode})")
    return _summary(cid)


# ============================================================ Warp / distort (PS-style)

@mcp.tool()
def warp_perspective(canvas_id: str, src_corners: list[list[float]],
                     dst_corners: list[list[float]], *,
                     output_width: int | None = None,
                     output_height: int | None = None,
                     new_canvas_id: str | None = None) -> dict:
    """4-corner perspective warp. ``src_corners`` and ``dst_corners`` are
    lists of FOUR ``[x, y]`` points in the SAME order (typical convention:
    top-left, top-right, bottom-right, bottom-left). The output is a NEW
    canvas; source untouched.

    Use cases: document straightening, perspective correction, projecting
    a flat graphic onto a tilted surface.

    Output size defaults to source size. Pixels remapped from outside the
    source are transparent."""
    img = store.compose(canvas_id)
    out_size = None
    if output_width or output_height:
        w, h = img.size
        out_size = (int(output_width or w), int(output_height or h))
    out = warp.warp_perspective(img, src_corners, dst_corners,
                                output_size=out_size)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name="Perspective warp")
    return _summary(cid)


@mcp.tool()
def warp_mesh(canvas_id: str, src_grid: list[list[float]],
              dst_grid: list[list[float]], *,
              regularization: float = 0.0,
              new_canvas_id: str | None = None) -> dict:
    """NxN mesh warp (Photoshop's Edit > Transform > Warp). Pass two lists
    of matching ``[x, y]`` control points; the source point moves to the
    destination point, with smooth thin-plate-spline interpolation between
    them. Need >= 3 points (4-9 is comfortable, 25 = a 5x5 grid).

    ``regularization`` ≥ 0 trades fidelity for smoothness — 0 hits the
    control points exactly, higher relaxes."""
    img = store.compose(canvas_id)
    out = warp.warp_mesh(img, src_grid, dst_grid,
                         regularization=regularization)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name="Mesh warp")
    return _summary(cid)


@mcp.tool()
def liquify(canvas_id: str, mode: str, points: list[list[float]], *,
            radius: float = 50.0, strength: float = 0.5,
            push_x: float | None = None, push_y: float | None = None,
            new_canvas_id: str | None = None) -> dict:
    """Local brush-based deformation (Photoshop's Filter > Liquify).

    ``mode``:
      - ``push`` — drag pixels along ``(push_x, push_y)`` (required for
        this mode). Use when you know the stroke direction.
      - ``twirl_cw`` / ``twirl_ccw`` — rotate pixels inside the brush
        around each point.
      - ``pucker`` — pull pixels toward the brush centre (shrink).
      - ``bloat`` — push pixels away (enlarge).

    ``radius`` is the brush radius in pixels. ``strength`` 0-1 nominal;
    over-1 over-pushes. ``points`` is a list of ``[x, y]`` brush centres —
    a single point for a stamp, many for a stroke."""
    img = store.compose(canvas_id)
    push_vec = None
    if mode == "push":
        if push_x is None or push_y is None:
            raise ValueError("mode='push' requires push_x and push_y")
        push_vec = (float(push_x), float(push_y))
    out = warp.liquify(img, mode, points, radius=radius, strength=strength,
                       push_vector=push_vec)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Liquify ({mode})")
    return _summary(cid)


@mcp.tool()
def distort(canvas_id: str, mode: str, *,
            amount: float | None = None,
            angle_deg: float | None = None,
            radius: float | None = None,
            amplitude: float | None = None,
            amplitude_x: float | None = None,
            amplitude_y: float | None = None,
            period: float | None = None,
            period_x: float | None = None,
            period_y: float | None = None,
            center_x: float | None = None,
            center_y: float | None = None,
            new_canvas_id: str | None = None) -> dict:
    """Procedural distortion filters (Photoshop's Filter > Distort family).

    ``mode`` selects the effect:
      - ``spherize`` — ``amount`` ∈ -1..1; positive bulges centre out,
        negative pinches in.
      - ``pinch`` — inverse-sign alias of spherize.
      - ``twirl`` — ``angle_deg`` (degrees), optional ``radius`` (default =
        half the shorter edge).
      - ``wave`` — sinusoidal displacement; ``amplitude_x`` / ``amplitude_y``
        (px) and ``period_x`` / ``period_y`` (px) control horizontal +
        vertical components independently.
      - ``ripple`` — radial sinusoid (pond ripple); ``amplitude`` (px) and
        ``period`` (px).
      - ``polar_to_rect`` / ``rect_to_polar`` — coordinate remap.

    All filters preserve image size; off-image samples replicate the edge."""
    img = store.compose(canvas_id)
    # Build a kwargs dict of only the params the user supplied so the
    # warp.distort dispatcher's defaults kick in.
    raw = {
        "amount": amount, "angle_deg": angle_deg, "radius": radius,
        "amplitude": amplitude,
        "amplitude_x": amplitude_x, "amplitude_y": amplitude_y,
        "period": period,
        "period_x": period_x, "period_y": period_y,
        "center_x": center_x, "center_y": center_y,
    }
    kwargs = {k: v for k, v in raw.items() if v is not None}
    out = warp.distort(img, mode, **kwargs)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Distort ({mode})")
    return _summary(cid)


@mcp.tool()
def displace_by_map(canvas_id: str, displacement_canvas_id: str, *,
                    scale_x: float = 10.0,
                    scale_y: float | None = None,
                    channel_mode: str = "luminance",
                    new_canvas_id: str | None = None) -> dict:
    """Displace ``canvas_id`` pixels using a displacement-map canvas
    (Photoshop's Filter > Distort > Displace).

    - ``channel_mode='luminance'`` (default): single grayscale channel
      drives both x and y. Most flexible — any greyscale image works.
    - ``channel_mode='rg'``: red channel → x displacement, green → y. Use
      this for purpose-built per-axis maps.

    Midpoint (128) = no displacement; ≷ 128 pushes ±. ``scale_x`` /
    ``scale_y`` are in pixels (PS default 10). ``scale_y`` defaults to
    ``scale_x``.

    The displacement map is resized to the source if its dimensions
    differ."""
    img = store.compose(canvas_id)
    disp = store.compose(displacement_canvas_id)
    out = warp.displace_by_map(img, disp,
                               scale_x=scale_x, scale_y=scale_y,
                               channel_mode=channel_mode)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name="Displace by map")
    return _summary(cid)


# ============================================================ Patterns (PS-style)

@mcp.tool()
def define_pattern(name: str, canvas_id: str, *,
                   x1: int | None = None, y1: int | None = None,
                   x2: int | None = None, y2: int | None = None) -> dict:
    """Store a region of a canvas as a named pattern in the in-process
    library. Subsequent ``fill_pattern`` / ``pattern_stamp`` /
    ``pattern_overlay`` calls reference it by ``name``.

    Omit ``x1/y1/x2/y2`` to use the entire canvas. Coordinates are
    clamped to the canvas; an empty bbox is rejected."""
    img = store.compose(canvas_id)
    if any(v is not None for v in (x1, y1, x2, y2)):
        w, h = img.size
        cx1 = max(0, int(x1 or 0))
        cy1 = max(0, int(y1 or 0))
        cx2 = min(w, int(x2 if x2 is not None else w))
        cy2 = min(h, int(y2 if y2 is not None else h))
        if cx2 <= cx1 or cy2 <= cy1:
            raise ValueError(f"empty bbox: ({x1}, {y1}, {x2}, {y2})")
        img = img.crop((cx1, cy1, cx2, cy2))
    return patterns.define_pattern(name, img)


@mcp.tool()
def define_pattern_from_file(name: str, path: str) -> dict:
    """Load a pattern tile from disk (any format Pillow / pillow-heif /
    cairosvg supports) and store it under ``name``."""
    img = io_formats.load_image(path)
    return patterns.define_pattern(name, img)


@mcp.tool()
def list_patterns() -> dict:
    """List the in-process pattern library — names + tile sizes."""
    return patterns.list_patterns()


@mcp.tool()
def delete_pattern(name: str) -> dict:
    """Remove a pattern from the library."""
    return patterns.delete_pattern(name)


@mcp.tool()
def fill_pattern(canvas_id: str, name: str, *,
                 x1: int | None = None, y1: int | None = None,
                 x2: int | None = None, y2: int | None = None,
                 scale: float = 1.0, rotation: float = 0.0,
                 offset_x: int = 0, offset_y: int = 0,
                 opacity: float = 1.0, blend_mode: str = "normal",
                 new_canvas_id: str | None = None) -> dict:
    """Tile a pattern across ``canvas_id`` (or the bbox region) with
    scale / rotation / opacity / blend-mode controls. Mirrors PS
    Edit > Fill > Pattern. Result is a NEW canvas; source untouched.

    ``blend_mode`` ∈ ``normal | multiply | screen | overlay``. ``scale``
    resamples the tile before placement (0.5 = halve tile, 2.0 = double).
    ``rotation`` rotates the tile (degrees CCW). ``offset_x/y`` shifts
    the tile origin to re-align seams."""
    img = store.compose(canvas_id)
    pat = patterns.get_pattern(name)
    bbox = None
    if any(v is not None for v in (x1, y1, x2, y2)):
        w, h = img.size
        bbox = (max(0, int(x1 or 0)),
                max(0, int(y1 or 0)),
                min(w, int(x2 if x2 is not None else w)),
                min(h, int(y2 if y2 is not None else h)))
    out = patterns.fill_with_pattern(
        img, pat, bbox=bbox,
        scale=scale, rotation=rotation,
        offset_x=offset_x, offset_y=offset_y,
        opacity=opacity, blend_mode=blend_mode,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Pattern fill ({name})")
    return _summary(cid)


@mcp.tool()
def pattern_stamp(canvas_id: str, name: str, points: list[list[int]], *,
                  brush_size: int = 64, hardness: float = 0.5,
                  scale: float = 1.0, rotation: float = 0.0,
                  opacity: float = 1.0,
                  new_canvas_id: str | None = None) -> dict:
    """Paint with a pattern as a soft brush along a polyline. Each point
    deposits a circular sample of the canvas-aligned tile — so overlapping
    strokes line up (matches PS Pattern Stamp tool).

    - ``points``: list of ``[x, y]`` pairs
    - ``brush_size``: diameter in pixels
    - ``hardness`` 0-1: 0 = soft gradient edge, 1 = hard edge
    - ``scale`` / ``rotation`` modify the underlying tile
    - ``opacity`` 0-1 modulates each stamp's contribution"""
    img = store.compose(canvas_id)
    pat = patterns.get_pattern(name)
    pts = [(int(p[0]), int(p[1])) for p in points]
    out = patterns.stamp_pattern(
        img, pat, pts,
        brush_size=brush_size, hardness=hardness,
        scale=scale, rotation=rotation, opacity=opacity,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name=f"Pattern stamp ({name})")
    return _summary(cid)


@mcp.tool()
def pattern_overlay(canvas_id: str, name: str, *,
                    scale: float = 1.0, rotation: float = 0.0,
                    offset_x: int = 0, offset_y: int = 0,
                    opacity: float = 1.0, blend_mode: str = "normal",
                    layer_name: str | None = None) -> dict:
    """Add a tiled-pattern layer ON TOP of the canvas (in-place), matching
    Photoshop's Pattern Overlay layer style. Returns the canvas summary.

    The overlay lands as a NEW layer above the active one, with its own
    blend mode + opacity — so you can re-tweak without re-tiling. To bake
    the result, ``flatten_canvas`` or ``merge_down``. To get a flattened
    output without touching the source, use ``fill_pattern`` instead.

    ``blend_mode`` is the layer-stack mode (``normal | multiply | screen |
    overlay | darken | lighten | color_dodge | color_burn | hard_light |
    soft_light | difference | exclusion | add | subtract``)."""
    if blend_mode not in layers_mod.BLEND_MODES:
        raise ValueError(
            f"unknown blend_mode {blend_mode!r}. "
            f"Choose from {layers_mod.BLEND_MODES}."
        )
    img = store.compose(canvas_id)
    pat = patterns.get_pattern(name)
    overlay = patterns.make_pattern_overlay_layer(
        pat, img.size,
        scale=scale, rotation=rotation,
        offset_x=offset_x, offset_y=offset_y,
        opacity=1.0,  # ``opacity`` is set on the layer itself below
    )
    # Add a new layer above the active one, paint the tiled overlay into it,
    # then dial in blend mode + opacity per the user's choice.
    idx = store.add_layer(
        canvas_id, name=layer_name or f"Pattern overlay ({name})",
    )
    store.replace_active_image(canvas_id, overlay)
    layer = store.entry(canvas_id).layers[idx]
    layer.blend_mode = blend_mode
    layer.opacity = float(opacity)
    return _summary(canvas_id)


@mcp.tool()
def make_seamless(canvas_id: str, *, blend_width: int | None = None,
                  new_canvas_id: str | None = None) -> dict:
    """Turn an image into a seamless tile via wrap-offset + edge cross-fade.
    Useful for prepping textures captured from photos before
    ``define_pattern``. Result is a NEW canvas the same size.

    ``blend_width`` is the width of the cross-fade strip (defaults to ~1/16
    of the shorter edge, clamped to 4-64 px)."""
    img = store.compose(canvas_id)
    out = patterns.make_seamless(img, blend_width=blend_width)
    cid = store.put_image(out, canvas_id=new_canvas_id,
                          layer_name="Seamless tile")
    return _summary(cid)


# ============================================================ Face swap + restore

@mcp.tool()
def face_status() -> dict:
    """Report what face-swap models are loaded, the ONNX providers in use,
    and how long they've been idle."""
    return face_swap.status()


@mcp.tool()
def face_load(*, detector_det_size: int = 640,
              swapper_path: str | None = None,
              restorer_model: str | None = "GFPGANv1.4.pth") -> dict:
    """Pre-warm the face-swap stack: InsightFace ``buffalo_l`` detector +
    ``inswapper_128.onnx`` swapper + GFPGAN restorer.

    ``detector_det_size`` is the longer edge of the detection input (640 is
    the default, 320 trades recall for speed on small faces).

    ``swapper_path`` may point to a local ``inswapper_128.onnx``; defaults
    to ``B:\\-AI-Stuff-\\ComfyUI\\models\\insightface\\inswapper_128.onnx``
    if present, else auto-download via insightface.

    ``restorer_model`` is ``GFPGANv1.4.pth`` (default) or ``GFPGANv1.3.pth``;
    pass ``null`` to skip GFPGAN (saves ~340 MB)."""
    return face_swap.load(
        detector_det_size=detector_det_size,
        swapper_path=swapper_path,
        restorer_model=restorer_model,
    )


@mcp.tool()
def face_unload() -> dict:
    """Drop the face-swap stack (detector + swapper + restorer). Frees ~1.2 GB."""
    return face_swap.unload()


@mcp.tool()
def face_set_idle_timeout(seconds: float) -> dict:
    """Auto-unload the face models after this many seconds of inactivity.
    Pass 0 to disable auto-eviction (manual ``face_unload`` only)."""
    return face_swap.set_idle_timeout(seconds)


@mcp.tool()
def face_detect(canvas_id: str, *, det_size: int = 640) -> dict:
    """Detect faces in the composited canvas. Returns a list sorted by bbox
    area (largest first) — that's the same index space the swap and restore
    tools use.

    Each face has: ``bbox`` ([x1, y1, x2, y2]), ``area``, ``score``, ``age``,
    ``sex``, ``kps`` (5-point landmarks)."""
    img = store.compose(canvas_id)
    faces = face_swap.detect_faces(img, det_size=det_size)
    return {"canvas_id": canvas_id, "n_faces": len(faces), "faces": faces}


@mcp.tool()
def face_transfer(source_canvas_id: str, target_canvas_id: str, *,
                  source_face_index: int = 0,
                  target_face_indices: list[int] | None = None,
                  restore: bool = False,
                  restorer_model: str = "GFPGANv1.4.pth",
                  restore_weight: float = 0.5,
                  swapper_path: str | None = None,
                  new_canvas_id: str | None = None) -> dict:
    """Swap face(s) from ``source_canvas_id`` onto ``target_canvas_id``.
    Result lands on a NEW canvas; originals untouched.

    - ``source_face_index`` picks which face from the source (sorted by
      bbox area, 0 = largest).
    - ``target_face_indices`` lists which target faces to overwrite. Omit
      or pass ``null`` to swap ALL detected target faces. ``[0]`` =
      largest only. ``[0, 1]`` = the two largest, etc.
    - ``restore=true`` runs GFPGAN over the result to clean up swap
      artefacts. ``restore_weight`` ∈ 0-1 blends restoration intensity."""
    src = store.compose(source_canvas_id)
    tgt = store.compose(target_canvas_id)
    out = face_swap.swap_face(
        src, tgt,
        source_face_index=source_face_index,
        target_face_indices=target_face_indices,
        swapper_path=swapper_path,
        restore=restore,
        restorer_model=restorer_model,
        restore_weight=restore_weight,
    )
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Face swap")
    return _summary(cid)


@mcp.tool()
def face_restore(canvas_id: str, *,
                 restorer_model: str = "GFPGANv1.4.pth",
                 weight: float = 0.5,
                 new_canvas_id: str | None = None) -> dict:
    """Run GFPGAN face restoration on every face in ``canvas_id``. Useful
    on small / blurry faces, on swap output, or as a generic face-cleanup
    pass. Result is a NEW canvas.

    ``weight`` ∈ 0-1 blends restored vs original (1.0 = full restoration,
    0.5 = balanced, 0.2 = subtle clean-up). ``restorer_model`` selects
    between ``GFPGANv1.4.pth`` (default, best quality) and
    ``GFPGANv1.3.pth`` (alternative)."""
    src = store.compose(canvas_id)
    out = face_swap.restore_faces(src, restorer_model=restorer_model,
                                  weight=weight)
    cid = store.put_image(out, canvas_id=new_canvas_id, layer_name="Face restore")
    return _summary(cid)


# ============================================================ entry

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
