"""MS Paint-style drawing primitives operating on a Pillow Image in place.

Every public function in this module takes a ``PIL.Image.Image`` as the first
argument and mutates it. The MCP tool layer (server.image_tools_server) is
responsible for calling ``store.snapshot(canvas_id)`` before invoking these,
so undo/redo Just Works.

Drawing on a canvas with no alpha channel: we draw directly. For canvases
with alpha, we use a transparent overlay + ``Image.alpha_composite`` so
semi-transparent strokes blend correctly rather than overwriting alpha.
"""
from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .colors import Color, parse_color


def _ensure_drawable(img: Image.Image) -> Image.Image:
    """Most ops need RGB or RGBA. Convert palette/grayscale up front so
    ``ImageDraw`` can handle RGBA colors uniformly."""
    if img.mode not in ("RGB", "RGBA", "L", "LA"):
        return img.convert("RGBA")
    return img


def _draw_with_alpha(img: Image.Image, do_draw) -> Image.Image:
    """Run ``do_draw(ImageDraw.Draw(overlay))`` on a transparent overlay,
    composite onto an RGBA copy of ``img``, then paste the result back into
    ``img`` so callers' references stay valid.

    Translucent strokes on an RGB image composite against the existing RGB
    pixels (alpha gets baked in) — same as MS Paint.
    """
    base = _ensure_drawable(img)
    if base.mode != "RGBA":
        base = base.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    do_draw(ImageDraw.Draw(overlay))
    composited = Image.alpha_composite(base, overlay)
    if composited.mode != img.mode:
        composited = composited.convert(img.mode)
    img.paste(composited)
    return img


# ---- pixel-level -----------------------------------------------------------

def draw_pixel(img: Image.Image, x: int, y: int, color) -> Image.Image:
    c = parse_color(color)

    def _do(d: ImageDraw.ImageDraw) -> None:
        d.point([(x, y)], fill=c)

    return _draw_with_alpha(img, _do)


def draw_line(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
              color, width: int = 1) -> Image.Image:
    c = parse_color(color)

    def _do(d: ImageDraw.ImageDraw) -> None:
        # Pillow rounds end caps oddly for width>1 lines; ``joint="curve"`` is
        # not exposed for ``line`` but the default is fine for single segments.
        d.line([(x1, y1), (x2, y2)], fill=c, width=max(1, int(width)))

    return _draw_with_alpha(img, _do)


def draw_rectangle(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
                   outline=None, fill=None, width: int = 1) -> Image.Image:
    if outline is None and fill is None:
        raise ValueError("draw_rectangle: provide outline, fill, or both")
    o = parse_color(outline) if outline is not None else None
    f = parse_color(fill) if fill is not None else None

    def _do(d: ImageDraw.ImageDraw) -> None:
        d.rectangle([(x1, y1), (x2, y2)], outline=o, fill=f, width=max(1, int(width)))

    return _draw_with_alpha(img, _do)


def draw_ellipse(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
                 outline=None, fill=None, width: int = 1) -> Image.Image:
    if outline is None and fill is None:
        raise ValueError("draw_ellipse: provide outline, fill, or both")
    o = parse_color(outline) if outline is not None else None
    f = parse_color(fill) if fill is not None else None

    def _do(d: ImageDraw.ImageDraw) -> None:
        d.ellipse([(x1, y1), (x2, y2)], outline=o, fill=f, width=max(1, int(width)))

    return _draw_with_alpha(img, _do)


def draw_polygon(img: Image.Image, points: Sequence[Sequence[int]],
                 outline=None, fill=None) -> Image.Image:
    if len(points) < 3:
        raise ValueError("draw_polygon: need at least 3 points")
    if outline is None and fill is None:
        raise ValueError("draw_polygon: provide outline, fill, or both")
    o = parse_color(outline) if outline is not None else None
    f = parse_color(fill) if fill is not None else None
    pts = [(int(p[0]), int(p[1])) for p in points]

    def _do(d: ImageDraw.ImageDraw) -> None:
        d.polygon(pts, outline=o, fill=f)

    return _draw_with_alpha(img, _do)


def draw_arc(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
             start: float, end: float, color, width: int = 1) -> Image.Image:
    c = parse_color(color)

    def _do(d: ImageDraw.ImageDraw) -> None:
        d.arc([(x1, y1), (x2, y2)], start=start, end=end, fill=c,
              width=max(1, int(width)))

    return _draw_with_alpha(img, _do)


# ---- text ------------------------------------------------------------------

def _resolve_font(font_name: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font by name or path; fall back to Pillow's
    bitmap default so text never silently fails."""
    if not font_name:
        # Pillow ships a built-in tiny bitmap font; better than nothing.
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()
    try:
        return ImageFont.truetype(font_name, size)
    except OSError as e:
        raise ValueError(
            f"font {font_name!r} not found. Pass an absolute .ttf path, a "
            f"font filename installed on the system, or omit to use the default."
        ) from e


def draw_text(img: Image.Image, x: int, y: int, text: str, *,
              font: str | None = None, size: int = 16,
              color="black", anchor: str = "la",
              stroke_width: int = 0, stroke_color=None) -> Image.Image:
    c = parse_color(color)
    sc = parse_color(stroke_color) if stroke_color is not None else None
    ft = _resolve_font(font, size)

    def _do(d: ImageDraw.ImageDraw) -> None:
        # ImageFont.load_default() doesn't accept anchor / stroke; degrade.
        kwargs: dict = {"fill": c}
        if isinstance(ft, ImageFont.FreeTypeFont):
            kwargs["font"] = ft
            kwargs["anchor"] = anchor
            if stroke_width > 0 and sc is not None:
                kwargs["stroke_width"] = int(stroke_width)
                kwargs["stroke_fill"] = sc
        else:
            kwargs["font"] = ft
        d.text((x, y), text, **kwargs)

    return _draw_with_alpha(img, _do)


# ---- brush / freehand ------------------------------------------------------

def draw_brush(img: Image.Image, points: Sequence[Sequence[int]],
               color, size: int = 4) -> Image.Image:
    """Smooth stroke through ``points`` using a round brush of ``size`` px.

    Implementation: draw a thick polyline plus filled circles at each vertex
    so corners don't pinch (Pillow's line caps are square).
    """
    if len(points) < 1:
        raise ValueError("draw_brush: need at least one point")
    c = parse_color(color)
    pts = [(int(p[0]), int(p[1])) for p in points]
    r = max(1, int(size)) // 2

    def _do(d: ImageDraw.ImageDraw) -> None:
        if len(pts) >= 2:
            d.line(pts, fill=c, width=max(1, int(size)))
        for x, y in pts:
            d.ellipse([(x - r, y - r), (x + r, y + r)], fill=c)

    return _draw_with_alpha(img, _do)


def eraser(img: Image.Image, points: Sequence[Sequence[int]],
           size: int = 8, background=None) -> Image.Image:
    """Erase along a stroke. On RGBA canvas erases to transparent; on RGB
    canvas paints with ``background`` (default white)."""
    if len(points) < 1:
        raise ValueError("eraser: need at least one point")
    base = _ensure_drawable(img)
    pts = [(int(p[0]), int(p[1])) for p in points]
    r = max(1, int(size)) // 2

    if base.mode == "RGBA":
        # Punch alpha to 0 — draw on a mask, then zero the source pixels.
        mask = Image.new("L", base.size, 0)
        md = ImageDraw.Draw(mask)
        if len(pts) >= 2:
            md.line(pts, fill=255, width=max(1, int(size)))
        for x, y in pts:
            md.ellipse([(x - r, y - r), (x + r, y + r)], fill=255)
        cleared = Image.new("RGBA", base.size, (0, 0, 0, 0))
        result = Image.composite(cleared, base, mask)
        if result.mode != img.mode:
            result = result.convert(img.mode)
        img.paste(result)
        return img

    bg = parse_color(background) if background is not None else (255, 255, 255, 255)

    def _do(d: ImageDraw.ImageDraw) -> None:
        if len(pts) >= 2:
            d.line(pts, fill=bg, width=max(1, int(size)))
        for x, y in pts:
            d.ellipse([(x - r, y - r), (x + r, y + r)], fill=bg)

    return _draw_with_alpha(img, _do)


# ---- flood fill / eyedropper -----------------------------------------------

def flood_fill(img: Image.Image, x: int, y: int, color,
               tolerance: int = 0) -> Image.Image:
    """Paint-bucket starting at (x, y). ``tolerance`` is per-channel."""
    c = parse_color(color)
    base = _ensure_drawable(img).convert("RGBA")
    ImageDraw.floodfill(base, (int(x), int(y)), c, thresh=max(0, int(tolerance)))
    if base.mode != img.mode:
        base = base.convert(img.mode)
    img.paste(base)
    return img


def pick_color(img: Image.Image, x: int, y: int) -> Color:
    """Eyedropper: return the pixel at (x, y) as (r, g, b, a)."""
    if not (0 <= x < img.width and 0 <= y < img.height):
        raise ValueError(f"({x}, {y}) is outside canvas {img.width}x{img.height}")
    px = img.convert("RGBA").getpixel((int(x), int(y)))
    return (int(px[0]), int(px[1]), int(px[2]), int(px[3]))
