"""Geometry transforms, filters, and color adjustments.

These ops can change the image's dimensions or mode, so they return a new
``Image`` rather than mutating in place. The MCP server layer calls
``store.replace(canvas_id, new_img)`` to swap the live image atomically.
"""
from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .colors import parse_color


# ---- geometry --------------------------------------------------------------

def crop(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    """Crop ``img`` to (x1,y1)-(x2,y2). Coordinates are normalized to a
    canonical (left,top,right,bottom) rectangle; a zero-area request raises.

    Out-of-bounds coordinates are allowed — Pillow's ``crop`` returns a
    region padded with background pixels (same behaviour as ``Image.crop``
    with negative or oversized boxes)."""
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    if x1 == x2 or y1 == y2:
        raise ValueError("crop: zero-area rectangle")
    return img.crop((x1, y1, x2, y2))


_RESAMPLE = {
    "nearest": Image.NEAREST,
    "box": Image.BOX,
    "bilinear": Image.BILINEAR,
    "hamming": Image.HAMMING,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def resize(img: Image.Image, width: int, height: int, *,
           resample: str = "lanczos") -> Image.Image:
    try:
        rs = _RESAMPLE[resample.lower()]
    except KeyError:
        raise ValueError(
            f"unknown resample {resample!r}. Choose one of {sorted(_RESAMPLE)}."
        )
    return img.resize((int(width), int(height)), rs)


def rotate(img: Image.Image, angle: float, *, expand: bool = True,
           background="transparent") -> Image.Image:
    """Counter-clockwise rotation in degrees. ``expand=True`` grows the canvas
    so corners aren't clipped; ``False`` keeps original dims."""
    bg = parse_color(background) if background is not None else None
    # ``fillcolor`` requires the image to support the color's mode. Convert
    # to RGBA if we're filling with anything translucent.
    base = img
    if bg is not None and bg[3] < 255 and base.mode != "RGBA":
        base = base.convert("RGBA")
    return base.rotate(float(angle), expand=bool(expand),
                       resample=Image.BICUBIC, fillcolor=bg)


def flip(img: Image.Image, axis: str) -> Image.Image:
    """Mirror the image. ``axis`` accepts ``horizontal``/``h``/``x`` for a
    left-right flip, ``vertical``/``v``/``y`` for a top-bottom flip."""
    a = axis.lower()
    if a in ("h", "horizontal", "x"):
        return ImageOps.mirror(img)
    if a in ("v", "vertical", "y"):
        return ImageOps.flip(img)
    raise ValueError(
        f"flip axis must be one of horizontal/h/x or vertical/v/y, got {axis!r}"
    )


def paste_region(dst: Image.Image, src: Image.Image, x: int, y: int) -> Image.Image:
    """Paste ``src`` onto a copy of ``dst`` at (x, y). Uses ``src``'s alpha
    channel as a mask if present."""
    out = dst.copy()
    if out.mode != "RGBA" and src.mode in ("RGBA", "LA"):
        out = out.convert("RGBA")
    mask = src if src.mode in ("RGBA", "LA") else None
    out.paste(src, (int(x), int(y)), mask=mask)
    return out


def copy_region(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    """Return a new image containing the rectangle (non-destructive)."""
    return crop(img, x1, y1, x2, y2).copy()


def clear_region(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
                 color="transparent") -> Image.Image:
    """Fill a rectangle with ``color``. For RGBA + transparent, this punches a
    hole; for RGB, paints with the color (falls back to white if transparent
    requested on RGB)."""
    c = parse_color(color)
    out = img.copy()
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    if out.mode != "RGBA" and c[3] < 255:
        out = out.convert("RGBA")
    region = Image.new("RGBA", (x2 - x1, y2 - y1), c)
    out.paste(region, (x1, y1), mask=region if c[3] < 255 else None)
    return out


# ---- filters ---------------------------------------------------------------

_FILTERS = {
    "blur": ImageFilter.BLUR,
    "sharpen": ImageFilter.SHARPEN,
    "smooth": ImageFilter.SMOOTH,
    "smooth_more": ImageFilter.SMOOTH_MORE,
    "edge_enhance": ImageFilter.EDGE_ENHANCE,
    "edge_enhance_more": ImageFilter.EDGE_ENHANCE_MORE,
    "find_edges": ImageFilter.FIND_EDGES,
    "contour": ImageFilter.CONTOUR,
    "emboss": ImageFilter.EMBOSS,
    "detail": ImageFilter.DETAIL,
}


def apply_filter(img: Image.Image, filter_name: str, *,
                 radius: float | None = None) -> Image.Image:
    name = filter_name.lower()
    if name == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius or 2.0))
    if name == "box_blur":
        return img.filter(ImageFilter.BoxBlur(radius or 2.0))
    if name == "unsharp_mask":
        return img.filter(ImageFilter.UnsharpMask(radius=radius or 2.0))
    if name in _FILTERS:
        return img.filter(_FILTERS[name])
    raise ValueError(
        f"unknown filter {filter_name!r}. Available: "
        f"{sorted(list(_FILTERS) + ['gaussian_blur', 'box_blur', 'unsharp_mask'])}"
    )


# ---- adjustments -----------------------------------------------------------

def adjust(img: Image.Image, *, brightness: float = 1.0, contrast: float = 1.0,
           color: float = 1.0, sharpness: float = 1.0) -> Image.Image:
    """Multiplicative adjustments. ``1.0`` is identity; ``0.0`` is fully off."""
    out = img
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(float(brightness))
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(float(contrast))
    if color != 1.0:
        out = ImageEnhance.Color(out).enhance(float(color))
    if sharpness != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(float(sharpness))
    return out


def invert(img: Image.Image) -> Image.Image:
    """Color-invert. Preserves alpha on RGBA inputs. Palette-mode (``P``)
    inputs are first promoted to ``RGB`` because Pillow's ``invert`` doesn't
    operate on indexed images directly — the result is RGB, not P."""
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        rgb = ImageOps.invert(rgb)
        return Image.merge("RGBA", (*rgb.split(), a))
    if img.mode == "P":
        return ImageOps.invert(img.convert("RGB"))
    return ImageOps.invert(img)


def grayscale(img: Image.Image) -> Image.Image:
    """Desaturate while preserving alpha. RGBA → RGBA grayscale; LA → LA."""
    if img.mode == "RGBA":
        gray = ImageOps.grayscale(img.convert("RGB")).convert("RGB")
        gray.putalpha(img.split()[-1])
        return gray.convert("RGBA")
    if img.mode == "LA":
        alpha = img.split()[-1]
        gray = img.convert("L")
        out = Image.merge("LA", (gray, alpha))
        return out
    return ImageOps.grayscale(img)


def posterize(img: Image.Image, bits: int) -> Image.Image:
    bits = max(1, min(8, int(bits)))
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        return Image.merge("RGBA", (*ImageOps.posterize(rgb, bits).split(), a))
    if img.mode != "RGB" and img.mode != "L":
        img = img.convert("RGB")
    return ImageOps.posterize(img, bits)


def add_border(img: Image.Image, width: int, color="black") -> Image.Image:
    c = parse_color(color)
    return ImageOps.expand(img.convert("RGBA") if c[3] < 255 else img,
                           border=int(width), fill=c)


def auto_crop_to_content(img: Image.Image, *,
                         alpha_threshold: int = 1,
                         bg_color: tuple | None = None,
                         tolerance: int = 5,
                         padding: int = 0) -> Image.Image:
    """Trim transparent / uniform-background borders.

    Two modes:
    - **RGBA**: trim pixels with alpha < ``alpha_threshold`` (default 1, so
      fully transparent edges are removed).
    - **RGB or palette**: trim pixels within ``tolerance`` of ``bg_color``
      (default: sample the four corners and use their median).

    ``padding`` pixels of margin are kept around the detected content.
    Returns the cropped image (no copy if nothing to trim)."""
    import numpy as _np
    if img.mode == "RGBA":
        a = _np.asarray(img.split()[-1])
        mask = a >= int(alpha_threshold)
    else:
        rgb = img.convert("RGB")
        arr = _np.asarray(rgb)
        if bg_color is None:
            # Median of the 4 corner pixels as the background reference.
            corners = _np.stack([
                arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1],
            ])
            bg = _np.median(corners, axis=0)
        else:
            bg = _np.asarray(bg_color[:3], dtype=arr.dtype)
        diff = _np.max(_np.abs(arr.astype(int) - bg.astype(int)), axis=-1)
        mask = diff > int(tolerance)
    rows = _np.any(mask, axis=1)
    cols = _np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img.copy()  # entire image is "background"
    y1, y2 = _np.argmax(rows), len(rows) - _np.argmax(rows[::-1])
    x1, x2 = _np.argmax(cols), len(cols) - _np.argmax(cols[::-1])
    pad = max(0, int(padding))
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(img.width, int(x2) + pad)
    y2 = min(img.height, int(y2) + pad)
    return img.crop((x1, y1, x2, y2))


def smart_crop_to_aspect(img: Image.Image, *,
                         aspect: float,
                         anchor: str = "center") -> Image.Image:
    """Crop to the largest rectangle of ``aspect = width / height`` that
    fits inside ``img``. ``anchor`` ∈ ``center | top | bottom | left |
    right | top_left | top_right | bottom_left | bottom_right`` controls
    which side / corner is preserved."""
    if aspect <= 0:
        raise ValueError(f"aspect must be > 0, got {aspect}")
    w, h = img.size
    target_w = w
    target_h = int(round(w / aspect))
    if target_h > h:
        target_h = h
        target_w = int(round(h * aspect))
    # Anchor placement
    anchors = {
        "center":       lambda: ((w - target_w) // 2, (h - target_h) // 2),
        "top":          lambda: ((w - target_w) // 2, 0),
        "bottom":       lambda: ((w - target_w) // 2, h - target_h),
        "left":         lambda: (0, (h - target_h) // 2),
        "right":        lambda: (w - target_w, (h - target_h) // 2),
        "top_left":     lambda: (0, 0),
        "top_right":    lambda: (w - target_w, 0),
        "bottom_left":  lambda: (0, h - target_h),
        "bottom_right": lambda: (w - target_w, h - target_h),
    }
    if anchor not in anchors:
        raise ValueError(
            f"unknown anchor {anchor!r}. Choose from {list(anchors)}"
        )
    x, y = anchors[anchor]()
    return img.crop((x, y, x + target_w, y + target_h))


def pixelate(img: Image.Image, *, block_size: int = 16,
             bbox: tuple[int, int, int, int] | None = None,
             mask: Image.Image | None = None) -> Image.Image:
    """Mosaic / pixelate. ``block_size`` is the size of each pixel-block in
    image coords. Optionally restrict to ``bbox = (x1, y1, x2, y2)``, or
    use ``mask`` (L-mode) for arbitrary regions (anything > 0 gets
    pixelated)."""
    if block_size < 2:
        raise ValueError("block_size must be >= 2")
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        region = rgba.crop((x1, y1, x2, y2))
    else:
        region = rgba
    rw, rh = region.size
    bw = max(1, rw // int(block_size))
    bh = max(1, rh // int(block_size))
    small = region.resize((bw, bh), Image.BILINEAR)
    pix = small.resize((rw, rh), Image.NEAREST)
    out = rgba.copy()
    if bbox is not None:
        out.paste(pix, (bbox[0], bbox[1]))
    elif mask is not None:
        m = mask.convert("L")
        if m.size != (w, h):
            m = m.resize((w, h), Image.BILINEAR)
        out = Image.composite(pix, rgba, m)
    else:
        out = pix
    return out
