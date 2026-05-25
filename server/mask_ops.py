"""Mask post-processing — feathering, expand/contract, refine.

Feathering softens binary mask edges so the resulting cutout blends naturally
when composited. The four modes here cover the practical workflows:

- ``gaussian``: symmetric blur — the classic "Photoshop feather" (edge spreads
  both ways from the original boundary).
- ``inside``:   blur only the inner edge — keeps the mask's outer silhouette
  intact, softens just the inside transition.
- ``outside``:  blur only the outer edge — extends the mask outward with a
  soft falloff; useful for grow-then-blur outlines.
- ``matte``:    use a guided filter against the source image for edge-aware
  refinement — produces hair-and-fur-friendly mattes from a coarse binary
  mask. Falls back to ``gaussian`` if opencv-contrib isn't available.

Expand/contract are pixel-wise morphological ops (dilate/erode) for nudging
the mask boundary before or after feathering. ``refine_mask`` is a one-shot
"clean up a binary mask" pipeline: small-hole fill + small-island remove +
optional feather.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _ensure_L(mask: Image.Image) -> Image.Image:
    """Coerce any input (binary, RGBA alpha, grayscale) to an L-mode mask."""
    if mask.mode == "L":
        return mask
    if mask.mode == "1":
        return mask.convert("L")
    if mask.mode in ("RGBA", "LA"):
        return mask.split()[-1]  # alpha channel
    return mask.convert("L")


def feather_mask(mask: Image.Image, *, radius: float = 4.0,
                 method: str = "gaussian",
                 source_image: Image.Image | None = None) -> Image.Image:
    """Soften mask edges.

    - ``radius``: blur radius in pixels. ``0`` returns the mask unchanged.
    - ``method``: ``gaussian`` | ``inside`` | ``outside`` | ``matte``.
    - ``source_image``: required for ``matte`` mode (guided-filter reference);
      ignored otherwise.
    """
    if radius <= 0:
        return _ensure_L(mask).copy()

    m = _ensure_L(mask)
    if method == "gaussian":
        return m.filter(ImageFilter.GaussianBlur(float(radius)))

    if method == "inside":
        # Blur the mask, then min-combine with original so the outer boundary
        # never grows: the result is original where mask was 0, and a
        # blurred-but-clipped version where mask was solid.
        blurred = m.filter(ImageFilter.GaussianBlur(float(radius)))
        arr_orig = np.asarray(m, dtype=np.uint8)
        arr_blur = np.asarray(blurred, dtype=np.uint8)
        result = np.minimum(arr_orig, arr_blur)
        return Image.fromarray(result, mode="L")

    if method == "outside":
        # Blur, then max-combine: original solid pixels stay solid, the
        # exterior gets a soft halo.
        blurred = m.filter(ImageFilter.GaussianBlur(float(radius)))
        arr_orig = np.asarray(m, dtype=np.uint8)
        arr_blur = np.asarray(blurred, dtype=np.uint8)
        result = np.maximum(arr_orig, arr_blur)
        return Image.fromarray(result, mode="L")

    if method == "matte":
        if source_image is None:
            # Without a guide image, fall back to gaussian rather than crash —
            # callers may be using a default and not realize matte needs one.
            return m.filter(ImageFilter.GaussianBlur(float(radius)))
        return _guided_filter_matte(m, source_image, radius=radius)

    raise ValueError(
        f"unknown method {method!r}; choose gaussian | inside | outside | matte"
    )


def _guided_filter_matte(mask: Image.Image, source: Image.Image, *,
                         radius: float) -> Image.Image:
    """Edge-aware refine via OpenCV's guided filter. Requires the
    ``opencv-contrib-python`` package; otherwise we degrade to gaussian."""
    try:
        import cv2
        if not hasattr(cv2, "ximgproc"):
            raise ImportError("cv2.ximgproc missing — install opencv-contrib-python")
    except Exception:
        return mask.filter(ImageFilter.GaussianBlur(float(radius)))

    src_rgb = np.asarray(source.convert("RGB"))
    if src_rgb.shape[:2] != (mask.height, mask.width):
        guide = np.asarray(source.convert("RGB").resize(mask.size,
                                                          Image.BILINEAR))
    else:
        guide = src_rgb
    mask_f = np.asarray(mask, dtype=np.float32) / 255.0
    # Radius in cv2 guided filter is an integer pixel radius; eps controls
    # smoothness vs edge-preservation tradeoff.
    refined = cv2.ximgproc.guidedFilter(
        guide=guide, src=mask_f,
        radius=max(1, int(radius * 2)),
        eps=1e-3,
    )
    refined = np.clip(refined * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(refined, mode="L")


def expand_mask(mask: Image.Image, *, pixels: int = 4) -> Image.Image:
    """Dilate the mask by ``pixels`` pixels (grows outward)."""
    m = _ensure_L(mask)
    if pixels <= 0:
        return m.copy()
    out = m
    step = ImageFilter.MaxFilter(3)
    for _ in range(int(pixels)):
        out = out.filter(step)
    return out


def contract_mask(mask: Image.Image, *, pixels: int = 4) -> Image.Image:
    """Erode the mask by ``pixels`` pixels (shrinks inward)."""
    m = _ensure_L(mask)
    if pixels <= 0:
        return m.copy()
    out = m
    step = ImageFilter.MinFilter(3)
    for _ in range(int(pixels)):
        out = out.filter(step)
    return out


def refine_mask(mask: Image.Image, *,
                fill_holes_px: int = 0,
                remove_islands_px: int = 0,
                feather_radius: float = 0.0,
                feather_method: str = "gaussian",
                source_image: Image.Image | None = None) -> Image.Image:
    """One-shot mask cleanup: fill small holes, remove small islands, feather.

    - ``fill_holes_px``: close holes smaller than this many pixels (closing op).
    - ``remove_islands_px``: drop isolated regions smaller than this (opening op).
    - ``feather_radius``: feather edges after the morphology cleanup.
    """
    m = _ensure_L(mask)
    if fill_holes_px > 0:
        m = expand_mask(m, pixels=fill_holes_px)
        m = contract_mask(m, pixels=fill_holes_px)
    if remove_islands_px > 0:
        m = contract_mask(m, pixels=remove_islands_px)
        m = expand_mask(m, pixels=remove_islands_px)
    if feather_radius > 0:
        m = feather_mask(m, radius=feather_radius,
                         method=feather_method, source_image=source_image)
    return m


def apply_mask_as_alpha(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Combine an RGB(A) image with a grayscale mask, producing an RGBA image
    where the mask becomes the alpha channel. Useful after feathering a
    segmentation mask to compose the cutout."""
    rgb = image.convert("RGB")
    m = _ensure_L(mask)
    if m.size != rgb.size:
        m = m.resize(rgb.size, Image.BILINEAR)
    r, g, b = rgb.split()
    return Image.merge("RGBA", (r, g, b, m))


_OVERLAY_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 0, 255),   # magenta
    (0, 255, 0),     # green
    (0, 200, 255),   # cyan-blue
    (255, 200, 0),   # amber
    (255, 0, 0),     # red
    (200, 100, 255), # violet
    (0, 255, 200),   # mint
    (255, 100, 100), # salmon
    (100, 200, 100), # sage
    (255, 150, 0),   # orange
)


def _parse_overlay_color(color: str | tuple | list | None,
                         default: tuple[int, int, int]) -> tuple[int, int, int]:
    """Loose color parser: hex / tuple / None. Used only for overlay tinting."""
    if color is None or color == "default":
        return default
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    if isinstance(color, str):
        s = color.lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) == 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    return default


def _mask_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    """Tight bbox of the non-zero region, or None if the mask is empty."""
    arr = np.asarray(_ensure_L(mask))
    ys, xs = np.nonzero(arr)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def magic_wand(image: Image.Image, x: int, y: int, *,
               tolerance: int = 16, contiguous: bool = True) -> Image.Image:
    """Select all pixels within ``tolerance`` of the colour at ``(x, y)``.
    Returns an L-mode mask the same size as ``image`` (255 = selected).

    - ``tolerance`` 0-255 is the max per-channel difference (PS's
      "Tolerance" slider).
    - ``contiguous=True`` (default) restricts to the connected region
      starting at ``(x, y)`` — like PS's Magic Wand. ``False`` selects
      every matching pixel regardless of connectivity (anti-magic-wand,
      Select > Colour Range).
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    h, w = rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"({x}, {y}) outside image {w}x{h}")
    target = rgb[int(y), int(x)]
    diff = np.max(np.abs(rgb - target), axis=-1)
    match = (diff <= int(tolerance)).astype(np.uint8)
    if not contiguous:
        return Image.fromarray(match * 255, mode="L")
    # Flood-fill the connected region from (x, y).
    try:
        import cv2
        # cv2.floodFill works on a single-channel uint8; build a 4-pad mask.
        flood = (1 - match).astype(np.uint8)  # 1 where NOT match
        # The mask passed to floodFill must be h+2, w+2.
        fmask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        fmask[1:-1, 1:-1] = flood
        # Fill from (x, y) onto a fresh canvas.
        canvas = np.zeros_like(match)
        cv2.floodFill(
            canvas, fmask, (int(x), int(y)), 1, loDiff=0, upDiff=0, flags=4,
        )
        return Image.fromarray(canvas * 255, mode="L")
    except ImportError:
        # Pure-numpy BFS flood fill (slower but always available).
        out = np.zeros_like(match)
        stack = [(int(y), int(x))]
        while stack:
            i, j = stack.pop()
            if not (0 <= i < h and 0 <= j < w):
                continue
            if out[i, j] or not match[i, j]:
                continue
            out[i, j] = 1
            stack.extend([(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])
        return Image.fromarray(out * 255, mode="L")


def blend_two(base: Image.Image, overlay: Image.Image, *,
              mode: str = "normal", opacity: float = 1.0
              ) -> Image.Image:
    """Composite ``overlay`` onto ``base`` with a chosen layer-stack blend
    mode. Both inputs are resized to ``base``'s size if they differ.
    Returns an RGBA image.

    ``mode`` ∈ the layer-stack blend modes (``normal | multiply | screen |
    overlay | darken | lighten | color_dodge | color_burn | hard_light |
    soft_light | difference | exclusion | add | subtract``). ``opacity``
    0-1 scales the overlay's contribution."""
    from . import layers as _layers
    base_rgba = base.convert("RGBA")
    ov = overlay.convert("RGBA")
    if ov.size != base_rgba.size:
        ov = ov.resize(base_rgba.size, Image.BILINEAR)
    if abs(opacity - 1.0) > 1e-3:
        # Pre-multiply alpha by opacity so the blend function honours it.
        a = ov.split()[-1].point(lambda v: int(v * max(0.0, min(1.0, opacity))))
        ov.putalpha(a)
    arr_b = np.asarray(base_rgba, dtype=np.uint8)
    arr_o = np.asarray(ov, dtype=np.uint8)
    blended = _layers._composite_over(arr_b, arr_o, mode)
    return Image.fromarray(blended, mode="RGBA")


def overlay_masks(
    image: Image.Image,
    masks: list[Image.Image],
    *,
    colors: list | None = None,
    alpha: float = 0.5,
    show_bbox: bool = True,
    bbox_width: int = 2,
    labels: list[str] | None = None,
    label_size: int = 14,
) -> Image.Image:
    """Composite one or more masks onto ``image`` as tinted overlays — a
    "screenshot" of the segmentation so you can judge mask quality before
    composing, feathering, or sending downstream.

    - ``masks``: list of L-mode images (or convertible). Resized to match
      ``image`` if needed.
    - ``colors``: per-mask RGB tuples or hex strings. ``None`` cycles
      through a 10-color palette.
    - ``alpha``: 0-1, overlay intensity. 0.5 keeps the original content
      visible under the tint.
    - ``show_bbox``: draw a 2-px rectangle around each mask's tight bbox.
    - ``labels``: optional list of strings drawn near each bbox corner.

    Returns an RGB image the same size as ``image``."""
    from PIL import ImageDraw, ImageFont
    base = image.convert("RGB")
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be 0-1, got {alpha}")
    if not masks:
        return base.copy()

    w, h = base.size
    out_arr = np.asarray(base, dtype=np.float32)
    bboxes: list[tuple[int, int, int, int] | None] = []
    for i, m in enumerate(masks):
        cs = colors[i] if (colors and i < len(colors)) else None
        rgb = _parse_overlay_color(cs, _OVERLAY_PALETTE[i % len(_OVERLAY_PALETTE)])
        m_L = _ensure_L(m)
        if m_L.size != (w, h):
            m_L = m_L.resize((w, h), Image.BILINEAR)
        bboxes.append(_mask_bbox(m_L))
        m_arr = np.asarray(m_L, dtype=np.float32) / 255.0
        weight = (m_arr * alpha)[..., None]
        tint = np.array(rgb, dtype=np.float32)[None, None, :]
        out_arr = out_arr * (1.0 - weight) + tint * weight

    out = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8), mode="RGB")

    if not (show_bbox or labels):
        return out

    drw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", int(label_size))
    except Exception:
        font = ImageFont.load_default()
    for i, bb in enumerate(bboxes):
        if bb is None:
            continue
        cs = colors[i] if (colors and i < len(colors)) else None
        rgb = _parse_overlay_color(cs, _OVERLAY_PALETTE[i % len(_OVERLAY_PALETTE)])
        x1, y1, x2, y2 = bb
        if show_bbox:
            drw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=rgb,
                          width=max(1, int(bbox_width)))
        text = labels[i] if (labels and i < len(labels)) else None
        if text:
            pad = 3
            try:
                bx, by, ex, ey = drw.textbbox((x1 + pad, y1 + pad), text,
                                              font=font)
                drw.rectangle([bx - pad, by - pad, ex + pad, ey + pad],
                              fill=(0, 0, 0))
            except Exception:
                pass
            drw.text((x1 + pad, y1 + pad), text, fill=rgb, font=font)
    return out
