"""Stateless file-in / file-out editing.

A pipeline of operations is applied to a loaded image and the result written
back to disk in one tool call — no canvas_id, no undo stack, no session state.
Every op is a dict with an ``op`` key naming the helper to invoke and the
remaining keys passed as kwargs.

Ops dispatch to the same drawing/transforms helpers the canvas tools use, so
behaviour is identical between stateless and canvas-based workflows.

Example operations list::

    [
        {"op": "resize", "width": 800, "height": 600},
        {"op": "adjust", "brightness": 1.1, "contrast": 1.2},
        {"op": "draw_text", "x": 10, "y": 10, "text": "© 2026", "size": 24,
         "color": "white", "stroke_width": 1, "stroke_color": "black"},
    ]
"""
from __future__ import annotations

from typing import Any, Callable

from PIL import Image

from . import drawing, transforms


# Two flavours of helper:
# - "draw" ops mutate the image in place (drawing.* return the same object).
# - "transform" ops return a new image (potentially with new dimensions).
# The dispatcher handles both by always rebinding ``img`` to whatever the
# helper returned.

_DRAW_OPS: dict[str, Callable[..., Image.Image]] = {
    "draw_pixel": drawing.draw_pixel,
    "draw_line": drawing.draw_line,
    "draw_rectangle": drawing.draw_rectangle,
    "draw_ellipse": drawing.draw_ellipse,
    "draw_polygon": drawing.draw_polygon,
    "draw_arc": drawing.draw_arc,
    "draw_text": drawing.draw_text,
    "draw_brush": drawing.draw_brush,
    "eraser": drawing.eraser,
    "flood_fill": drawing.flood_fill,
}

_TRANSFORM_OPS: dict[str, Callable[..., Image.Image]] = {
    "crop": transforms.crop,
    "resize": transforms.resize,
    "rotate": transforms.rotate,
    "flip": transforms.flip,
    "clear_region": transforms.clear_region,
    "apply_filter": transforms.apply_filter,
    "adjust": transforms.adjust,
    "invert": transforms.invert,
    "grayscale": transforms.grayscale,
    "posterize": transforms.posterize,
    "add_border": transforms.add_border,
}

ALL_OPS = sorted(set(_DRAW_OPS) | set(_TRANSFORM_OPS) | {"thumbnail"})


def _apply_op(img: Image.Image, spec: dict[str, Any]) -> Image.Image:
    """Dispatch one op dict against the helper registries."""
    if not isinstance(spec, dict) or "op" not in spec:
        raise ValueError(
            f"each operation must be a dict with an 'op' key, got {spec!r}"
        )
    name = spec["op"]
    kwargs = {k: v for k, v in spec.items() if k != "op"}

    if name == "thumbnail":
        # Common-case helper: scale to fit within max_size while preserving
        # aspect. Not in drawing/transforms because it doesn't take fixed dims.
        max_size = int(kwargs.get("max_size", 512))
        if img.width <= max_size and img.height <= max_size:
            return img.copy()
        ratio = max_size / max(img.width, img.height)
        return img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
            Image.LANCZOS,
        )

    if name in _DRAW_OPS:
        # Draw ops mutate; first arg is the image, rest are positional/keyword.
        return _DRAW_OPS[name](img, **kwargs)

    if name in _TRANSFORM_OPS:
        return _TRANSFORM_OPS[name](img, **kwargs)

    raise ValueError(
        f"unknown op {name!r}. Available: {ALL_OPS}"
    )


def apply_pipeline(img: Image.Image, operations: list[dict[str, Any]]) -> Image.Image:
    """Run ``operations`` in order. Each op sees the result of the previous."""
    if not isinstance(operations, list):
        raise ValueError("operations must be a list of dicts")
    out = img
    for i, spec in enumerate(operations):
        try:
            out = _apply_op(out, spec)
        except Exception as e:
            raise ValueError(
                f"operation {i} ({spec.get('op', '?') if isinstance(spec, dict) else '?'}) "
                f"failed: {type(e).__name__}: {e}"
            ) from e
    return out
