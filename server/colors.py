"""Color parsing.

Tools accept color in several forms — a CSS-style hex string ("#ff8800"), a
named color ("red", "transparent"), or an [R,G,B] / [R,G,B,A] tuple. Normalize
to a 4-tuple so downstream drawing code never branches on input form.
"""
from __future__ import annotations

from typing import Any

from PIL import ImageColor

Color = tuple[int, int, int, int]


def parse_color(value: Any, *, default_alpha: int = 255) -> Color:
    if value is None:
        raise ValueError("color is required")

    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("transparent", "none", "clear"):
            return (0, 0, 0, 0)
        # ImageColor.getcolor(.., "RGBA") always returns a 4-tuple; defensive
        # length check covers older Pillow versions that returned 3-tuples.
        rgba = ImageColor.getcolor(s, "RGBA")
        if len(rgba) == 3:
            return (rgba[0], rgba[1], rgba[2], default_alpha)
        return (rgba[0], rgba[1], rgba[2], rgba[3])

    if isinstance(value, (list, tuple)):
        if len(value) == 3:
            r, g, b = (_clamp_channel(c) for c in value)
            return (r, g, b, default_alpha)
        if len(value) == 4:
            r, g, b, a = (_clamp_channel(c) for c in value)
            return (r, g, b, a)
        raise ValueError(f"color tuple must have 3 or 4 ints, got {len(value)}")

    raise TypeError(f"unsupported color type: {type(value).__name__}")


def _clamp_channel(c: Any) -> int:
    """Coerce to int and clamp to 0..255. Out-of-range values would otherwise
    propagate into Pillow / numpy and produce garbage colors."""
    return max(0, min(255, int(c)))
