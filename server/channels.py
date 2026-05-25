"""Channel ops: extract / merge / split-to-layers.

Photoshop's Channels panel lets you treat the R, G, B, and A planes as
independent grayscale documents — useful for masks, advanced compositing,
and color-correction tricks.
"""
from __future__ import annotations

from typing import Literal

from PIL import Image


CHANNELS = ["R", "G", "B", "A", "L"]


def extract_channel(img: Image.Image,
                    channel: Literal["R", "G", "B", "A", "L"]) -> Image.Image:
    """Pull a single channel out as a grayscale (``L``) image of the same size.

    ``L`` returns the BT.601 luminance, not Pillow's perceived-luminance
    (which is what ``ImageOps.grayscale`` produces) so it matches the same
    formula used by adjustments / threshold / vibrance internally.
    """
    rgba = img.convert("RGBA")
    if channel == "R":
        return rgba.getchannel("R")
    if channel == "G":
        return rgba.getchannel("G")
    if channel == "B":
        return rgba.getchannel("B")
    if channel == "A":
        return rgba.getchannel("A")
    if channel == "L":
        import numpy as np

        arr = np.asarray(rgba, dtype=np.float32)
        lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return Image.fromarray(lum.astype("uint8"), mode="L")
    raise ValueError(f"unknown channel {channel!r}. Choose from {CHANNELS}.")


def merge_channels(r: Image.Image, g: Image.Image, b: Image.Image,
                   a: Image.Image | None = None) -> Image.Image:
    """Combine grayscale channel images into one RGB / RGBA image.

    All inputs must be the same size. They're converted to ``L`` first so
    the caller can feed RGB images directly (in which case we use each one's
    luminance) — convenient when iterating in pixel space.
    """
    size = r.size
    if not (g.size == b.size == size):
        raise ValueError(
            f"channel sizes must match: R={r.size}, G={g.size}, B={b.size}"
        )
    if a is not None and a.size != size:
        raise ValueError(f"alpha size {a.size} != channel size {size}")
    R = r.convert("L")
    G = g.convert("L")
    B = b.convert("L")
    if a is None:
        return Image.merge("RGB", (R, G, B))
    A = a.convert("L")
    return Image.merge("RGBA", (R, G, B, A))


def split_to_grayscales(img: Image.Image) -> dict[str, Image.Image]:
    """Return the four channels as a dict: R, G, B, A (always 4 entries).
    Single image with no alpha gets an all-255 A."""
    rgba = img.convert("RGBA")
    return {
        "R": rgba.getchannel("R"),
        "G": rgba.getchannel("G"),
        "B": rgba.getchannel("B"),
        "A": rgba.getchannel("A"),
    }
