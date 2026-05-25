"""ControlNet preprocessors — turn a regular image into the conditioning
image a ControlNet expects (Canny edges, depth maps, normal maps, etc.).

Today: canny edges + simple grayscale-as-depth. Future home for openpose,
zoe-depth, normal maps, MLSD line detection, etc.

These are stateless image ops — they take a ``PIL.Image`` and return a
new ``PIL.Image``. They don't touch the canvas store directly; the MCP
layer wires that up.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def canny_edges(image: Image.Image, *, low: int = 100, high: int = 200,
                blur_radius: int = 0) -> Image.Image:
    """Canny edge detection. Returns an RGB image (the diffusers ControlNet
    pipelines expect a 3-channel input even when the content is monochrome).

    ``low`` / ``high`` are the hysteresis thresholds. Defaults match the
    ``lllyasviel/sd-controlnet-canny`` reference implementation.

    ``blur_radius`` optionally pre-blurs the input (gaussian, 1-px sigma per
    unit). 0 disables, 1-3 is typical for noisy / high-frequency inputs.
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "canny_edges needs opencv-python. "
            "`pip install opencv-python` or install the [yolo] / [seg] extra."
        ) from e

    low = int(low)
    high = int(high)
    if not (0 <= low < high <= 255):
        raise ValueError(
            f"canny thresholds must satisfy 0 <= low < high <= 255, "
            f"got low={low}, high={high}"
        )
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if blur_radius and blur_radius > 0:
        # cv2.GaussianBlur kernel must be odd; sigma=0 lets cv2 derive it.
        k = max(1, int(blur_radius) * 2 + 1)
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(gray, int(low), int(high))
    out = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(out, mode="RGB")


def depth_from_grayscale(image: Image.Image, *, invert: bool = False
                         ) -> Image.Image:
    """Cheap depth-like map — just the luminance, optionally inverted. Useful
    as a placeholder ControlNet input when you don't have a real depth
    model loaded. For accurate depth use a model like Marigold or ZoeDepth
    (TODO — add as separate preprocessor)."""
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    if invert:
        arr = 255 - arr
    out = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(out, mode="RGB")
